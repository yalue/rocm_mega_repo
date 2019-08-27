/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "inc/roctracer.h"
#include "inc/roctracer_hcc.h"
#include "inc/roctracer_hip.h"
#define PROF_API_IMPL 1
#include "inc/roctracer_hsa.h"

#include <atomic>
#include <mutex>
#include <dirent.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>

#include "core/loader.h"
#include "proxy/tracker.h"
#include "ext/hsa_rt_utils.hpp"
#include "util/exception.h"
#include "util/hsa_rsrc_factory.h"
#include "util/logger.h"

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

#define PTHREAD_CALL(call)                                                                         \
  do {                                                                                             \
    int err = call;                                                                                \
    if (err != 0) {                                                                                \
      errno = err;                                                                                 \
      perror(#call);                                                                               \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

#define HIPAPI_CALL(call)                                                                          \
  do {                                                                                             \
    hipError_t err = call;                                                                         \
    if (err != hipSuccess)                                                                         \
      HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, #call " error(" << err << ")");                \
  } while (0)

#define API_METHOD_PREFIX                                                                          \
  roctracer_status_t err = ROCTRACER_STATUS_SUCCESS;                                               \
  try {

#define API_METHOD_SUFFIX                                                                          \
  }                                                                                                \
  catch (std::exception & e) {                                                                     \
    ERR_LOGGING(__FUNCTION__ << "(), " << e.what());                                               \
    err = roctracer::GetExcStatus(e);                                                              \
  }                                                                                                \
  return err;

#define API_METHOD_CATCH(X)                                                                        \
  }                                                                                                \
  catch (std::exception & e) {                                                                     \
    ERR_LOGGING(__FUNCTION__ << "(), " << e.what());                                               \
  }                                                                                                \
  (void)err;                                                                                       \
  return X;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Internal library methods
//
namespace roctracer {
decltype(hsa_amd_memory_async_copy)* hsa_amd_memory_async_copy_fn;
decltype(hsa_amd_memory_async_copy_rect)* hsa_amd_memory_async_copy_rect_fn;

namespace hsa_support {
// callbacks table
cb_table_t cb_table;
// asyc copy activity callback
activity_async_callback_t async_copy_callback_fun = NULL;
void* async_copy_callback_arg = NULL;
bool async_copy_callback_enabled = false;
// Table of function pointers to HSA Core Runtime
CoreApiTable CoreApiTable_saved{};
// Table of function pointers to AMD extensions
AmdExtTable AmdExtTable_saved{};
// Table of function pointers to HSA Image Extension
ImageExtTable ImageExtTable_saved{};
}

roctracer_status_t GetExcStatus(const std::exception& e) {
  const util::exception* roctracer_exc_ptr = dynamic_cast<const util::exception*>(&e);
  return (roctracer_exc_ptr) ? static_cast<roctracer_status_t>(roctracer_exc_ptr->status()) : ROCTRACER_STATUS_ERROR;
}

class GlobalCounter {
  public:
  typedef std::mutex mutex_t;
  typedef uint64_t counter_t;

  static counter_t Increment() {
    std::lock_guard<mutex_t> lock(mutex_);
    return ++counter_;
  }

  private:
  static mutex_t mutex_;
  static counter_t counter_;
};
GlobalCounter::mutex_t GlobalCounter::mutex_;
GlobalCounter::counter_t GlobalCounter::counter_ = 0;

class MemoryPool {
  public:
  typedef std::mutex mutex_t;

  static void allocator_default(char** ptr, size_t size, void* arg) {
    (void)arg;
    if (*ptr == NULL) {
      *ptr = reinterpret_cast<char*>(malloc(size));
    } else if (size != 0) {
      *ptr = reinterpret_cast<char*>(realloc(ptr, size));
    } else {
      free(*ptr); 
      *ptr = NULL;
    }
  }

  MemoryPool(const roctracer_properties_t& properties) { 
    // Assigning pool allocator
    alloc_fun_ = allocator_default;
    alloc_arg_ = NULL;
    if (properties.alloc_fun != NULL) {
      alloc_fun_ = properties.alloc_fun;
      alloc_arg_ = properties.alloc_arg;
    }

    // Pool definition
    buffer_size_ = properties.buffer_size;
    const size_t pool_size = 2 * buffer_size_;
    pool_begin_ = NULL;
    alloc_fun_(&pool_begin_, pool_size, alloc_arg_);
    if (pool_begin_ == NULL) EXC_ABORT(ROCTRACER_STATUS_ERROR, "pool allocator failed");
    pool_end_ = pool_begin_ + pool_size;
    buffer_begin_ = pool_begin_;
    buffer_end_ = buffer_begin_ + buffer_size_;
    write_ptr_ = buffer_begin_;

    // Consuming read thread
    read_callback_fun_ = properties.buffer_callback_fun;
    read_callback_arg_ = properties.buffer_callback_arg;
    consumer_arg_.set(this, NULL, NULL, true);
    PTHREAD_CALL(pthread_mutex_init(&read_mutex_, NULL));
    PTHREAD_CALL(pthread_cond_init(&read_cond_, NULL));
    PTHREAD_CALL(pthread_create(&consumer_thread_, NULL, reader_fun, &consumer_arg_));
  }

  ~MemoryPool() {
    Flush();
    PTHREAD_CALL(pthread_cancel(consumer_thread_));
    void *res;
    PTHREAD_CALL(pthread_join(consumer_thread_, &res));
    if (res != PTHREAD_CANCELED) EXC_ABORT(ROCTRACER_STATUS_ERROR, "consumer thread wasn't stopped correctly");
    allocator_default(&pool_begin_, 0, alloc_arg_);
  }

  template <typename Record>
  void Write(const Record& record) {
    std::lock_guard<mutex_t> lock(write_mutex_);
    getRecord<Record>(record);
  }

  void Flush() {
    std::lock_guard<mutex_t> lock(write_mutex_);
    if (write_ptr_ > buffer_begin_) {
      spawn_reader(buffer_begin_, write_ptr_);
      sync_reader(&consumer_arg_);
      buffer_begin_ = (buffer_end_ == pool_end_) ? pool_begin_ : buffer_end_;
      buffer_end_ = buffer_begin_ + buffer_size_;
      write_ptr_ = buffer_begin_;
    }
  }

  private:
  struct consumer_arg_t {
    MemoryPool* obj;
    const char* begin;
    const char* end;
    volatile std::atomic<bool> valid;
    void set(MemoryPool* obj_p, const char* begin_p, const char* end_p, bool valid_p) {
      obj = obj_p;
      begin = begin_p;
      end = end_p;
      valid.store(valid_p);
    }
  };

  template <typename Record>
  Record* getRecord(const Record& init) {
    char* next = write_ptr_ + sizeof(Record);
    if (next > buffer_end_) {
      if (write_ptr_ == buffer_begin_) EXC_ABORT(ROCTRACER_STATUS_ERROR, "buffer size(" << buffer_size_ << ") is less then the record(" << sizeof(Record) << ")");
      spawn_reader(buffer_begin_, write_ptr_);
      buffer_begin_ = (buffer_end_ == pool_end_) ? pool_begin_ : buffer_end_;
      buffer_end_ = buffer_begin_ + buffer_size_;
      write_ptr_ = buffer_begin_;
      next = write_ptr_ + sizeof(Record);
    }

    Record* ptr = reinterpret_cast<Record*>(write_ptr_);
    write_ptr_ = next;

    *ptr = init;
    return ptr;
  }

  static void reset_reader(consumer_arg_t* arg) {
    arg->valid.store(false);
  }

  static void sync_reader(const consumer_arg_t* arg) {
    while(arg->valid.load() == true) PTHREAD_CALL(pthread_yield());
  }

  static void* reader_fun(void* consumer_arg) {
    consumer_arg_t* arg = reinterpret_cast<consumer_arg_t*>(consumer_arg);
    roctracer::MemoryPool* obj = arg->obj;

    reset_reader(arg);

    while (1) {
      PTHREAD_CALL(pthread_mutex_lock(&(obj->read_mutex_)));
      while (arg->valid.load() == false) {
        PTHREAD_CALL(pthread_cond_wait(&(obj->read_cond_), &(obj->read_mutex_)));
      }

      obj->read_callback_fun_(arg->begin, arg->end, obj->read_callback_arg_);
      reset_reader(arg);
      PTHREAD_CALL(pthread_mutex_unlock(&(obj->read_mutex_)));
    }

    return NULL;
  }

  void spawn_reader(const char* data_begin, const char* data_end) {
    sync_reader(&consumer_arg_);
    PTHREAD_CALL(pthread_mutex_lock(&read_mutex_));
    consumer_arg_.set(this, data_begin, data_end, true);
    PTHREAD_CALL(pthread_cond_signal(&read_cond_));
    PTHREAD_CALL(pthread_mutex_unlock(&read_mutex_));
  }

  // pool allocator
  roctracer_allocator_t alloc_fun_;
  void* alloc_arg_;

  // Pool definition
  size_t buffer_size_;
  char* pool_begin_;
  char* pool_end_;
  char* buffer_begin_;
  char* buffer_end_;
  char* write_ptr_;
  mutex_t write_mutex_;

  // Consuming read thread
  roctracer_buffer_callback_t read_callback_fun_;
  void* read_callback_arg_;
  consumer_arg_t consumer_arg_;
  pthread_t consumer_thread_;
  pthread_mutex_t read_mutex_;
  pthread_cond_t read_cond_;
};

CONSTRUCTOR_API void constructor() {
  util::Logger::Create();
}

DESTRUCTOR_API void destructor() {
  ::util::HsaRsrcFactory::Destroy();
  util::Logger::Destroy();
}

// Correlation id storage
static thread_local activity_correlation_id_t correlation_id_tls = 0;
typedef std::map<activity_correlation_id_t, activity_correlation_id_t> correlation_id_map_t;
typedef std::mutex correlation_id_mutex_t;
correlation_id_map_t* correlation_id_map = NULL;
correlation_id_mutex_t correlation_id_mutex;

static inline void CorrelationIdRegistr(const activity_correlation_id_t& correlation_id) {
  std::lock_guard<correlation_id_mutex_t> lck(correlation_id_mutex);
  if (correlation_id_map == NULL) correlation_id_map = new correlation_id_map_t;
  const auto ret = correlation_id_map->insert({correlation_id, correlation_id_tls});
  if (ret.second == false) EXC_ABORT(ROCTRACER_STATUS_ERROR, "HCC activity id is not unique(" << correlation_id << ")");
}

static inline activity_correlation_id_t CorrelationIdLookup(const activity_correlation_id_t& correlation_id) {
  auto it = correlation_id_map->find(correlation_id);
  if (it == correlation_id_map->end()) EXC_ABORT(ROCTRACER_STATUS_ERROR, "HCC activity id lookup failed(" << correlation_id << ")");
  return it->second;
}

roctracer_record_t* HIP_SyncActivityCallback(
    uint32_t op_id,
    roctracer_record_t* record,
    const void* callback_data,
    void* arg)
{
  static hsa_rt_utils::Timer timer;

  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
  MemoryPool* pool = reinterpret_cast<MemoryPool*>(arg);
  if (pool == NULL) EXC_ABORT(ROCTRACER_STATUS_ERROR, "ActivityCallback pool is NULL");
  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    record->domain = ACTIVITY_DOMAIN_HIP_API;
    record->op = op_id;
    record->begin_ns = timer.timestamp_ns();
    // Correlation ID generating
    uint64_t correlation_id = data->correlation_id;
    if (correlation_id == 0) {
      correlation_id = GlobalCounter::Increment();
      const_cast<hip_api_data_t*>(data)->correlation_id = correlation_id;
    }
    record->correlation_id = correlation_id;
    // Passing correlatin ID
    correlation_id_tls = correlation_id;
    return record;
  } else {
    record->end_ns = timer.timestamp_ns();
    record->process_id = syscall(__NR_getpid);
    record->thread_id = syscall(__NR_gettid);
    pool->Write(*record);
    // Clearing correlatin ID
    correlation_id_tls = 0;
    return NULL;
  }
}

void HCC_ActivityIdCallback(activity_correlation_id_t correlation_id) {
  CorrelationIdRegistr(correlation_id);
}

void HCC_AsyncActivityCallback(uint32_t op_id, void* record, void* arg) {
  static hsa_rt_utils::Timer timer;

  MemoryPool* pool = reinterpret_cast<MemoryPool*>(arg);
  roctracer_record_t* record_ptr = reinterpret_cast<roctracer_record_t*>(record);
  record_ptr->domain = ACTIVITY_DOMAIN_HCC_OPS;
  record_ptr->correlation_id = CorrelationIdLookup(record_ptr->correlation_id);
  pool->Write(*record_ptr);
}

bool hsa_async_copy_handler(hsa_signal_value_t value, void* arg) {
  ::proxy::Tracker::entry_t* entry = reinterpret_cast<::proxy::Tracker::entry_t*>(arg);

  activity_record_t record{};
  record.domain = ACTIVITY_DOMAIN_HSA_OPS;   // activity domain id
  record.correlation_id = entry->index;      // activity ID
  record.begin_ns = entry->record->begin;    // host begin timestamp
  record.end_ns = entry->record->end;        // host end timestamp
  record.device_id = 0;                      // device id

  hsa_support::async_copy_callback_fun(hsa_support::HSA_OP_ID_async_copy, &record, hsa_support::async_copy_callback_arg);
  return false;
}

hsa_status_t hsa_amd_memory_async_copy_interceptor(
    void* dst, hsa_agent_t dst_agent, const void* src,
    hsa_agent_t src_agent, size_t size, uint32_t num_dep_signals,
    const hsa_signal_t* dep_signals, hsa_signal_t completion_signal)
{
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (hsa_support::async_copy_callback_enabled) {
    ::proxy::Tracker* tracker = &::proxy::Tracker::Instance();
    ::proxy::Tracker::entry_t* tracker_entry = tracker->Alloc(hsa_agent_t{}, completion_signal);
    status = hsa_amd_memory_async_copy_fn(dst, dst_agent, src,
                                          src_agent, size, num_dep_signals,
                                          dep_signals, tracker_entry->signal);
    if (status == HSA_STATUS_SUCCESS) {
      tracker->EnableMemcopy(tracker_entry, hsa_async_copy_handler, reinterpret_cast<void*>(tracker_entry));
    } else {
      tracker->Delete(tracker_entry);
    }
  } else {
    status = hsa_amd_memory_async_copy_fn(dst, dst_agent, src,
                                          src_agent, size, num_dep_signals,
                                          dep_signals, completion_signal);
  }
  return status;
}

hsa_status_t hsa_amd_memory_async_copy_rect_interceptor(
    const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset, const hsa_pitched_ptr_t* src,
    const hsa_dim3_t* src_offset, const hsa_dim3_t* range, hsa_agent_t copy_agent,
    hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal)
{
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (hsa_support::async_copy_callback_enabled) {
    ::proxy::Tracker* tracker = &::proxy::Tracker::Instance();
    ::proxy::Tracker::entry_t* tracker_entry = tracker->Alloc(hsa_agent_t{}, completion_signal);
    status = hsa_amd_memory_async_copy_rect_fn(dst, dst_offset, src,
                                               src_offset, range, copy_agent,
                                               dir, num_dep_signals, dep_signals,
                                               tracker_entry->signal);
    if (status == HSA_STATUS_SUCCESS) {
      tracker->EnableMemcopy(tracker_entry, hsa_async_copy_handler, reinterpret_cast<void*>(tracker_entry));
    } else {
      tracker->Delete(tracker_entry);
    }
  } else {
    status = hsa_amd_memory_async_copy_rect_fn(dst, dst_offset, src,
                                               src_offset, range, copy_agent,
                                               dir, num_dep_signals, dep_signals,
                                               completion_signal);
  }
  return status;
}

util::Logger::mutex_t util::Logger::mutex_;
std::atomic<util::Logger*> util::Logger::instance_{};
MemoryPool* memory_pool = NULL;
typedef std::recursive_mutex memory_pool_mutex_t;
memory_pool_mutex_t memory_pool_mutex;
}

LOADER_INSTANTIATE();

std::atomic<proxy::Tracker*> proxy::Tracker::instance_{};
proxy::Tracker::mutex_t proxy::Tracker::glob_mutex_;
proxy::Tracker::counter_t proxy::Tracker::counter_ = 0;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Public library methods
//
extern "C" {

// Returns library vesrion
PUBLIC_API uint32_t roctracer_version_major() { return ROCTRACER_VERSION_MAJOR; }
PUBLIC_API uint32_t roctracer_version_minor() { return ROCTRACER_VERSION_MINOR; }

// Returns the last error
PUBLIC_API const char* roctracer_error_string() {
  return strdup(roctracer::util::Logger::LastMessage().c_str());
}

// Return Op string by given domain and activity/API codes
// NULL returned on the error and the library errno is set
PUBLIC_API const char* roctracer_op_string(
    uint32_t domain,
    uint32_t op,
    uint32_t kind)
{
  API_METHOD_PREFIX
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_API: {
      return roctracer::hsa_support::GetApiName(op);
      break;
    }
    case ACTIVITY_DOMAIN_HCC_OPS: {
      return roctracer::HccLoader::Instance().GetCmdName(kind);
      break;
    }
    case ACTIVITY_DOMAIN_HIP_API: {
      return hipApiName(op);
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
  API_METHOD_CATCH(NULL)
}

// Return Op code and kind by given string
PUBLIC_API roctracer_status_t roctracer_op_code(
    uint32_t domain,
    const char* str,
    uint32_t* op,
    uint32_t* kind)
{
  API_METHOD_PREFIX
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_API: {
      *op = roctracer::hsa_support::GetApiCode(str);
      if (kind != NULL) *kind = 0;
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "limited domain ID(" << domain << ")");
  }
  API_METHOD_SUFFIX
}

static inline uint32_t get_op_num(const uint32_t& domain) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS: return 1;
    case ACTIVITY_DOMAIN_HSA_API: return HSA_API_ID_NUMBER;
    case ACTIVITY_DOMAIN_HCC_OPS: return hc::HSA_OP_ID_NUMBER;
    case ACTIVITY_DOMAIN_HIP_API: return HIP_API_ID_NUMBER;
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
  return 0;
}

// Enable runtime API callbacks
static void roctracer_enable_callback_impl(
    uint32_t domain,
    uint32_t op,
    roctracer_rtapi_callback_t callback,
    void* user_data)
{
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS: break;
    case ACTIVITY_DOMAIN_HSA_API: {
      roctracer::hsa_support::cb_table.set(op, callback, user_data);
      break;
    }
    case ACTIVITY_DOMAIN_HCC_OPS: break;
    case ACTIVITY_DOMAIN_HIP_API: {
      hipError_t hip_err = roctracer::HipLoader::Instance().RegisterApiCallback(op, (void*)callback, user_data);
      if (hip_err != hipSuccess) HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, "hipRegisterApiCallback error(" << hip_err << ")");
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
}

PUBLIC_API roctracer_status_t roctracer_enable_op_callback(
    roctracer_domain_t domain,
    uint32_t op,
    roctracer_rtapi_callback_t callback,
    void* user_data)
{
  API_METHOD_PREFIX
  roctracer_enable_callback_impl(domain, op, callback, user_data);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_enable_domain_callback(
    roctracer_domain_t domain,
    roctracer_rtapi_callback_t callback,
    void* user_data)
{
  API_METHOD_PREFIX
  const uint32_t op_num = get_op_num(domain);
  for (uint32_t op = 0; op < op_num; op++) roctracer_enable_callback_impl(domain, op, callback, user_data);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_enable_callback(
    roctracer_rtapi_callback_t callback,
    void* user_data)
{
  API_METHOD_PREFIX
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; domain++) {
    const uint32_t op_num = get_op_num(domain);
    for (uint32_t op = 0; op < op_num; op++) roctracer_enable_callback_impl(domain, op, callback, user_data);
  }
  API_METHOD_SUFFIX
}

// Disable runtime API callbacks
static void roctracer_disable_callback_impl(
    uint32_t domain,
    uint32_t op)
{
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS: break;
    case ACTIVITY_DOMAIN_HSA_API: break;
    case ACTIVITY_DOMAIN_HCC_OPS: break;
    case ACTIVITY_DOMAIN_HIP_API: {
      hipError_t hip_err = roctracer::HipLoader::Instance().RemoveApiCallback(op);
      if (hip_err != hipSuccess) HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, "hipRemoveApiCallback error(" << hip_err << ")");
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
}

PUBLIC_API roctracer_status_t roctracer_disable_op_callback(
    roctracer_domain_t domain,
    uint32_t op)
{
  API_METHOD_PREFIX
  roctracer_disable_callback_impl(domain, op);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_disable_domain_callback(
    roctracer_domain_t domain)
{
  API_METHOD_PREFIX
  const uint32_t op_num = get_op_num(domain);
  for (uint32_t op = 0; op < op_num; op++) roctracer_disable_callback_impl(domain, op);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_disable_callback()
{
  API_METHOD_PREFIX
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; domain++) {
    const uint32_t op_num = get_op_num(domain);
    for (uint32_t op = 0; op < op_num; op++) roctracer_disable_callback_impl(domain, op);
  }
  API_METHOD_SUFFIX
}

// Return default pool and set new one if parameter pool is not NULL.
PUBLIC_API roctracer_pool_t* roctracer_default_pool(roctracer_pool_t* pool) {
  std::lock_guard<roctracer::memory_pool_mutex_t> lock(roctracer::memory_pool_mutex);
  roctracer_pool_t* p = reinterpret_cast<roctracer_pool_t*>(roctracer::memory_pool);
  if (pool != NULL) roctracer::memory_pool = reinterpret_cast<roctracer::MemoryPool*>(pool);
  return p;
}

// Open memory pool
PUBLIC_API roctracer_status_t roctracer_open_pool(
    const roctracer_properties_t* properties,
    roctracer_pool_t** pool)
{
  API_METHOD_PREFIX
  std::lock_guard<roctracer::memory_pool_mutex_t> lock(roctracer::memory_pool_mutex);
  if ((pool == NULL) && (roctracer::memory_pool != NULL)) {
    EXC_RAISING(ROCTRACER_STATUS_ERROR, "default pool already set");
  }
  roctracer::MemoryPool* p = new roctracer::MemoryPool(*properties);
  if (p == NULL) EXC_RAISING(ROCTRACER_STATUS_ERROR, "MemoryPool() error");
  if (pool != NULL) *pool = p;
  else roctracer::memory_pool = p;
  API_METHOD_SUFFIX
}

// Close memory pool
PUBLIC_API roctracer_status_t roctracer_close_pool(roctracer_pool_t* pool) {
  API_METHOD_PREFIX
  std::lock_guard<roctracer::memory_pool_mutex_t> lock(roctracer::memory_pool_mutex);
  roctracer_pool_t* ptr = (pool == NULL) ? roctracer_default_pool() : pool;
  roctracer::MemoryPool* memory_pool = reinterpret_cast<roctracer::MemoryPool*>(ptr);
  delete(memory_pool);
  if (pool == NULL) roctracer::memory_pool = NULL;
  API_METHOD_SUFFIX
}

// Enable activity records logging
static void roctracer_enable_activity_impl(
    uint32_t domain,
    uint32_t op,
    roctracer_pool_t* pool)
{
  if (pool == NULL) pool = roctracer_default_pool();
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS: {
      roctracer::hsa_support::async_copy_callback_enabled = true;
      break;
    }
    case ACTIVITY_DOMAIN_HSA_API: break;
    case ACTIVITY_DOMAIN_HCC_OPS: {
      if (roctracer::HccLoader::GetRef() == NULL) {
        roctracer::HccLoader::Instance().InitActivityCallback((void*)roctracer::HCC_ActivityIdCallback,
                                                              (void*)roctracer::HCC_AsyncActivityCallback,
                                                              (void*)pool);
      }
      const bool succ = roctracer::HccLoader::Instance().EnableActivityCallback(op, true);
      if (succ == false) HCC_EXC_RAISING(ROCTRACER_STATUS_HCC_OPS_ERR, "HCC::EnableActivityCallback error");
      break;
    }
    case ACTIVITY_DOMAIN_HIP_API: {
      const hipError_t hip_err = roctracer::HipLoader::Instance().RegisterActivityCallback(op, (void*)roctracer::HIP_SyncActivityCallback, (void*)pool);
      if (hip_err != hipSuccess) HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, "hipRegisterActivityCallback error(" << hip_err << ")");
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
}

PUBLIC_API roctracer_status_t roctracer_enable_op_activity(
    roctracer_domain_t domain,
    uint32_t op,
    roctracer_pool_t* pool)
{
  API_METHOD_PREFIX
  roctracer_enable_activity_impl(domain, op, pool);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_enable_domain_activity(
    roctracer_domain_t domain,
    roctracer_pool_t* pool)
{
  API_METHOD_PREFIX
  const uint32_t op_num = get_op_num(domain);
  for (uint32_t op = 0; op < op_num; op++) roctracer_enable_activity_impl(domain, op, pool);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_enable_activity(
    roctracer_pool_t* pool)
{
  API_METHOD_PREFIX
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; domain++) {
    const uint32_t op_num = get_op_num(domain);
    for (uint32_t op = 0; op < op_num; op++) roctracer_enable_activity_impl(domain, op, pool);
  }
  API_METHOD_SUFFIX
}

// Disable activity records logging
static void roctracer_disable_activity_impl(
    uint32_t domain,
    uint32_t op)
{
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS: {
      roctracer::hsa_support::async_copy_callback_enabled = false;
      break;
    }
    case ACTIVITY_DOMAIN_HSA_API: break;
    case ACTIVITY_DOMAIN_HCC_OPS: {
      const bool succ = roctracer::HccLoader::Instance().EnableActivityCallback(op, false);
      if (succ == false) HCC_EXC_RAISING(ROCTRACER_STATUS_HCC_OPS_ERR, "HCC::EnableActivityCallback(NULL) error domain(" << domain << ") op(" << op << ")");
      break;
    }
    case ACTIVITY_DOMAIN_HIP_API: {
      const hipError_t hip_err = roctracer::HipLoader::Instance().RemoveActivityCallback(op);
      if (hip_err != hipSuccess) HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, "hipRemoveActivityCallback error(" << hip_err << ")");
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
}

PUBLIC_API roctracer_status_t roctracer_disable_op_activity(
    roctracer_domain_t domain,
    uint32_t op)
{
  API_METHOD_PREFIX
  roctracer_disable_activity_impl(domain, op);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_disable_domain_activity(
    roctracer_domain_t domain)
{
  API_METHOD_PREFIX
  const uint32_t op_num = get_op_num(domain);
  for (uint32_t op = 0; op < op_num; op++) roctracer_disable_activity_impl(domain, op);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_disable_activity()
{
  API_METHOD_PREFIX
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; domain++) {
    const uint32_t op_num = get_op_num(domain);
    for (uint32_t op = 0; op < op_num; op++) roctracer_disable_activity_impl(domain, op);
  }
  API_METHOD_SUFFIX
}

// Flush available activity records
PUBLIC_API roctracer_status_t roctracer_flush_activity(roctracer_pool_t* pool) {
  API_METHOD_PREFIX
  if (pool == NULL) pool = roctracer_default_pool();
  roctracer::MemoryPool* memory_pool = reinterpret_cast<roctracer::MemoryPool*>(pool);
  memory_pool->Flush();
  API_METHOD_SUFFIX
}

// Set properties
PUBLIC_API roctracer_status_t roctracer_set_properties(
    roctracer_domain_t domain,
    void* properties)
{
  API_METHOD_PREFIX
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS: {
      // HSA OPS properties
      roctracer::hsa_ops_properties_t* ops_properties = reinterpret_cast<roctracer::hsa_ops_properties_t*>(properties);
      roctracer::hsa_support::async_copy_callback_fun = ops_properties->async_copy_callback_fun;
      roctracer::hsa_support::async_copy_callback_arg = ops_properties->async_copy_callback_arg;
      break;
    }
    case ACTIVITY_DOMAIN_HSA_API: {
      // HSA API properties
      HsaApiTable* table = reinterpret_cast<HsaApiTable*>(properties);
      roctracer::hsa_support::intercept_CoreApiTable(table->core_);
      roctracer::hsa_support::intercept_AmdExtTable(table->amd_ext_);
      roctracer::hsa_support::intercept_ImageExtTable(table->image_ext_);
      break;
    }
    case ACTIVITY_DOMAIN_HCC_OPS:
    case ACTIVITY_DOMAIN_HIP_API:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "properties are not supported, domain ID(" << domain << ")");
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
  API_METHOD_SUFFIX
}

// HSA-runtime tool on-load method
PUBLIC_API bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                       const char* const* failed_tool_names) {
  // enabled HSA async-copy tracing
  hsa_status_t status = hsa_amd_profiling_async_copy_enable(true);
  if (status != HSA_STATUS_SUCCESS) EXC_ABORT(status, "hsa_amd_profiling_async_copy_enable");
  roctracer::hsa_amd_memory_async_copy_fn = table->amd_ext_->hsa_amd_memory_async_copy_fn;
  roctracer::hsa_amd_memory_async_copy_rect_fn = table->amd_ext_->hsa_amd_memory_async_copy_rect_fn;
  table->amd_ext_->hsa_amd_memory_async_copy_fn = roctracer::hsa_amd_memory_async_copy_interceptor;
  table->amd_ext_->hsa_amd_memory_async_copy_rect_fn = roctracer::hsa_amd_memory_async_copy_rect_interceptor;

  return true;
}

}  // extern "C"
