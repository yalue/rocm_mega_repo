#ifndef SRC_CORE_TRACE_BUFFER_H_
#define SRC_CORE_TRACE_BUFFER_H_

#include <list>
#include <mutex>
#include <pthread.h>
#include <string.h>
#include <unistd.h>

#define PTHREAD_CALL(call)                                                                         \
  do {                                                                                             \
    int err = call;                                                                                \
    if (err != 0) {                                                                                \
      errno = err;                                                                                 \
      perror(#call);                                                                               \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

namespace roctracer {
enum {
  TRACE_ENTRY_INV = 0,
  TRACE_ENTRY_INIT = 1,
  TRACE_ENTRY_COMPL = 2
};

enum {
  API_ENTRY_TYPE,
  COPY_ENTRY_TYPE,
  KERNEL_ENTRY_TYPE
};

struct trace_entry_t {
  std::atomic<uint32_t> valid;
  uint32_t type;
  uint64_t dispatch;
  uint64_t begin;                                      // kernel begin timestamp, ns
  uint64_t end;                                        // kernel end timestamp, ns
  uint64_t complete;
  hsa_agent_t agent;
  uint32_t dev_index;
  hsa_signal_t orig;
  hsa_signal_t signal;
  union {
    struct {
    } copy;
    struct {
      const char* name;
      hsa_agent_t agent;
      uint32_t tid;
    } kernel;
  };
};

template <typename Entry>
class TraceBuffer {
  public:
  typedef void (*callback_t)(Entry*);
  typedef TraceBuffer<Entry> Obj;
  typedef uint64_t pointer_t;
  typedef std::mutex mutex_t;

  struct flush_prm_t {
    uint32_t type;
    callback_t fun;
  };

  TraceBuffer(const char* name, uint32_t size, flush_prm_t* flush_prm_arr, uint32_t flush_prm_count) :
    is_flushed_(ATOMIC_FLAG_INIT)
  {
    name_ = strdup(name);
    size_ = size;
    data_ = allocate_fun();
    next_ = NULL;
    read_pointer_ = 0;
    end_pointer_ = size;
    buf_list_.push_back(data_);

    flush_prm_arr_ = flush_prm_arr;
    flush_prm_count_ = flush_prm_count;

    PTHREAD_CALL(pthread_mutex_init(&work_mutex_, NULL));
    PTHREAD_CALL(pthread_cond_init(&work_cond_, NULL));
    PTHREAD_CALL(pthread_create(&work_thread_, NULL, allocate_worker, this));
  }

  ~TraceBuffer() {
    PTHREAD_CALL(pthread_cancel(work_thread_));
    void *res;
    PTHREAD_CALL(pthread_join(work_thread_, &res));
    if (res != PTHREAD_CANCELED) abort_run("~TraceBuffer: consumer thread wasn't stopped correctly");

    Flush();
  }


  Entry* GetEntry() {
    const pointer_t pointer = read_pointer_.fetch_add(1);
    if (pointer >= end_pointer_) wrap_buffer(pointer);
    if (pointer >= end_pointer_) abort_run("pointer >= end_pointer_ after buffer wrap");
    return data_ + (pointer + size_ - end_pointer_);
  }

  void Flush() {
    flush_buf();
  }

  private:
  void flush_buf() {
    std::lock_guard<mutex_t> lck(mutex_);
    const bool is_flushed = atomic_flag_test_and_set_explicit(&is_flushed_, std::memory_order_acquire);

    if (is_flushed == false) {
      for (flush_prm_t* prm = flush_prm_arr_; prm < flush_prm_arr_ + flush_prm_count_; prm++) {
        uint32_t type = prm->type;
        callback_t fun = prm->fun;
        pointer_t pointer = 0;
        for (Entry* ptr : buf_list_) {
          Entry* end = ptr + size_;
          while ((ptr < end) && (pointer < read_pointer_)) {
            if (ptr->type == type) {
              if (ptr->valid == TRACE_ENTRY_COMPL) {
                fun(ptr);
              }
            }
            ptr++;
            pointer++;
          }
        }
      }
    }
  }

  inline Entry* allocate_fun() {
    Entry* ptr = (Entry*) malloc(size_ * sizeof(Entry));
    if (ptr == NULL) abort_run("TraceBuffer::allocate_fun: calloc failed");
    //memset(ptr, 0, size_ * sizeof(Entry));
    return ptr;
  }

  static void* allocate_worker(void* arg) {
    Obj* obj = (Obj*)arg;

    while (1) {
      PTHREAD_CALL(pthread_mutex_lock(&(obj->work_mutex_)));
      while (obj->next_ != NULL) {
        PTHREAD_CALL(pthread_cond_wait(&(obj->work_cond_), &(obj->work_mutex_)));
      }
      obj->next_ = obj->allocate_fun();
      PTHREAD_CALL(pthread_mutex_unlock(&(obj->work_mutex_)));
    }

    return NULL;
  }

  void wrap_buffer(const pointer_t pointer) {
    std::lock_guard<mutex_t> lck(mutex_);
    PTHREAD_CALL(pthread_mutex_lock(&work_mutex_));
    if (pointer >= end_pointer_) {
      data_ = next_;
      next_ = NULL;
      PTHREAD_CALL(pthread_cond_signal(&work_cond_));
      end_pointer_ += size_;
      if (end_pointer_ == 0) abort_run("TraceBuffer::wrap_buffer: pointer overflow");
      buf_list_.push_back(data_);
    }
    PTHREAD_CALL(pthread_mutex_unlock(&work_mutex_));
  }

  void abort_run(const char* str) {
    fprintf(stderr, "%s\n", str);
    fflush(stderr);
    abort();
  }

  const char* name_;
  uint32_t size_;
  Entry* data_;
  Entry* next_;
  volatile std::atomic<pointer_t> read_pointer_;
  volatile std::atomic<pointer_t> end_pointer_;
  std::list<Entry*> buf_list_;

  flush_prm_t* flush_prm_arr_;
  uint32_t flush_prm_count_;
  volatile std::atomic_flag is_flushed_;

  pthread_t work_thread_;
  pthread_mutex_t work_mutex_;
  pthread_cond_t work_cond_;

  mutex_t mutex_;
};
}  // namespace roctracer

#endif  // SRC_CORE_TRACE_BUFFER_H_
