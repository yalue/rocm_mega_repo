#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ags_communication.h>
#include "inc/hsa.h"
#include "ags_hsa_state.h"

// Tracks state of the connection to AGS. Initialized during
// InitializeAGSConnection.
AGSHSAState *ags_state = NULL;

void LockAGS(void) {
  pthread_mutex_lock(&(ags_state->mutex));
}

void UnlockAGS(void) {
  pthread_mutex_unlock(&(ags_state->mutex));
}

// Should be called to free the given SharedCPUAllocation. Returns false on
// error. Unmaps and closes the shared memory, and zeroes the struct. Does not
// free the struct itself, though.
static bool CleanupSharedCPUAllocation(SharedCPUAllocation *a) {
  int result = munmap(a->va, a->size);
  bool to_return = true;
  if (result != 0) {
    printf("Failed unmapping shared CPU allocation: %s\n", strerror(errno));
    to_return = false;
  }
  close(a->fd);
  memset(a, 0, sizeof(*a));
  return to_return;
}

// Closes any connections to pipes and deletes the process-specific pipe if
// possible.
static void CleanupAGSState(void) {
  if (!ags_state) return;
  close(ags_state->fd);
  ags_state->fd = -1;
  pthread_mutex_destroy(&(ags_state->mutex));
  // Hopefully all of the CPU allocations should be freed if we're cleaning up
  // AGS' state, but we'll free anything remaining as a sanity check and print
  // a message.
  auto allocations = ags_state->cpu_allocations;
  for (auto it = allocations->begin(); it != allocations->end(); ++it) {
    SharedCPUAllocation *a = it->second;
    printf("Unmapping %lu-byte remaining shared CPU allocation @%p.\n",
      (unsigned long) a->size, a->va);
    CleanupSharedCPUAllocation(a);
    free(a);
  }
  allocations->clear();
  delete ags_state->cpu_allocations;
  free(ags_state);
  ags_state = NULL;
}

// A sanity-checking function to check that the response from AGS is for the
// given request. Prints a message and exits on error.
static void VerifyRequestIDs(AGSRequest *request,
    AGSResponse *response) {
  if (request->request_id == response->request_id) return;
  printf("Error: Response request_id doesn't match the request!\n");
  printf("Request (data omitted):\n");
  request->data_size = 0;
  PrintRequestInformation(request, NULL, "  ");
  printf("Request ID in response = %lu\n",
    (unsigned long) response->request_id);
  exit(1);
}

bool DoAGSTransaction(AGSRequest *request, void *request_data,
    AGSResponse *response, uint32_t response_data_size, void *response_data) {
  if (!ags_state) {
    printf("Called DoAGSTransaction when not connected to AGS.\n");
    response->prevent_default = 0;
    return false;
  }
  request->pid = getpid();
  request->tid = GetTID();
  LockAGS();
  // We assign the request ID here, behind the lock.
  request->request_id = ags_state->request_id++;
  if (send(ags_state->fd, request, sizeof(*request), 0) != sizeof(*request)) {
    printf("Failed sending request to AGS: %s\n", strerror(errno));
    UnlockAGS();
    CleanupAGSState();
    return false;
  }
  if (request->data_size != 0) {
    if (send(ags_state->fd, request_data, request->data_size, 0) !=
      request->data_size) {
      printf("Failed sending request data to AGS: %s\n", strerror(errno));
      UnlockAGS();
      CleanupAGSState();
      return false;
    }
  }
  if (recv(ags_state->fd, response, sizeof(*response), 0) !=
    sizeof(*response)) {
    printf("Failed receiving response header from AGS: %s\n", strerror(errno));
    UnlockAGS();
    CleanupAGSState();
    return false;
  }
  VerifyRequestIDs(request, response);
  if (response->data_size == 0) {
    UnlockAGS();
    return true;
  }
  if (response->data_size > response_data_size) {
    printf("Insufficient buffer size to receive %d-byte response from AGS.\n",
      (int) response->data_size);
    UnlockAGS();
    CleanupAGSState();
    return false;
  }
  if (recv(ags_state->fd, response_data, response->data_size, 0) !=
    response->data_size) {
    printf("Failed receiving response from AGS: %s\n", strerror(errno));
    UnlockAGS();
    CleanupAGSState();
    return false;
  }
  UnlockAGS();
  return true;
}

bool InitializeAGSConnection(void) {
  int ags_fd;
  AGSHSAState *tmp = NULL;

  // Make sure we don't re-initialize AGS if it's already set up.
  if (ags_state) return true;

  // Try connecting to AGS
  ags_fd = OpenAGSSocket();
  if (ags_fd == -1) {
    // We couldn't connect to AGS, this isn't an error.
    return true;
  }

  tmp = (AGSHSAState *) calloc(1, sizeof(AGSHSAState));
  if (!tmp) {
    printf("Failed allocating AGS HSA state.\n");
    return false;
  }
  tmp->fd = ags_fd;
  tmp->request_id = 1;
  pthread_mutex_init(&(tmp->mutex), NULL);
  tmp->cpu_allocations = new std::map<uint64_t, SharedCPUAllocation*>();
  ags_state = tmp;
  return true;
}

bool SendInitialMessage(hsa_status_t *result, bool *prevent_default) {
  AGSRequest request;
  AGSResponse response;
  *prevent_default = 0;
  if (!ags_state) return true;
  memset(&request, 0, sizeof(request));
  request.request_type = AGS_NEW_PROCESS;
  request.data_size = 0;
  printf("Sending initial request to AGS for PID %d.\n", (int) request.pid);
  if (!DoAGSTransaction(&request, NULL, &response, 0, NULL)) {
    printf("Failed notifying AGS of process creation.\n");
    return false;
  }
  *result = (hsa_status_t) response.hsa_status;
  *prevent_default = response.prevent_default;
  return true;
}

bool EndAGSConnection(hsa_status_t *result) {
  AGSRequest request;
  AGSResponse response;
  if (!ags_state) return true;
  memset(&request, 0, sizeof(request));
  request.request_type = AGS_PROCESS_QUITTING;
  request.data_size = 0;
  if (!DoAGSTransaction(&request, NULL, &response, 0, NULL)) {
    printf("Failed notifying AGS of process ending.\n");
    return true;
  }
  CleanupAGSState();
  if (response.prevent_default) {
    *result = (hsa_status_t) response.hsa_status;
    return false;
  }
  return true;
}

bool SendAGSPlaceholderRequest(const char *file, const char *func, int line,
    hsa_status_t *result) {
  AGSRequest header;
  AGSResponse response;
  char data[256];

  // Do nothing if AGS isn't running.
  if (!ags_state) return true;

  memset(&header, 0, sizeof(header));
  memset(data, 0, sizeof(data));
  // Remember that snprintf does not include the null character in its return
  // value.
  header.data_size = snprintf(data, sizeof(data) - 1, "%s, at line %d in %s",
    func, line, file) + 1;
  if (header.data_size > sizeof(data)) {
    // Just let the application keep running if this happens--it's our fault
    // and needs to be adjusted at compile time.
    printf("Warning: not enough buffer space to send placeholder request.\n");
    return true;
  }
  header.request_type = AGS_PLACEHOLDER_REQUEST;

  if (!DoAGSTransaction(&header, data, &response, 0, NULL)) {
    printf("Failed sending placeholder request header to AGS.\n");
    return true;
  }
  if (response.prevent_default) {
    *result = (hsa_status_t) response.hsa_status;
    return false;
  }
  return true;
}

bool AGSHandleIterateAgents(hsa_status_t (*callback)(hsa_agent_t agent,
    void *data), void *data, hsa_status_t *result) {
  AGSRequest header;
  AGSResponse response;
  hsa_agent_t agents[AGS_MAX_HSA_AGENT_COUNT];
  int agent_count = 0;
  int i;
  if (!ags_state) return true;
  memset(&header, 0, sizeof(header));
  header.request_type = AGS_HSA_ITERATE_AGENTS;
  if (!DoAGSTransaction(&header, NULL, &response, sizeof(agents), agents)) {
    printf("Failed getting hsa_iterate_agents response from AGS.\n");
    return true;
  }
  if (!response.prevent_default) {
    printf("AGS error: Expected prevent_default for hsa_iterate_agents.\n");
    CleanupAGSState();
    return true;
  }
  // At this point, AGS has already handled this request.
  *result = (hsa_status_t) response.hsa_status;
  if (response.hsa_status != HSA_STATUS_SUCCESS) {
    return false;
  }
  agent_count = response.data_size / sizeof(hsa_agent_t);
  for (i = 0; i < agent_count; i++) {
    *result = callback(agents[i], data);
    if (*result != HSA_STATUS_SUCCESS) return false;
  }
  return false;
}

bool AGSHandleAgentGetInfo(hsa_agent_t agent, hsa_agent_info_t attribute,
    void *data, hsa_status_t *result) {
  AGSRequest header;
  AGSResponse response;
  AGSAgentGetInfoRequest args;
  uint8_t response_data[128];
  if (!ags_state) return true;
  args.agent = agent;
  args.attribute = attribute;
  header.request_type = AGS_HSA_AGENT_GET_INFO;
  header.data_size = sizeof(args);
  if (!DoAGSTransaction(&header, &args, &response, sizeof(response_data),
    response_data)) {
    printf("Failed getting hsa_agent_get_info response from AGS.\n");
    return true;
  }
  if (!response.prevent_default) {
    printf("AGS error: Expected prevent_default for hsa_agent_get_info.\n");
    CleanupAGSState();
    return true;
  }
  *result = (hsa_status_t) response.hsa_status;
  if (response.hsa_status != HSA_STATUS_SUCCESS) return false;
  memcpy(data, response_data, response.data_size);
  return false;
}

bool AGSHandleAgentIterateRegions(hsa_agent_t agent, hsa_status_t (*callback)(
    hsa_region_t region, void *data), void *data, hsa_status_t *result) {
  AGSRequest header;
  AGSResponse response;
  hsa_region_t *regions = NULL;
  int result_buffer_size = AGS_MAX_HSA_REGION_COUNT * sizeof(hsa_region_t);
  int region_count = 0;
  int i;
  if (!ags_state) return true;
  regions = (hsa_region_t *) malloc(result_buffer_size);
  if (!regions) {
    printf("Failed allocating buffer for hsa_agent_iterate_regions result.\n");
    return true;
  }
  memset(&header, 0, sizeof(header));
  header.request_type = AGS_HSA_AGENT_ITERATE_REGIONS;
  header.data_size = sizeof(agent);
  if (!DoAGSTransaction(&header, &agent, &response, result_buffer_size,
    regions)) {
    printf("Failed getting hsa_agent_iterate_regions response from AGS.\n");
    free(regions);
    return true;
  }
  if (!response.prevent_default) {
    printf("Error: Expected prevent_default for hsa_agent_iterate_regions.\n");
    CleanupAGSState();
    free(regions);
    return true;
  }

  // At this point, AGS has already handled this request.
  *result = (hsa_status_t) response.hsa_status;
  if (response.hsa_status != HSA_STATUS_SUCCESS) {
    free(regions);
    return false;
  }
  region_count = response.data_size / sizeof(hsa_region_t);
  for (i = 0; i < region_count; i++) {
    *result = callback(regions[i], data);
    if (*result != HSA_STATUS_SUCCESS) break;
  }
  free(regions);
  return false;
}

bool AGSHandleRegionGetInfo(hsa_region_t region, hsa_region_info_t attribute,
    void *data, hsa_status_t *result) {
  AGSRequest request;
  AGSResponse response;
  AGSRegionGetInfoRequest request_args;
  uint8_t response_data[64];
  if (!ags_state) return true;

  request_args.region = region;
  request_args.attribute = attribute;
  request.data_size = sizeof(request_args);
  request.request_type = AGS_HSA_REGION_GET_INFO;
  if (!DoAGSTransaction(&request, &request_args, &response,
    sizeof(response_data), response_data)) {
    printf("Failed getting hsa_region_get_info response from AGS.\n");
    CleanupAGSState();
    return true;
  }

  if (!response.prevent_default) {
    printf("Error: Expected prevent_default for hsa_region_get_info.\n");
    CleanupAGSState();
    return true;
  }
  *result = (hsa_status_t) response.hsa_status;
  if (*result != HSA_STATUS_SUCCESS) return false;
  memcpy(data, response_data, response.data_size);
  return false;
}

bool AGSHandleAMDAgentMemoryPoolGetInfo(hsa_agent_t agent,
    hsa_amd_memory_pool_t memory_pool,
    hsa_amd_agent_memory_pool_info_t attribute, void *value,
    hsa_status_t *result) {
  AGSRequest request;
  AGSResponse response;
  AGSAMDAgentMemoryPoolGetInfoRequest request_args;
  void *response_data = NULL;
  if (!ags_state) return true;

  response_data = calloc(1, AGS_MAX_DATA_SIZE);
  if (!response_data) {
    printf("Failed allocating response data for amd_agent_memory_pool...\n");
    CleanupAGSState();
    return true;
  }
  request_args.agent = agent;
  request_args.memory_pool = memory_pool;
  request_args.attribute = attribute;
  request.data_size = sizeof(request_args);
  request.request_type = AGS_AMD_AGENT_MEMORY_POOL_GET_INFO;
  if (!DoAGSTransaction(&request, &request_args, &response, AGS_MAX_DATA_SIZE,
    response_data)) {
    printf("Failed getting hsa_amd_agent_memory_pool_get_info response.\n");
    CleanupAGSState();
    free(response_data);
    return true;
  }
  if (!response.prevent_default) {
    printf("Error: Expected prevent_default for amd_agent_memory_pool...\n");
    CleanupAGSState();
    free(response_data);
    return true;
  }
  *result = (hsa_status_t) response.hsa_status;
  if (*result != HSA_STATUS_SUCCESS) {
    free(response_data);
    return false;
  }
  memcpy(value, response_data, response.data_size);
  free(response_data);
  return false;
}

bool AGSHandleAMDProfilingAsyncCopyEnable(bool value, hsa_status_t *result) {
  AGSRequest request;
  AGSResponse response;
  if (!ags_state) return true;
  uint8_t arg = 0;
  if (value) arg = 1;
  request.data_size = 1;
  request.request_type = AGS_AMD_PROFILING_ASYNC_COPY_ENABLE;
  if (!DoAGSTransaction(&request, &arg, &response, 0, NULL)) {
    printf("Failed getting hsa_amd_profiling_async_copy_enable response.\n");
    CleanupAGSState();
    return true;
  }
  // Maybe I won't call this one an error if it doesn't set prevent_default,
  // since it doesn't take any handle arguments that could cause crashes.
  *result = (hsa_status_t) response.hsa_status;
  return response.prevent_default == 0;
}

bool AGSHandleAMDAgentIterateMemoryPools(hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void *data),
    void *data, hsa_status_t *result) {
  AGSRequest request;
  AGSResponse response;
  hsa_amd_memory_pool_t *memory_pools = NULL;
  int response_data_size, memory_pool_count, i;
  if (!ags_state) return true;

  response_data_size = AGS_MAX_AMD_AGENT_MEMORY_POOLS *
    sizeof(hsa_amd_memory_pool_t);
  memory_pools = (hsa_amd_memory_pool_t *) malloc(response_data_size);
  if (!memory_pools) {
    printf("Failed allocating space for AMD agent memory pools.\n");
    CleanupAGSState();
    return true;
  }
  request.request_type = AGS_AMD_AGENT_ITERATE_MEMORY_POOLS;
  request.data_size = sizeof(agent);
  if (!DoAGSTransaction(&request, &agent, &response, response_data_size,
    memory_pools)) {
    printf("Failed getting hsa_amd_agent_iterate_memory_pools response.\n");
    free(memory_pools);
    CleanupAGSState();
    return true;
  }
  if (!response.prevent_default) {
    printf("Expected prevent_default for amd_agent_iterate_memory...\n");
    free(memory_pools);
    CleanupAGSState();
    return true;
  }

  *result = (hsa_status_t) response.hsa_status;
  if (*result != HSA_STATUS_SUCCESS) {
    free(memory_pools);
    return false;
  }
  memory_pool_count = response.data_size / sizeof(hsa_amd_memory_pool_t);
  for (i = 0; i < memory_pool_count; i++) {
    *result = callback(memory_pools[i], data);
    if (*result != HSA_STATUS_SUCCESS) {
      free(memory_pools);
      return false;
    }
  }
  free(memory_pools);
  return false;
}

bool AGSHandleAMDMemoryPoolGetInfo(hsa_amd_memory_pool_t memory_pool,
  hsa_amd_memory_pool_info_t attribute, void *value, hsa_status_t *result) {
  AGSRequest request;
  AGSResponse response;
  AGSAMDMemoryPoolGetInfoRequest args;
  uint8_t response_data[32];
  if (!ags_state) return true;

  args.memory_pool = memory_pool;
  args.attribute = attribute;
  request.data_size = sizeof(args);
  request.request_type = AGS_AMD_MEMORY_POOL_GET_INFO;
  if (!DoAGSTransaction(&request, &args, &response, sizeof(response_data),
    response_data)) {
    printf("Failed getting hsa_amd_memory_pool_get_info response.\n");
    CleanupAGSState();
    return true;
  }

  if (!response.prevent_default) {
    printf("Expected prevent_default for hsa_amd_memory_pool_get_info.\n");
    CleanupAGSState();
    return true;
  }
  *result = (hsa_status_t) response.hsa_status;
  memcpy(value, response_data, response.data_size);
  return false;
}

// This is called if we're handling a CPU allocation, and will send the VA
// where we mapped the memory back to AGS. Returns false on error.
static bool NotifyAGSWithCPUAddress(AGSMemoryAllocateResponse *r,
    SharedCPUAllocation *allocation) {
  AGSRequest request;
  AGSResponse response;
  AGSClientMappedMemoryAddress args;
  args.ags_cpu_va = r->ags_cpu_va;
  args.client_va = allocation->va;
  request.data_size = sizeof(args);
  request.request_type = AGS_CLIENT_MAPPED_MEMORY_ADDRESS;

  if (!DoAGSTransaction(&request, &args, &response, 0, NULL)) {
    printf("Didn't get response when notifying AGS of shared-memory VA.\n");
    CleanupAGSState();
    return false;
  }
  if (response.hsa_status != HSA_STATUS_SUCCESS) {
    printf("Didn't get HSA_STATUS_SUCCESS after notifying AGS of shared-memory"
      " VA. Got status = %d.\n", (int) response.hsa_status);
    CleanupAGSState();
    return false;
  }
  return true;
}

// Handles a memory allocation result returned by AGS, including potentially
// mapping a shared memory object into userspace.
static bool FollowUpAGSMemoryAllocation(AGSMemoryAllocateResponse *r,
    size_t size, void **ptr, hsa_status_t *result) {
  SharedCPUAllocation *allocation = NULL;
  void *va = NULL;
  *result = HSA_STATUS_ERROR;
  *ptr = NULL;

  // Return now if this was a GPU allocation.
  if (!r->is_cpu_allocation) {
    *ptr = r->gpu_va;
    *result = HSA_STATUS_SUCCESS;
    return false;
  }

  // We'll now need to map the shared memory file into our local address space.
  allocation = (SharedCPUAllocation *) calloc(1, sizeof(*allocation));
  if (!allocation) {
    printf("Failed allocating space to track shared-memory allocation.\n");
    CleanupAGSState();
    return true;
  }
  allocation->fd = shm_open(r->shared_memory_name, O_RDWR, 0666);
  if (allocation->fd < 0) {
    printf("Failed opening shared memory object %s: %s\n",
      r->shared_memory_name, strerror(errno));
    free(allocation);
    CleanupAGSState();
    return true;
  }
  allocation->size = size;
  va = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, allocation->fd, 0);
  if (va == MAP_FAILED) {
    printf("Failed mapping shared-memory object %s: %s\n",
      r->shared_memory_name, strerror(errno));
    close(allocation->fd);
    free(allocation);
    CleanupAGSState();
    return true;
  }
  allocation->va = va;

  // Let AGS know where we mapped the memory in our own address space.
  if (!NotifyAGSWithCPUAddress(r, allocation)) {
    CleanupSharedCPUAllocation(allocation);
    free(allocation);
    CleanupAGSState();
    return true;
  }

  // Now, insert the record of the allocation into the map and return the VA.
  ags_state->cpu_allocations->insert(std::make_pair((uint64_t) va,
    allocation));
  *ptr = va;
  *result = HSA_STATUS_SUCCESS;
  return false;
}

bool AGSHandleAMDMemoryPoolAllocate(hsa_amd_memory_pool_t memory_pool,
    size_t size, uint32_t flags, void **ptr, hsa_status_t *result) {
  AGSRequest request;
  AGSResponse response;
  AGSAMDMemoryPoolAllocateRequest args;
  AGSMemoryAllocateResponse response_data;
  if (!ags_state) return true;

  args.memory_pool = memory_pool;
  args.size = size;
  args.flags = flags;
  request.data_size = sizeof(args);
  request.request_type = AGS_AMD_MEMORY_POOL_ALLOCATE;
  if (!DoAGSTransaction(&request, &args, &response, sizeof(response_data),
    &response_data)) {
    printf("Failed getting hsa_amd_memory_pool_allocate response.\n");
    CleanupAGSState();
    return true;
  }

  if (!response.prevent_default) {
    printf("Expected prevent_default for hsa_amd_memory_pool_allocate.\n");
    CleanupAGSState();
    return true;
  }
  *result = (hsa_status_t) response.hsa_status;
  // Don't do further processing if an error occurred.
  if (response.hsa_status != HSA_STATUS_SUCCESS) {
    return false;
  }
  return FollowUpAGSMemoryAllocation(&response_data, size, ptr, result);
}

bool AGSHandleHSAMemoryAllocate(hsa_region_t region, size_t size, void **ptr,
    hsa_status_t *result) {
  AGSRequest request;
  AGSResponse response;
  AGSMemoryAllocateRequest args;
  AGSMemoryAllocateResponse response_data;
  if (!ags_state) return true;
  args.region = region;
  args.size = size;
  request.data_size = sizeof(args);
  request.request_type = AGS_HSA_MEMORY_ALLOCATE;
  if (!DoAGSTransaction(&request, &args, &response, sizeof(response_data),
    &response_data)) {
    printf("Failed getting hsa_memory_allocate response.\n");
    CleanupAGSState();
    return true;
  }

  if (!response.prevent_default) {
    printf("Expected prevent_default for hsa_memory_allocate.\n");
    CleanupAGSState();
    return true;
  }
  *result = (hsa_status_t) response.hsa_status;
  if (response.hsa_status != HSA_STATUS_SUCCESS) {
    return false;
  }
  return FollowUpAGSMemoryAllocation(&response_data, size, ptr, result);
  // TODO (next): Test hsa_memory_allocate.
}

bool AGSHandleAMDAgentsAllowAccess(uint32_t num_agents,
    const hsa_agent_t *agents, const uint32_t *flags, const void *ptr,
    hsa_status_t *result) {
  AGSRequest request;
  AGSResponse response;
  AGSAMDAgentsAllowAccessRequest args;

  if (!ags_state) return true;
  memset(&args, 0, sizeof(args));
  args.num_agents = num_agents;
  memcpy(args.agents, agents, num_agents * sizeof(hsa_agent_t));
  if (flags) memcpy(args.flags, flags, num_agents * sizeof(uint32_t));
  memcpy(&args.ptr, &ptr, sizeof(ptr));
  request.data_size = sizeof(args);
  request.request_type = AGS_AMD_AGENTS_ALLOW_ACCESS;
  if (!DoAGSTransaction(&request, &args, &response, 0, NULL)) {
    printf("Failed getting hsa_amd_agents_allow_access response.\n");
    CleanupAGSState();
    return true;
  }

  if (!response.prevent_default) {
    printf("Expected prevent_default for hsa_amd_agents_allow_access.\n");
    CleanupAGSState();
    return true;
  }
  *result = (hsa_status_t) response.hsa_status;
  return false;
}

bool AGSHandleHSASignalCreate(hsa_signal_value_t initial_value,
    uint32_t num_consumers, const hsa_agent_t *consumers,
    hsa_signal_t *signal, hsa_status_t *result) {
  AGSRequest request;
  AGSResponse response;
  AGSSignalCreateRequest args;

  if (!ags_state) return true;
  memset(&args, 0, sizeof(args));
  args.initial_value = initial_value;
  args.num_consumers = num_consumers;
  if (num_consumers >= AGS_MAX_HSA_AGENT_COUNT) {
    printf("hsa_signal_create: Too many agents for AGS to handle.\n");
    return true;
  }
  memcpy(args.agents, consumers, num_consumers * sizeof(hsa_agent_t));
  request.data_size = sizeof(args);
  request.request_type = AGS_HSA_SIGNAL_CREATE;
  if (!DoAGSTransaction(&request, &args, &response, sizeof(*signal), signal)) {
    printf("Failed getting hsa_signal_create response.\n");
    CleanupAGSState();
    return true;
  }

  if (!response.prevent_default) {
    printf("Expected prevent_default for hsa_signal_create.\n");
    CleanupAGSState();
    return true;
  }
  *result = (hsa_status_t) response.hsa_status;
  return false;
}

bool AGSHandleAMDMemoryLock(void *host_ptr, size_t size, hsa_agent_t *agents,
    int num_agents, void **agent_ptr, hsa_status_t *result) {
  AGSRequest request;
  AGSResponse response;
  AGSAMDMemoryLockRequest args;

  if (!ags_state) return true;
  memset(&args, 0, sizeof(args));
  args.cpu_ptr = host_ptr;
  args.size = size;
  if (num_agents >= AGS_MAX_HSA_AGENT_COUNT) {
    printf("hsa_amd_memory_lock: Too many agents for AGS.\n");
    return true;
  }
  memcpy(args.agents, agents, num_agents * sizeof(hsa_agent_t));
  args.num_agents = num_agents;
  request.data_size = sizeof(args);
  request.request_type = AGS_AMD_MEMORY_LOCK;
  if (!DoAGSTransaction(&request, &args, &response, sizeof(*agent_ptr),
    agent_ptr)) {
    printf("Failed getting hsa_amd_memory_lock response.\n");
    CleanupAGSState();
    return true;
  }

  if (!response.prevent_default) {
    printf("Expected prevent_default for hsa_amd_memory_lock.\n");
    CleanupAGSState();
    return true;
  }
  *result = (hsa_status_t) response.hsa_status;
  return false;
}

bool AGSHandleSystemGetInfo(hsa_system_info_t attribute, void *value,
    hsa_status_t *result) {
  AGSRequest request;
  AGSResponse response;
  uint8_t response_data[128];

  if (!ags_state) return true;
  request.data_size = sizeof(attribute);
  request.request_type = AGS_HSA_SYSTEM_GET_INFO;
  if (!DoAGSTransaction(&request, &attribute, &response, sizeof(response_data),
    response_data)) {
    printf("Failed getting hsa_system_get_info response.\n");
    CleanupAGSState();
    return true;
  }

  if (!response.prevent_default) {
    printf("Expected prevent_default for hsa_system_get_info.\n");
    CleanupAGSState();
    return true;
  }
  memcpy(value, response_data, response.data_size);
  *result = (hsa_status_t) response.hsa_status;
  return false;
}

bool AGSHandleFreeOrUnlock(void *ptr, AGSRequestType request_type,
    hsa_status_t *result) {
  // TODO (next, 2): Update our code for freeing memory:
  //  - Unmap shared memory if the address is in the cpu_allocations map.
  //  - Send the VA back to AGS properly.
  AGSRequest request;
  AGSResponse response;
  const char *request_name = NULL;

  if (!ags_state) return true;
  request.data_size = sizeof(ptr);
  request.request_type = request_type;
  request_name = GetRequestTypeName(request_type);
  if (!DoAGSTransaction(&request, &ptr, &response, 0, NULL)) {
    printf("Failed getting response for %s\n", request_name);
    CleanupAGSState();
    return true;
  }
  if (!response.prevent_default) {
    printf("Expected prevent_default for %s\n", request_name);
    CleanupAGSState();
    return true;
  }
  *result = (hsa_status_t) response.hsa_status;
  return false;
}

bool AGSHandleHSASignalDestroy(hsa_signal_t signal, hsa_status_t *result) {
  AGSRequest request;
  AGSResponse response;
  if (!ags_state) return true;
  request.data_size = sizeof(signal);
  request.request_type = AGS_HSA_SIGNAL_DESTROY;
  if (!DoAGSTransaction(&request, &signal, &response, 0, NULL)) {
    printf("Failed getting response for hsa_signal_destroy.\n");
    CleanupAGSState();
    return true;
  }
  if (!response.prevent_default) {
    printf("Expected prevent_default for hsa_signal_destroy.\n");
    CleanupAGSState();
    return true;
  }
  *result = (hsa_status_t) response.hsa_status;
  return false;
}

bool AGSHandleHSAISAGetInfoAlt(hsa_isa_t isa, hsa_isa_info_t attribute,
    void *value, hsa_status_t *result) {
  AGSRequest request;
  AGSResponse response;
  AGSISAGetInfoAltRequest args;
  // Arbitrarily chosen.
  int max_response_data_size = 256;
  if (!ags_state) return true;
  args.isa = isa;
  args.attribute = attribute;
  request.data_size = sizeof(args);
  request.request_type = AGS_HSA_ISA_GET_INFO_ALT;
  if (!DoAGSTransaction(&request, &args, &response, max_response_data_size,
    value)) {
    printf("Failed getting response for hsa_isa_get_info_alt.\n");
    CleanupAGSState();
    return true;
  }
  if (!response.prevent_default) {
    printf("Expected prevent_default for hsa_isa_get_info_alt.\n");
    CleanupAGSState();
    return true;
  }
  *result = (hsa_status_t) response.hsa_status;
  return false;
}
