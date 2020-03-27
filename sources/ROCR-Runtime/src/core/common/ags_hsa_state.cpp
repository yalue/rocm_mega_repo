#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <pthread.h>
#include <ags_communication.h>
#include "hsa.h"
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

// Closes any connections to pipes and deletes the process-specific pipe if
// possible.
static void CleanupAGSState(void) {
  if (!ags_state) return;
  close(ags_state->fd);
  ags_state->fd = -1;
  pthread_mutex_destroy(&(ags_state->mutex));
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
  //__asm__ __volatile__ ("int $3");

  // Make sure we don't re-initialize AGS if it's already set up.
  if (ags_state) return true;

  // Try connecting to AGS
  ags_fd = OpenAGSSocket();
  if (ags_fd == -1) {
    // We couldn't connect to AGS, this isn't an error.
    return true;
  }

  ags_state = (AGSHSAState *) malloc(sizeof(AGSHSAState));
  if (!ags_state) {
    printf("Failed allocating AGS HSA state.\n");
    return false;
  }
  memset(ags_state, 0, sizeof(AGSHSAState));
  ags_state->fd = ags_fd;
  ags_state->request_id = 1;
  pthread_mutex_init(&(ags_state->mutex), NULL);
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
    callback(agents[i], data);
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
    callback(regions[i], data);
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
