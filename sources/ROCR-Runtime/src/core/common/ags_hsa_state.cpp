#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ags_communication.h>
#include "hsa.h"
#include "ags_hsa_state.h"

// Tracks state of the connection to AGS. Initialized during
// InitializeAGSConnection.
AGSHSAState *ags_state = NULL;

// Closes any connections to pipes and deletes the process-specific pipe if
// possible.
void CleanupAGSState(void) {
  if (!ags_state) return;
  close(ags_state->fd);
  ags_state->fd = -1;
  free(ags_state);
  ags_state = NULL;
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
  return true;
}

bool SendInitialMessage(hsa_status_t *result, bool *prevent_default) {
  AGSRequestHeader request;
  AGSResponse response;
  *prevent_default = 0;
  if (!ags_state) return true;
  memset(&request, 0, sizeof(request));
  request.pid = getpid();
  request.tid = GetTID();
  request.request_type = AGS_NEW_PROCESS;
  request.data_size = 0;
  request.request_id = ags_state->request_id++;
  printf("Sending initial request to AGS for PID %d.\n", (int) request.pid);
  if (send(ags_state->fd, &request, sizeof(request), 0) != sizeof(request)) {
    printf("Failed notifying AGS of process creation: %s\n", strerror(errno));
    CleanupAGSState();
    return false;
  }
  if (!GetAGSResponse(&response, 0, NULL)) {
    printf("Failed receiving initial response from AGS: %s\n",
      strerror(errno));
    CleanupAGSState();
    return false;
  }
  VerifyRequestIDs(&request, &response);
  *result = (hsa_status_t) response.hsa_status;
  *prevent_default = response.prevent_default;
  return true;
}

bool EndAGSConnection(hsa_status_t *result) {
  AGSRequestHeader request;
  AGSResponse response;
  if (!ags_state) return true;
  memset(&request, 0, sizeof(request));
  request.pid = getpid();
  request.tid = GetTID();
  request.request_type = AGS_PROCESS_QUITTING;
  request.data_size = 0;
  request.request_id = ags_state->request_id++;
  if (send(ags_state->fd, &request, sizeof(request), 0) != sizeof(request)) {
    printf("Failed notifying AGS of process quitting: %s\n", strerror(errno));
    CleanupAGSState();
    return true;
  }
  if (recv(ags_state->fd, &response, sizeof(response), 0) !=
    sizeof(response)) {
    printf("Failed receiving final response from AGS: %s\n", strerror(errno));
    CleanupAGSState();
    return true;
  }
  VerifyRequestIDs(&request, &response);
  CleanupAGSState();
  if (response.prevent_default) {
    *result = (hsa_status_t) response.hsa_status;
    return false;
  }
  return true;
}

bool GetAGSResponse(AGSResponse *response, uint32_t data_size, void *data) {
  if (!ags_state) {
    memset(response, 0, sizeof(*response));
    return true;
  }
  if (recv(ags_state->fd, response, sizeof(*response), 0) !=
      sizeof(*response)) {
    printf("Failed receiving response from AGS: %s\n", strerror(errno));
    CleanupAGSState();
    return false;
  }
  if (response->data_size == 0) return true;
  if (response->data_size > data_size) {
    printf("Insufficient buffer size to read data from pipe.\n");
    CleanupAGSState();
    return false;
  }
  if (recv(ags_state->fd, data, response->data_size, 0) !=
    response->data_size) {
    printf("Failed receiving response data: %s\n", strerror(errno));
    CleanupAGSState();
    return false;
  }
  return true;
}

bool SendAGSPlaceholderRequest(const char *file, const char *func, int line,
    hsa_status_t *result) {
  AGSRequestHeader header;
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
  header.pid = getpid();
  header.tid = GetTID();
  header.request_type = AGS_PLACEHOLDER_REQUEST;
  header.request_id = ags_state->request_id++;

  if (send(ags_state->fd, &header, sizeof(header), 0) != sizeof(header)) {
    printf("Failed sending placeholder request header to AGS: %s\n",
      strerror(errno));
    CleanupAGSState();
    return true;
  }
  if (send(ags_state->fd, data, header.data_size, 0) != header.data_size) {
    printf("Failed sending placeholder request data to AGS: %s\n",
      strerror(errno));
    CleanupAGSState();
    return true;
  }
  if (!GetAGSResponse(&response, 0, NULL)) return true;
  VerifyRequestIDs(&header, &response);
  if (response.prevent_default) {
    *result = (hsa_status_t) response.hsa_status;
    return false;
  }
  return true;
}

void VerifyRequestIDs(AGSRequestHeader *request, AGSResponse *response) {
  if (request->request_id == response->request_id) return;
  printf("Error: Response request_id doesn't match the request!\n");
  printf("Request (data omitted):\n");
  request->data_size = 0;
  PrintRequestInformation(request, NULL, "  ");
  printf("Request ID in response = %lu\n",
    (unsigned long) response->request_id);
  exit(1);
}
