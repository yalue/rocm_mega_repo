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
static AGSHSAState *ags_state = NULL;

AGSHSAState* GetAGSState(void) {
  return ags_state;
}

// Closes any connections to pipes and deletes the process-specific pipe if
// possible.
void CleanupAGSState(void) {
  if (!ags_state) return;
  if (ags_state->main_pipe) {
    fclose(ags_state->main_pipe);
    ags_state->main_pipe = NULL;
  }
  if (ags_state->process_pipe) {
    fclose(ags_state->process_pipe);
    DeletePipeWithPID(getpid());
    ags_state->process_pipe = NULL;
  }
  free(ags_state);
  ags_state = NULL;
}

// Creates a pipe for our process, at AGS_PIPE_DIR/<pid>
static FILE* CreateProcessSpecificPipe(void) {
  return CreatePipeWithName(std::to_string(getpid()).c_str(), "r+b");
}

bool InitializeAGSConnection(void) {
  AGSRequestHeader ags_request;
  ags_state = NULL;
  // Return with no error if AGS isn't running (we assume it's not running if
  // its pipe doesn't exist).
  if (!FileExists(AGS_PIPE_DIR AGS_MAIN_PIPE)) return true;
  ags_state = (AGSHSAState *) malloc(sizeof(AGSHSAState));
  if (!ags_state) {
    printf("Failed allocating AGS HSA state.\n");
    return false;
  }
  memset(ags_state, 0, sizeof(AGSHSAState));
  ags_state->main_pipe = fopen(AGS_PIPE_DIR AGS_MAIN_PIPE, "wb");
  if (!ags_state->main_pipe) {
    printf("Failed opening %s: %s\n", AGS_PIPE_DIR AGS_MAIN_PIPE,
      strerror(errno));
    free(ags_state);
    ags_state = NULL;
    return false;
  }
  ags_state->process_pipe = CreateProcessSpecificPipe();
  if (!ags_state->process_pipe) {
    printf("Failed opening process-specific pipe for PID %d\n", getpid());
    fclose(ags_state->main_pipe);
    free(ags_state);
    ags_state = NULL;
    return false;
  }
  memset(&ags_request, 0, sizeof(ags_request));
  ags_request.pid = getpid();
  ags_request.tid = GetTID();
  ags_request.request_type = AGS_NEW_PROCESS;
  ags_request.data_size = 0;
  if (fwrite(&ags_request, sizeof(ags_request), 1, ags_state->main_pipe) <= 0) {
    printf("Failed notifying AGS of process creation: %s\n", strerror(errno));
    fclose(ags_state->main_pipe);
    free(ags_state);
    ags_state = NULL;
    return false;
  }
  return true;
}

bool EndAGSConnection(hsa_status_t *result) {
  AGSRequestHeader header;
  AGSResponse response;
  if (!ags_state) return true;
  memset(&header, 0, sizeof(header));
  header.pid = getpid();
  header.tid = GetTID();
  header.request_type = AGS_PROCESS_QUITTING;
  header.data_size = 0;
  if (fwrite(&header, sizeof(header), 1, ags_state->main_pipe) < 1) {
    printf("Failed notifying AGS of process quitting: %s\n", strerror(errno));
    CleanupAGSState();
    return true;
  }
  if (fread(&response, sizeof(response), 1, ags_state->process_pipe) != 1) {
    printf("Failed reading response from AGS: %s\n", strerror(errno));
    CleanupAGSState();
    return true;
  }
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
  if (fread(response, sizeof(*response), 1, ags_state->process_pipe) < 1) {
    printf("Failed reading response from process pipe: %s\n", strerror(errno));
    return false;
  }
  if (response->data_size == 0) return true;
  if (response->data_size > data_size) {
    printf("Insufficient buffer size to read data from pipe.\n");
    return false;
  }
  if (fread(data, response->data_size, 1, ags_state->process_pipe) < 1) {
    printf("Failed reading response data: %s\n", strerror(errno));
    return false;
  }
  return true;
}

bool SendAGSPlaceholderRequest(const char *file, const char *func, int line,
    hsa_status_t *result) {
  uint8_t request[512];
  AGSResponse response;
  static_assert(sizeof(AGSRequestHeader) < sizeof(request),
    "request buffer too small");
  // The header and the data must be sent in a single write.
  AGSRequestHeader *header = (AGSRequestHeader *) request;
  uint8_t *data = request + sizeof(*header);
  int max_data_size = sizeof(request) - sizeof(*header) - 1;
  int actual_data_size = 0;

  // Do nothing if AGS isn't running.
  if (!ags_state->main_pipe) return true;

  // Set up the request with the header immediately being followed by a "data"
  // string containing the location in HSA code that made the request.
  memset(request, 0, sizeof(request));
  actual_data_size = snprintf((char *) data, max_data_size,
    "%s, at line %d in %s", func, line, file);
  if (actual_data_size > max_data_size) {
    // Just let HSA keep running if this happens (we'll just need to resize the
    // buffer at compile time if this happens).
    printf("Warning: not enough buffer space to send placeholder request.\n");
    return true;
  }
  header->pid = getpid();
  header->tid = GetTID();
  header->request_type = AGS_PLACEHOLDER_REQUEST;
  // Make sure to include the null character.
  header->data_size = actual_data_size + 1;

  // Send the request on the main pipe.
  if (fwrite(request, sizeof(AGSRequestHeader) + header->data_size, 1,
    ags_state->main_pipe) != 1) {
    printf("Failed writing placeholder request to AGS main pipe: %s\n",
      strerror(errno));
    return true;
  }

  if (!GetAGSResponse(&response, 0, NULL)) return true;
  if (response.prevent_default) {
    *result = (hsa_status_t) response.hsa_status;
    return false;
  }
  return true;
}
