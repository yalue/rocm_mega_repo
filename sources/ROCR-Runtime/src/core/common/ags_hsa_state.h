#ifndef AGS_HSA_STATE_H
#define AGS_HSA_STATE_H
#include <cstdlib>
#include <cstdint>
#include <pthread.h>
// This file contains the functions and struct definitions used for managing an
// HSA application's connection to the Arbiter for GPU Sharing (AGS).
#include <ags_communication.h>

// If defined, ENABLE_FULL_AGS_INTERCEPTION will cause all of the HSA API
// functions to execute in the context of AGS' process rather than the client
// processes. If false, the HSA clients will only send placeholder requests to
// AGS, which are useful for tracing and ensuring that nothing's broken.
//#define ENABLE_FULL_AGS_INTERCEPTION (1)

// Sends a placeholder request to AGS. Returns from the surrounding function if
// AGS' response sets prevent_default (AGS should not do this, though). Allows
// the surrounding function to continue as usual in all other cases, including
// if an error occurs or if AGS is not running.
#define DoAGSPlaceholderRequest() do { \
    hsa_status_t result_tmp; \
    if (!SendAGSPlaceholderRequest(__FILE__, __func__, __LINE__, \
      &result_tmp)) { \
      return result_tmp; \
    } \
  } while (0);

// The same as DoAGSPlaceholderRequest, but simply exits if prevent_default is
// nonzero.
#define DoAGSPlaceholderRequestNoReturn() do { \
    hsa_status_t result_tmp; \
    if (!SendAGSPlaceholderRequest(__FILE__, __func__, __LINE__, \
      &result_tmp)) { \
      printf("Got error/prevent_default from SendAGSPlaceholderRequest.\n"); \
      exit(1); \
    } \
  } while (0);

// Maintains the state of the current process' connection to AGS.
typedef struct {
  // The file descriptor for our socket to AGS. Will be -1 if AGS isn't
  // running.
  int fd;
  // Contains a unique ID to be associated with each request. Will be
  // incremented after each request.
  uint64_t request_id;
  // This is used to ensure that we wait for any pending response from AGS
  // before trying to send a new request.
  pthread_mutex_t mutex;
} AGSHSAState;

// A single AGS state instance, non-NULL if AGS is connected. Initialized by
// InitializeAGSConnection.
extern AGSHSAState *ags_state;

// Acquire the AGS mutex.
void LockAGS(void);

// Release the AGS mutex.
void UnlockAGS(void);

// This should only be called once--during application initialization. It
// opens the AGS socket and allocates space to hold AGS' state. Returns false
// on error. If AGS isn't running (i.e. we can't connect to the server), then
// AGS' state won't be allocated.
bool InitializeAGSConnection(void);

// Sends the initial message to AGS. Should generally be called once,
// immediately after InitializeAGSConnection. Returns false on error. On
// success, it returns true and sets the prevent_default and result fields from
// AGS' response, to potentially prevent normal execution of hsa_init.
bool SendInitialMessage(hsa_status_t *result, bool *prevent_default);

// This should be called when the HSA runtime is exiting. It sends the message
// to AGS that the process is exiting, waits for the final message from AGS,
// then closes our end of the connection. If AGS sets prvent_default to true,
// then this returns false and sets *result to the value that hsa_shut_down
// should return.
bool EndAGSConnection(hsa_status_t *result);

// This function takes care of sending a request to AGS and receiving the
// response. Handles acquiring and releasing the AGS lock. Returns false on
// error, cleaning up the AGS state and disconnecting from AGS. All buffers
// must be allocated by the caller, and the response data buffer must be large
// enough to hold all of the response data. Both request_data and response_data
// may be NULL if they aren't needed.
bool DoAGSTransaction(AGSRequest *request, void *request_data,
    AGSResponse *response, uint32_t response_data_size, void *response_data);

// Sends a AGS_PLACEHOLDER_REQUEST. If the received response has
// prevent_default set to 1, then this returns false and sets *result to the
// hsa_status specified in AGS' response.
bool SendAGSPlaceholderRequest(const char *file, const char *func, int line,
    hsa_status_t *result);

// Sends an AGS_HSA_ITERATE_AGENTS request. After receiving the response from
// AGS, this returns false and sets *result to the result from AGS. Otherwise,
// this returns true and the normal HSA function should continue.
bool AGSHandleIterateAgents(hsa_status_t (*callback)(hsa_agent_t agent,
    void *data), void *data, hsa_status_t *result);

// Sends an AGS_AGENT_GET_INFO request. Returns the response from AGS to the
// caller. The return boolean and result argument work like the other functions
// here.
bool AGSHandleAgentGetInfo(hsa_agent_t agent, hsa_agent_info_t attribute,
    void *data, hsa_status_t *result);

bool AGSHandleAgentIterateRegions(hsa_agent_t agent, hsa_status_t (*callback)(
    hsa_region_t region, void *data), void *data, hsa_status_t *result);

bool AGSHandleRegionGetInfo(hsa_region_t region, hsa_region_info_t attribute,
    void *data, hsa_status_t *result);

#endif  // AGS_HSA_STATE_H
