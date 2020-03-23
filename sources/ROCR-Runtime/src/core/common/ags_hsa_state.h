#ifndef AGS_HSA_STATE_H
#define AGS_HSA_STATE_H
#include <cstdlib>
#include <cstdint>
// This file contains the functions and struct definitions used for managing an
// HSA application's connection to the Arbiter for GPU Sharing (AGS).
#include <ags_communication.h>

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
} AGSHSAState;

// A single AGS state instance, non-NULL if AGS is connected. Initialized by
// InitializeAGSConnection.
extern AGSHSAState *ags_state;

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

// Fills in the AGSResponse struct as well as the data buffer, which the caller
// must ensure is able to hold enough bytes for any possible response data
// (the needed size will depend on context, and the caller should usually be
// able to make a reasonable estimate, e.g. an expected struct holding
// additional results from AGS). The actual data size will be held in the
// response's data_size field; the data_size argument to this function instead
// must contain the number of bytes the caller has allocated for the given
// data buffer. Returns false on error. Fills in the response even if AGS isn't
// connected, but fills in the prevent_default field to 0.
bool GetAGSResponse(AGSResponse *response, uint32_t data_size, void *data);

// Sends a AGS_PLACEHOLDER_REQUEST. If the received response has
// prevent_default set to 1, then this returns false and sets *result to the
// hsa_status specified in AGS' response.
bool SendAGSPlaceholderRequest(const char *file, const char *func, int line,
    hsa_status_t *result);

// A sanity-checking function to check that the response from AGS is for the
// given request. Prints a message and exits on error.
void VerifyRequestIDs(AGSRequestHeader *request, AGSResponse *response);

#endif  // AGS_HSA_STATE_H
