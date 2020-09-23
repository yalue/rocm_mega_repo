// This file contains structs and definitions used when interfacing with the
// GPU locking module.

#ifndef GPU_LOCKING_MODULE_H
#define GPU_LOCKING_MODULE_H
#include <linux/ioctl.h>

#ifdef __cplusplus
extern "C" {
#endif

// A pointer to this struct must be passed as the first and only argument to
// the AGSIOC_MAP_SHARED_MEMORY ioctl.
typedef struct {
  // The ID of the partition to acquire.
  int partition_id;
  // The relative deadline, in milliseconds, from the time this request is
  // submitted.
  unsigned deadline;
} AcquireGPULockArgs;

#define GPULOCKIOC_ACQUIRE_LOCK _IOWR('h', 0xaa, AcquireGPULockArgs)

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // GPU_LOCKING_MODULE_H

