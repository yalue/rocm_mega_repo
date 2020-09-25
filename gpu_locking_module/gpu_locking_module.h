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
  // The request's priority. Lower = higher priority.
  uint64_t priority;
} GPULockArgs;

// Send this ioctl to acquire the lock. This will return -EAGAIN *without*
// acquiring the lock, if a signal is received before the lock is acquired.
#define GPULOCK_IOC_ACQUIRE_LOCK _IOW('h', 0xaa, GPULockArgs)

// Send this ioctl to release the lock, after it has been acquired. Returns an
// error if the requesting process doesn't hold the lock for the specified
// partition ID.
#define GPULOCK_IOC_RELEASE_LOCK _IOW('h', 0xab, GPULockArgs)

// Send this ioctl to release all locks associated with the specified
// partition.
#define GPULOCK_IOC_RELEASE_ALL_PARTITION _IOW('h', 0xac, GPULockArgs)

// Send this ioctl to release all locks associated with all partitions.
// Intended to be used for a failsafe mechanism to unblock any stuck waiters.
#define GPULOCK_IOC_RELEASE_ALL _IO('h', 0xad)

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // GPU_LOCKING_MODULE_H

