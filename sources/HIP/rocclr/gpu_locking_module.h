// This file contains structs and definitions used when interfacing with the
// GPU locking module.

#ifndef GPU_LOCKING_MODULE_H
#define GPU_LOCKING_MODULE_H
#include <linux/ioctl.h>

#ifdef __cplusplus
extern "C" {
#endif

// Determines the number of locks that the module creates. The lock_id in the
// GPULockArgs struct must be less than this number.
#define GPU_LOCK_COUNT (4)

// A pointer to this struct is used as an argument for several ioctls to the
// module, including acquiring and releasing locks.
typedef struct {
  // The ID of the lock to acquire.
  uint32_t lock_id;
} GPULockArgs;

// A pointer to this struct is used as an argument to the ioctl for setting
// a deadline associated with a given thread-group ID (PID from userspace). If
// no deadline has been set, then the lock will be granted in best-effort FIFO
// order (any process with an explicitly-set deadline will have higher priority
// than one for which no deadline has been set).
typedef struct {
  // The PID of the process for which the deadline should be set. (Must match
  // the tgid of the task making the LOCK_ACQUIRE ioctl.)
  int pid;
  // The deadline to set. This must be a number of *nanoseconds relative to the
  // current time*.  Set this to 0 to unset task's deadline, making it revert
  // back to best-effort FIFO if it attempts to acquire the lock again.
  uint64_t deadline;
} SetDeadlineArgs;

// Send this ioctl to acquire the lock. This will return -EINTR *without*
// acquiring the lock, if a signal is received before the lock is acquired.
#define GPU_LOCK_ACQUIRE_IOC _IOW('h', 0xaa, GPULockArgs)

// Send this ioctl to release the lock, after it has been acquired. Returns an
// error if the requesting process doesn't hold the lock for the specified
// partition ID.
#define GPU_LOCK_RELEASE_IOC _IOW('h', 0xab, GPULockArgs)

// Send this ioctl to set or remove a deadline associated with a particular
// PID. Does not check that the PID is a process that actually exists when
// setting a deadline.  Does *not* affect any lock-acquire requests made before
// this ioctl--only subsequent ones.  The caller *must* call this to unset any
// deadline that has been previously set, when the deadline is no longer
// needed.  Failure to do so will result in subsequent invocations of this
// ioctl failing, as the module can only track a small number of deadlines for
// now.
#define GPU_LOCK_SET_DEADLINE_IOC _IOW('h', 0xac, SetDeadlineArgs)

// A fail-safe mechanism to unset all deadlines being tracked by the module.
#define GPU_LOCK_CLEAR_ALL_DEADLINES_IOC _IO('h', 0xad)

// Send this ioctl to release all waiters on the given lock ID.
#define GPU_LOCK_RELEASE_ALL_IOC _IOW('h', 0xae, GPULockArgs)

// Send this ioctl to release all waiters for all locks. Intended to be used
// as a failsafe mechanism when testing.
#define PRIORITY_DEBUG_RELEASE_ALL_IOC _IO('h', 0xaf)

// TODO (next): Test these. Also, probably just copy this file to the HIP
// source directory.

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // GPU_LOCKING_MODULE_H

