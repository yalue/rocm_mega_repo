// This file simply serves to test the interface for the GPU locking module. I
// won't include it in the Makefile, just build it using:
// gcc -O3 -Wall -Werror -o userspace_test ./userspace_test.c -lpthread
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include "gpu_locking_module.h"

static int GetTID(void) {
  pid_t tid = syscall(SYS_gettid);
  return (int) tid;
}

// Acquires the lock for the given partition, with the specified priority.
// Returns 0 on error.
static int AcquireLock(int fd, int partition, uint64_t priority) {
  GPULockArgs args;
  int result;
  args.partition_id = partition;
  args.priority = priority;
  result = ioctl(fd, GPULOCK_IOC_ACQUIRE_LOCK, &args);
  if (result != 0) {
    printf("Acquire ioctl returned an error: %s\n", strerror(errno));
    return 0;
  }
  return 1;
}

// Releases the lock for the given partition. Returns 0 on error. Returns an
// error if the lock for the given partition isn't held by the caller.
static int ReleaseLock(int fd, int partition) {
  GPULockArgs args;
  int result;
  args.partition_id = partition;
  args.priority = 0;
  result = ioctl(fd, GPULOCK_IOC_RELEASE_LOCK, &args);
  if (result != 0) {
    printf("Release ioctl returned an error: %s\n", strerror(errno));
    return 0;
  }
  return 1;
}

// Tests acquiring and releasing the lock. Returns 0 if an error occurs at any
// step of the process.
static int TestIoctl(int fd) {
  char my_info[256];
  memset(my_info, 0, sizeof(my_info));
  snprintf(my_info, sizeof(my_info) - 1, "PID %d, TID %d", (int) getpid(),
    GetTID());
  printf("%s attempting to acquire lock.\n", my_info);
  if (!AcquireLock(fd, 0, 1)) {
    printf("%s failed acquiring lock.\n", my_info);
    return 0;
  }
  printf("%s acquired lock successfully. Sleeping a bit.\n", my_info);
  sleep(1);
  if (!ReleaseLock(fd, 0)) {
    printf("%s failed releasing lock.\n", my_info);
    return 0;
  }
  printf("%s released lock successfully.\n", my_info);
  return 1;
}

// Simply wraps TestIoctl, but matches the function signature expected by
// pthread_create.
static void* TestIoctlThreadEntry(void *arg) {
  if (!TestIoctl((int) ((uintptr_t) arg))) return NULL;
  return (void *) 1;
}

// Tests whether the ioctl works in a separate thread. Returns the thread, or 0
// on error.
static pthread_t TestIoctlInThread(int fd) {
  pthread_t thread;
  int result = pthread_create(&thread, NULL, TestIoctlThreadEntry,
    (void *) ((uintptr_t) fd));
  if (result != 0) {
    printf("pthread_create failed: %s\n", strerror(result));
    return 0;
  }
  return thread;
}

// Waits for the ioctl test thread to finish. Returns 0 on error, or if the
// thread exits with an error.
static int WaitForThread(pthread_t thread) {
  void *thread_result = NULL;
  int result = pthread_join(thread, &thread_result);
  if (result != 0) {
    printf("pthread_join failed: %s\n", strerror(result));
    return 0;
  }
  if (!thread_result) {
    printf("Child thread failed testing ioctl.\n");
    return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  int fd;
  pthread_t thread;
  fd = open("/dev/gpu_locking_module", O_RDWR);
  if (fd < 0) {
    printf("Error opening /dev/gpu_locking_module: %s\n", strerror(errno));
    return 1;
  }
  printf("Parent PID %d, TID %d\n", (int) getpid(), GetTID());
  thread = TestIoctlInThread(fd);
  if (!thread) {
    printf("Failed starting test thread.\n");
    return 1;
  }
  if (!TestIoctl(fd)) {
    printf("Test failed.\n");
  } else {
    printf("Test succeeded!\n");
  }
  if (!WaitForThread(thread)) {
    printf("The test thread failed.\n");
  } else {
    printf("The test thread succeeded!\n");
  }
  close(fd);
  return 0;
}
