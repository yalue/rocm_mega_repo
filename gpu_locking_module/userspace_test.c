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
#include <unistd.h>
#include "gpu_locking_module.h"

// Tests sending an ioctl to AGS. Requires a file descriptor to /dev/ags and a
// PID for a "target" process.
static int TestIoctl(int fd) {
  GPULockArgs args;
  int result;
  args.partition_id = 0;
  args.priority = 1337;
  result = ioctl(fd, GPULOCK_IOC_ACQUIRE_LOCK, &args);
  if (result != 0) {
    printf("ioctl returned an error: %s\n", strerror(errno));
    return 0;
  }

  return 1;
}

// Simply wraps TestIoctl, but matches the function signature expected by
// pthread_create.
static void* TestIoctlThreadEntry(void *arg) {
  printf("Testing ioctl in separate thread.\n");
  if (!TestIoctl((int) ((uintptr_t) arg))) {
    return NULL;
  }
  return (void *) 1;
}

// Tests whether the ioctl works in a separate thread.
static int TestIoctlInThread(int fd) {
  pthread_t thread;
  void *thread_result = NULL;
  int result = pthread_create(&thread, NULL, TestIoctlThreadEntry,
    (void *) ((uintptr_t) fd));
  if (result != 0) {
    printf("pthread_create failed: %s\n", strerror(result));
    return 0;
  }
  result = pthread_join(thread, &thread_result);
  if (result != 0) {
    printf("pthread_join failed: %s\n", strerror(result));
    return 0;
  }
  if (!thread_result) {
    printf("TestIoctl failed in child thread.\n");
    return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  int fd;
  fd = open("/dev/gpu_locking_module", O_RDWR);
  if (fd < 0) {
    printf("Error opening /dev/gpu_locking_module: %s\n", strerror(errno));
    return 1;
  }
  if (!TestIoctlInThread(fd)) {
    printf("Thread test failed.\n");
  } else {
    printf("Thread test succeeded!\n");
  }
  if (!TestIoctl(fd)) {
    printf("Test failed.\n");
  } else {
    printf("Test succeeded!\n");
  }
  close(fd);
  return 0;
}
