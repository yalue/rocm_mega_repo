// This file simply serves to test the interface for the GPU locking module. I
// won't include it in the Makefile, just build it using:
// gcc -O3 -Wall -Werror -o userspace_test ./userspace_test.c
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#include "gpu_locking_module.h"

// Tests sending an ioctl to AGS. Requires a file descriptor to /dev/ags and a
// PID for a "target" process.
static int TestIoctl(int fd) {
  AcquireGPULockArgs args;
  int result;
  args.partition_id = 0;
  args.deadline = 1337;
  result = ioctl(fd, GPULOCKIOC_ACQUIRE_LOCK, &args);
  if (result != 0) {
    printf("ioctl returned an error: %s\n", strerror(errno));
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
  if (!TestIoctl(fd)) {
    printf("Test failed.\n");
  } else {
    printf("Test succeeded!\n");
  }
  close(fd);
  return 0;
}
