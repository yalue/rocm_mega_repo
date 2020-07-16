#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

// The device number on which to perform the computation.
#define DEVICE_ID (0)

// The number of 64-bit integers in each vectors to add.
#define VECTOR_LENGTH (200 * 1024 * 1024)

// This macro takes a hipError_t value and exits if it isn't equal to
// hipSuccess.
#define CheckHIPError(val) (InternalHIPErrorCheck((val), #val, __FILE__, __LINE__))

static void InternalHIPErrorCheck(hipError_t result, const char *fn,
    const char *file, int line) {
  if (result == hipSuccess) return;
  printf("HIP error %d: %s. In %s, line %d (%s)\n", (int) result,
    hipGetErrorString(result), file, line, fn);
  exit(1);
}

static double CurrentSeconds(void) {
  double to_return = 0;
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  to_return = t.tv_sec;
  to_return += t.tv_nsec * 1e-9;
  return to_return;
}

// Spins in a busy loop for the number of times given by "count". Set dummy to
// (uint64_t *) NULL; it's only there to prevent optimizations.
__global__ void CounterSpin(uint64_t count, uint64_t *dummy) {
  uint64_t index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  uint64_t accumulator = 0;
  uint64_t i;
  if (index != 0) return;
  for (i = 0; i < count; i++) {
    // Random junk to try to prevent optimizations
    if ((i + index) == (count % 413)) continue;
    accumulator += index + 1;
  }
  if (dummy) *dummy = accumulator;
}

// Sets the HIP device and prints out some device info.
static void SetupDevice(void) {
  printf("Setting device...\n");
  hipDeviceProp_t props;
  CheckHIPError(hipSetDevice(DEVICE_ID));
  CheckHIPError(hipSetDeviceFlags(hipDeviceScheduleYield));
  CheckHIPError(hipGetDeviceProperties(&props, DEVICE_ID));
  printf("Running on device: %s\n", props.name);
  printf("Using AMD GPU architecture %d. Has %d multiprocessors, "
    "supporting %d threads each.\n", props.gcnArch, props.multiProcessorCount,
    props.maxThreadsPerMultiProcessor);
  printf("Total global device memory: %llu MB\n",
    ((unsigned long long) props.totalGlobalMem) / (1024 * 1024));
}

int main(int argc, char **argv) {
  double start_time, after_launch, after_sync;
  hipStream_t stream;
  int block_count = 1024;
  int thread_count = 512;
  uint64_t spin_iterations = 100000000;

  // It looks like the stupid optimizer doesn't even call the kernel if we
  // don't get results from it...
  uint64_t *device_junk = NULL;
  uint64_t host_junk;

  SetupDevice();
  CheckHIPError(hipStreamCreate(&stream));
  CheckHIPError(hipMalloc(&device_junk, sizeof(*device_junk)));

  // Do a warm-up iteration of the kernel before we take a time measurement.
  hipLaunchKernelGGL(CounterSpin, block_count, thread_count, 0, stream,
    10, device_junk);
  CheckHIPError(hipStreamSynchronize(stream));
  CheckHIPError(hipMemcpyAsync(&host_junk, device_junk, sizeof(host_junk),
    hipMemcpyDeviceToHost, stream));
  CheckHIPError(hipStreamSynchronize(stream));
  printf("Warm-up accumulator: %lu\n", (unsigned long) host_junk);

  // Measure the time for an initial test (to help with manual calibration).
  start_time = CurrentSeconds();
  hipLaunchKernelGGL(CounterSpin, block_count, thread_count, 0, stream,
    spin_iterations, device_junk);
  after_launch = CurrentSeconds();
  // Intentionally *don't* use hipStreamSynchronize here, since we're going to
  // experiment with destroying the stream in the later version. Therefore, we
  // can't sync on the stream; it will have been destroyed.
  CheckHIPError(hipDeviceSynchronize());
  after_sync = CurrentSeconds();
  CheckHIPError(hipMemcpy(&host_junk, device_junk, sizeof(host_junk),
    hipMemcpyDeviceToHost));
  CheckHIPError(hipDeviceSynchronize());
  printf("Running the kernel for %lu iterations took %f seconds.\n",
    (unsigned long) spin_iterations, after_sync - start_time);
  printf("It took %f seconds for hipLaunchKernelGGL to return.\n",
    after_launch - start_time);
  printf("Accumulator result: %lu\n", (unsigned long) host_junk);

  // Measure the time if we kill the stream during kernel execution.
  start_time = CurrentSeconds();
  hipLaunchKernelGGL(CounterSpin, block_count, thread_count, 0, stream,
    spin_iterations, device_junk);
  after_launch = CurrentSeconds();
  // The key part of the test: destroy the stream after the kernel launch, but
  // before it completes.
  //CheckHIPError(hipStreamDestroy(stream));  // This didn't crash!!
  auto queue = hipStreamGetHSAQueue(stream);
  if (!queue) {
    printf("hipStreamGetHSAQueue failed\n");
    exit(1);
  }
  printf("Inactivating stream HSA queue @ %p\n", queue);
  hsa_status_t res2 = hsa_queue_inactivate(queue);
  printf("hsa_queue_inactivate returned %d\n", (int) res2);
  // Intentionally *don't* use hipStreamSynchronize here, since we're going to
  // experiment with destroying the stream in the later version. Therefore, we
  // can't sync on the stream; it will have been destroyed.
  CheckHIPError(hipDeviceSynchronize());
  after_sync = CurrentSeconds();
  CheckHIPError(hipMemcpy(&host_junk, device_junk, sizeof(host_junk),
    hipMemcpyDeviceToHost));
  CheckHIPError(hipDeviceSynchronize());
  printf("Running the kernel for %lu iterations took %f seconds.\n",
    (unsigned long) spin_iterations, after_sync - start_time);
  printf("It took %f seconds for hipLaunchKernelGGL to return.\n",
    after_launch - start_time);
  printf("Accumulator result: %lu\n", (unsigned long) host_junk);

  printf("Done. Cleaning up.\n");
  CheckHIPError(hipDeviceSynchronize());
  CheckHIPError(hipFree(device_junk));
  return 0;
}
