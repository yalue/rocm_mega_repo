#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <hip/hip_runtime.h>

// The device number on which to perform the computation.
#define DEVICE_ID (0)

// The number of 64-bit integers in each vectors to add.
#define VECTOR_LENGTH (200 * 1024 * 1024)

// This macro takes a hipError_t value and exits if it isn't equal to
// hipSuccess.
#define CheckHIPError(val) (InternalHIPErrorCheck((val), #val, __FILE__, __LINE__))

// This struct just keeps the vectors we'll be adding in an easy-to-access
// place.
typedef struct {
  uint64_t *host_a;
  uint64_t *host_b;
  uint64_t *host_result;
  uint64_t *device_a;
  uint64_t *device_b;
  uint64_t *device_result;
} Vectors;

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

// Should be called to clean up and free memory when no longer needed.
static void CleanupVectors(Vectors *v) {
  if (v->host_a) hipHostFree(v->host_a);
  if (v->host_b) hipHostFree(v->host_b);
  if (v->host_result) hipHostFree(v->host_result);
  if (v->device_a) hipFree(v->device_a);
  if (v->device_b) hipFree(v->device_b);
  if (v->device_result) hipFree(v->device_result);
  memset(v, 0, sizeof(*v));
}

// Fills the given vector with the specified number of random 64-bit integers.
static void RandomFill(uint64_t *buffer, int count) {
  int i;
  // I know this may not use the high bits, but this should be sufficient for a
  // sanity-check like this program.
  for (i = 0; i < count; i++) {
    buffer[i] = rand();
  }
}

// Allocates memory buffers in the given Vectors struct, and randomly fills
// the buffers.
static void InitializeVectors(Vectors *v) {
  size_t vector_size = VECTOR_LENGTH * sizeof(uint64_t);
  memset(v, 0, sizeof(*v));
  CheckHIPError(hipHostMalloc(&(v->host_a), vector_size));
  CheckHIPError(hipHostMalloc(&(v->host_b), vector_size));
  CheckHIPError(hipHostMalloc(&(v->host_result), vector_size));
  CheckHIPError(hipMalloc(&(v->device_a), vector_size));
  CheckHIPError(hipMalloc(&(v->device_b), vector_size));
  CheckHIPError(hipMalloc(&(v->device_result), vector_size));
  RandomFill(v->host_a, VECTOR_LENGTH);
  RandomFill(v->host_b, VECTOR_LENGTH);
  memset(v->host_result, 0, vector_size);
  CheckHIPError(hipMemset(v->device_result, 0, vector_size));
  CheckHIPError(hipMemcpy(v->device_a, v->host_a, vector_size,
    hipMemcpyHostToDevice));
  CheckHIPError(hipMemcpy(v->device_b, v->host_b, vector_size,
    hipMemcpyHostToDevice));
  CheckHIPError(hipDeviceSynchronize());
}

static void CPUVectorAdd(Vectors *v) {
  int i;
  for (i = 0; i < VECTOR_LENGTH; i++) {
    v->host_result[i] = v->host_a[i] + v->host_b[i];
  }
}

__global__ void GPUVectorAdd(Vectors v) {
  uint64_t index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (index >= VECTOR_LENGTH) return;
  v.device_result[index] = v.device_a[index] + v.device_b[index];
}

// Sets the HIP device and prints out some device info.
static void SetupDevice(void) {
  printf("Setting device...\n");
  hipDeviceProp_t props;
  CheckHIPError(hipSetDevice(DEVICE_ID));
  CheckHIPError(hipGetDeviceProperties(&props, DEVICE_ID));
  printf("Running on device: %s\n", props.name);
  printf("Using AMD GPU architecture %d. Has %d multiprocessors, "
    "supporting %d threads each.\n", props.gcnArch, props.multiProcessorCount,
    props.maxThreadsPerMultiProcessor);
  printf("Total global device memory: %llu MB\n",
    ((unsigned long long) props.totalGlobalMem) / (1024 * 1024));
}

int main(int argc, char **argv) {
  double start_time, end_time;
  int block_count, i, all_ok;
  Vectors v;
  SetupDevice();
  InitializeVectors(&v);
  start_time = CurrentSeconds();
  CPUVectorAdd(&v);
  end_time = CurrentSeconds();
  printf("CPU vector add took %f seconds.\n", end_time - start_time);
  block_count = VECTOR_LENGTH / 256;
  if ((VECTOR_LENGTH % 256) != 0) block_count++;

  hipStream_t stream;
  CheckHIPError(hipStreamCreate(&stream));
  //CheckHIPError(hipStreamSetComputeUnitMask(stream, 7));
  // Do a warm-up iteration of the kernel before we take a time measurement.
  hipLaunchKernelGGL(GPUVectorAdd, block_count, 256, 0, 0, v);
  CheckHIPError(hipDeviceSynchronize());
  // Reset the result vector.
  CheckHIPError(hipMemset(v.device_result, 0, VECTOR_LENGTH *
    sizeof(uint64_t)));

  start_time = CurrentSeconds();
  hipLaunchKernelGGL(GPUVectorAdd, block_count, 256, 0, stream, v);
  CheckHIPError(hipStreamSynchronize(stream));
  CheckHIPError(hipDeviceSynchronize());
  end_time = CurrentSeconds();
  printf("GPU vector add took %f seconds.\n", end_time - start_time);
  // Just re-use one of the host "input" vectors to hold the output from the
  // device, so we can verify the results.
  CheckHIPError(hipMemcpy(v.host_a, v.device_result,
    VECTOR_LENGTH * sizeof(uint64_t), hipMemcpyDeviceToHost));
  CheckHIPError(hipDeviceSynchronize());
  all_ok = 1;
  for (i = 0; i < VECTOR_LENGTH; i++) {
    if (v.host_a[i] != v.host_result[i]) {
      printf("Results mismatch at item %d.\n", i);
      all_ok = 0;
      break;
    }
  }
  if (all_ok) {
    printf("The device computation produced the correct result!\n");
  }
  CheckHIPError(hipStreamDestroy(stream));
  CleanupVectors(&v);
  return 0;
}
