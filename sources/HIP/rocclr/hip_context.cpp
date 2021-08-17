/* Copyright (c) 2015-present Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
#include "hip_platform.hpp"
#include "platform/runtime.hpp"
#include "utils/flags.hpp"
#include "utils/versions.hpp"

// Pull in some junk needed to issue our ioctls.
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
// Ensure this is an exact copy from the running version of the module.
#include "gpu_locking_module.h"

std::vector<hip::Device*> g_devices;

namespace hip {

thread_local Device* g_device = nullptr;
thread_local std::stack<Device*> g_ctxtStack;
thread_local hipError_t g_lastError = hipSuccess;
std::once_flag g_ihipInitialized;
Device* host_device = nullptr;

int gpu_lock_fd = -1;
int gpu_lock_id = 0;
int simple_hip_trace = 0;

// Takes the name of an environment variable expected to contain a single
// integer. The integer may be specified in hex or octal (it must be a format
// that strtol can handle).  If the variable exists and contains a valid
// integer, then *value will be set to the integer's value and this function
// will return true. In any other case this will return false.
static bool GetEnvVarValue(const char *name, long *value) {
  char *s = getenv(name);
  char *end = NULL;
  long v = 0;
  if (!s) return false;
  v = strtol(s, &end, 0);
  // The string was either empty or contained a non-numeric character.
  if ((s == end) || (*end != 0)) {
    printf("Env var %s was defined, but wasn't a valid number!\n", name);
    return false;
  }
  *value = v;
  return true;
}

void init() {
  // (otternes): Hacky code to connect to my kernel-module chardev for
  // priority-based locking of the GPU. We check for the chardev's presence as
  // well as the environment-variable settings here.
  long v;
  bool found_env_var;
  gpu_lock_fd = -1;
  gpu_lock_id = 0;
  found_env_var = GetEnvVarValue("IGNORE_GPU_LOCK_CHARDEV", &v);
  if (!found_env_var || (v <= 0)) {
    gpu_lock_fd = open("/dev/gpu_locking_module", O_RDWR);
    if (gpu_lock_fd < 0) {
      printf("Error opening /dev/gpu_locking_module: %s\n", strerror(errno));
      gpu_lock_fd = -1;
    } else {
      printf("/dev/gpu_locking_module is available. Will try to use it.\n");
    }
  }
  if (GetEnvVarValue("GPU_LOCK_ID", &v)) {
    gpu_lock_id = v;
    if (gpu_lock_id < 0) {
      printf("Likely error: Setting a negative GPU lock ID %d\n", gpu_lock_id);
    }
  }
  if (GetEnvVarValue("SIMPLE_HIP_TRACE", &v)) {
    simple_hip_trace = 1;
  }

  if (!amd::Runtime::initialized()) {
    amd::IS_HIP = true;
    GPU_NUM_MEM_DEPENDENCY = 0;
    amd::Runtime::init();
  }

  const std::vector<amd::Device*>& devices = amd::Device::getDevices(CL_DEVICE_TYPE_GPU, false);

  for (unsigned int i=0; i<devices.size(); i++) {
    const std::vector<amd::Device*> device(1, devices[i]);
    amd::Context* context = new amd::Context(device, amd::Context::Info());
    if (!context) return;

    // Enable active wait on the device by default
    devices[i]->SetActiveWait(true);

    if (context && CL_SUCCESS != context->create(nullptr)) {
      context->release();
    } else {
      g_devices.push_back(new Device(context, i));
    }
  }

  amd::Context* hContext = new amd::Context(devices, amd::Context::Info());
  if (!hContext) return;

  if (CL_SUCCESS != hContext->create(nullptr)) {
    hContext->release();
  }
  host_device = new Device(hContext, -1);

  PlatformState::instance().init();
}

// Used to acquire a lock before any kernel launch. See hip_internal.hpp.
void AcquireGPULock() {
  // Silently do nothing if the locking module wasn't loaded.
  if (gpu_lock_fd < 0) return;
  GPULockArgs args;
  int result;
  // This lock id was set using the GPU_LOCK_ID env var in hip::init().
  args.lock_id = gpu_lock_id;
  result = ioctl(gpu_lock_fd, GPU_LOCK_ACQUIRE_IOC, &args);
  if (result != 0) {
    printf("AcquireGPULock failed for PID %d: %s\n", getpid(),
      strerror(errno));
    exit(1);
  }
}

// Releases the GPU lock. Must be called after a kernel completes. See
// hip_internal.hpp.
void ReleaseGPULock() {
  if (gpu_lock_fd < 0) return;
  GPULockArgs args;
  int result;
  // See comment in AcquireGPULock.
  args.lock_id = gpu_lock_id;
  result = ioctl(gpu_lock_fd, GPU_LOCK_RELEASE_IOC, &args);
  if (result != 0) {
    printf("ReleaseGPULock failed: %s\n", strerror(errno));
    exit(1);
  }
}

Device* getCurrentDevice() {
  return g_device;
}

void setCurrentDevice(unsigned int index) {
  assert(index<g_devices.size());
  g_device = g_devices[index];
}

amd::HostQueue* getQueue(hipStream_t stream) {
 if (stream == nullptr) {
    return getNullStream();
  } else {
    constexpr bool WaitNullStreamOnly = true;
    amd::HostQueue* queue = reinterpret_cast<hip::Stream*>(stream)->asHostQueue();
    if (!(reinterpret_cast<hip::Stream*>(stream)->Flags() & hipStreamNonBlocking)) {
      iHipWaitActiveStreams(queue, WaitNullStreamOnly);
    }
    return queue;
  }
}

// ================================================================================================
amd::HostQueue* getNullStream(amd::Context& ctx) {
  for (auto& it : g_devices) {
    if (it->asContext() == &ctx) {
      return it->NullStream();
    }
  }
  // If it's a pure SVM allocation with system memory access, then it shouldn't matter which device
  // runtime selects by default
  if (hip::host_device->asContext() == &ctx) {
    // Return current...
    return getNullStream();
  }
  return nullptr;
}

// ================================================================================================
amd::HostQueue* getNullStream() {
  Device* device = getCurrentDevice();
  return device ? device->NullStream() : nullptr;
}

};

using namespace hip;

hipError_t hipInit(unsigned int flags) {
  HIP_INIT_API(hipInit, flags);

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxCreate(hipCtx_t *ctx, unsigned int flags,  hipDevice_t device) {
  HIP_INIT_API(hipCtxCreate, ctx, flags, device);

  if (static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *ctx = reinterpret_cast<hipCtx_t>(g_devices[device]);

  // Increment ref count for device primary context
  g_devices[device]->retain();
  g_ctxtStack.push(g_devices[device]);

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
  HIP_INIT_API(hipCtxSetCurrent, ctx);

  if (ctx == nullptr) {
    if(!g_ctxtStack.empty()) {
      g_ctxtStack.pop();
    }
  } else {
    hip::g_device = reinterpret_cast<hip::Device*>(ctx);
    if(!g_ctxtStack.empty()) {
      g_ctxtStack.pop();
    }
    g_ctxtStack.push(hip::getCurrentDevice());
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxGetCurrent(hipCtx_t* ctx) {
  HIP_INIT_API(hipCtxGetCurrent, ctx);

  *ctx = reinterpret_cast<hipCtx_t>(hip::getCurrentDevice());

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig) {
  HIP_INIT_API(hipCtxGetSharedMemConfig, pConfig);

  *pConfig = hipSharedMemBankSizeFourByte;

  HIP_RETURN(hipSuccess);
}

hipError_t hipRuntimeGetVersion(int *runtimeVersion) {
  HIP_INIT_API(hipRuntimeGetVersion, runtimeVersion);

  if (!runtimeVersion) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *runtimeVersion = AMD_PLATFORM_BUILD_NUMBER;

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxDestroy(hipCtx_t ctx) {
  HIP_INIT_API(hipCtxDestroy, ctx);

  hip::Device* dev = reinterpret_cast<hip::Device*>(ctx);
  if (dev == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Need to remove the ctx of calling thread if its the top one
  if (!g_ctxtStack.empty() && g_ctxtStack.top() == dev) {
    g_ctxtStack.pop();
  }

  // Remove context from global context list
  for (unsigned int i = 0; i < g_devices.size(); i++) {
    if (g_devices[i] == dev) {
      // Decrement ref count for device primary context
      dev->release();
    }
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxPopCurrent(hipCtx_t* ctx) {
  HIP_INIT_API(hipCtxPopCurrent, ctx);

  hip::Device** dev = reinterpret_cast<hip::Device**>(ctx);
  if (!g_ctxtStack.empty()) {
    if (dev != nullptr) {
      *dev = g_ctxtStack.top();
    }
    g_ctxtStack.pop();
  } else {
    DevLogError("Context Stack empty \n");
    HIP_RETURN(hipErrorInvalidContext);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
  HIP_INIT_API(hipCtxPushCurrent, ctx);

  hip::Device* dev = reinterpret_cast<hip::Device*>(ctx);
  if (dev == nullptr) {
    HIP_RETURN(hipErrorInvalidContext);
  }

  hip::g_device = dev;
  g_ctxtStack.push(hip::getCurrentDevice());

  HIP_RETURN(hipSuccess);
}

hipError_t hipDriverGetVersion(int* driverVersion) {
  HIP_INIT_API(hipDriverGetVersion, driverVersion);

  auto* deviceHandle = g_devices[0]->devices()[0];
  const auto& info = deviceHandle->info();

  if (driverVersion) {
    *driverVersion = AMD_PLATFORM_BUILD_NUMBER * 100 +
                     AMD_PLATFORM_REVISION_NUMBER;
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxGetDevice(hipDevice_t* device) {
  HIP_INIT_API(hipCtxGetDevice, device);

  if (device != nullptr) {
    *device = hip::getCurrentDevice()->deviceId();
    HIP_RETURN(hipSuccess);
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipErrorInvalidContext);
}

hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion) {
  HIP_INIT_API(hipCtxGetApiVersion, apiVersion);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipCtxGetCacheConfig(hipFuncCache_t* cacheConfig) {
  HIP_INIT_API(hipCtxGetCacheConfig, cacheConfig);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) {
  HIP_INIT_API(hipCtxSetCacheConfig, cacheConfig);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
  HIP_INIT_API(hipCtxSetSharedMemConfig, config);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipCtxSynchronize(void) {
  HIP_INIT_API(hipCtxSynchronize, 1);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipCtxGetFlags(unsigned int* flags) {
  HIP_INIT_API(hipCtxGetFlags, flags);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active) {
  HIP_INIT_API(hipDevicePrimaryCtxGetState, dev, flags, active);

  if (static_cast<unsigned int>(dev) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (flags != nullptr) {
    *flags = 0;
  }

  if (active != nullptr) {
    *active = (g_devices[dev] == hip::getCurrentDevice())? 1 : 0;
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
  HIP_INIT_API(hipDevicePrimaryCtxRelease, dev);

  if (static_cast<unsigned int>(dev) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
  HIP_INIT_API(hipDevicePrimaryCtxRetain, pctx, dev);

  if (static_cast<unsigned int>(dev) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }
  if (pctx == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pctx = reinterpret_cast<hipCtx_t>(g_devices[dev]);

  HIP_RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
  HIP_INIT_API(hipDevicePrimaryCtxReset, dev);

  HIP_RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags) {
  HIP_INIT_API(hipDevicePrimaryCtxSetFlags, dev, flags);

  if (static_cast<unsigned int>(dev) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  } else {
    HIP_RETURN(hipErrorContextAlreadyInUse);
  }
}
