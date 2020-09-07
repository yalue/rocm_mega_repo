/* Copyright (c) 2008-present Advanced Micro Devices, Inc.

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

#ifndef WITHOUT_HSA_BACKEND

#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "os/os.hpp"
#include "utils/debug.hpp"
#include "utils/flags.hpp"
#include "utils/options.hpp"
#include "utils/versions.hpp"
#include "thread/monitor.hpp"
#include "CL/cl_ext.h"

#include "vdi_common.hpp"
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocblit.hpp"
#include "device/rocm/rocvirtual.hpp"
#include "device/rocm/rocprogram.hpp"
#include "device/rocm/rocmemory.hpp"
#include "device/rocm/rocglinterop.hpp"
#ifdef WITH_AMDGPU_PRO
#include "pro/prodriver.hpp"
#endif
#include "platform/sampler.hpp"
#include "rochostcall.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numaif.h>
#endif  // WITHOUT_HSA_BACKEND

#define OPENCL_VERSION_STR XSTR(OPENCL_MAJOR) "." XSTR(OPENCL_MINOR)
#define OPENCL_C_VERSION_STR XSTR(OPENCL_C_MAJOR) "." XSTR(OPENCL_C_MINOR)

#ifndef WITHOUT_HSA_BACKEND
namespace device {
extern const char* BlitSourceCode;
}

namespace roc {
amd::Device::Compiler* NullDevice::compilerHandle_;
bool roc::Device::isHsaInitialized_ = false;
std::vector<hsa_agent_t> roc::Device::gpu_agents_;
std::vector<AgentInfo> roc::Device::cpu_agents_;

const bool roc::Device::offlineDevice_ = false;
const bool roc::NullDevice::offlineDevice_ = true;
address Device::mg_sync_ = nullptr;

static HsaDeviceId getHsaDeviceId(hsa_agent_t device, uint32_t& pci_id) {
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(device, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CHIP_ID, &pci_id)) {
    return HSA_INVALID_DEVICE_ID;
  }

  char agent_name[64] = {0};

  if (HSA_STATUS_SUCCESS != hsa_agent_get_info(device, HSA_AGENT_INFO_NAME, agent_name)) {
    return HSA_INVALID_DEVICE_ID;
  }

  if (::strncmp(agent_name, "gfx", 3) != 0) {
    return HSA_INVALID_DEVICE_ID;
  }

  for (uint i = 0; i < sizeof(DeviceInfo) / sizeof(AMDDeviceInfo); ++i) {
    if (::strcmp(agent_name, DeviceInfo[i].machineTargetLC_) == 0) {
      return i;
    }
  }

  return HSA_INVALID_DEVICE_ID;
}

bool NullDevice::create(const AMDDeviceInfo& deviceInfo) {
  online_ = false;
  deviceInfo_ = deviceInfo;
  // Mark the device as GPU type
  info_.type_ = CL_DEVICE_TYPE_GPU;
  info_.vendorId_ = 0x1002;

  settings_ = new Settings();
  roc::Settings* hsaSettings = static_cast<roc::Settings*>(settings_);
  if ((hsaSettings == nullptr) || !hsaSettings->create(false, deviceInfo_.gfxipMajor_, deviceInfo_.gfxipMinor_)) {
    LogError("Error creating settings for nullptr HSA device");
    return false;
  }

  if (!ValidateComgr()) {
    LogError("Code object manager initialization failed!");
    return false;
  }

  // Report the device name
  ::strcpy(info_.name_, "AMD HSA Device");
  info_.extensions_ = getExtensionString();
  info_.maxWorkGroupSize_ = hsaSettings->maxWorkGroupSize_;
  ::strcpy(info_.vendor_, "Advanced Micro Devices, Inc.");
  info_.oclcVersion_ = "OpenCL C " OPENCL_C_VERSION_STR " ";
  info_.spirVersions_ = "";
  strcpy(info_.driverVersion_, "1.0 Provisional (hsa)");
  info_.version_ = "OpenCL " OPENCL_VERSION_STR " ";
  return true;
}

Device::Device(hsa_agent_t bkendDevice)
    : mapCacheOps_(nullptr)
    , mapCache_(nullptr)
    , _bkendDevice(bkendDevice)
    , gpuvm_segment_max_alloc_(0)
    , alloc_granularity_(0)
    , context_(nullptr)
    , xferQueue_(nullptr)
    , xferRead_(nullptr)
    , xferWrite_(nullptr)
    , pro_device_(nullptr)
    , pro_ena_(false)
    , freeMem_(0)
    , vgpusAccess_("Virtual GPU List Ops Lock", true)
    , hsa_exclusive_gpu_access_(false)
    , queuePool_(QueuePriority::Total)
    , numOfVgpus_(0) {
  group_segment_.handle = 0;
  system_segment_.handle = 0;
  system_coarse_segment_.handle = 0;
  gpuvm_segment_.handle = 0;
  gpu_fine_grained_segment_.handle = 0;
  prefetch_signal_.handle = 0;
}

void Device::setupCpuAgent() {
  uint32_t numaDistance = std::numeric_limits<uint32_t>::max();
  uint32_t index = 0; // 0 as default
  auto size = cpu_agents_.size();
  for (uint32_t i = 0; i < size; i++) {
    uint32_t hops = 0;
    uint32_t link_type = 0;
    uint32_t distance = 0;
    if (getNumaInfo(cpu_agents_[i].fine_grain_pool, &hops, &link_type, &distance)) {
      if (distance < numaDistance) {
        numaDistance = distance;
        index = i;
      }
    }
  }

  cpu_agent_ = cpu_agents_[index].agent;
  system_segment_ = cpu_agents_[index].fine_grain_pool;
  system_coarse_segment_ = cpu_agents_[index].coarse_grain_pool;
  ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Numa select cpu agent[%zu]=0x%zx(fine=0x%zx,coarse=0x%zx) "
          "for gpu agent=0x%zx",
          index, cpu_agent_.handle, system_segment_.handle, system_coarse_segment_.handle,
          _bkendDevice.handle);
}

Device::~Device() {
#ifdef WITH_AMDGPU_PRO
  delete pro_device_;
#endif

  // Release cached map targets
  for (uint i = 0; mapCache_ != nullptr && i < mapCache_->size(); ++i) {
    if ((*mapCache_)[i] != nullptr) {
      (*mapCache_)[i]->release();
    }
  }
  delete mapCache_;
  delete mapCacheOps_;

  if (nullptr != p2p_stage_) {
    p2p_stage_->release();
    p2p_stage_ = nullptr;
  }
  if (nullptr != mg_sync_) {
    amd::SvmBuffer::free(GlbCtx(), mg_sync_);
    mg_sync_ = nullptr;
  }
  if (glb_ctx_ != nullptr) {
      glb_ctx_->release();
      glb_ctx_ = nullptr;
  }

  // Destroy temporary buffers for read/write
  delete xferRead_;
  delete xferWrite_;

  // Destroy transfer queue
  if (xferQueue_ && xferQueue_->terminate()) {
    delete xferQueue_;
    xferQueue_ = nullptr;
  }

  if (blitProgram_) {
    delete blitProgram_;
    blitProgram_ = nullptr;
  }

  if (context_ != nullptr) {
    context_->release();
  }

  if (info_.extensions_) {
    delete[] info_.extensions_;
    info_.extensions_ = nullptr;
  }

  if (settings_) {
    delete settings_;
    settings_ = nullptr;
  }

  delete[] p2p_agents_list_;
}
bool NullDevice::initCompiler(bool isOffline) {
#if defined(WITH_COMPILER_LIB)
  // Initialize the compiler handle if has already not been initialized
  // This is destroyed in Device::teardown
  acl_error error;
  if (!compilerHandle_) {
    aclCompilerOptions opts = {
      sizeof(aclCompilerOptions_0_8), "libamdoclcl64.so",
      NULL, NULL, NULL, NULL, NULL, NULL
    };
    compilerHandle_ = aclCompilerInit(&opts, &error);
    if (!GPU_ENABLE_LC && error != ACL_SUCCESS) {
      LogError("Error initializing the compiler handle");
      return false;
    }
  }
#endif // defined(WITH_COMPILER_LIB)
  return true;
}

bool NullDevice::destroyCompiler() {
#if defined(WITH_COMPILER_LIB)
  if (compilerHandle_ != nullptr) {
    acl_error error = aclCompilerFini(compilerHandle_);
    if (error != ACL_SUCCESS) {
      LogError("Error closing the compiler");
      return false;
    }
  }
#endif // defined(WITH_COMPILER_LIB)
  return true;
}

void NullDevice::tearDown() { destroyCompiler(); }
bool NullDevice::init() {
  // Initialize the compiler
  if (!initCompiler(offlineDevice_)) {
    return false;
  }

  // Return without initializing offline device list
  return true;

#if defined(WITH_COMPILER_LIB)
  // If there is an HSA enabled device online then skip any offline device
  std::vector<Device*> devices;
  devices = getDevices(CL_DEVICE_TYPE_GPU, false);

  // Load the offline devices
  // Iterate through the set of available offline devices
  for (uint id = 0; id < sizeof(DeviceInfo) / sizeof(AMDDeviceInfo); id++) {
    bool isOnline = false;
    // Check if the particular device is online
    for (unsigned int i = 0; i < devices.size(); i++) {
      if (static_cast<NullDevice*>(devices[i])->deviceInfo_.hsaDeviceId_ ==
          DeviceInfo[id].hsaDeviceId_) {
        isOnline = true;
      }
    }
    if (isOnline) {
      continue;
    }
    NullDevice* nullDevice = new NullDevice();
    if (!nullDevice->create(DeviceInfo[id])) {
      LogError("Error creating new instance of Device.");
      delete nullDevice;
      return false;
    }
    nullDevice->registerDevice();
  }
#endif  // defined(WITH_COMPILER_LIB)
  return true;
}

NullDevice::~NullDevice() {
  if (info_.extensions_) {
    delete[] info_.extensions_;
    info_.extensions_ = nullptr;
  }

  if (settings_) {
    delete settings_;
    settings_ = nullptr;
  }
}

hsa_status_t Device::iterateAgentCallback(hsa_agent_t agent, void* data) {
  hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

  hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }

  if (dev_type == HSA_DEVICE_TYPE_CPU) {
    AgentInfo info = { agent, { 0 }, { 0 }};
    stat = hsa_amd_agent_iterate_memory_pools(agent, Device::iterateCpuMemoryPoolCallback,
                                              reinterpret_cast<void*>(&info));
    if (stat == HSA_STATUS_INFO_BREAK) {
      cpu_agents_.push_back(info);
      stat = HSA_STATUS_SUCCESS;
    }
  } else if (dev_type == HSA_DEVICE_TYPE_GPU) {
    gpu_agents_.push_back(agent);
  }

  return stat;
}

hsa_ven_amd_loader_1_00_pfn_t Device::amd_loader_ext_table = {nullptr};

hsa_status_t Device::loaderQueryHostAddress(const void* device, const void** host) {
  return amd_loader_ext_table.hsa_ven_amd_loader_query_host_address
      ? amd_loader_ext_table.hsa_ven_amd_loader_query_host_address(device, host)
      : HSA_STATUS_ERROR;
}

Device::XferBuffers::~XferBuffers() {
  // Destroy temporary buffer for reads
  for (const auto& buf : freeBuffers_) {
    delete buf;
  }
  freeBuffers_.clear();
}

bool Device::XferBuffers::create() {
  Memory* xferBuf = nullptr;
  bool result = false;

  // Create a buffer object
  xferBuf = new Buffer(dev(), bufSize_);

  // Try to allocate memory for the transfer buffer
  if ((nullptr == xferBuf) || !xferBuf->create()) {
    delete xferBuf;
    xferBuf = nullptr;
    LogError("Couldn't allocate a transfer buffer!");
  } else {
    result = true;
    freeBuffers_.push_back(xferBuf);
  }

  return result;
}

Memory& Device::XferBuffers::acquire() {
  Memory* xferBuf = nullptr;
  size_t listSize;

  // Lock the operations with the staged buffer list
  amd::ScopedLock l(lock_);
  listSize = freeBuffers_.size();

  // If the list is empty, then attempt to allocate a staged buffer
  if (listSize == 0) {
    // Allocate memory
    xferBuf = new Buffer(dev(), bufSize_);

    // Allocate memory for the transfer buffer
    if ((nullptr == xferBuf) || !xferBuf->create()) {
      delete xferBuf;
      xferBuf = nullptr;
      LogError("Couldn't allocate a transfer buffer!");
    } else {
      ++acquiredCnt_;
    }
  }

  if (xferBuf == nullptr) {
    xferBuf = *(freeBuffers_.begin());
    freeBuffers_.erase(freeBuffers_.begin());
    ++acquiredCnt_;
  }

  return *xferBuf;
}

void Device::XferBuffers::release(VirtualGPU& gpu, Memory& buffer) {
  // Make sure buffer isn't busy on the current VirtualGPU, because
  // the next aquire can come from different queue
  //    buffer.wait(gpu);
  // Lock the operations with the staged buffer list
  amd::ScopedLock l(lock_);
  freeBuffers_.push_back(&buffer);
  --acquiredCnt_;
}

bool Device::init() {
  ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Initializing HSA stack.");

  // Initialize the compiler
  if (!initCompiler(offlineDevice_)) {
    return false;
  }

  if (HSA_STATUS_SUCCESS != hsa_init()) {
    LogError("hsa_init failed.");
    return false;
  }

  hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(amd_loader_ext_table),
                                       &amd_loader_ext_table);

  if (HSA_STATUS_SUCCESS != hsa_iterate_agents(iterateAgentCallback, nullptr)) {
    return false;
  }

  std::string ordinals = amd::IS_HIP ? ((HIP_VISIBLE_DEVICES[0] != '\0') ?
                         HIP_VISIBLE_DEVICES : CUDA_VISIBLE_DEVICES)
                         : GPU_DEVICE_ORDINAL;
  if (ordinals[0] != '\0') {
    size_t end, pos = 0;
    std::vector<hsa_agent_t> valid_agents;
    do {
      bool deviceIdValid = true;
      end = ordinals.find_first_of(',', pos);
      int index = atoi(ordinals.substr(pos, end - pos).c_str());
      if (index < 0 || static_cast<size_t>(index) >= gpu_agents_.size()) {
        deviceIdValid = false;
      }

      if (!deviceIdValid) {
        // Exit the loop as anything to the right of invalid deviceId
        // has to be discarded
        break;
      } else {
        valid_agents.push_back(gpu_agents_[index]);
      }
      pos = end + 1;
    } while (end != std::string::npos);
    gpu_agents_ = valid_agents;
  }

  for (auto agent : gpu_agents_) {
    std::unique_ptr<Device> roc_device(new Device(agent));

    if (!roc_device) {
      LogError("Error creating new instance of Device on then heap.");
      return false;
    }

    uint32_t pci_id;
    HsaDeviceId deviceId = getHsaDeviceId(agent, pci_id);
    if (deviceId == HSA_INVALID_DEVICE_ID) {
      LogPrintfError("Invalid HSA device %x", pci_id);
      continue;
    }

    roc_device->deviceInfo_ = DeviceInfo[deviceId];
    roc_device->deviceInfo_.pciDeviceId_ = pci_id;

    if (!roc_device->create()) {
      LogError("Error creating new instance of Device.");
      continue;
    }

    // Setup System Memory to be Non-Coherent per user
    // request via environment variable. By default the
    // System Memory is setup to be Coherent
    if (roc_device->settings().enableNCMode_) {
      hsa_status_t err = hsa_amd_coherency_set_type(agent, HSA_AMD_COHERENCY_TYPE_NONCOHERENT);
      if (err != HSA_STATUS_SUCCESS) {
        LogError("Unable to set NC memory policy!");
        continue;
      }
    }

    roc_device.release()->registerDevice();
  }

  if (0 != Device::numDevices(CL_DEVICE_TYPE_GPU, false)) {
    // Loop through all available devices
    for (auto device1: Device::devices()) {
      // Find all agents that can have access to the current device
      for (auto agent: static_cast<Device*>(device1)->p2pAgents()) {
        // Find cl_device_id associated with the current agent
        for (auto device2: Device::devices()) {
          if (agent.handle == static_cast<Device*>(device2)->getBackendDevice().handle) {
            // Device2 can have access to device1
            device2->p2pDevices_.push_back(as_cl(device1));
            device1->p2p_access_devices_.push_back(device2);
          }
        }
      }
    }
  }

  return true;
}

extern const char* SchedulerSourceCode;
extern const char* GwsInitSourceCode;

void Device::tearDown() {
  NullDevice::tearDown();
  hsa_shut_down();
}

bool Device::create() {
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_PROFILE, &agent_profile_)) {
    return false;
  }

  uint32_t coop_groups = 0;
  // Check cooperative groups for HIP only
  if (amd::IS_HIP && (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice,
        static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES), &coop_groups))) {
    return false;
  }

  // Create HSA settings
  settings_ = new Settings();
  roc::Settings* hsaSettings = static_cast<roc::Settings*>(settings_);
  if ((hsaSettings == nullptr) ||
      !hsaSettings->create((agent_profile_ == HSA_PROFILE_FULL),
        deviceInfo_.gfxipMajor_, deviceInfo_.gfxipMinor_, coop_groups)) {
    return false;
  }

  if (!ValidateComgr()) {
    LogError("Code object manager initialization failed!");
    return false;
  }

  if (!amd::Device::create()) {
    return false;
  }

  uint32_t hsa_bdf_id = 0;
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice,
        static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID), &hsa_bdf_id)) {
    return false;
  }

  info_.deviceTopology_.pcie.type = CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD;
  info_.deviceTopology_.pcie.bus = (hsa_bdf_id & (0xFF << 8)) >> 8;
  info_.deviceTopology_.pcie.device = (hsa_bdf_id & (0x1F << 3)) >> 3;
  info_.deviceTopology_.pcie.function = (hsa_bdf_id & 0x07);

#ifdef WITH_AMDGPU_PRO
  // Create amdgpu-pro device interface for SSG support
  pro_device_ = IProDevice::Init(
      info_.deviceTopology_.pcie.bus,
      info_.deviceTopology_.pcie.device,
      info_.deviceTopology_.pcie.function);
  if (pro_device_ != nullptr) {
    pro_ena_ = true;
    settings_->enableExtension(ClAMDLiquidFlash);
    pro_device_->GetAsicIdAndRevisionId(&info_.pcieDeviceId_, &info_.pcieRevisionId_);
  }
#endif

  if (populateOCLDeviceConstants() == false) {
    LogError("populateOCLDeviceConstants failed!");
    return false;
  }

  amd::Context::Info info = {0};
  std::vector<amd::Device*> devices;
  devices.push_back(this);

  // Create a dummy context
  context_ = new amd::Context(devices, info);
  if (context_ == nullptr) {
    return false;
  }

  mapCacheOps_ = new amd::Monitor("Map Cache Lock", true);
  if (nullptr == mapCacheOps_) {
    return false;
  }

  mapCache_ = new std::vector<amd::Memory*>();
  if (mapCache_ == nullptr) {
    return false;
  }
  // Use just 1 entry by default for the map cache
  mapCache_->push_back(nullptr);

  if ((glb_ctx_ == nullptr) && (gpu_agents_.size() >= 1) &&
      // Allow creation for the last device in the list.
      (gpu_agents_[gpu_agents_.size() - 1].handle == _bkendDevice.handle)) {
    std::vector<amd::Device*> devices;
    uint32_t numDevices = amd::Device::numDevices(CL_DEVICE_TYPE_GPU, false);
    // Add all PAL devices
    for (uint32_t i = 0; i < numDevices; ++i) {
      devices.push_back(amd::Device::devices()[i]);
    }
    // Add current
    devices.push_back(this);
    // Create a dummy context
    glb_ctx_ = new amd::Context(devices, info);
    if (glb_ctx_ == nullptr) {
      return false;
    }

    if ((p2p_agents_.size() < (devices.size()-1)) && (devices.size() > 1)) {
      amd::Buffer* buf = new (GlbCtx()) amd::Buffer(GlbCtx(), CL_MEM_ALLOC_HOST_PTR, kP2PStagingSize);
      if ((buf != nullptr) && buf->create()) {
        p2p_stage_ = buf;
      }
      else {
        delete buf;
        return false;
      }
    }
    // Check if sync buffer wasn't allocated yet
    if (amd::IS_HIP && mg_sync_ == nullptr) {
      mg_sync_ = reinterpret_cast<address>(amd::SvmBuffer::malloc(
          GlbCtx(), (CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS),
          kMGInfoSizePerDevice * GlbCtx().devices().size(), kMGInfoSizePerDevice));
      if (mg_sync_ == nullptr) {
        return false;
      }
    }
  }

  if (settings().stagedXferSize_ != 0) {
    // Initialize staged write buffers
    if (settings().stagedXferWrite_) {
      xferWrite_ = new XferBuffers(*this, amd::alignUp(settings().stagedXferSize_, 4 * Ki));
      if ((xferWrite_ == nullptr) || !xferWrite_->create()) {
        LogError("Couldn't allocate transfer buffer objects for read");
        return false;
      }
    }

    // Initialize staged read buffers
    if (settings().stagedXferRead_) {
      xferRead_ = new XferBuffers(*this, amd::alignUp(settings().stagedXferSize_, 4 * Ki));
      if ((xferRead_ == nullptr) || !xferRead_->create()) {
        LogError("Couldn't allocate transfer buffer objects for write");
        return false;
      }
    }
  }

  // Create signal for HMM prefetch operation on device
  if (HSA_STATUS_SUCCESS != hsa_signal_create(InitSignalValue, 0, nullptr, &prefetch_signal_)) {
    return false;
  }

  return true;
}

device::Program* NullDevice::createProgram(amd::Program& owner, amd::option::Options* options) {
  device::Program* program;
  if (settings().useLightning_) {
    program = new LightningProgram(*this, owner);
  } else {
    program = new HSAILProgram(*this, owner);
  }

  if (program == nullptr) {
    LogError("Memory allocation has failed!");
  }

  return program;
}

bool Device::AcquireExclusiveGpuAccess() {
  // Lock the virtual GPU list
  vgpusAccess().lock();

  // Find all available virtual GPUs and lock them
  // from the execution of commands
  for (uint idx = 0; idx < vgpus().size(); ++idx) {
    vgpus()[idx]->execution().lock();
    // Make sure a wait is done
    vgpus()[idx]->releaseGpuMemoryFence();
  }
  if (!hsa_exclusive_gpu_access_) {
    // @todo call rocr
    hsa_exclusive_gpu_access_ = true;
  }
  return true;
}

void Device::ReleaseExclusiveGpuAccess(VirtualGPU& vgpu) const {
  // Make sure the operation is done
  vgpu.releaseGpuMemoryFence();

  // Find all available virtual GPUs and unlock them
  // for the execution of commands
  for (uint idx = 0; idx < vgpus().size(); ++idx) {
    vgpus()[idx]->execution().unlock();
  }

  // Unock the virtual GPU list
  vgpusAccess().unlock();
}

bool Device::createBlitProgram() {
  bool result = true;
  const char* scheduler = nullptr;

#if defined(USE_COMGR_LIBRARY)
  std::string sch = SchedulerSourceCode;
  if (settings().useLightning_) {
    if (info().cooperativeGroups_) {
      sch.append(GwsInitSourceCode);
    }
    scheduler = sch.c_str();
  }
#endif  // USE_COMGR_LIBRARY

  blitProgram_ = new BlitProgram(context_);
  // Create blit programs
  if (blitProgram_ == nullptr || !blitProgram_->create(this, scheduler)) {
    delete blitProgram_;
    blitProgram_ = nullptr;
    LogError("Couldn't create blit kernels!");
    return false;
  }

  return result;
}

device::Program* Device::createProgram(amd::Program& owner, amd::option::Options* options) {
  device::Program* program;
  if (settings().useLightning_) {
    program = new LightningProgram(*this, owner);
  } else {
    program = new HSAILProgram(*this, owner);
  }

  if (program == nullptr) {
    LogError("Memory allocation has failed!");
  }

  return program;
}

hsa_status_t Device::iterateGpuMemoryPoolCallback(hsa_amd_memory_pool_t pool, void* data) {
  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_region_segment_t segment_type = (hsa_region_segment_t)0;
  hsa_status_t stat =
      hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment_type);
  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }

  // TODO: system and device local segment
  Device* dev = reinterpret_cast<Device*>(data);
  switch (segment_type) {
    case HSA_REGION_SEGMENT_GLOBAL: {
      if (dev->settings().enableLocalMemory_) {
        uint32_t global_flag = 0;
        hsa_status_t stat =
            hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flag);
        if (stat != HSA_STATUS_SUCCESS) {
          return stat;
        }

        if ((global_flag & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0) {
          dev->gpu_fine_grained_segment_ = pool;
        } else if ((global_flag & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0) {
          dev->gpuvm_segment_ = pool;

          // If cpu agent cannot access this pool, the device does not support large bar.
          hsa_amd_memory_pool_access_t tmp{};
          hsa_amd_agent_memory_pool_get_info(
            dev->cpu_agent_,
            pool,
            HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
            &tmp);

          if (tmp == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
            dev->info_.largeBar_ = false;
          } else {
            dev->info_.largeBar_ = ROC_ENABLE_LARGE_BAR;
          }
        }

        if (dev->gpuvm_segment_.handle == 0) {
          dev->gpuvm_segment_ = pool;
        }
      }
      break;
    }
    case HSA_REGION_SEGMENT_GROUP:
      dev->group_segment_ = pool;
      break;
    default:
      break;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t Device::iterateCpuMemoryPoolCallback(hsa_amd_memory_pool_t pool, void* data) {
  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_region_segment_t segment_type = (hsa_region_segment_t)0;
  hsa_status_t stat =
      hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment_type);
  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }
  AgentInfo* agentInfo = reinterpret_cast<AgentInfo*>(data);

  switch (segment_type) {
    case HSA_REGION_SEGMENT_GLOBAL: {
      uint32_t global_flag = 0;
      stat = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                          &global_flag);
      if (stat != HSA_STATUS_SUCCESS) {
        break;
      }

      if ((global_flag & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0) {
        agentInfo->fine_grain_pool = pool;
      } else {
        agentInfo->coarse_grain_pool = pool;
      }

      if (agentInfo->fine_grain_pool.handle != 0 && agentInfo->coarse_grain_pool.handle != 0) {
        stat = HSA_STATUS_INFO_BREAK;
      }
      break;
    }
    default:
      break;
  }

  return stat;
}

bool Device::createSampler(const amd::Sampler& owner, device::Sampler** sampler) const {
  *sampler = nullptr;
  Sampler* gpuSampler = new Sampler(*this);
  if ((nullptr == gpuSampler) || !gpuSampler->create(owner)) {
    delete gpuSampler;
    return false;
  }
  *sampler = gpuSampler;
  return true;
}

void Sampler::fillSampleDescriptor(hsa_ext_sampler_descriptor_t& samplerDescriptor,
                                   const amd::Sampler& sampler) const {
  samplerDescriptor.filter_mode = sampler.filterMode() == CL_FILTER_NEAREST
      ? HSA_EXT_SAMPLER_FILTER_MODE_NEAREST
      : HSA_EXT_SAMPLER_FILTER_MODE_LINEAR;
  samplerDescriptor.coordinate_mode = sampler.normalizedCoords()
      ? HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED
      : HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED;
  switch (sampler.addressingMode()) {
    case CL_ADDRESS_CLAMP_TO_EDGE:
      samplerDescriptor.address_mode = HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE;
      break;
    case CL_ADDRESS_REPEAT:
      samplerDescriptor.address_mode = HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT;
      break;
    case CL_ADDRESS_CLAMP:
      samplerDescriptor.address_mode = HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER;
      break;
    case CL_ADDRESS_MIRRORED_REPEAT:
      samplerDescriptor.address_mode = HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT;
      break;
    case CL_ADDRESS_NONE:
      samplerDescriptor.address_mode = HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED;
      break;
    default:
      return;
  }
}

bool Sampler::create(const amd::Sampler& owner) {
  hsa_ext_sampler_descriptor_t samplerDescriptor;
  fillSampleDescriptor(samplerDescriptor, owner);

  hsa_status_t status = hsa_ext_sampler_create(dev_.getBackendDevice(), &samplerDescriptor, &hsa_sampler);

  if (HSA_STATUS_SUCCESS != status) {
    DevLogPrintfError("Sampler creation failed with status: %d \n", status);
    return false;
  }

  hwSrd_ = reinterpret_cast<uint64_t>(hsa_sampler.handle);
  hwState_ = reinterpret_cast<address>(hsa_sampler.handle);

  return true;
}

Sampler::~Sampler() {
  hsa_ext_sampler_destroy(dev_.getBackendDevice(), hsa_sampler);
}

Memory* Device::getGpuMemory(amd::Memory* mem) const {
  return static_cast<roc::Memory*>(mem->getDeviceMemory(*this));
}

bool Device::populateOCLDeviceConstants() {
  info_.available_ = true;

  roc::Settings* hsa_settings = static_cast<roc::Settings*>(settings_);

  hsa_isa_t isa = {0};
  if (hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_ISA, &isa) != HSA_STATUS_SUCCESS) {
    return false;
  }

  uint32_t isaNameLength = 0;
  if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &isaNameLength) != HSA_STATUS_SUCCESS) {
    return false;
  }

  if ((isaNameLength + 1) > sizeof(info_.targetId_)) {
    return false;
  }

  if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, info_.targetId_) != HSA_STATUS_SUCCESS) {
    return false;
  }
  info_.targetId_[isaNameLength] = '\0';

  char *gfxSubString = ::strstr(info_.targetId_, "gfx");
  if (nullptr == gfxSubString) {
    return false;
  }
  ::strcpy(info_.name_, gfxSubString);

  info_.gfxipMajor_ = deviceInfo_.gfxipMajor_;
  info_.gfxipMinor_ = deviceInfo_.gfxipMinor_;
  info_.gfxipStepping_ = deviceInfo_.gfxipStepping_;

  // TODO: gfxipVersion_ and sramEccEnabled_ will be removed when Target ID
  // feature is fully implemented
  info_.gfxipVersion_ = info_.gfxipMajor_ * 100 + info_.gfxipMinor_ * 10 + info_.gfxipStepping_;

  if ((info_.gfxipMajor_ == 9) && (info_.gfxipMinor_ == 0)) {
    switch (info_.gfxipStepping_) {
    case 6:
    case 8:
      info_.sramEccEnabled_ = true;
      break;
    default:
      info_.sramEccEnabled_ = false;
      break;
    }
  }

  char device_name[64] = {0};
  if (HSA_STATUS_SUCCESS == hsa_agent_get_info(_bkendDevice,
                                               (hsa_agent_info_t)HSA_AMD_AGENT_INFO_PRODUCT_NAME,
                                               device_name)) {
    ::strcpy(info_.boardName_, device_name);
  }

  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
                         &info_.maxComputeUnits_)) {
    return false;
  }
  assert(info_.maxComputeUnits_ > 0);

  info_.maxComputeUnits_ = settings().enableWgpMode_
      ? info_.maxComputeUnits_ / 2
      : info_.maxComputeUnits_;

  if (HSA_STATUS_SUCCESS != hsa_agent_get_info(_bkendDevice,
                                               (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CACHELINE_SIZE,
                                               &info_.globalMemCacheLineSize_)) {
    return false;
  }
  assert(info_.globalMemCacheLineSize_ > 0);

  uint32_t cachesize[4] = {0};
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_CACHE_SIZE, cachesize)) {
    return false;
  }
  assert(cachesize[0] > 0);
  info_.globalMemCacheSize_ = cachesize[0];

  info_.globalMemCacheType_ = CL_READ_WRITE_CACHE;

  info_.type_ = CL_DEVICE_TYPE_GPU;

  info_.extensions_ = getExtensionString();
  info_.nativeVectorWidthDouble_ = info_.preferredVectorWidthDouble_ =
      (settings().doublePrecision_) ? 1 : 0;

  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY,
                         &info_.maxEngineClockFrequency_)) {
    return false;
  }

  //TODO: add the assert statement for Raven
  if ((info_.gfxipMajor_*100 + info_.gfxipMinor_*10 + info_.gfxipStepping_) != 902) {
    assert(info_.maxEngineClockFrequency_ > 0);
  }

  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY,
          &info_.maxMemoryClockFrequency_)) {
      return false;
  }

  setupCpuAgent();

  assert(system_segment_.handle != 0);
  if (HSA_STATUS_SUCCESS != hsa_amd_agent_iterate_memory_pools(
                                _bkendDevice, Device::iterateGpuMemoryPoolCallback, this)) {
    return false;
  }

  assert(group_segment_.handle != 0);

  for (auto agent: gpu_agents_) {
    if (agent.handle != _bkendDevice.handle) {
      hsa_status_t err;
      // Can another GPU (agent) have access to the current GPU memory pool (gpuvm_segment_)?
      hsa_amd_memory_pool_access_t access;
      err = hsa_amd_agent_memory_pool_get_info(agent, gpuvm_segment_, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
      if (err != HSA_STATUS_SUCCESS) {
        continue;
      }

      // Find accessible p2p agents - i.e != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED
      if (HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT == access ||
          HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT == access) {
        // Agent can have access to the current gpuvm_segment_
        p2p_agents_.push_back(agent);
      }
    }
  }

  /* Keep track of all P2P Agents in a Array including current device handle for IPC */
  p2p_agents_list_ = new hsa_agent_t[1 + p2p_agents_.size()];
  p2p_agents_list_[0] = getBackendDevice();
  for (size_t agent_idx = 0; agent_idx < p2p_agents_.size(); ++agent_idx) {
    p2p_agents_list_[1 + agent_idx] = p2p_agents_[agent_idx];
  }

  size_t group_segment_size = 0;
  if (HSA_STATUS_SUCCESS != hsa_amd_memory_pool_get_info(group_segment_,
                                                         HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                                         &group_segment_size)) {
    return false;
  }
  assert(group_segment_size > 0);

  info_.localMemSizePerCU_ = group_segment_size;
  info_.localMemSize_ = group_segment_size;

  info_.maxWorkItemDimensions_ = 3;

  if (settings().enableLocalMemory_ && gpuvm_segment_.handle != 0) {
    size_t global_segment_size = 0;
    if (HSA_STATUS_SUCCESS != hsa_amd_memory_pool_get_info(gpuvm_segment_,
                                                           HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                                           &global_segment_size)) {
      return false;
    }

    assert(global_segment_size > 0);
    info_.globalMemSize_ = static_cast<uint64_t>(global_segment_size);

    gpuvm_segment_max_alloc_ =
        uint64_t(info_.globalMemSize_ * std::min(GPU_SINGLE_ALLOC_PERCENT, 100u) / 100u);
    assert(gpuvm_segment_max_alloc_ > 0);

    info_.maxMemAllocSize_ = static_cast<uint64_t>(gpuvm_segment_max_alloc_);

    if (HSA_STATUS_SUCCESS !=
        hsa_amd_memory_pool_get_info(gpuvm_segment_, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                     &alloc_granularity_)) {
      return false;
    }

    assert(alloc_granularity_ > 0);
  } else {
    // We suppose half of physical memory can be used by GPU in APU system
    info_.globalMemSize_ =
        uint64_t(sysconf(_SC_PAGESIZE)) * uint64_t(sysconf(_SC_PHYS_PAGES)) / 2;
    info_.globalMemSize_ = std::max(info_.globalMemSize_, uint64_t(1 * Gi));
    info_.maxMemAllocSize_ =
        uint64_t(info_.globalMemSize_ * std::min(GPU_SINGLE_ALLOC_PERCENT, 100u) / 100u);

    if (HSA_STATUS_SUCCESS !=
        hsa_amd_memory_pool_get_info(
            system_segment_, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &alloc_granularity_)) {
      return false;
    }
  }

  freeMem_ = info_.globalMemSize_;

  // Make sure the max allocation size is not larger than the available
  // memory size.
  info_.maxMemAllocSize_ = std::min(info_.maxMemAllocSize_, info_.globalMemSize_);

  /*make sure we don't run anything over 8 params for now*/
  info_.maxParameterSize_ = 1024;  // [TODO]: CAL stack values: 1024*
  // constant

  uint32_t max_work_group_size = 0;
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &max_work_group_size)) {
    return false;
  }
  assert(max_work_group_size > 0);
  max_work_group_size =
      std::min(max_work_group_size, static_cast<uint32_t>(settings().maxWorkGroupSize_));
  info_.maxWorkGroupSize_ = max_work_group_size;

  uint16_t max_workgroup_size[3] = {0, 0, 0};
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_WORKGROUP_MAX_DIM, &max_workgroup_size)) {
    return false;
  }
  assert(max_workgroup_size[0] != 0 && max_workgroup_size[1] != 0 && max_workgroup_size[2] != 0);

  uint16_t max_work_item_size = static_cast<uint16_t>(max_work_group_size);
  info_.maxWorkItemSizes_[0] = std::min(max_workgroup_size[0], max_work_item_size);
  info_.maxWorkItemSizes_[1] = std::min(max_workgroup_size[1], max_work_item_size);
  info_.maxWorkItemSizes_[2] = std::min(max_workgroup_size[2], max_work_item_size);
  info_.preferredWorkGroupSize_ = settings().preferredWorkGroupSize_;

  info_.nativeVectorWidthChar_ = info_.preferredVectorWidthChar_ = 4;
  info_.nativeVectorWidthShort_ = info_.preferredVectorWidthShort_ = 2;
  info_.nativeVectorWidthInt_ = info_.preferredVectorWidthInt_ = 1;
  info_.nativeVectorWidthLong_ = info_.preferredVectorWidthLong_ = 1;
  info_.nativeVectorWidthFloat_ = info_.preferredVectorWidthFloat_ = 1;

  if (agent_profile_ == HSA_PROFILE_FULL) {  // full-profile = participating in coherent memory,
                                             // base-profile = NUMA based non-coherent memory
    info_.hostUnifiedMemory_ = CL_TRUE;
  }
  info_.memBaseAddrAlign_ =
      8 * (flagIsDefault(MEMOBJ_BASE_ADDR_ALIGN) ? sizeof(int64_t[16]) : MEMOBJ_BASE_ADDR_ALIGN);
  info_.minDataTypeAlignSize_ = sizeof(int64_t[16]);

  info_.maxConstantArgs_ = 8;
  info_.preferredConstantBufferSize_ = 16 * Ki;
  info_.maxConstantBufferSize_ = info_.maxMemAllocSize_;
  info_.localMemType_ = CL_LOCAL;
  info_.errorCorrectionSupport_ = false;
  info_.profilingTimerResolution_ = 1;
  info_.littleEndian_ = true;
  info_.compilerAvailable_ = true;
  info_.executionCapabilities_ = CL_EXEC_KERNEL;
  info_.queueProperties_ = CL_QUEUE_PROFILING_ENABLE;
  info_.platform_ = AMD_PLATFORM;
  info_.profile_ = "FULL_PROFILE";
  strcpy(info_.vendor_, "Advanced Micro Devices, Inc.");

  info_.addressBits_ = LP64_SWITCH(32, 64);
  info_.maxSamplers_ = 16;
  info_.bufferFromImageSupport_ = CL_FALSE;
  info_.oclcVersion_ = "OpenCL C " OPENCL_C_VERSION_STR " ";
  info_.spirVersions_ = "";

  uint16_t major, minor;
  if (hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_VERSION_MAJOR, &major) !=
          HSA_STATUS_SUCCESS ||
      hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_VERSION_MINOR, &minor) !=
          HSA_STATUS_SUCCESS) {
    return false;
  }
  std::stringstream ss;
  ss << AMD_BUILD_STRING " (HSA" << major << "." << minor << "," << (settings().useLightning_ ? "LC" : "HSAIL");
  ss <<  ")";

  strcpy(info_.driverVersion_, ss.str().c_str());

  // Enable OpenCL 2.0 for Vega10+
  if (deviceInfo_.gfxipMajor_ >= 9) {
    info_.version_ = "OpenCL " /*OPENCL_VERSION_STR*/"2.0" " ";
  } else {
    info_.version_ = "OpenCL " /*OPENCL_VERSION_STR*/"1.2" " ";
  }

  info_.builtInKernels_ = "";
  info_.linkerAvailable_ = true;
  info_.preferredInteropUserSync_ = true;
  info_.printfBufferSize_ = PrintfDbg::WorkitemDebugSize * info().maxWorkGroupSize_;
  info_.vendorId_ = 0x1002;  // AMD's PCIe vendor id

  info_.maxGlobalVariableSize_ = static_cast<size_t>(info_.maxMemAllocSize_);
  info_.globalVariablePreferredTotalSize_ = static_cast<size_t>(info_.globalMemSize_);

  // Populate the single config setting.
  info_.singleFPConfig_ =
      CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_FMA;

  if (hsa_settings->doublePrecision_) {
    info_.doubleFPConfig_ = info_.singleFPConfig_ | CL_FP_DENORM;
    info_.singleFPConfig_ |= CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
  }

  if (hsa_settings->singleFpDenorm_) {
    info_.singleFPConfig_ |= CL_FP_DENORM;
  }

  info_.preferredPlatformAtomicAlignment_ = 0;
  info_.preferredGlobalAtomicAlignment_ = 0;
  info_.preferredLocalAtomicAlignment_ = 0;

  uint8_t hsa_extensions[128];
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_EXTENSIONS, hsa_extensions)) {
    return false;
  }

  assert(HSA_EXTENSION_IMAGES < 8);
  const bool image_is_supported = ((hsa_extensions[0] & (1 << HSA_EXTENSION_IMAGES)) != 0);
  if (image_is_supported) {
    // Images
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS),
                           &info_.maxSamplers_)) {
      return false;
    }

    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES),
                           &info_.maxReadImageArgs_)) {
      return false;
    }

    // TODO: no attribute for write image.
    info_.maxWriteImageArgs_ = 8;

    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES),
                           &info_.maxReadWriteImageArgs_)) {
      return false;
    }

    uint32_t image_max_dim[3];
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS),
                           &image_max_dim)) {
      return false;
    }

    info_.image2DMaxWidth_ = image_max_dim[0];
    info_.image2DMaxHeight_ = image_max_dim[1];

    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS),
                           &image_max_dim)) {
      return false;
    }

    info_.image3DMaxWidth_ = image_max_dim[0];
    info_.image3DMaxHeight_ = image_max_dim[1];
    info_.image3DMaxDepth_ = image_max_dim[2];

    uint32_t max_array_size = 0;
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS),
                           &max_array_size)) {
      return false;
    }

    info_.imageMaxArraySize_ = max_array_size;

    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS),
                           &image_max_dim)) {
      return false;
    }
    info_.imageMaxBufferSize_ = image_max_dim[0];

    info_.imagePitchAlignment_ = 256;

    info_.imageBaseAddressAlignment_ = 256;

    info_.bufferFromImageSupport_ = CL_FALSE;

    info_.imageSupport_ = (info_.maxReadWriteImageArgs_ > 0) ? CL_TRUE : CL_FALSE;
  }

  // Enable SVM Capabilities of Hsa device. Ensure
  // user has not setup memory to be non-coherent
  info_.svmCapabilities_ = 0;
  if (hsa_settings->enableNCMode_ == false) {
    info_.svmCapabilities_ = CL_DEVICE_SVM_COARSE_GRAIN_BUFFER;
    info_.svmCapabilities_ |= CL_DEVICE_SVM_FINE_GRAIN_BUFFER;
    // Report fine-grain system only on full profile
    if (agent_profile_ == HSA_PROFILE_FULL) {
      info_.svmCapabilities_ |= CL_DEVICE_SVM_FINE_GRAIN_SYSTEM;
    }
    if (amd::IS_HIP) {
      // Report atomics capability based on GFX IP, control on Hawaii
      if (info_.hostUnifiedMemory_ || deviceInfo_.gfxipMajor_ >= 8) {
        info_.svmCapabilities_ |= CL_DEVICE_SVM_ATOMICS;
      }
    }
    else if (!settings().useLightning_) {
      // Report atomics capability based on GFX IP, control on Hawaii
      // and Vega10.
      if (info_.hostUnifiedMemory_ || (deviceInfo_.gfxipMajor_ == 8)) {
        info_.svmCapabilities_ |= CL_DEVICE_SVM_ATOMICS;
      }
    }
  }

  if (settings().checkExtension(ClAmdDeviceAttributeQuery)) {
    info_.simdPerCU_ = settings().enableWgpMode_
                       ? (2 * deviceInfo_.simdPerCU_)
                       : deviceInfo_.simdPerCU_;
    info_.simdWidth_ = deviceInfo_.simdWidth_;
    info_.simdInstructionWidth_ = deviceInfo_.simdInstructionWidth_;
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_WAVEFRONT_SIZE, &info_.wavefrontWidth_)) {
      return false;
    }
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MEMORY_WIDTH),
                           &info_.vramBusBitWidth_)) {
      return false;
    }

    uint32_t max_waves_per_cu;
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU),
                           &max_waves_per_cu)) {
      return false;
    }

    info_.maxThreadsPerCU_ = info_.wavefrontWidth_ * max_waves_per_cu;
    uint32_t cache_sizes[4];
    /* FIXIT [skudchad] -  Seems like hardcoded in HSA backend so 0*/
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_AGENT_INFO_CACHE_SIZE),
                           cache_sizes)) {
      return false;
    }

    uint32_t asic_revision;
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION),
                           &asic_revision)) {
      return false;
    }
    info_.asicRevision_ = asic_revision;

    info_.l2CacheSize_ = cache_sizes[1];
    info_.timeStampFrequency_ = 1000000;
    info_.globalMemChannelBanks_ = 4;
    info_.globalMemChannelBankWidth_ = deviceInfo_.memChannelBankWidth_;
    info_.localMemSizePerCU_ = deviceInfo_.localMemSizePerCU_;
    info_.localMemBanks_ = deviceInfo_.localMemBanks_;
    info_.numAsyncQueues_ = kMaxAsyncQueues;
    info_.numRTQueues_ = info_.numAsyncQueues_;
    info_.numRTCUs_ = info_.maxComputeUnits_;

    //TODO: set to true once thread trace support is available
    info_.threadTraceEnable_ = false;
    info_.pcieDeviceId_ = deviceInfo_.pciDeviceId_;
    info_.cooperativeGroups_ = settings().enableCoopGroups_;
    info_.cooperativeMultiDeviceGroups_ = settings().enableCoopMultiDeviceGroups_;
  }

  info_.maxPipePacketSize_ = info_.maxMemAllocSize_;
  info_.maxPipeActiveReservations_ = 16;
  info_.maxPipeArgs_ = 16;

  info_.queueOnDeviceProperties_ =
      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE;
  info_.queueOnDevicePreferredSize_ = 256 * Ki;
  info_.queueOnDeviceMaxSize_ = 8 * Mi;
  info_.maxOnDeviceQueues_ = 1;
  info_.maxOnDeviceEvents_ = settings().numDeviceEvents_;

  return true;
}

device::VirtualDevice* Device::createVirtualDevice(amd::CommandQueue* queue) {
  amd::ScopedLock lock(vgpusAccess());

  bool profiling = (queue != nullptr) && queue->properties().test(CL_QUEUE_PROFILING_ENABLE);
  bool cooperative = false;

  // If amd command queue is null, then it's an internal device queue
  if (queue == nullptr) {
    // In HIP mode the device queue will be allocated for the cooperative launches only
    cooperative = amd::IS_HIP && settings().enableCoopGroups_;
    profiling = amd::IS_HIP;
  }

  // Initialization of heap and other resources occur during the command
  // queue creation time.
  const std::vector<uint32_t> defaultCuMask = {};
  bool q = (queue != nullptr);
  VirtualGPU* virtualDevice = new VirtualGPU(*this, profiling, cooperative,
                                            q ? queue->cuMask() : defaultCuMask,
                                            q ? queue->priority()
                                              : amd::CommandQueue::Priority::Normal);

  if (!virtualDevice->create()) {
    delete virtualDevice;
    return nullptr;
  }

  return virtualDevice;
}

bool Device::globalFreeMemory(size_t* freeMemory) const {
  const uint TotalFreeMemory = 0;
  const uint LargestFreeBlock = 1;

  freeMemory[TotalFreeMemory] = freeMem_ / Ki;
  freeMemory[TotalFreeMemory] -= (freeMemory[TotalFreeMemory] > HIP_HIDDEN_FREE_MEM * Ki) ?
                                  HIP_HIDDEN_FREE_MEM * Ki : 0;
  // since there is no memory heap on ROCm, the biggest free block is
  // equal to total free local memory
  freeMemory[LargestFreeBlock] = freeMemory[TotalFreeMemory];

  return true;
}

bool Device::bindExternalDevice(uint flags, void* const gfxDevice[], void* gfxContext,
                                bool validateOnly) {
#if defined(_WIN32)
  return false;
#else
  if ((flags & amd::Context::GLDeviceKhr) == 0) return false;

  MesaInterop::MESA_INTEROP_KIND kind = MesaInterop::MESA_INTEROP_NONE;
  MesaInterop::DisplayHandle display;
  MesaInterop::ContextHandle context;

  if ((flags & amd::Context::EGLDeviceKhr) != 0) {
    kind = MesaInterop::MESA_INTEROP_EGL;
    display.eglDisplay = reinterpret_cast<EGLDisplay>(gfxDevice[amd::Context::GLDeviceKhrIdx]);
    context.eglContext = reinterpret_cast<EGLContext>(gfxContext);
  } else {
    kind = MesaInterop::MESA_INTEROP_GLX;
    display.glxDisplay = reinterpret_cast<Display*>(gfxDevice[amd::Context::GLDeviceKhrIdx]);
    context.glxContext = reinterpret_cast<GLXContext>(gfxContext);
  }

  mesa_glinterop_device_info info;
  info.version = MESA_GLINTEROP_DEVICE_INFO_VERSION;
  if (!MesaInterop::Init(kind)) {
    return false;
  }

  if (!MesaInterop::GetInfo(info, kind, display, context)) {
    return false;
  }

  bool match = true;
  match &= info_.deviceTopology_.pcie.bus == info.pci_bus;
  match &= info_.deviceTopology_.pcie.device == info.pci_device;
  match &= info_.deviceTopology_.pcie.function == info.pci_function;
  match &= info_.vendorId_ == info.vendor_id;
  match &= deviceInfo_.pciDeviceId_ == info.device_id;

  return match;
#endif
}

bool Device::unbindExternalDevice(uint flags, void* const gfxDevice[], void* gfxContext,
                                  bool validateOnly) {
#if defined(_WIN32)
  return false;
#else
  if ((flags & amd::Context::GLDeviceKhr) == 0) return false;
  return true;
#endif
}

amd::Memory* Device::findMapTarget(size_t size) const {
  // Must be serialised for access
  amd::ScopedLock lk(*mapCacheOps_);

  amd::Memory* map = nullptr;
  size_t minSize = 0;
  size_t maxSize = 0;
  uint mapId = mapCache_->size();
  uint releaseId = mapCache_->size();

  // Find if the list has a map target of appropriate size
  for (uint i = 0; i < mapCache_->size(); i++) {
    if ((*mapCache_)[i] != nullptr) {
      // Requested size is smaller than the entry size
      if (size < (*mapCache_)[i]->getSize()) {
        if ((minSize == 0) || (minSize > (*mapCache_)[i]->getSize())) {
          minSize = (*mapCache_)[i]->getSize();
          mapId = i;
        }
      }
      // Requeted size matches the entry size
      else if (size == (*mapCache_)[i]->getSize()) {
        mapId = i;
        break;
      } else {
        // Find the biggest map target in the list
        if (maxSize < (*mapCache_)[i]->getSize()) {
          maxSize = (*mapCache_)[i]->getSize();
          releaseId = i;
        }
      }
    }
  }

  // Check if we found any map target
  if (mapId < mapCache_->size()) {
    map = (*mapCache_)[mapId];
    (*mapCache_)[mapId] = nullptr;
  }
  // If cache is full, then release the biggest map target
  else if (releaseId < mapCache_->size()) {
    (*mapCache_)[releaseId]->release();
    (*mapCache_)[releaseId] = nullptr;
  }

  return map;
}

bool Device::addMapTarget(amd::Memory* memory) const {
  // Must be serialised for access
  amd::ScopedLock lk(*mapCacheOps_);

  // the svm memory shouldn't be cached
  if (!memory->canBeCached()) {
    return false;
  }
  // Find if the list has a map target of appropriate size
  for (uint i = 0; i < mapCache_->size(); ++i) {
    if ((*mapCache_)[i] == nullptr) {
      (*mapCache_)[i] = memory;
      return true;
    }
  }

  // Add a new entry
  mapCache_->push_back(memory);

  return true;
}

Memory* Device::getRocMemory(amd::Memory* mem) const {
  return static_cast<roc::Memory*>(mem->getDeviceMemory(*this));
}


device::Memory* Device::createMemory(amd::Memory& owner) const {
  roc::Memory* memory = nullptr;
  if (owner.asBuffer()) {
    memory = new roc::Buffer(*this, owner);
  } else if (owner.asImage()) {
    memory = new roc::Image(*this, owner);
  } else {
    LogError("Unknown memory type");
  }

  if (memory == nullptr) {
    return nullptr;
  }

  bool result = memory->create();

  if (!result) {
    LogError("Failed creating memory");
    delete memory;
    return nullptr;
  }

  // Initialize if the memory is a pipe object
  if (owner.getType() == CL_MEM_OBJECT_PIPE) {
    // Pipe initialize in order read_idx, write_idx, end_idx. Refer clk_pipe_t structure.
    // Init with 3 DWORDS for 32bit addressing and 6 DWORDS for 64bit
    size_t pipeInit[3] = { 0, 0, owner.asPipe()->getMaxNumPackets() };
    xferMgr().writeBuffer((void *)pipeInit, *memory, amd::Coord3D(0), amd::Coord3D(sizeof(pipeInit)));
  }

  // Transfer data only if OCL context has one device.
  // Cache coherency layer will update data for multiple devices
  if (!memory->isHostMemDirectAccess() && owner.asImage() && (owner.parent() == nullptr) &&
      (owner.getMemFlags() & CL_MEM_COPY_HOST_PTR) && (owner.getContext().devices().size() == 1)) {
    // To avoid recurssive call to Device::createMemory, we perform
    // data transfer to the view of the image.
    amd::Image* imageView = owner.asImage()->createView(
        owner.getContext(), owner.asImage()->getImageFormat(), xferQueue());

    if (imageView == nullptr) {
      LogError("[OCL] Fail to allocate view of image object");
      return nullptr;
    }

    Image* devImageView = new roc::Image(static_cast<const Device&>(*this), *imageView);
    if (devImageView == nullptr) {
      LogError("[OCL] Fail to allocate device mem object for the view");
      imageView->release();
      return nullptr;
    }

    if (devImageView != nullptr && !devImageView->createView(static_cast<roc::Image&>(*memory))) {
      LogError("[OCL] Fail to create device mem object for the view");
      delete devImageView;
      imageView->release();
      return nullptr;
    }

    imageView->replaceDeviceMemory(this, devImageView);

    result = xferMgr().writeImage(owner.getHostMem(), *devImageView, amd::Coord3D(0, 0, 0),
                                  imageView->getRegion(), 0, 0, true);

    // Release host memory, since runtime copied data
    owner.setHostMem(nullptr);

    imageView->release();
  }

  // Prepin sysmem buffer for possible data synchronization between CPU and GPU
  if (!memory->isHostMemDirectAccess() &&
      (owner.getHostMem() != nullptr) &&
      (owner.getSvmPtr() == nullptr)) {
    memory->pinSystemMemory(owner.getHostMem(), owner.getSize());
  }

  if (!result) {
    delete memory;
    DevLogError("Cannot Write Image \n");
    return nullptr;
  }

  return memory;
}

void* Device::hostAlloc(size_t size, size_t alignment, bool atomics) const {
  void* ptr = nullptr;
  const hsa_amd_memory_pool_t segment = (!atomics)
      ? (system_coarse_segment_.handle != 0) ? system_coarse_segment_ : system_segment_
      : system_segment_;
  assert(segment.handle != 0);
  hsa_status_t stat = hsa_amd_memory_pool_allocate(segment, size, 0, &ptr);
  if (stat != HSA_STATUS_SUCCESS) {
    LogError("Fail allocation host memory");
    return nullptr;
  }

  stat = hsa_amd_agents_allow_access(gpu_agents_.size(), &gpu_agents_[0], nullptr, ptr);
  if (stat != HSA_STATUS_SUCCESS) {
    LogError("Fail hsa_amd_agents_allow_access");
    hostFree(ptr, size);
    return nullptr;
  }

  return ptr;
}

void* Device::hostAgentAlloc(size_t size, const AgentInfo& agentInfo, bool atomics) const {
  void* ptr = nullptr;
  const hsa_amd_memory_pool_t segment =
      (!atomics) ?
          (agentInfo.coarse_grain_pool.handle != 0) ?
              agentInfo.coarse_grain_pool : agentInfo.fine_grain_pool
          : agentInfo.fine_grain_pool;
  assert(segment.handle != 0);
  hsa_status_t stat = hsa_amd_memory_pool_allocate(segment, size, 0, &ptr);
  if (stat != HSA_STATUS_SUCCESS) {
    LogPrintfError("Fail allocation host memory with err %d", stat);
    return nullptr;
  }

  stat = hsa_amd_agents_allow_access(gpu_agents_.size(), &gpu_agents_[0], nullptr, ptr);
  if (stat != HSA_STATUS_SUCCESS) {
    LogPrintfError("Fail hsa_amd_agents_allow_access with err %d", stat);
    hostFree(ptr, size);
    return nullptr;
  }

  return ptr;
}

void* Device::hostNumaAlloc(size_t size, size_t alignment, bool atomics) const {
  void* ptr = nullptr;
  int mode = MPOL_DEFAULT;
  unsigned long nodeMask = 0;
  auto cpuCount = cpu_agents_.size();

  constexpr unsigned long maxNode = sizeof(nodeMask) * 8;
  long res = get_mempolicy(&mode, &nodeMask, maxNode, NULL, 0);
  if (res) {
    LogPrintfError("get_mempolicy failed with error %ld", res);
    return ptr;
  }
  ClPrint(amd::LOG_INFO, amd::LOG_RESOURCE,
          "get_mempolicy() succeed with mode %d, nodeMask 0x%lx, cpuCount %zu",
          mode, nodeMask, cpuCount);

  switch (mode) {
    // For details, see "man get_mempolicy".
    case MPOL_BIND:
    case MPOL_PREFERRED:
      // We only care about the first CPU node
      for (unsigned int i = 0; i < cpuCount; i++) {
        if ((1u << i) & nodeMask) {
          ptr = hostAgentAlloc(size, cpu_agents_[i], atomics);
          break;
        }
      }
      break;
    default:
      //  All other modes fall back to default mode
      ptr = hostAlloc(size, alignment, atomics);
  }
  return ptr;
}

void Device::hostFree(void* ptr, size_t size) const { memFree(ptr, size); }

void* Device::deviceLocalAlloc(size_t size, bool atomics) const {
  const hsa_amd_memory_pool_t& pool = (atomics)? gpu_fine_grained_segment_ : gpuvm_segment_;

  if (pool.handle == 0 || gpuvm_segment_max_alloc_ == 0) {
    DevLogPrintfError("Invalid argument, pool_handle: 0x%x , max_alloc: %u \n",
                      pool.handle, gpuvm_segment_max_alloc_);
    return nullptr;
  }

  void* ptr = nullptr;
  hsa_status_t stat = hsa_amd_memory_pool_allocate(pool, size, 0, &ptr);
  if (stat != HSA_STATUS_SUCCESS) {
    LogError("Fail allocation local memory");
    return nullptr;
  }

  if (p2pAgents().size() > 0) {
    stat = hsa_amd_agents_allow_access(p2pAgents().size(), p2pAgents().data(), nullptr, ptr);
    if (stat != HSA_STATUS_SUCCESS) {
      LogError("Allow p2p access for memory allocation");
      memFree(ptr, size);
      return nullptr;
    }
  }

  return ptr;
}

void Device::memFree(void* ptr, size_t size) const {
  hsa_status_t stat = hsa_amd_memory_pool_free(ptr);
  if (stat != HSA_STATUS_SUCCESS) {
    LogError("Fail freeing local memory");
  }
}

void Device::updateFreeMemory(size_t size, bool free) {
  if (free) {
    freeMem_ += size;
  }
  else {
    freeMem_ -= size;
  }
  ClPrint(amd::LOG_INFO, amd::LOG_MEM, "device=0x%lx, freeMem_ = 0x%x", this, freeMem_.load());
}

bool Device::IpcCreate(void* dev_ptr, size_t* mem_size, void* handle) {
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;

  amd::Memory* amd_mem_obj = amd::MemObjMap::FindMemObj(dev_ptr);
  if (amd_mem_obj == nullptr) {
    DevLogPrintfError("Cannot retrieve amd_mem_obj for dev_ptr: 0x%x \n", dev_ptr);
    return false;
  }

  // Get the starting pointer from the amd::Memory object
  void* orig_dev_ptr = nullptr;
  if (amd_mem_obj->getSvmPtr() != nullptr) {
    orig_dev_ptr = amd_mem_obj->getSvmPtr();
  } else if (amd_mem_obj->getHostMem() != nullptr) {
    orig_dev_ptr = amd_mem_obj->getHostMem();
  } else {
    ShouldNotReachHere();
  }

  if (orig_dev_ptr != dev_ptr) {
    DevLogPrintfError("Handle can only be created for Original Dev Ptr: 0x%x", orig_dev_ptr);
    return false;
  }

  *mem_size = amd_mem_obj->getSize();

  // Pass the pointer and memory size to retrieve the handle
  hsa_status = hsa_amd_ipc_memory_create(dev_ptr, amd::alignUp(*mem_size, alloc_granularity()),
                                         reinterpret_cast<hsa_amd_ipc_memory_t*>(handle));

  if (hsa_status != HSA_STATUS_SUCCESS) {
    LogPrintfError("Failed to create memory for IPC, failed with hsa_status: %d \n", hsa_status);
    return false;
  }

  return true;
}

bool Device::IpcAttach(const void* handle, size_t mem_size, unsigned int flags,
                       void** dev_ptr) const {
  amd::Memory* amd_mem_obj = nullptr;
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;

  // Retrieve the devPtr from the handle
  hsa_status
    = hsa_amd_ipc_memory_attach(reinterpret_cast<const hsa_amd_ipc_memory_t*>(handle),
                                mem_size, (1 + p2p_agents_.size()), p2p_agents_list_, dev_ptr);

  if (hsa_status != HSA_STATUS_SUCCESS) {
    LogPrintfError("HSA failed to attach IPC memory with status: %d \n", hsa_status);
    return false;
  }

  // Create an amd Memory object for the pointer
  amd_mem_obj = new (context()) amd::Buffer(context(), flags, mem_size, *dev_ptr);
  if (amd_mem_obj == nullptr) {
    LogError("failed to create a mem object!");
    return false;
  }

  if (!amd_mem_obj->create(nullptr)) {
    LogError("failed to create a svm hidden buffer!");
    amd_mem_obj->release();
    return false;
  }

  // Add the memory to the MemObjMap
  amd::MemObjMap::AddMemObj(*dev_ptr, amd_mem_obj);

  return true;
}

bool Device::IpcDetach (void* dev_ptr) const {
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;

  amd::Memory* amd_mem_obj = amd::MemObjMap::FindMemObj(dev_ptr);
  if (amd_mem_obj == nullptr) {
    DevLogPrintfError("Memory object for the ptr: 0x%x cannot be null \n", dev_ptr);
    return false;
  }

  // Detach the memory from HSA
  hsa_status = hsa_amd_ipc_memory_detach(dev_ptr);
  if (hsa_status != HSA_STATUS_SUCCESS) {
    LogPrintfError("HSA failed to detach memory with status: %d \n", hsa_status);
    return false;
  }

  amd::MemObjMap::RemoveMemObj(dev_ptr);
  amd_mem_obj->release();

  return true;
}

void* Device::svmAlloc(amd::Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags,
                       void* svmPtr) const {
  amd::Memory* mem = nullptr;

  if (nullptr == svmPtr) {
    // create a hidden buffer, which will allocated on the device later
    mem = new (context) amd::Buffer(context, flags, size, reinterpret_cast<void*>(1));
    if (mem == nullptr) {
      LogError("failed to create a svm mem object!");
      return nullptr;
    }

    if (!mem->create(nullptr)) {
      LogError("failed to create a svm hidden buffer!");
      mem->release();
      return nullptr;
    }
    // if the device supports SVM FGS, return the committed CPU address directly.
    Memory* gpuMem = getRocMemory(mem);

    // add the information to context so that we can use it later.
    amd::MemObjMap::AddMemObj(mem->getSvmPtr(), mem);
    svmPtr = mem->getSvmPtr();
  } else {
    // Find the existing amd::mem object
    mem = amd::MemObjMap::FindMemObj(svmPtr);
    if (nullptr == mem) {
      DevLogPrintfError("Cannot find svm_ptr: 0x%x \n", svmPtr);
      return nullptr;
    }

    svmPtr = mem->getSvmPtr();
  }

  return svmPtr;
}

// ================================================================================================
bool Device::SetSvmAttributes(const void* dev_ptr, size_t count,
                              amd::MemoryAdvice advice, bool first_alloc) const {
  if ((settings().hmmFlags_ & Settings::Hmm::EnableSvmTracking) && !first_alloc) {
    amd::Memory* svm_mem = amd::MemObjMap::FindMemObj(dev_ptr);
    if (nullptr == svm_mem) {
      LogPrintfError("SetSvmAttributes received unknown memory for update: %p!", dev_ptr);
      return false;
    }
  }
#if AMD_HMM_SUPPORT
  std::vector<hsa_amd_svm_attribute_pair_t> attr;
  if (first_alloc) {
    attr.push_back({HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG, HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED});
  }

  switch (advice) {
    case amd::MemoryAdvice::SetReadMostly:
      attr.push_back({HSA_AMD_SVM_ATTRIB_READ_ONLY, true});
      break;
    case amd::MemoryAdvice::UnsetReadMostly:
      attr.push_back({HSA_AMD_SVM_ATTRIB_READ_ONLY, false});
      break;
    case amd::MemoryAdvice::SetPreferredLocation:
      attr.push_back({HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, getBackendDevice().handle});
      break;
    case amd::MemoryAdvice::UnsetPreferredLocation:
      // Note: The current behavior doesn't match hip spec precisely
      attr.push_back({HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, getCpuAgent().handle});
      break;
    case amd::MemoryAdvice::SetAccessedBy:
      attr.push_back({HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE, getBackendDevice().handle});
      break;
    case amd::MemoryAdvice::UnsetAccessedBy:
      // @note: The current behavior doesn't match hip spec precisely
      attr.push_back({HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE, getCpuAgent().handle});
      break;
    default:
      return false;
    break;
  }

  hsa_status_t status = hsa_amd_svm_attributes_set(const_cast<void*>(dev_ptr), count,
                                                   attr.data(), attr.size());
  if (status != HSA_STATUS_SUCCESS) {
    LogError("hsa_amd_svm_attributes_set() failed");
    return false;
  }
#endif // AMD_HMM_SUPPORT
  return true;
}

// ================================================================================================
bool Device::GetSvmAttributes(void** data, size_t* data_sizes, int* attributes,
                              size_t num_attributes, const void* dev_ptr, size_t count) const {
  if (settings().hmmFlags_ & Settings::Hmm::EnableSvmTracking) {
    amd::Memory* svm_mem = amd::MemObjMap::FindMemObj(dev_ptr);
    if (nullptr == svm_mem) {
      LogPrintfError("GetSvmAttributes received unknown memory %p for state!", dev_ptr);
      return false;
    }
  }
#if AMD_HMM_SUPPORT
  std::vector<hsa_amd_svm_attribute_pair_t> attr;

  for (int i = 0; i < num_attributes; ++i) {
    switch (attributes[i]) {
      case amd::MemRangeAttribute::ReadMostly:
        attr.push_back({HSA_AMD_SVM_ATTRIB_READ_ONLY, 0});
        break;
      case amd::MemRangeAttribute::PreferredLocation:
        attr.push_back({HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, 0});
        break;
      case amd::MemRangeAttribute::AccessedBy:
        attr.push_back({HSA_AMD_SVM_ATTRIB_ACCESS_QUERY, 0});
        break;
      case amd::MemRangeAttribute::LastPrefetchLocation:
        attr.push_back({HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION, 0});
        break;
      default:
        return false;
      break;
    }
  }

  hsa_status_t status = hsa_amd_svm_attributes_get(const_cast<void*>(dev_ptr), count,
                                                   attr.data(), attr.size());
  if (status != HSA_STATUS_SUCCESS) {
    LogError("hsa_amd_svm_attributes_get() failed");
    return false;
  }

  uint32_t idx = 0;
  for (auto& it : attr) {
    switch (it.attribute) {
      case HSA_AMD_SVM_ATTRIB_READ_ONLY:
        if (data_sizes[idx] != sizeof(uint32_t)) {
          return false;
        }
        // Cast ROCr value into the hip format
        *reinterpret_cast<uint32_t*>(data[idx]) = static_cast<uint32_t>(it.value);
        break;
      // The logic should be identical for the both queries
      case HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION:
      case HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION:
        if (data_sizes[idx] != sizeof(uint32_t)) {
          return false;
        }
        *reinterpret_cast<int32_t*>(data[idx]) = static_cast<int32_t>(amd::InvalidDeviceId);
        // Find device agent returned by ROCr
        for (auto& device : devices()) {
          if (static_cast<Device*>(device)->getBackendDevice().handle == it.value) {
            *reinterpret_cast<uint32_t*>(data[idx]) = static_cast<uint32_t>(device->index());
          }
        }
        // Find CPU agent returned by ROCr
        for (auto& agent_info : getCpuAgents()) {
          if (agent_info.agent.handle == it.value) {
            *reinterpret_cast<int32_t*>(data[idx]) = static_cast<int32_t>(amd::CpuDeviceId);
          }
        }
        break;
      case HSA_AMD_SVM_ATTRIB_ACCESS_QUERY:
        // Make sure it's multiple of 4
        if (data_sizes[idx] % 4 != 0) {
          return false;
        }
        // @note currently it's a nop
        break;
      default:
        return false;
      break;
    }
    // Find the next location in the query
    ++idx;
  }
#endif // AMD_HMM_SUPPORT
  return true;
}

// ================================================================================================
bool Device::SvmAllocInit(void* memory, size_t size) const {
  amd::MemoryAdvice advice = amd::MemoryAdvice::SetAccessedBy;
  constexpr bool kFirstAlloc = true;
  SetSvmAttributes(memory, size, advice, kFirstAlloc);

  if (settings().hmmFlags_ & Settings::Hmm::EnableSystemMemory) {
    advice = amd::MemoryAdvice::UnsetPreferredLocation;
    SetSvmAttributes(memory, size, advice);
  } else {
    advice = amd::MemoryAdvice::SetPreferredLocation;
    SetSvmAttributes(memory, size, advice);
  }

  if ((settings().hmmFlags_ & Settings::Hmm::EnableMallocPrefetch) == 0) {
    return true;
  }

#if AMD_HMM_SUPPORT
  // Initialize signal for the barrier
  hsa_signal_store_relaxed(prefetch_signal_, InitSignalValue);

  // Initiate a prefetch command which should force memory update in HMM
  hsa_status_t status = hsa_amd_svm_prefetch_async(memory, size, getBackendDevice(),
                                                   0, nullptr, prefetch_signal_);
  if (status != HSA_STATUS_SUCCESS) {
    LogError("hsa_amd_svm_attributes_get() failed");
    return false;
  }

  // Wait for the prefetch
  if (hsa_signal_wait_scacquire(prefetch_signal_, HSA_SIGNAL_CONDITION_EQ, 0, uint64_t(-1),
                              HSA_WAIT_STATE_BLOCKED) != 0) {
    LogError("Barrier packet submission failed");
    return false;
  }
#endif // AMD_HMM_SUPPORT
  return true;
}

// ================================================================================================
void Device::svmFree(void* ptr) const {
  amd::Memory* svmMem = nullptr;
  svmMem = amd::MemObjMap::FindMemObj(ptr);
  if (nullptr != svmMem) {
    amd::MemObjMap::RemoveMemObj(svmMem->getSvmPtr());
    svmMem->release();
  }
}

VirtualGPU* Device::xferQueue() const {
  if (!xferQueue_) {
    // Create virtual device for internal memory transfer
    Device* thisDevice = const_cast<Device*>(this);
    thisDevice->xferQueue_ = reinterpret_cast<VirtualGPU*>(thisDevice->createVirtualDevice());
    if (!xferQueue_) {
      LogError("Couldn't create the device transfer manager!");
    }
  }
  xferQueue_->enableSyncBlit();
  return xferQueue_;
}

bool Device::SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput,
  cl_set_device_clock_mode_output_amd* pSetClockModeOutput) {
  bool result = true;
  return result;
}

static void callbackQueue(hsa_status_t status, hsa_queue_t* queue, void* data) {
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    // Abort on device exceptions.
    ClPrint(amd::LOG_NONE, amd::LOG_ALWAYS, "Device::callbackQueue aborting with status: 0x%x",
            status);
    abort();
  }
}

hsa_queue_t* Device::getQueueFromPool(const uint qIndex) {
  if (qIndex < QueuePriority::Total && queuePool_[qIndex].size() > 0) {
    typedef decltype(queuePool_)::value_type::const_reference PoolRef;
    auto lowest = std::min_element(queuePool_[qIndex].begin(),
        queuePool_[qIndex].end(), [] (PoolRef A, PoolRef B) {
          return A.second.refCount < B.second.refCount;
        });
    ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
        "selected queue with least refCount: %p (%d)", lowest->first,
        lowest->second.refCount);
    lowest->second.refCount++;
    return lowest->first;
  } else {
    return nullptr;
  }
}

hsa_queue_t* Device::acquireQueue(uint32_t queue_size_hint, bool coop_queue,
                                  const std::vector<uint32_t>& cuMask,
                                  amd::CommandQueue::Priority priority) {
  assert(queuePool_[QueuePriority::Low].size() <= GPU_MAX_HW_QUEUES ||
         queuePool_[QueuePriority::Normal].size() <= GPU_MAX_HW_QUEUES ||
         queuePool_[QueuePriority::High].size() <= GPU_MAX_HW_QUEUES);

  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "number of allocated hardware queues with low priority: %d,"
      " with normal priority: %d, with high priority: %d, maximum per priority is: %d",
      queuePool_[QueuePriority::Low].size(),
      queuePool_[QueuePriority::Normal].size(),
      queuePool_[QueuePriority::High].size(), GPU_MAX_HW_QUEUES);

  hsa_amd_queue_priority_t queue_priority;
  uint qIndex;
  switch (priority) {
    case amd::CommandQueue::Priority::Low:
      queue_priority = HSA_AMD_QUEUE_PRIORITY_LOW;
      qIndex = QueuePriority::Low;
      break;
    case amd::CommandQueue::Priority::High:
      queue_priority = HSA_AMD_QUEUE_PRIORITY_HIGH;
      qIndex = QueuePriority::High;
      break;
    case amd::CommandQueue::Priority::Normal:
    case amd::CommandQueue::Priority::Medium:
    default:
      queue_priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
      qIndex = QueuePriority::Normal;
      break;
  }

  // If we have reached the max number of queues, reuse an existing queue with the matching queue priority,
  // choosing the one with the least number of users.
  // Note: Don't attempt to reuse the cooperative queue, since it's single per device
  if (!coop_queue && (cuMask.size() == 0) && (queuePool_[qIndex].size() == GPU_MAX_HW_QUEUES)) {
    return getQueueFromPool(qIndex);
  }

  // Else create a new queue. This also includes the initial state where there
  // is no queue.
  uint32_t queue_max_packets = 0;
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max_packets)) {
    DevLogError("Cannot get hsa agent info \n");
    return nullptr;
  }
  auto queue_size = (queue_max_packets < queue_size_hint) ? queue_max_packets : queue_size_hint;

  hsa_queue_t* queue;
  auto queue_type = HSA_QUEUE_TYPE_MULTI;

  // Enable cooperative queue for the device queue
  if (coop_queue) {
    queue_type = HSA_QUEUE_TYPE_COOPERATIVE;
  }

  while (hsa_queue_create(_bkendDevice, queue_size, queue_type, callbackQueue, this,
                          std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max(),
                          &queue) != HSA_STATUS_SUCCESS) {
    queue_size >>= 1;
    if (queue_size < 64) {
      // if a queue with the same requested priority available from the pool, returns it here
      if (!coop_queue && (cuMask.size() == 0) && (queuePool_[qIndex].size() > 0)) {
        return getQueueFromPool(qIndex);
      }
      DevLogError("Device::acquireQueue: hsa_queue_create failed!");
      return nullptr;
    }
  }

  // default priority is normal so no need to set it again
  if (queue_priority != HSA_AMD_QUEUE_PRIORITY_NORMAL) {
    hsa_status_t st = HSA_STATUS_SUCCESS;
    st = hsa_amd_queue_set_priority(queue, queue_priority);
    if (st != HSA_STATUS_SUCCESS) {
      DevLogError("Device::acquireQueue: hsa_amd_queue_set_priority failed!");
      hsa_queue_destroy(queue);
      return nullptr;
    }
  }

  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "created hardware queue %p with size %d with priority %d,"
      " cooperative: %i", queue, queue_size, queue_priority, coop_queue);

  hsa_amd_profiling_set_profiler_enabled(queue, 1);
  if (cuMask.size() != 0) {
    std::stringstream ss;
    ss << std::hex;
    for (int i = cuMask.size() - 1; i >= 0; i--) {
      ss << cuMask[i];
    }
    ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "setting custom CU mask 0x%s for hardware queue %p",
            ss.str().c_str(), queue);
    hsa_status_t status = HSA_STATUS_SUCCESS;
    status = hsa_amd_queue_cu_set_mask(queue, cuMask.size() * 32, cuMask.data());
    if (status != HSA_STATUS_SUCCESS) {
      DevLogError("Device::acquireQueue: hsa_amd_queue_cu_set_mask failed!");
      hsa_queue_destroy(queue);
      return nullptr;
    }
    // Skip queue recycling for queues with custom CU mask
    return queue;
  }
  if (coop_queue) {
    // Skip queue recycling for cooperative queues, since it should be just one
    // per device.
    return queue;
  }
  auto result = queuePool_[qIndex].emplace(std::make_pair(queue, QueueInfo()));
  assert(result.second && "QueueInfo already exists");
  auto &qInfo = result.first->second;
  qInfo.refCount = 1;
  return queue;
}

void Device::releaseQueue(hsa_queue_t* queue) {
  for (auto& it : queuePool_) {
    auto qIter = it.find(queue);
    if (qIter != it.end()) {
      auto &qInfo = qIter->second;
      assert(qInfo.refCount > 0);
      qInfo.refCount--;
      if (qInfo.refCount != 0) {
        return;
      }
      ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
          "deleting hardware queue %p with refCount 0", queue);

      if (qInfo.hostcallBuffer_) {
        ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
            "deleting hostcall buffer %p for hardware queue %p",
            qInfo.hostcallBuffer_, queue);
        disableHostcalls(qInfo.hostcallBuffer_);
        context().svmFree(qInfo.hostcallBuffer_);
      }

      ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
          "deleting hardware queue %p with refCount 0", queue);
      it.erase(qIter);
      break;
    }
  }
  hsa_queue_destroy(queue);
}

void* Device::getOrCreateHostcallBuffer(hsa_queue_t* queue) {
  decltype(queuePool_)::value_type::iterator qIter;
  for (auto& it : queuePool_) {
    qIter = it.find(queue);
    if (qIter != it.end()) {
      break;
    }
  }

  assert(qIter != queuePool_[QueuePriority::High].end());

  auto& qInfo = qIter->second;
  if (qInfo.hostcallBuffer_) {
    return qInfo.hostcallBuffer_;
  }

  // The number of packets required in each buffer is at least equal to the
  // maximum number of waves supported by the device.
  auto wavesPerCu = info().maxThreadsPerCU_ / info().wavefrontWidth_;
  auto numPackets = info().maxComputeUnits_ * wavesPerCu;

  auto size = getHostcallBufferSize(numPackets);
  auto align = getHostcallBufferAlignment();

  void* buffer = context().svmAlloc(size, align, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS);
  if (!buffer) {
    ClPrint(amd::LOG_ERROR, amd::LOG_QUEUE,
            "Failed to create hostcall buffer for hardware queue %p", queue);
    return nullptr;
  }
  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "Created hostcall buffer %p for hardware queue %p", buffer,
          queue);
  qInfo.hostcallBuffer_ = buffer;
  if (!enableHostcalls(buffer, numPackets)) {
    ClPrint(amd::LOG_ERROR, amd::LOG_QUEUE, "Failed to register hostcall buffer %p with listener",
            buffer);
    return nullptr;
  }
  return buffer;
}

bool Device::findLinkTypeAndHopCount(amd::Device* other_device,
                                     uint32_t* link_type, uint32_t* hop_count) {
  uint32_t distance = 0;
  return getNumaInfo((static_cast<roc::Device*>(other_device))->gpuvm_segment_,
                     hop_count, link_type, &distance);
}

bool Device::getNumaInfo(const hsa_amd_memory_pool_t& pool, uint32_t* hop_count,
                         uint32_t* link_type, uint32_t* numa_distance) const {
  uint32_t hops = 0;

  if (!pool.handle) {
    return false;
  }

  hsa_status_t res = hsa_amd_agent_memory_pool_get_info(_bkendDevice, pool,
      HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS, &hops);
  if (res != HSA_STATUS_SUCCESS) {
    return false;
  }

  if (hops < 0) {
    return false;
  } else if (hops == 0) {
    //This pool is on its agent
    *hop_count = 0;  // No hop
    *link_type = -1; // No link, so type is meaningless, caller should ignore it.
    *numa_distance = 0;
    return true;
  }

  hsa_amd_memory_pool_link_info_t *link_info = new hsa_amd_memory_pool_link_info_t[hops];

  res = hsa_amd_agent_memory_pool_get_info(_bkendDevice, pool,
        HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO, link_info);

  if (res == HSA_STATUS_SUCCESS) {
    // Now RocR always set hops=1 between two different devices.
    // If RocR changes the behavior, we need revisit here.
    *link_type = link_info[0].link_type;

    uint32_t distance = 0;
    for (uint32_t i = 0; i < hops; i++) {
      distance += link_info[i].numa_distance;
    }
    *numa_distance = distance;

    // The following logics will be subject to change in rocm3.7
    uint32_t oneHopDistance = 20; // Default to PCIE
    if (*link_type == HSA_AMD_LINK_INFO_TYPE_XGMI) {
      oneHopDistance = 15;
    }
    *hop_count = distance/oneHopDistance;
  }

  delete [] link_info;
  return res == HSA_STATUS_SUCCESS;
}

} // namespace roc
#endif  // WITHOUT_HSA_BACKEND
