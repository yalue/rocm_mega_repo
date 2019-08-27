/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>

#include <assert.h>
#include <sys/stat.h>
#include <stdint.h>
#include <string>
#include <map>
#include <fstream>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>

#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_exception.h"

extern "C" {
#include "shared_mutex.h"  // NOLINT
};

namespace amd {
namespace smi {

// Sysfs file names
static const char *kDevPerfLevelFName = "power_dpm_force_performance_level";
static const char *kDevDevIDFName = "device";
static const char *kDevVendorIDFName = "vendor";
static const char *kDevSubSysDevIDFName = "subsystem_device";
static const char *kDevSubSysVendorIDFName = "subsystem_vendor";
static const char *kDevOverDriveLevelFName = "pp_sclk_od";
static const char *kDevGPUSClkFName = "pp_dpm_sclk";
static const char *kDevGPUMClkFName = "pp_dpm_mclk";
static const char *kDevDCEFClkFName = "pp_dpm_dcefclk";
static const char *kDevFClkFName = "pp_dpm_fclk";
static const char *kDevSOCClkFName = "pp_dpm_socclk";
static const char *kDevGPUPCIEClkFname = "pp_dpm_pcie";
static const char *kDevPowerProfileModeFName = "pp_power_profile_mode";
static const char *kDevPowerODVoltageFName = "pp_od_clk_voltage";
static const char *kDevUsageFName = "gpu_busy_percent";
static const char *kDevVBiosVerFName = "vbios_version";
static const char *kDevPCIEThruPutFName = "pcie_bw";
static const char *kDevErrCntSDMAFName = "ras/sdma_err_count";
static const char *kDevErrCntUMCFName = "ras/umc_err_count";
static const char *kDevErrCntGFXFName = "ras/gfx_err_count";
static const char *kDevErrCntFeaturesFName = "ras/features";
static const char *kDevMemTotGTTFName = "mem_info_gtt_total";
static const char *kDevMemTotVisVRAMFName = "mem_info_vis_vram_total";
static const char *kDevMemTotVRAMFName = "mem_info_vram_total";
static const char *kDevMemUsedGTTFName = "mem_info_gtt_used";
static const char *kDevMemUsedVisVRAMFName = "mem_info_vis_vram_used";
static const char *kDevMemUsedVRAMFName = "mem_info_vram_used";
static const char *kDevPCIEReplayCountFName = "pcie_replay_count";

// Strings that are found within sysfs files
static const char *kDevPerfLevelAutoStr = "auto";
static const char *kDevPerfLevelLowStr = "low";
static const char *kDevPerfLevelHighStr = "high";
static const char *kDevPerfLevelManualStr = "manual";
static const char *kDevPerfLevelStandardStr = "profile_standard";
static const char *kDevPerfLevelMinMClkStr = "profile_min_mclk";
static const char *kDevPerfLevelMinSClkStr = "profile_min_sclk";
static const char *kDevPerfLevelPeakStr = "profile_peak";
static const char *kDevPerfLevelUnknownStr = "unknown";

static const std::map<DevInfoTypes, const char *> kDevAttribNameMap = {
    {kDevPerfLevel, kDevPerfLevelFName},
    {kDevOverDriveLevel, kDevOverDriveLevelFName},
    {kDevDevID, kDevDevIDFName},
    {kDevVendorID, kDevVendorIDFName},
    {kDevSubSysDevID, kDevSubSysDevIDFName},
    {kDevSubSysVendorID, kDevSubSysVendorIDFName},
    {kDevGPUMClk, kDevGPUMClkFName},
    {kDevGPUSClk, kDevGPUSClkFName},
    {kDevDCEFClk, kDevDCEFClkFName},
    {kDevFClk, kDevFClkFName},
    {kDevSOCClk, kDevSOCClkFName},
    {kDevPCIEClk, kDevGPUPCIEClkFname},
    {kDevPowerProfileMode, kDevPowerProfileModeFName},
    {kDevUsage, kDevUsageFName},
    {kDevPowerODVoltage, kDevPowerODVoltageFName},
    {kDevVBiosVer, kDevVBiosVerFName},
    {kDevPCIEThruPut, kDevPCIEThruPutFName},
    {kDevErrCntSDMA, kDevErrCntSDMAFName},
    {kDevErrCntUMC, kDevErrCntUMCFName},
    {kDevErrCntGFX, kDevErrCntGFXFName},
    {kDevErrCntFeatures, kDevErrCntFeaturesFName},
    {kDevMemTotGTT, kDevMemTotGTTFName},
    {kDevMemTotVisVRAM, kDevMemTotVisVRAMFName},
    {kDevMemTotVRAM, kDevMemTotVRAMFName},
    {kDevMemUsedGTT, kDevMemUsedGTTFName},
    {kDevMemUsedVisVRAM, kDevMemUsedVisVRAMFName},
    {kDevMemUsedVRAM, kDevMemUsedVRAMFName},
    {kDevPCIEReplayCount, kDevPCIEReplayCountFName},
};

static const std::map<rsmi_dev_perf_level, const char *> kDevPerfLvlMap = {
    {RSMI_DEV_PERF_LEVEL_AUTO, kDevPerfLevelAutoStr},
    {RSMI_DEV_PERF_LEVEL_LOW, kDevPerfLevelLowStr},
    {RSMI_DEV_PERF_LEVEL_HIGH, kDevPerfLevelHighStr},
    {RSMI_DEV_PERF_LEVEL_MANUAL, kDevPerfLevelManualStr},
    {RSMI_DEV_PERF_LEVEL_STABLE_STD, kDevPerfLevelStandardStr},
    {RSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK, kDevPerfLevelMinMClkStr},
    {RSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK, kDevPerfLevelMinSClkStr},
    {RSMI_DEV_PERF_LEVEL_STABLE_PEAK, kDevPerfLevelPeakStr},

    {RSMI_DEV_PERF_LEVEL_UNKNOWN, kDevPerfLevelUnknownStr},
};

static bool isRegularFile(std::string fname) {
  struct stat file_stat;
  stat(fname.c_str(), &file_stat);
  return S_ISREG(file_stat.st_mode);
}

#define RET_IF_NONZERO(X) { \
  if (X) return X; \
}

Device::Device(std::string p, RocmSMI_env_vars const *e) : path_(p), env_(e) {
  monitor_ = nullptr;

  // Get the device name
  size_t i = path_.rfind('/', path_.length());
  std::string dev = path_.substr(i + 1, path_.length() - i);

  std::string m_name("/rocm_smi_");
  m_name += dev;
  m_name += '_';
  m_name += std::to_string(geteuid());

  mutex_ = shared_mutex_init(m_name.c_str(), 0777);

  if (mutex_.ptr == nullptr) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
                                       "Failed to create shared mem. mutex.");
  }
}

Device:: ~Device() {
  shared_mutex_close(mutex_);
}

template <typename T>
int Device::openSysfsFileStream(DevInfoTypes type, T *fs, const char *str) {
  auto sysfs_path = path_;

  if (env_->path_DRM_root_override && type == env_->enum_override) {
    sysfs_path = env_->path_DRM_root_override;

    if (str) {
      sysfs_path += ".write";
    }
  }

  sysfs_path += "/device/";
  sysfs_path += kDevAttribNameMap.at(type);

  DBG_FILE_ERROR(sysfs_path, str);
  if (!isRegularFile(sysfs_path)) {
    return ENOENT;
  }

  fs->open(sysfs_path);

  if (!fs->is_open()) {
      return errno;
  }

  return 0;
}

int Device::readDevInfoStr(DevInfoTypes type, std::string *retStr) {
  std::ifstream fs;
  int ret = 0;

  assert(retStr != nullptr);

  ret = openSysfsFileStream(type, &fs);
  if (ret != 0) {
    return ret;
  }

  fs >> *retStr;
  fs.close();

  return 0;
}

int Device::writeDevInfoStr(DevInfoTypes type, std::string valStr) {
  auto tempPath = path_;
  std::ofstream fs;
  int ret;

  ret = openSysfsFileStream(type, &fs, valStr.c_str());
  if (ret != 0) {
    return ret;
  }

  try {
    fs << valStr;
  } catch (...) {
    std::cout << "Write to file threw exception" << std::endl;
  }
  fs.close();

  return 0;
}

rsmi_dev_perf_level Device::perfLvlStrToEnum(std::string s) {
  rsmi_dev_perf_level pl;

  for (pl = RSMI_DEV_PERF_LEVEL_FIRST; pl <= RSMI_DEV_PERF_LEVEL_LAST; ) {
    if (s == kDevPerfLvlMap.at(pl)) {
      return pl;
    }
    pl = static_cast<rsmi_dev_perf_level>(static_cast<uint32_t>(pl) + 1);
  }
  return RSMI_DEV_PERF_LEVEL_UNKNOWN;
}

int Device::writeDevInfo(DevInfoTypes type, uint64_t val) {
  switch (type) {
    // The caller is responsible for making sure "val" is within a valid range
    case kDevOverDriveLevel:  // integer between 0 and 20
    case kDevPowerProfileMode:
      return writeDevInfoStr(type, std::to_string(val));
      break;

    case kDevPerfLevel:  // string: "auto", "low", "high", "manual", ...
      return writeDevInfoStr(type,
                                 kDevPerfLvlMap.at((rsmi_dev_perf_level)val));
      break;

    default:
      break;
  }

  return -1;
}

int Device::writeDevInfo(DevInfoTypes type, std::string val) {
  switch (type) {
    case kDevGPUMClk:
    case kDevDCEFClk:
    case kDevFClk:
    case kDevGPUSClk:
    case kDevPCIEClk:
    case kDevPowerODVoltage:
    case kDevSOCClk:
      return writeDevInfoStr(type, val);

    default:
      break;
  }

  return -1;
}

int Device::readDevInfoLine(DevInfoTypes type, std::string *line) {
  int ret;
  std::ifstream fs;

  assert(line != nullptr);

  ret = openSysfsFileStream(type, &fs);
  if (ret != 0) {
    return ret;
  }

  std::getline(fs, *line);

  return 0;
}

int Device::readDevInfoMultiLineStr(DevInfoTypes type,
                                           std::vector<std::string> *retVec) {
  std::string line;
  int ret;
  std::ifstream fs;

  assert(retVec != nullptr);

  ret = openSysfsFileStream(type, &fs);
  if (ret != 0) {
    return ret;
  }

  while (std::getline(fs, line)) {
    retVec->push_back(line);
  }

  if (retVec->size() == 0) {
    return EPERM;
  }
  // Remove any *trailing* empty (whitespace) lines
  while (retVec->back().find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
    retVec->pop_back();
  }
  return 0;
}

int Device::readDevInfo(DevInfoTypes type, uint64_t *val) {
  assert(val != nullptr);

  std::string tempStr;
  int ret;
  switch (type) {
    case kDevDevID:
    case kDevSubSysDevID:
    case kDevSubSysVendorID:
    case kDevVendorID:
      ret = readDevInfoStr(type, &tempStr);
      RET_IF_NONZERO(ret);
      *val = std::stoi(tempStr, 0, 16);
      break;

    case kDevUsage:
    case kDevOverDriveLevel:
    case kDevMemTotGTT:
    case kDevMemTotVisVRAM:
    case kDevMemTotVRAM:
    case kDevMemUsedGTT:
    case kDevMemUsedVisVRAM:
    case kDevMemUsedVRAM:
    case kDevPCIEReplayCount:
      ret = readDevInfoStr(type, &tempStr);
      RET_IF_NONZERO(ret);
      *val = std::stoul(tempStr, 0);
      break;

    default:
      return -1;
  }
  return 0;
}

int Device::readDevInfo(DevInfoTypes type, std::vector<std::string> *val) {
  assert(val != nullptr);

  switch (type) {
    case kDevGPUMClk:
    case kDevGPUSClk:
    case kDevDCEFClk:
    case kDevFClk:
    case kDevPCIEClk:
    case kDevSOCClk:
    case kDevPowerProfileMode:
    case kDevPowerODVoltage:
    case kDevErrCntSDMA:
    case kDevErrCntUMC:
    case kDevErrCntGFX:
    case kDevErrCntFeatures:
      return readDevInfoMultiLineStr(type, val);
      break;

    default:
      return -1;
  }

  return 0;
}

int Device::readDevInfo(DevInfoTypes type, std::string *val) {
  assert(val != nullptr);

  switch (type) {
    case kDevPerfLevel:
    case kDevUsage:
    case kDevOverDriveLevel:
    case kDevDevID:
    case kDevSubSysDevID:
    case kDevSubSysVendorID:
    case kDevVendorID:
    case kDevVBiosVer:
    case kDevPCIEThruPut:
      return readDevInfoStr(type, val);
      break;

    default:
      return -1;
  }
  return 0;
}

#undef RET_IF_NONZERO
}  // namespace smi
}  // namespace amd
