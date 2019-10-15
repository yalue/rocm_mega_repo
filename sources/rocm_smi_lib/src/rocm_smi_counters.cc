/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2019, Advanced Micro Devices, Inc.
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

#include <assert.h>
#include <string.h>
#include <linux/perf_event.h>
#include <unistd.h>
#include <asm/unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <stdio.h>

#include <map>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <unordered_set>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_counters.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_device.h"

namespace amd {
namespace smi {
namespace evt {

static const char *kPathDeviceEventRoot = "/sys/bus/event_source/devices";

// Event group names
static const char *kEvGrpDataFabricFName = "amdgpu_df_#";

// Data Fabric event file names
static const char *kDFEvtCake0FtiReqAllocFName = "cake0_ftiinstat_reqalloc";
static const char *kDFEvtCake0FtiRspAllocFName = "cake0_ftiinstat_rspalloc";
static const char *kDFEvtCake0PcsOutTxDataFName = "cake0_pcsout_txdata";
static const char *kDFEvtCake0PcsOutTxMetaFName = "cake0_pcsout_txmeta";
static const char *kDFEvtCake1FtiReqAllocFName = "cake1_ftiinstat_reqalloc";
static const char *kDFEvtCake1FtiRspAllocFName = "cake1_ftiinstat_rspalloc";
static const char *kDFEvtCake1PcsOutTxDataFName = "cake1_pcsout_txdata";
static const char *kDFEvtCake1PcsOutTxMetaFName = "cake1_pcsout_txmeta";


static const std::map<rsmi_event_type_t, const char *> kEventFNameMap = {
  {RSMI_EVNT_XGMI_0_NOP_TX,      kDFEvtCake0PcsOutTxMetaFName},
  {RSMI_EVNT_XGMI_0_REQUEST_TX,  kDFEvtCake0FtiReqAllocFName},
  {RSMI_EVNT_XGMI_0_RESPONSE_TX, kDFEvtCake0FtiRspAllocFName},
  {RSMI_EVNT_XGMI_0_BEATS_TX,    kDFEvtCake0PcsOutTxDataFName},
  {RSMI_EVNT_XGMI_1_NOP_TX,      kDFEvtCake1PcsOutTxMetaFName},
  {RSMI_EVNT_XGMI_1_REQUEST_TX,  kDFEvtCake1FtiReqAllocFName},
  {RSMI_EVNT_XGMI_1_RESPONSE_TX, kDFEvtCake1FtiRspAllocFName},
  {RSMI_EVNT_XGMI_1_BEATS_TX,    kDFEvtCake1PcsOutTxDataFName},
};

static const std::map<rsmi_event_group_t, const char *> kEvtGrpFNameMap = {
    {RSMI_EVNT_GRP_XGMI, kEvGrpDataFabricFName},
    {RSMI_EVNT_GRP_INVALID, "bogus"},
};

static rsmi_event_group_t EvtGrpFromEvtID(rsmi_event_type_t evnt) {
#define EVNT_GRP_RANGE_CHK(EVGRP_SHORT, EVGRP_ENUM) \
  if (evnt >= RSMI_EVNT_##EVGRP_SHORT##_FIRST && \
                                   evnt <= RSMI_EVNT_##EVGRP_SHORT##_LAST) { \
    return EVGRP_ENUM; \
  }
  EVNT_GRP_RANGE_CHK(XGMI, RSMI_EVNT_GRP_XGMI);

  return RSMI_EVNT_GRP_INVALID;
}

// Note below that dev_num is not the same as the usual dv_ind.
// dev_num is the number of the device (e.g., 1 for card1) whereas dv_ind
// is usually the index into the vector of devices
void
GetSupportedEventGroups(uint32_t dev_num, dev_evt_grp_set_t *supported_grps) {
  assert(supported_grps != nullptr);

  std::string grp_path_base;
  std::string grp_path;
  uint32_t ret;

  grp_path_base = kPathDeviceEventRoot;
  grp_path_base += '/';
  struct stat file_stat;

  for (auto g : kEvtGrpFNameMap) {
    grp_path = grp_path_base;
    grp_path += g.second;

    std::replace(grp_path.begin(), grp_path.end(), '#',
                                            static_cast<char>('0' + dev_num));

    ret = stat(grp_path.c_str(), &file_stat);

    if (ret) {
      assert(errno == ENOENT);
      continue;
    }
    if (S_ISDIR(file_stat.st_mode)) {
      supported_grps->insert(g.first);
    }
  }
}
//  /sys/bus/event_source/devices/<hw block>_<instance>/type
Event::Event(rsmi_event_type_t event, uint32_t dev_ind)  :
                                       event_type_(event) {
  fd_ = -1;
  rsmi_event_group_t grp = EvtGrpFromEvtID(event);
  assert(grp != RSMI_EVNT_GRP_INVALID);  // This should have failed before now

  evt_path_root_ = kPathDeviceEventRoot;
  evt_path_root_ += '/';
  evt_path_root_ += kEvtGrpFNameMap.at(grp);


  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();
  assert(dev_ind < smi.monitor_devices().size());
  std::shared_ptr<amd::smi::Device> dev = smi.monitor_devices()[dev_ind];
  assert(dev != nullptr);


  dev_ind_ = dev_ind;
  dev_file_ind_ = dev->index();
  std::replace(evt_path_root_.begin(), evt_path_root_.end(), '#',
                                      static_cast<char>('0' + dev_file_ind_));
}
Event::~Event(void) {
  int ret;
  if (fd_ != -1) {
    ret = close(fd_);

    if (ret == -1) {
      perror("Failed to close file descriptor.");
    }
  }
}

static void
parse_field_config(std::string fstr, evnt_info_t *val) {
  std::stringstream ss(fstr);
  std::stringstream fs;
  std::string config_ln;
  std::string field_name;
  uint32_t start_bit;
  uint32_t end_bit;
  char jnk;

  assert(val != nullptr);

  getline(ss, config_ln, ':');
  ss >> start_bit;
  ss >> jnk;
  assert(jnk == '-');
  ss >> end_bit;

  val->start_bit = start_bit;
  val->field_size = end_bit - start_bit + 1;
}

static uint32_t
get_event_bitfield_info(std::string *config_path, evnt_info_t *val) {
  uint32_t err;

  std::string fstr;

  err = ReadSysfsStr(*config_path, &fstr);
  if (err) {
    return err;
  }
  parse_field_config(fstr, val);

  return 0;
}

uint32_t
Event::get_event_file_info(void) {
  uint32_t err;

  std::string fn = evt_path_root_;
  std::string fstr;

  fn += "/events/";
  fn += kEventFNameMap.at(event_type_);
  err = ReadSysfsStr(fn, &fstr);
  if (err) {
    return err;
  }
  // parse_perf_attr(fstr, &event_id_.event_field_vals);
  std::stringstream ss(fstr);
  std::stringstream fs;
  std::string field_assgn;
  std::string field_name;
  evnt_info_t ev_info;

  while (ss.rdbuf()->in_avail() != 0) {
    ev_info = {};
    getline(ss, field_assgn, ',');
    fs.clear();
    fs << field_assgn;
    getline(fs, field_name, '=');
    fs >> std::hex >> ev_info.value;
    assert(fs.rdbuf()->in_avail() == 0);

    // Now, get the corresponding bitfield
    std::string config_path = evt_path_root_;
    config_path += "/format/";
    config_path += field_name;
    err = get_event_bitfield_info(&config_path, &ev_info);
    if (err) {
      return err;
    }
    event_info_.push_back(ev_info);
  }
  return 0;
}

uint32_t
Event::get_event_type(uint32_t *ev_type) {
  assert(ev_type != nullptr);
  if (ev_type == nullptr) {
    return EINVAL;
  }
  std::string fn = evt_path_root_;
  std::string fstr;

  fn += "/type";

  std::ifstream fs;
  fs.open(fn);

  if (!fs.is_open()) {
    return errno;
  }
  fs >> *ev_type;
  fs.close();
  return 0;
}

static uint64_t
get_perf_attr_config(std::vector<evnt_info_t> *ev_info) {
  uint64_t ret_val = 0;

  assert(ev_info != nullptr);

  for (const evnt_info_t& ev : *ev_info) {
    ret_val |= ev.value << ev.start_bit;
  }
  return ret_val;
}

uint32_t
amd::smi::evt::Event::openPerfHandle(void) {
  uint32_t ret;

  memset(&attr_, 0, sizeof(struct perf_event_attr));

  ret = get_event_file_info();
  if (ret) {
    return ret;
  }
  ret = get_event_type(&attr_.type);
  if (ret) {
    return ret;
  }

  attr_.size = sizeof(struct perf_event_attr);
  attr_.config = get_perf_attr_config(&event_info_);
  attr_.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED |
                          PERF_FORMAT_TOTAL_TIME_RUNNING;
  attr_.disabled = 1;
  attr_.inherit = 1;

  fd_ = syscall(__NR_perf_event_open, &attr_,
                           -1, 0, -1, PERF_FLAG_FD_NO_GROUP);

  if (fd_ < 0) {
    return errno;
  }
  return 0;
}

uint32_t
amd::smi::evt::Event::startCounter(void) {
  int32_t ret;

  if (fd_ == -1) {
    ret = openPerfHandle();
    if (ret != 0) {
      return ret;
    }
  }
  ret = ioctl(fd_, PERF_EVENT_IOC_ENABLE, NULL);

  if (ret == -1) {
    return errno;
  }

  assert(ret == 0);  // We're expecting the ioctl call to return -1 or 0
  return 0;
}

uint32_t
amd::smi::evt::Event::stopCounter(void) {
  int32_t ret;

  if (fd_ == -1) {
    return EBADF;
  }
  ret = ioctl(fd_, PERF_EVENT_IOC_DISABLE, NULL);

  if (ret == -1) {
    return errno;
  }

  assert(ret == 0);  // We're expecting the ioctl call to return -1 or 0
  return 0;
}

static ssize_t
readn(int fd, void *buf, size_t n) {
  size_t left = n;
  ssize_t bytes;

  while (left) {
    bytes = read(fd, buf, left);
    if (!bytes) /* reach EOF */
      return (n - left);
    if (bytes < 0) {
      if (errno == EINTR) /* read got interrupted */
        continue;
      else
        return -errno;
    }
    left -= bytes;
    buf = reinterpret_cast<void *>((reinterpret_cast<uint8_t *>(buf) + bytes));
  }
  return n;
}

uint32_t
amd::smi::evt::Event::getValue(rsmi_counter_value_t *val) {
  assert(val != nullptr);
  ssize_t ret;

  perf_read_format_t pvalue;
  ret = readn(fd_, &pvalue, sizeof(perf_read_format_t));
  if (ret < 0) {
    return -ret;
  }

  if (ret != sizeof(perf_read_format_t)) {
    return EIO;
  }

  val->value = pvalue.value;
  val->time_enabled = pvalue.enabled_time;
  val->time_running = pvalue.run_time;

  return 0;
}

}  // namespace evt
}  // namespace smi
}  // namespace amd
