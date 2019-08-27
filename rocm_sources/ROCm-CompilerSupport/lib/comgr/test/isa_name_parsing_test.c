/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void parseIsaName(amd_comgr_action_info_t dataAction, const char *isaName,
                  amd_comgr_status_t expectedStatus) {
  amd_comgr_status_t trueStatus =
      amd_comgr_action_info_set_isa_name(dataAction, isaName);
  if (trueStatus != expectedStatus) {
    amd_comgr_status_t status;
    const char *trueStatusString, *expectedStatusString;
    status = amd_comgr_status_string(trueStatus, &trueStatusString);
    checkError(status, "amd_comgr_status_string");
    status = amd_comgr_status_string(expectedStatus, &expectedStatusString);
    checkError(status, "amd_comgr_status_string");
    printf("Parsing \"%s\" resulted in \"%s\"; expected \"%s\"\n", isaName,
           trueStatusString, expectedStatusString);
    exit(1);
  }
}

int main(int argc, char *argv[]) {
  amd_comgr_status_t status;
  amd_comgr_action_info_t dataAction;

  status = amd_comgr_create_action_info(&dataAction);
  checkError(status, "amd_comgr_create_action_info");

  parseIsaName(dataAction, "amdgcn-amd-amdhsa--gfx803",
               AMD_COMGR_STATUS_SUCCESS);
  parseIsaName(dataAction, "amdgcn-amd-amdhsa--gfx801+xnack",
               AMD_COMGR_STATUS_SUCCESS);
  parseIsaName(dataAction, "", AMD_COMGR_STATUS_SUCCESS);
  parseIsaName(dataAction, NULL, AMD_COMGR_STATUS_SUCCESS);
  parseIsaName(dataAction, "amdgcn-amd-amdhsa-opencl-gfx803",
               AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
  parseIsaName(dataAction, "amdgcn-amd-amdhsa-gfx803",
               AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
  parseIsaName(dataAction, "gfx803", AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
  parseIsaName(dataAction, " amdgcn-amd-amdhsa--gfx803",
               AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
  parseIsaName(dataAction, " amdgcn-amd-amdhsa--gfx803 ",
               AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
  parseIsaName(dataAction, "amdgcn-amd-amdhsa--gfx803 ",
               AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
  parseIsaName(dataAction, "   amdgcn-amd-amdhsa--gfx803  ",
               AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT);

  status = amd_comgr_destroy_action_info(dataAction);
  checkError(status, "amd_comgr_destroy_action_info");
}
