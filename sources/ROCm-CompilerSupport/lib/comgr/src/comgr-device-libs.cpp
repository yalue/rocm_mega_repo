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

#include "comgr-device-libs.h"
#include "comgr.h"
#include "libraries.inc"
#include <cstdint>

using namespace llvm;

namespace COMGR {

static amd_comgr_status_t addObject(DataSet *DataSet,
                                    amd_comgr_data_kind_t Kind,
                                    const char *Name, const void *Data,
                                    size_t Size) {
  DataObject *Obj = DataObject::allocate(Kind);
  if (!Obj)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  if (auto Status = Obj->setName(Name))
    return Status;
  if (auto Status =
          Obj->setData(StringRef(reinterpret_cast<const char *>(Data), Size)))
    return Status;
  DataSet->DataObjects.insert(Obj);
  return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t
addOCLCObject(DataSet *DataSet,
              std::tuple<const char *, const void *, size_t> OCLCLib) {
  return addObject(DataSet, AMD_COMGR_DATA_KIND_BC, std::get<0>(OCLCLib),
                   std::get<1>(OCLCLib), std::get<2>(OCLCLib));
}

amd_comgr_status_t addPrecompiledHeaders(DataAction *ActionInfo,
                                         DataSet *ResultSet) {
  switch (ActionInfo->Language) {
  case AMD_COMGR_LANGUAGE_OPENCL_1_2:
    return addObject(ResultSet, AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER,
                     "opencl1.2-c.pch", opencl1_2_c, opencl1_2_c_size);
  case AMD_COMGR_LANGUAGE_OPENCL_2_0:
    return addObject(ResultSet, AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER,
                     "opencl2.0-c.pch", opencl2_0_c, opencl2_0_c_size);
  default:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

amd_comgr_status_t addDeviceLibraries(DataAction *ActionInfo,
                                      DataSet *ResultSet) {
  if (ActionInfo->Language != AMD_COMGR_LANGUAGE_OPENCL_1_2 &&
      ActionInfo->Language != AMD_COMGR_LANGUAGE_OPENCL_2_0)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (auto Status = addObject(ResultSet, AMD_COMGR_DATA_KIND_BC,
                              "opencl_lib.bc", opencl_lib, opencl_lib_size))
    return Status;
  if (auto Status = addObject(ResultSet, AMD_COMGR_DATA_KIND_BC, "ocml_lib.bc",
                              ocml_lib, ocml_lib_size))
    return Status;
  if (auto Status = addObject(ResultSet, AMD_COMGR_DATA_KIND_BC, "ockl_lib.bc",
                              ockl_lib, ockl_lib_size))
    return Status;

  TargetIdentifier Ident;
  if (auto Status = parseTargetIdentifier(ActionInfo->IsaName, Ident))
    return Status;
  if (!Ident.Processor.consume_front("gfx"))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  uint32_t GFXIP;
  if (Ident.Processor.getAsInteger<uint32_t>(10, GFXIP))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  if (auto Status = addOCLCObject(ResultSet, get_oclc_isa_version(GFXIP)))
    return Status;

  StringRef OptionsString(ActionInfo->Options);
  SmallVector<StringRef, 5> Options;
  OptionsString.split(Options, ',', -1, false);
  bool CorrectlyRoundedSqrt = false, DazOpt = false, FiniteOnly = false,
       UnsafeMath = false;
  for (auto &Option : Options) {
    bool *Flag = StringSwitch<bool *>(Option)
                     .Case("correctly_rounded_sqrt", &CorrectlyRoundedSqrt)
                     .Case("daz_opt", &DazOpt)
                     .Case("finite_only", &FiniteOnly)
                     .Case("unsafe_math", &UnsafeMath)
                     .Default(nullptr);
    // It is invalid to provide an unknown option and to repeat an option.
    if (!Flag || *Flag)
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    *Flag = true;
  }

  if (auto Status = addOCLCObject(
          ResultSet, get_oclc_correctly_rounded_sqrt(CorrectlyRoundedSqrt)))
    return Status;
  if (auto Status = addOCLCObject(ResultSet, get_oclc_daz_opt(DazOpt)))
    return Status;
  if (auto Status = addOCLCObject(ResultSet, get_oclc_finite_only(FiniteOnly)))
    return Status;
  if (auto Status = addOCLCObject(ResultSet, get_oclc_unsafe_math(UnsafeMath)))
    return Status;

  return AMD_COMGR_STATUS_SUCCESS;
}

} // namespace COMGR
