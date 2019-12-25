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

#include "comgr-disassembly.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;
using namespace COMGR;

amd_comgr_status_t
DisassemblyInfo::create(const TargetIdentifier &Ident,
                        ReadMemoryCallback ReadMemory,
                        PrintInstructionCallback PrintInstruction,
                        PrintAddressAnnotationCallback PrintAddressAnnotation,
                        amd_comgr_disassembly_info_t *DisassemblyInfoT) {
  std::string TT = (Twine(Ident.Arch) + "-" + Ident.Vendor + "-" + Ident.OS +
                    "-" + Ident.Environ)
                       .str();
  std::string Features;

  bool EnableXNACK = false;
  bool EnableSRAMECC = false;

  for (auto &Feature : Ident.Features)
    if (Feature == "xnack")
      EnableXNACK = true;
    else if (Feature == "sram-ecc")
      EnableSRAMECC = true;
    else
      return AMD_COMGR_STATUS_ERROR;

  SmallVector<StringRef, 2> FeaturesVec;
  FeaturesVec.push_back(EnableXNACK ? "+xnack" : "-xnack");
  FeaturesVec.push_back(EnableSRAMECC ? "+sram-ecc" : "-sram-ecc");

  Features = join(FeaturesVec, ",");

  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
  if (!TheTarget)
    return AMD_COMGR_STATUS_ERROR;

  std::unique_ptr<const MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
  if (!MRI)
    return AMD_COMGR_STATUS_ERROR;

  llvm::MCTargetOptions MCOptions;
  std::unique_ptr<const MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TT, MCOptions));
  if (!MAI)
    return AMD_COMGR_STATUS_ERROR;

  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII)
    return AMD_COMGR_STATUS_ERROR;

  std::unique_ptr<const MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TT, Ident.Processor, Features));
  if (!STI)
    return AMD_COMGR_STATUS_ERROR;

  std::unique_ptr<MCContext> Ctx(new (std::nothrow)
                                     MCContext(MAI.get(), MRI.get(), nullptr));
  if (!Ctx)
    return AMD_COMGR_STATUS_ERROR;

  std::unique_ptr<const MCDisassembler> DisAsm(
      TheTarget->createMCDisassembler(*STI, *Ctx));
  if (!DisAsm)
    return AMD_COMGR_STATUS_ERROR;

  // Optional; currently AMDGPU does not implement this.
  std::unique_ptr<const MCInstrAnalysis> MIA(
      TheTarget->createMCInstrAnalysis(MII.get()));

  std::unique_ptr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(
      Triple(TT), MAI->getAssemblerDialect(), *MAI, *MII, *MRI));
  if (!IP)
    return AMD_COMGR_STATUS_ERROR;

  DisassemblyInfo *DI = new (std::nothrow) DisassemblyInfo(
      ReadMemory, PrintInstruction, PrintAddressAnnotation, TheTarget,
      std::move(MAI), std::move(MRI), std::move(STI), std::move(MII),
      std::move(Ctx), std::move(DisAsm), std::move(MIA), std::move(IP));
  if (!DI)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  *DisassemblyInfoT = DisassemblyInfo::convert(DI);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t DisassemblyInfo::disassembleInstruction(uint64_t Address,
                                                           void *UserData,
                                                           uint64_t &Size) {
  uint64_t ReadSize = MAI->getMaxInstLength();
  SmallVector<uint8_t, 16> Buffer(ReadSize);

  uint64_t ActualSize = ReadMemory(
      Address, reinterpret_cast<char *>(Buffer.data()), ReadSize, UserData);
  if (!ActualSize || ActualSize > ReadSize)
    return AMD_COMGR_STATUS_ERROR;

  Buffer.resize(ActualSize);

  MCInst Inst;
  std::string Annotations;
  raw_string_ostream AnnotationsStream(Annotations);
  if (DisAsm->getInstruction(Inst, Size, Buffer, Address, nulls(),
                             AnnotationsStream) != MCDisassembler::Success)
    return AMD_COMGR_STATUS_ERROR;

  std::string InstStr;
  raw_string_ostream InstStream(InstStr);
  IP->printInst(&Inst, InstStream, AnnotationsStream.str(), *STI);

  PrintInstruction(InstStream.str().c_str(), UserData);

  if (MIA && (MIA->isCall(Inst) || MIA->isUnconditionalBranch(Inst) ||
              MIA->isConditionalBranch(Inst))) {
    uint64_t Target;
    if (MIA->evaluateBranch(Inst, Address, Size, Target))
      PrintAddressAnnotation(Target, UserData);
  }

  return AMD_COMGR_STATUS_SUCCESS;
}
