//===--  PromotePointerKernargsToGlobal.cpp - Promote Pointers To Global --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares and defines a pass which uses the double-cast trick (
// flat-to-global and global-to-flat) for pointers that reside in the
// __constant__ address space. For example, given __constant__ int** foo, all
// single dereferences of foo will be promoted to yield a global int*, as
// opposed to a flat int*. It is preferable to execute SelectAcceleratorCode
// before, as this reduces the workload by pruning functions that are not
// reachable by an accelerator. It is mandatory to run InferAddressSpaces after,
// otherwise no benefit shall be obtained (the spurious casts do get removed).
//===----------------------------------------------------------------------===//
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {
class PromoteConstant : public ModulePass {
  // TODO: query the address spaces robustly.
  static constexpr unsigned int FlatAddrSpace{0u};
  static constexpr unsigned int GlobalAddrSpace{1u};
  static constexpr unsigned int ConstantAddrSpace{4u};

  // TODO: this should be hoisted to a common header with HC utility functions
  //       once the related work on PromotePointerKernArgsToGlobal gets merged
  void createPromotableCast(IRBuilder<>& Builder, Value *From, Value *To) {
    To->dump();
    From->replaceAllUsesWith(To);

    Value *FToG = Builder.CreateAddrSpaceCast(
      From,
      cast<PointerType>(
        From->getType())->getElementType()->getPointerTo(GlobalAddrSpace));
    Value *GToF = Builder.CreateAddrSpaceCast(FToG, From->getType());

    To->replaceAllUsesWith(GToF);
  }

  // TODO: this should be hoisted to a common header with HC utility functions
  //       once the related work on PromotePointerKernArgsToGlobal gets merged
  bool maybePromoteUse(IRBuilder<>& Builder, Instruction *UI) {
    if (!UI)
      return false;

    Builder.SetInsertPoint(UI->getNextNonDebugInstruction());

    // We cannot use IRBuilder since it might do the obvious folding, which
    // would yield an undef value of a possibly primitive type, which cannot be
    // disambiguated from other undefs of the same primitive type and would
    // cause havoc when replaced with the promotable cast created later.
    Value *UD = UndefValue::get(Builder.getInt8Ty());
    Value *Tmp = CastInst::CreateBitOrPointerCast(UD, UI->getType(), "Tmp",
                                                  &*Builder.GetInsertPoint());

    createPromotableCast(Builder, UI, Tmp);

    return true;
  }
  // TODO: Whilst ConstantExpr and Operator handling could obviously be folded
  //       into a single function, we leave them separate for now to allow
  //       possible additional development.
  bool maybeHandleInstruction(IRBuilder<>& Builder, Instruction *I) {
    if (!I)
      return false;

    if (!I->getType()->isPointerTy())
      return false;
    if (I->getType()->getPointerAddressSpace() != FlatAddrSpace)
      return false;

    if (auto LI = dyn_cast<LoadInst>(I))
      return maybePromoteUse(Builder, LI);
    if (auto PHI = dyn_cast<PHINode>(I)) {
      return false;
    }
    if (auto SEL = dyn_cast<SelectInst>(I))
      return false;

    return false;
  }

  bool maybeHandleOperator(IRBuilder<>& Builder, Operator *Op) {
    if (!Op)
      return false;
    if (!Op->getType()->isPointerTy())
      return false;

    bool Modified = false;
    for (auto &&U : Op->users()) {
      if (maybeHandleConstantExpr(Builder, dyn_cast<ConstantExpr>(U)))
        Modified = true;
      else if (maybeHandleOperator(Builder, dyn_cast<Operator>(U)))
        Modified = true;
      else if (maybeHandleInstruction(Builder, dyn_cast<Instruction>(U)))
        Modified = true;
    }

    return Modified;
  }

  bool maybeHandleConstantExpr(IRBuilder<>& Builder, ConstantExpr *CE) {
    if (!CE)
      return false;
    if (!CE->getType()->isPointerTy())
      return false;

    bool Modified = false;
    for (auto &&U : CE->users()) {
      if (maybeHandleConstantExpr(Builder, dyn_cast<ConstantExpr>(U)))
        Modified = true;
      else if (maybeHandleInstruction(Builder, dyn_cast<Instruction>(U)))
        Modified = true;
      else if (maybeHandleOperator(Builder, dyn_cast<Operator>(U)))
        Modified = true;
    }

    return Modified;
  }
public:
  static char ID;
  PromoteConstant() : ModulePass{ID} {}

  bool runOnModule(Module &M) override {
    SmallVector<GlobalVariable *, 8u> PromotableGlobals;
    for (auto &&GV : M.globals())
      if (GV.getAddressSpace() == ConstantAddrSpace)
        PromotableGlobals.push_back(&GV);

    if (PromotableGlobals.empty())
      return false;

    IRBuilder<> Builder(M.getContext());

    bool Modified = false;
    for (auto &&GV : PromotableGlobals) {
      for (auto &&U : GV->users()) {
        if (maybeHandleConstantExpr(Builder, dyn_cast<ConstantExpr>(U)))
          Modified = true;
        else if (maybeHandleInstruction(Builder, dyn_cast<Instruction>(U)))
          Modified = true;
        else if (maybeHandleOperator(Builder, dyn_cast<Operator>(U)))
          Modified = true;
      }
    }

    return Modified;
  }
};
char PromoteConstant::ID = 0;

static RegisterPass<PromoteConstant> X{
  "promote-constant",
  "Promotes users of variables annotated with __constant__ to refer to the "
  "global address space iff the user produces a flat pointer.",
  false,
  false};
}