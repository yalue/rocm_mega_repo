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
// generic-to-global and global-to-generic) for the formal arguments of pointer
// type of a kernel (i.e. pfe trampoline or HIP __global__ function). This
// transformation is valid due to the invariants established by both HC and HIP
// in accordance with an address passed to a kernel can only reside in the
// global address space. It is preferable to execute SelectAcceleratorCode
// before, as this reduces the workload by pruning functions that are not
// reachable by an accelerator. It is mandatory to run InferAddressSpaces after,
// otherwise no benefit shall be obtained (the spurious casts do get removed).
//===----------------------------------------------------------------------===//
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {
class PromotePointerKernArgsToGlobal : public FunctionPass {
    // TODO: query the address space robustly.
    static constexpr unsigned int GenericAddrSpace{0u};
    static constexpr unsigned int GlobalAddrSpace{1u};

    void createPromotableCast(IRBuilder<>& Builder, Value *From, Value *To) {
        From->replaceAllUsesWith(To);

        Value *FToG = Builder.CreateAddrSpaceCast(
            From,
            cast<PointerType>(From->getType())
                ->getElementType()->getPointerTo(GlobalAddrSpace));
        Value *GToF = Builder.CreateAddrSpaceCast(FToG, From->getType());

        To->replaceAllUsesWith(GToF);
    }

    void maybePromoteUse(IRBuilder<>& Builder, Instruction *UI) {
        if (!UI)
            return;

        Builder.SetInsertPoint(UI->getNextNonDebugInstruction());

        Value *Tmp = Builder.CreateBitCast(UndefValue::get(UI->getType()),
                                           UI->getType());
        createPromotableCast(Builder, UI, Tmp);
    }

    void promoteArgument(IRBuilder<>& Builder, Argument *Arg) {
        Value *Tmp = Builder.CreateBitCast(UndefValue::get(Arg->getType()),
                                           Arg->getType());
        createPromotableCast(Builder, Arg, Tmp);
    }
public:
    static char ID;
    PromotePointerKernArgsToGlobal() : FunctionPass{ID} {}

    bool runOnFunction(Function &F) override
    {
        if (F.getCallingConv() != CallingConv::AMDGPU_KERNEL)
            return false;

        SmallVector<Argument *, 8> PromotableArgs;
        SmallVector<User *, 8> PromotableUses;
        for (auto &&Arg : F.args()) {
            for (auto &&U : Arg.users()) {
                if (!U->getType()->isPointerTy())
                    continue;
                if (U->getType()->getPointerAddressSpace() != GenericAddrSpace)
                    continue;

                PromotableUses.push_back(U);
            }

            if (!Arg.getType()->isPointerTy())
                continue;
            if (Arg.getType()->getPointerAddressSpace() != GenericAddrSpace)
                continue;

            PromotableArgs.push_back(&Arg);
        }

        if (PromotableArgs.empty() && PromotableUses.empty())
            return false;

        static IRBuilder<> Builder{F.getContext()};
        for (auto &&PU : PromotableUses)
            maybePromoteUse(Builder, dyn_cast<Instruction>(PU));

        Builder.SetInsertPoint(&F.getEntryBlock().front());
        for (auto &&Arg : PromotableArgs)
            promoteArgument(Builder, Arg);
        return true;
    }
};
char PromotePointerKernArgsToGlobal::ID = 0;

static RegisterPass<PromotePointerKernArgsToGlobal> X{
    "promote-pointer-kernargs-to-global",
    "Promotes kernel formals of pointer type to point to the global address "
    "space, since the actuals can only represent a global address.",
    false,
    false};
}