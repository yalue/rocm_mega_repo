target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"
target triple = "amdgcn-amd-amdhsa"

;;;
;;; TODO add synchscope(N)
;;;

;;;;; Load
define protected i32 @__llvm_ld_atomic_a1_x_dev_i32(i32 addrspace(1)* nocapture readonly) #0 {
  %2 = load atomic volatile i32, i32 addrspace(1)* %0 monotonic, align 4
  ret i32 %2
}

define protected i64 @__llvm_ld_atomic_a1_x_dev_i64(i64 addrspace(1)* nocapture readonly) #0 {
  %2 = load atomic volatile i64, i64 addrspace(1)* %0 monotonic, align 8
  ret i64 %2
}

define protected i32 @__llvm_ld_atomic_a3_x_wg_i32(i32 addrspace(3)* nocapture readonly) #0 {
  %2 = load atomic volatile i32, i32 addrspace(3)* %0 monotonic, align 4
  ret i32 %2
}

define protected i64 @__llvm_ld_atomic_a3_x_wg_i64(i64 addrspace(3)* nocapture readonly) #0 {
  %2 = load atomic volatile i64, i64 addrspace(3)* %0 monotonic, align 8
  ret i64 %2
}

;;;;; Store
define protected void @__llvm_st_atomic_a1_x_dev_i32(i32 addrspace(1)* nocapture, i32) #1 {
  store atomic volatile i32 %1, i32 addrspace(1)* %0 monotonic, align 4
  ret void
}

define protected void @__llvm_st_atomic_a1_x_dev_i64(i64 addrspace(1)* nocapture, i64) #1 {
  store atomic volatile i64 %1, i64 addrspace(1)* %0 monotonic, align 8
  ret void
}

define protected void @__llvm_st_atomic_a3_x_wg_i32(i32 addrspace(3)* nocapture, i32) #1 {
  store atomic volatile i32 %1, i32 addrspace(3)* %0 monotonic, align 4
  ret void
}

define protected void @__llvm_st_atomic_a3_x_wg_i64(i64 addrspace(3)* nocapture, i64) #1 {
  store atomic volatile i64 %1, i64 addrspace(3)* %0 monotonic, align 8
  ret void
}

;;;;; Add
define protected i32 @__llvm_atomic_add_a1_x_dev_i32(i32 addrspace(1)* nocapture, i32) #1 {
  %3 = atomicrmw volatile add i32 addrspace(1)* %0, i32 %1 monotonic
  ret i32 %3
}

define protected i64 @__llvm_atomic_add_a1_x_dev_i64(i64 addrspace(1)* nocapture, i64) #1 {
  %3 = atomicrmw volatile add i64 addrspace(1)* %0, i64 %1 monotonic
  ret i64 %3
}

define protected i32 @__llvm_atomic_add_a3_x_wg_i32(i32 addrspace(3)* nocapture, i32) #1 {
  %3 = atomicrmw volatile add i32 addrspace(3)* %0, i32 %1 monotonic
  ret i32 %3
}

define protected i64 @__llvm_atomic_add_a3_x_wg_i64(i64 addrspace(3)* nocapture, i64) #1 {
  %3 = atomicrmw volatile add i64 addrspace(3)* %0, i64 %1 monotonic
  ret i64 %3
}

;;;;; And
define protected i32 @__llvm_atomic_and_a1_x_dev_i32(i32 addrspace(1)* nocapture, i32) #1 {
  %3 = atomicrmw volatile and i32 addrspace(1)* %0, i32 %1 monotonic
  ret i32 %3
}

define protected i64 @__llvm_atomic_and_a1_x_dev_i64(i64 addrspace(1)* nocapture, i64) #1 {
  %3 = atomicrmw volatile and i64 addrspace(1)* %0, i64 %1 monotonic
  ret i64 %3
}

define protected i32 @__llvm_atomic_and_a3_x_wg_i32(i32 addrspace(3)* nocapture, i32) #1 {
  %3 = atomicrmw volatile and i32 addrspace(3)* %0, i32 %1 monotonic
  ret i32 %3
}

define protected i64 @__llvm_atomic_and_a3_x_wg_i64(i64 addrspace(3)* nocapture, i64) #1 {
  %3 = atomicrmw volatile and i64 addrspace(3)* %0, i64 %1 monotonic
  ret i64 %3
}

;;;;; Or
define protected i32 @__llvm_atomic_or_a1_x_dev_i32(i32 addrspace(1)* nocapture, i32) #1 {
  %3 = atomicrmw volatile or i32 addrspace(1)* %0, i32 %1 monotonic
  ret i32 %3
}

define protected i64 @__llvm_atomic_or_a1_x_dev_i64(i64 addrspace(1)* nocapture, i64) #1 {
  %3 = atomicrmw volatile or i64 addrspace(1)* %0, i64 %1 monotonic
  ret i64 %3
}

define protected i32 @__llvm_atomic_or_a3_x_wg_i32(i32 addrspace(3)* nocapture, i32) #1 {
  %3 = atomicrmw volatile or i32 addrspace(3)* %0, i32 %1 monotonic
  ret i32 %3
}

define protected i64 @__llvm_atomic_or_a3_x_wg_i64(i64 addrspace(3)* nocapture, i64) #1 {
  %3 = atomicrmw volatile or i64 addrspace(3)* %0, i64 %1 monotonic
  ret i64 %3
}

;;;;; Max
define protected i32 @__llvm_atomic_max_a1_x_dev_i32(i32 addrspace(1)* nocapture, i32) #0 {
  %3 = atomicrmw volatile max i32 addrspace(1)* %0, i32 %1 monotonic
  ret i32 %3
}

define protected i32 @__llvm_atomic_umax_a1_x_dev_i32(i32 addrspace(1)* nocapture, i32) #1 {
  %3 = atomicrmw volatile umax i32 addrspace(1)* %0, i32 %1 monotonic
  ret i32 %3
}

define protected i64 @__llvm_atomic_max_a1_x_dev_i64(i64 addrspace(1)* nocapture, i64) #1 {
  %3 = atomicrmw volatile max i64 addrspace(1)* %0, i64 %1 monotonic
  ret i64 %3
}

define protected i64 @__llvm_atomic_umax_a1_x_dev_i64(i64 addrspace(1)* nocapture, i64) #1 {
  %3 = atomicrmw volatile umax i64 addrspace(1)* %0, i64 %1 monotonic
  ret i64 %3
}

define protected i32 @__llvm_atomic_max_a3_x_wg_i32(i32 addrspace(3)* nocapture, i32) #1 {
  %3 = atomicrmw volatile max i32 addrspace(3)* %0, i32 %1 monotonic
  ret i32 %3
}

define protected i32 @__llvm_atomic_umax_a3_x_wg_i32(i32 addrspace(3)* nocapture, i32) #1 {
  %3 = atomicrmw volatile umax i32 addrspace(3)* %0, i32 %1 monotonic
  ret i32 %3
}

define protected i64 @__llvm_atomic_max_a3_x_wg_i64(i64 addrspace(3)* nocapture, i64) #1 {
  %3 = atomicrmw volatile max i64 addrspace(3)* %0, i64 %1 monotonic
  ret i64 %3
}

define protected i64 @__llvm_atomic_umax_a3_x_wg_i64(i64 addrspace(3)* nocapture, i64) #1 {
  %3 = atomicrmw volatile umax i64 addrspace(3)* %0, i64 %1 monotonic
  ret i64 %3
}

;;;;; Min
define protected i32 @__llvm_atomic_min_a1_x_dev_i32(i32 addrspace(1)* nocapture, i32) #1 {
  %3 = atomicrmw volatile min i32 addrspace(1)* %0, i32 %1 monotonic
  ret i32 %3
}

define protected i32 @__llvm_atomic_umin_a1_x_dev_i32(i32 addrspace(1)* nocapture, i32) #1 {
  %3 = atomicrmw volatile umin i32 addrspace(1)* %0, i32 %1 monotonic
  ret i32 %3
}

define protected i64 @__llvm_atomic_min_a1_x_dev_i64(i64 addrspace(1)* nocapture, i64) #1 {
  %3 = atomicrmw volatile min i64 addrspace(1)* %0, i64 %1 monotonic
  ret i64 %3
}

define protected i64 @__llvm_atomic_umin_a1_x_dev_i64(i64 addrspace(1)* nocapture, i64) #1 {
  %3 = atomicrmw volatile umin i64 addrspace(1)* %0, i64 %1 monotonic
  ret i64 %3
}

define protected i32 @__llvm_atomic_min_a3_x_wg_i32(i32 addrspace(3)* nocapture, i32) #1 {
  %3 = atomicrmw volatile min i32 addrspace(3)* %0, i32 %1 monotonic
  ret i32 %3
}

define protected i32 @__llvm_atomic_umin_a3_x_wg_i32(i32 addrspace(3)* nocapture, i32) #1 {
  %3 = atomicrmw volatile umin i32 addrspace(3)* %0, i32 %1 monotonic
  ret i32 %3
}

define protected i64 @__llvm_atomic_min_a3_x_wg_i64(i64 addrspace(3)* nocapture, i64) #1 {
  %3 = atomicrmw volatile min i64 addrspace(3)* %0, i64 %1 monotonic
  ret i64 %3
}

define protected i64 @__llvm_atomic_umin_a3_x_wg_i64(i64 addrspace(3)* nocapture, i64) #1 {
  %3 = atomicrmw volatile umin i64 addrspace(3)* %0, i64 %1 monotonic
  ret i64 %3
}

;;;;; cmpxchg
define protected i32 @__llvm_cmpxchg_a1_x_x_dev_i32(i32 addrspace(1)* nocapture, i32, i32) #0 {
  %4 = cmpxchg volatile i32 addrspace(1)* %0, i32 %1, i32 %2 monotonic monotonic
  %5 = extractvalue { i32, i1 } %4, 0
  ret i32 %5
}

define protected i64 @__llvm_cmpxchg_a1_x_x_dev_i64(i64 addrspace(1)* nocapture, i64, i64) #1 {
  %4 = cmpxchg volatile i64 addrspace(1)* %0, i64 %1, i64 %2 monotonic monotonic
  %5 = extractvalue { i64, i1 } %4, 0
  ret i64 %5
}

define protected i32 @__llvm_cmpxchg_a3_x_x_wg_i32(i32 addrspace(3)* nocapture, i32, i32) #1 {
  %4 = cmpxchg volatile i32 addrspace(3)* %0, i32 %1, i32 %2 monotonic monotonic
  %5 = extractvalue { i32, i1 } %4, 0
  ret i32 %5
}

define protected i64 @__llvm_cmpxchg_a3_x_x_wg(i64 addrspace(3)* nocapture, i64, i64) #1 {
  %4 = cmpxchg volatile i64 addrspace(3)* %0, i64 %1, i64 %2 monotonic monotonic
  %5 = extractvalue { i64, i1 } %4, 0
  ret i64 %5
}

attributes #0 = { alwaysinline argmemonly norecurse nounwind readonly }
attributes #1 = { alwaysinline argmemonly norecurse nounwind }
