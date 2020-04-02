include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/AMDGPU
  ${LLVM_BINARY_DIR}/lib/Target/AMDGPU
  )

set(LLVM_LINK_COMPONENTS
  AMDGPUCodeGen
  AMDGPUDesc
  AMDGPUInfo
  MC
  Support
  )

add_llvm_target_unittest(AMDGPUTests
  DwarfRegMappings.cpp
  )
