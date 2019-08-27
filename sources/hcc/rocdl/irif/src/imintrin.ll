target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_load_1d_v4f32_i32(i32 %arg1, <8 x i32> %arg2) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 15, i32 %arg1, <8 x i32> %arg2, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_load_2d_v4f32_i32(i32 %arg1, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 15, i32 %arg1, i32 %arg2, <8 x i32> %arg3, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_load_3d_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.3d.v4f32.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.3d.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_load_cube_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.cube.v4f32.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.cube.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_load_1darray_v4f32_i32(i32 %arg1, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.1darray.v4f32.i32(i32 15, i32 %arg1, i32 %arg2, <8 x i32> %arg3, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.1darray.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_load_2darray_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.2darray.v4f32.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.2darray.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.2dmsaa.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.2darraymsaa.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_load_mip_1d_v4f32_i32(i32 %arg1, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.mip.1d.v4f32.i32(i32 15, i32 %arg1, i32 %arg2, <8 x i32> %arg3, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.mip.1d.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_load_mip_2d_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_load_mip_3d_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.mip.3d.v4f32.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.mip.3d.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_load_mip_cube_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.mip.cube.v4f32.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.mip.cube.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_load_mip_1darray_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.mip.1darray.v4f32.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.mip.1darray.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_load_mip_2darray_v4f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.load.mip.2darray.v4f32.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.mip.2darray.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_load_1d_v4f16_i32(i32 %arg1, <8 x i32> %arg2) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.1d.v4f16.i32(i32 15, i32 %arg1, <8 x i32> %arg2, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.load.1d.v4f16.i32(i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_load_2d_v4f16_i32(i32 %arg1, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.2d.v4f16.i32(i32 15, i32 %arg1, i32 %arg2, <8 x i32> %arg3, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.load.2d.v4f16.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_load_3d_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.3d.v4f16.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.load.3d.v4f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_load_cube_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.cube.v4f16.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.load.cube.v4f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_load_1darray_v4f16_i32(i32 %arg1, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.1darray.v4f16.i32(i32 15, i32 %arg1, i32 %arg2, <8 x i32> %arg3, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.load.1darray.v4f16.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_load_2darray_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.2darray.v4f16.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.load.2darray.v4f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.load.2dmsaa.v4f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.load.2darraymsaa.v4f16.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_load_mip_1d_v4f16_i32(i32 %arg1, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.mip.1d.v4f16.i32(i32 15, i32 %arg1, i32 %arg2, <8 x i32> %arg3, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.load.mip.1d.v4f16.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_load_mip_2d_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.mip.2d.v4f16.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.load.mip.2d.v4f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_load_mip_3d_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.mip.3d.v4f16.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.load.mip.3d.v4f16.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_load_mip_cube_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.mip.cube.v4f16.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.load.mip.cube.v4f16.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_load_mip_1darray_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.mip.1darray.v4f16.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.load.mip.1darray.v4f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_load_mip_2darray_v4f16_i32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.load.mip.2darray.v4f16.i32(i32 15, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.load.mip.2darray.v4f16.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.load.1d.f32.i32(i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected float @__llvm_amdgcn_image_load_2d_f32_i32(i32 %arg1, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.load.2d.f32.i32(i32 1, i32 %arg1, i32 %arg2, <8 x i32> %arg3, i32 0, i32 0) #1
  ret float %tmp
}

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.load.2d.f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.load.3d.f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.load.cube.f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.load.1darray.f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected float @__llvm_amdgcn_image_load_2darray_f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.load.2darray.f32.i32(i32 1, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #1
  ret float %tmp
}

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.load.2darray.f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.load.2darraymsaa.f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.load.mip.1d.f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected float @__llvm_amdgcn_image_load_mip_2d_f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.load.mip.2d.f32.i32(i32 1, i32 %arg1, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #1
  ret float %tmp
}

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.load.mip.2d.f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.load.mip.3d.f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.load.mip.cube.f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.load.mip.1darray.f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected float @__llvm_amdgcn_image_load_mip_2darray_f32_i32(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.load.mip.2darray.f32.i32(i32 1, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #1
  ret float %tmp
}

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.load.mip.2darray.f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_1d_v4f32_i32(<4 x float> %arg, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %arg, i32 15, i32 %arg2, <8 x i32> %arg3, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float>, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_2d_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float> %arg, i32 15, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_3d_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.3d.v4f32.i32(<4 x float> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.3d.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_cube_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.cube.v4f32.i32(<4 x float> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.cube.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_1darray_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.1darray.v4f32.i32(<4 x float> %arg, i32 15, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.1darray.v4f32.i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_2darray_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.2darray.v4f32.i32(<4 x float> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.2darray.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.2dmsaa.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.2darraymsaa.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_mip_1d_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.1d.v4f32.i32(<4 x float> %arg, i32 15, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.1d.v4f32.i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_mip_2d_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.2d.v4f32.i32(<4 x float> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.2d.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_mip_3d_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.3d.v4f32.i32(<4 x float> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.3d.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_mip_cube_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.cube.v4f32.i32(<4 x float> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.cube.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_mip_1darray_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.1darray.v4f32.i32(<4 x float> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.1darray.v4f32.i32(<4 x float>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_mip_2darray_v4f32_i32(<4 x float> %arg, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.2darray.v4f32.i32(<4 x float> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.2darray.v4f32.i32(<4 x float>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_1d_v4f16_i32(<4 x half> %arg, i32 %arg2, <8 x i32> %arg3) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.1d.v4f16.i32(<4 x half> %arg, i32 15, i32 %arg2, <8 x i32> %arg3, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.1d.v4f16.i32(<4 x half>, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_2d_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.2d.v4f16.i32(<4 x half> %arg, i32 15, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.2d.v4f16.i32(<4 x half>, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_3d_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.3d.v4f16.i32(<4 x half> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.3d.v4f16.i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_cube_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.cube.v4f16.i32(<4 x half> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.cube.v4f16.i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_1darray_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.1darray.v4f16.i32(<4 x half> %arg, i32 15, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.1darray.v4f16.i32(<4 x half>, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_2darray_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.2darray.v4f16.i32(<4 x half> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.2darray.v4f16.i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.2dmsaa.v4f16.i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.2darraymsaa.v4f16.i32(<4 x half>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_mip_1d_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.1d.v4f16.i32(<4 x half> %arg, i32 15, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.1d.v4f16.i32(<4 x half>, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_mip_2d_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.2d.v4f16.i32(<4 x half> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.2d.v4f16.i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_mip_3d_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.3d.v4f16.i32(<4 x half> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.3d.v4f16.i32(<4 x half>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_mip_cube_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.cube.v4f16.i32(<4 x half> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.cube.v4f16.i32(<4 x half>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_mip_1darray_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.1darray.v4f16.i32(<4 x half> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.1darray.v4f16.i32(<4 x half>, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_mip_2darray_v4f16_i32(<4 x half> %arg, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.2darray.v4f16.i32(<4 x half> %arg, i32 15, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.2darray.v4f16.i32(<4 x half>, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.1d.f32.i32(float, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_2d_f32_i32(float %arg, i32 %arg2, i32 %arg3, <8 x i32> %arg4) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.2d.f32.i32(float %arg, i32 15, i32 %arg2, i32 %arg3, <8 x i32> %arg4, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.2d.f32.i32(float, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.3d.f32.i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.cube.f32.i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.1darray.f32.i32(float, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_2darray_f32_i32(float %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.2darray.f32.i32(float %arg, i32 1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.2darray.f32.i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.2dmsaa.f32.i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.2darraymsaa.f32.i32(float, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

declare void @llvm.amdgcn.image.store.mip.1d.f32.i32(float, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_mip_2d_f32_i32(float %arg, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.2d.f32.i32(float %arg, i32 1, i32 %arg2, i32 %arg3, i32 %arg4, <8 x i32> %arg5, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.2d.f32.i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.3d.f32.i32(float, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.cube.f32.i32(float, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.1darray.f32.i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind writeonly
define protected void @__llvm_amdgcn_image_store_mip_2darray_f32_i32(float %arg, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6) local_unnamed_addr #2 {
bb:
  tail call void @llvm.amdgcn.image.store.mip.2darray.f32.i32(float %arg, i32 1, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, <8 x i32> %arg6, i32 0, i32 0) #3
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.mip.2darray.f32.i32(float, i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #3

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_lz_1d_v4f32_f32(float %arg1, <8 x i32> %arg2, <4 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.lz.1d.v4f32.f32(i32 15, float %arg1, <8 x i32> %arg2, <4 x i32> %arg3, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.lz.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_l_1d_v4f32_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.l.1d.v4f32.f32(i32 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.l.1d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_d_1d_v4f32_f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.d.1d.v4f32.f32.f32(i32 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.d.1d.v4f32.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_lz_2d_v4f32_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_l_2d_v4f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.l.2d.v4f32.f32(i32 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.l.2d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_d_2d_v4f32_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, <8 x i32> %arg7, <4 x i32> %arg8) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f32.f32(i32 15, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, <8 x i32> %arg7, <4 x i32> %arg8, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f32.f32(i32, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_lz_3d_v4f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.lz.3d.v4f32.f32(i32 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.lz.3d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_l_3d_v4f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.l.3d.v4f32.f32(i32 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.l.3d.v4f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_d_3d_v4f32_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, float %arg8, float %arg9, <8 x i32> %arg10, <4 x i32> %arg11) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.d.3d.v4f32.f32.f32(i32 15, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, float %arg8, float %arg9, <8 x i32> %arg10, <4 x i32> %arg11, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.d.3d.v4f32.f32.f32(i32, float, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_lz_cube_v4f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.lz.cube.v4f32.f32(i32 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.lz.cube.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_l_cube_v4f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.l.cube.v4f32.f32(i32 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.l.cube.v4f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.d.cube.v4f32.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_lz_1darray_v4f32_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.lz.1darray.v4f32.f32(i32 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.lz.1darray.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_l_1darray_v4f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.l.1darray.v4f32.f32(i32 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.l.1darray.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_d_1darray_v4f32_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.d.1darray.v4f32.f32.f32(i32 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.d.1darray.v4f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_lz_2darray_v4f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.lz.2darray.v4f32.f32(i32 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.lz.2darray.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_l_2darray_v4f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.l.2darray.v4f32.f32(i32 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.l.2darray.v4f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_sample_d_2darray_v4f32_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, <8 x i32> %arg8, <4 x i32> %arg9) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.sample.d.2darray.v4f32.f32.f32(i32 15, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, <8 x i32> %arg8, <4 x i32> %arg9, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.d.2darray.v4f32.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_lz_1d_v4f16_f32(float %arg1, <8 x i32> %arg2, <4 x i32> %arg3) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.lz.1d.v4f16.f32(i32 15, float %arg1, <8 x i32> %arg2, <4 x i32> %arg3, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.lz.1d.v4f16.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_l_1d_v4f16_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.l.1d.v4f16.f32(i32 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.l.1d.v4f16.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_d_1d_v4f16_f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.d.1d.v4f16.f32.f32(i32 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.d.1d.v4f16.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_lz_2d_v4f16_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.lz.2d.v4f16.f32(i32 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.lz.2d.v4f16.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_l_2d_v4f16_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.l.2d.v4f16.f32(i32 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.l.2d.v4f16.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_d_2d_v4f16_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, <8 x i32> %arg7, <4 x i32> %arg8) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.d.2d.v4f16.f32.f32(i32 15, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, <8 x i32> %arg7, <4 x i32> %arg8, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.d.2d.v4f16.f32.f32(i32, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_lz_3d_v4f16_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.lz.3d.v4f16.f32(i32 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.lz.3d.v4f16.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_l_3d_v4f16_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.l.3d.v4f16.f32(i32 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.l.3d.v4f16.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_d_3d_v4f16_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, float %arg8, float %arg9, <8 x i32> %arg10, <4 x i32> %arg11, i32 %arg13, i32 %arg14) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.d.3d.v4f16.f32.f32(i32 15, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, float %arg8, float %arg9, <8 x i32> %arg10, <4 x i32> %arg11, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.d.3d.v4f16.f32.f32(i32, float, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_lz_cube_v4f16_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.lz.cube.v4f16.f32(i32 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.lz.cube.v4f16.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_l_cube_v4f16_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.l.cube.v4f16.f32(i32 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.l.cube.v4f16.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.d.cube.v4f16.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_lz_1darray_v4f16_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.lz.1darray.v4f16.f32(i32 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.lz.1darray.v4f16.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_l_1darray_v4f16_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.l.1darray.v4f16.f32(i32 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.l.1darray.v4f16.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_d_1darray_v4f16_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.d.1darray.v4f16.f32.f32(i32 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.d.1darray.v4f16.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_lz_2darray_v4f16_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.lz.2darray.v4f16.f32(i32 15, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.lz.2darray.v4f16.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_l_2darray_v4f16_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.l.2darray.v4f16.f32(i32 15, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.l.2darray.v4f16.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x half> @__llvm_amdgcn_image_sample_d_2darray_v4f16_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, <8 x i32> %arg8, <4 x i32> %arg9) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x half> @llvm.amdgcn.image.sample.d.2darray.v4f16.f32.f32(i32 15, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, <8 x i32> %arg8, <4 x i32> %arg9, i1 false, i32 0, i32 0) #1
  ret <4 x half> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x half> @llvm.amdgcn.image.sample.d.2darray.v4f16.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.lz.1d.f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.l.1d.f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.d.1d.f32.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected float @__llvm_amdgcn_image_sample_lz_2d_f32_f32(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.sample.lz.2d.f32.f32(i32 1, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 false, i32 0, i32 0) #1
  ret float %tmp
}

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.lz.2d.f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected float @__llvm_amdgcn_image_sample_l_2d_f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.sample.l.2d.f32.f32(i32 1, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 false, i32 0, i32 0) #1
  ret float %tmp
}

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.l.2d.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected float @__llvm_amdgcn_image_sample_d_2d_f32_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, <8 x i32> %arg7, <4 x i32> %arg8) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.sample.d.2d.f32.f32.f32(i32 1, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, <8 x i32> %arg7, <4 x i32> %arg8, i1 false, i32 0, i32 0) #1
  ret float %tmp
}

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.d.2d.f32.f32.f32(i32, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.lz.3d.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.l.3d.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.d.3d.f32.f32.f32(i32, float, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.lz.cube.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.l.cube.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.d.cube.f32.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.lz.1darray.f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.l.1darray.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.d.1darray.f32.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected float @__llvm_amdgcn_image_sample_lz_2darray_f32_f32(float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.sample.lz.2darray.f32.f32(i32 1, float %arg1, float %arg2, float %arg3, <8 x i32> %arg4, <4 x i32> %arg5, i1 false, i32 0, i32 0) #1
  ret float %tmp
}

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.lz.2darray.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected float @__llvm_amdgcn_image_sample_l_2darray_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.sample.l.2darray.f32.f32(i32 1, float %arg1, float %arg2, float %arg3, float %arg4, <8 x i32> %arg5, <4 x i32> %arg6, i1 false, i32 0, i32 0) #1
  ret float %tmp
}

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.l.2darray.f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected float @__llvm_amdgcn_image_sample_d_2darray_f32_f32_f32(float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, <8 x i32> %arg8, <4 x i32> %arg9, i32 %arg11, i32 %arg12) local_unnamed_addr #0 {
bb:
  %tmp = tail call float @llvm.amdgcn.image.sample.d.2darray.f32.f32.f32(i32 1, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, <8 x i32> %arg8, <4 x i32> %arg9, i1 false, i32 0, i32 0) #1
  ret float %tmp
}

; Function Attrs: nounwind readonly
declare float @llvm.amdgcn.image.sample.d.2darray.f32.f32.f32(i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: alwaysinline nounwind readonly
define protected <4 x float> @__llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_r(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32(i32 1, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

define protected <4 x float> @__llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_g(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32(i32 2, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

define protected <4 x float> @__llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_b(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32(i32 4, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

define protected <4 x float> @__llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_a(float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) local_unnamed_addr #0 {
bb:
  %tmp = tail call <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32(i32 8, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 false, i32 0, i32 0) #1
  ret <4 x float> %tmp
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.image.amdgcn.gather4.l.2d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.gather4.lz.cube.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.gather4.l.cube.v4f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.gather4.lz.2darray.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.gather4.l.2darray.v4f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.gather.4h.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.image.amdgcn.gather.4h.cube.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.image.amdgcn.gather.4h.2darray.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

attributes #0 = { alwaysinline nounwind readonly }
attributes #1 = { nounwind readonly }
attributes #2 = { alwaysinline nounwind writeonly }
attributes #3 = { nounwind writeonly }
