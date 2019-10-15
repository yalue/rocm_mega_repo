/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "ockl.h"
#include "oclc.h"

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define RATTR __attribute__((pure))
#define WATTR
#define GATTR __attribute__((const))

// TSHARP/SSHARP access
#define FIELD(P,B,W) ((P[B >> 5] >> (B & 0x1f)) & ((1 << W) - 1))
#define WORD(P,I) P[I]
#define LOAD_TSHARP(I) *(__constant uint8 *)I
#define LOAD_VSHARP(I) *(__constant uint4 *)I
#define LOAD_SSHARP(S) *(__constant uint4 *)S

// Adjustments for hardware precision limits
#define ADJUST_X(C,I,S) do { \
    float _w = (float)WORD(I,10); \
    float _p = FIELD(S,15,1) ? 1.0f : _w; \
    float _x = __builtin_floorf(C * _p) * __builtin_amdgcn_rcpf(_p); \
    C = FIELD(S,84,1) ? C : _x; \
} while (0)

#define ADJUST_XY(C,I,S) do { \
    float _w = (float)WORD(I,10); \
    float _h = (float)(FIELD(I,78,14) + 1U); \
    bool _f = FIELD(S,15,1); \
    float _p = _f ? 1.0f : _w; \
    float _q = _f ? 1.0f : _h; \
    float _x = __builtin_floorf(C.x * _p) * __builtin_amdgcn_rcpf(_p); \
    float _y = __builtin_floorf(C.y * _q) * __builtin_amdgcn_rcpf(_q); \
    bool _m = FIELD(S,84,1); \
    C.x = _m ? C.x : _x; \
    C.y = _m ? C.y : _y; \
} while (0)

#define ADJUST_XYZ(C,I,S) do { \
    float _w = (float)WORD(I,10); \
    float _h = (float)(FIELD(I,78,14) + 1U); \
    float _d = (float)(FIELD(I, 128, 13) + 1U); \
    bool _f = FIELD(S,15,1); \
    float _p = _f ? 1.0f : _w; \
    float _q = _f ? 1.0f : _h; \
    float _r = _f ? 1.0f : _d; \
    float _x = __builtin_floorf(C.x * _p) * __builtin_amdgcn_rcpf(_p); \
    float _y = __builtin_floorf(C.y * _q) * __builtin_amdgcn_rcpf(_q); \
    float _z = __builtin_floorf(C.z * _r) * __builtin_amdgcn_rcpf(_r); \
    bool _m = FIELD(S,84,1); \
    C.x = _m ? C.x : _x; \
    C.y = _m ? C.y : _y; \
    C.z = _m ? C.z : _z; \
} while (0)

GATTR
static float fmuladd_f32(float a, float b, float c)
{
    #pragma OPENCL FP_CONTRACT ON
    return a * b + c;
}

#define LS_ARRAY_FACE(I,F) (6 * (((I) << 8) >> 8) + (F))
#define SAMPLE_ARRAY_FACE(I, F) fmuladd_f32(__builtin_rintf(I), 8.0f, F)

#define CUBE_PREP(C) do { \
    float _vx = C.x; \
    float _vy = C.y; \
    float _vz = C.z; \
    float _rl = __builtin_amdgcn_rcpf(__builtin_amdgcn_cubema(_vx, _vy, _vz)); \
    C.x = fmuladd_f32(__builtin_amdgcn_cubesc(_vx, _vy, _vz), _rl, 0.5f); \
    C.y = fmuladd_f32(__builtin_amdgcn_cubetc(_vx, _vy, _vz), _rl, 0.5f); \
    C.z = __builtin_amdgcn_cubeid(_vx, _vy, _vz); \
} while (0)

RATTR float4
OCKL_MANGLE_T(image_load,1D)(TSHARP i, int c)
{
    return __llvm_amdgcn_image_load_1d_v4f32_i32(c, LOAD_TSHARP(i));
}

RATTR float4
OCKL_MANGLE_T(image_load,1Da)(TSHARP i, int2 c)
{
    return __llvm_amdgcn_image_load_1darray_v4f32_i32(c.x, c.y, LOAD_TSHARP(i));
}

RATTR float4
OCKL_MANGLE_T(image_load,1Db)(TSHARP i, int c)
{
    return __llvm_amdgcn_buffer_load_format_v4f32(LOAD_VSHARP(i), c, 0, false, false);
}

RATTR float4
OCKL_MANGLE_T(image_load,2D)(TSHARP i, int2 c)
{
    return __llvm_amdgcn_image_load_2d_v4f32_i32(c.x, c.y, LOAD_TSHARP(i));
}

RATTR float4
OCKL_MANGLE_T(image_load,2Da)(TSHARP i, int4 c)
{
    return __llvm_amdgcn_image_load_2darray_v4f32_i32(c.x, c.y, c.z, LOAD_TSHARP(i));
}

RATTR float
OCKL_MANGLE_T(image_load,2Dad)(TSHARP i, int4 c)
{
    return __llvm_amdgcn_image_load_2darray_f32_i32(c.x, c.y, c.z, LOAD_TSHARP(i));
}

RATTR float
OCKL_MANGLE_T(image_load,2Dd)(TSHARP i, int2 c)
{
    return __llvm_amdgcn_image_load_2d_f32_i32(c.x, c.y, LOAD_TSHARP(i));
}

RATTR float4
OCKL_MANGLE_T(image_load,3D)(TSHARP i, int4 c)
{
    return __llvm_amdgcn_image_load_3d_v4f32_i32(c.x, c.y, c.z, LOAD_TSHARP(i));
}

RATTR float4
OCKL_MANGLE_T(image_load,CM)(TSHARP i, int2 c, int f)
{
    return __llvm_amdgcn_image_load_cube_v4f32_i32(c.x, c.y, f, LOAD_TSHARP(i));
}

RATTR float4
OCKL_MANGLE_T(image_load,CMa)(TSHARP i, int4 c, int f)
{
    f = LS_ARRAY_FACE(c.z, f);
    return __llvm_amdgcn_image_load_cube_v4f32_i32(c.x, c.y, f, LOAD_TSHARP(i));
}

RATTR float4
OCKL_MANGLE_T(image_load_lod,1D)(TSHARP i, int c, int l)
{
    return __llvm_amdgcn_image_load_mip_1d_v4f32_i32(c, l, LOAD_TSHARP(i));
}

RATTR float4
OCKL_MANGLE_T(image_load_lod,1Da)(TSHARP i, int2 c, int l)
{
    return __llvm_amdgcn_image_load_mip_1darray_v4f32_i32(c.x, c.y, l, LOAD_TSHARP(i));
}

RATTR float4
OCKL_MANGLE_T(image_load_lod,2D)(TSHARP i, int2 c, int l)
{
    return __llvm_amdgcn_image_load_mip_2d_v4f32_i32(c.x, c.y, l, LOAD_TSHARP(i));
}

RATTR float4
OCKL_MANGLE_T(image_load_lod,2Da)(TSHARP i, int4 c, int l)
{
    return __llvm_amdgcn_image_load_mip_2darray_v4f32_i32(c.x, c.y, c.z, l, LOAD_TSHARP(i));
}

RATTR float
OCKL_MANGLE_T(image_load_lod,2Dad)(TSHARP i, int4 c, int l)
{
    return __llvm_amdgcn_image_load_mip_2darray_f32_i32(c.x, c.y, c.z, l, LOAD_TSHARP(i));
}

RATTR float
OCKL_MANGLE_T(image_load_lod,2Dd)(TSHARP i, int2 c, int l)
{
    return __llvm_amdgcn_image_load_mip_2d_f32_i32(c.x, c.y, l, LOAD_TSHARP(i));
}

RATTR float4
OCKL_MANGLE_T(image_load_lod,3D)(TSHARP i, int4 c, int l)
{
    return __llvm_amdgcn_image_load_mip_3d_v4f32_i32(c.x, c.y, c.z, l, LOAD_TSHARP(i));
}

RATTR float4
OCKL_MANGLE_T(image_load_lod,CM)(TSHARP i, int2 c, int f, int l)
{
    return __llvm_amdgcn_image_load_mip_cube_v4f32_i32(c.x, c.y, f, l, LOAD_TSHARP(i));
}

RATTR float4
OCKL_MANGLE_T(image_load_lod,CMa)(TSHARP i, int4 c, int f, int l)
{
    f = LS_ARRAY_FACE(c.z, f);
    return __llvm_amdgcn_image_load_mip_cube_v4f32_i32(c.x, c.y, f, l, LOAD_TSHARP(i));
}

RATTR half4
OCKL_MANGLE_T(image_loadh,1D)(TSHARP i, int c)
{
    return __llvm_amdgcn_image_load_1d_v4f16_i32(c, LOAD_TSHARP(i));
}

RATTR half4
OCKL_MANGLE_T(image_loadh,1Da)(TSHARP i, int2 c)
{
    return __llvm_amdgcn_image_load_1darray_v4f16_i32(c.x, c.y, LOAD_TSHARP(i));
}

RATTR half4
OCKL_MANGLE_T(image_loadh,1Db)(TSHARP i, int c)
{
    return __llvm_amdgcn_buffer_load_format_v4f16(LOAD_VSHARP(i), c, 0, false, false);
}

RATTR half4
OCKL_MANGLE_T(image_loadh,2D)(TSHARP i, int2 c)
{
    return __llvm_amdgcn_image_load_2d_v4f16_i32(c.x, c.y, LOAD_TSHARP(i));
}

RATTR half4
OCKL_MANGLE_T(image_loadh,2Da)(TSHARP i, int4 c)
{
    return __llvm_amdgcn_image_load_2darray_v4f16_i32(c.x, c.y, c.z, LOAD_TSHARP(i));
}

RATTR half4
OCKL_MANGLE_T(image_loadh,3D)(TSHARP i, int4 c)
{
    return __llvm_amdgcn_image_load_3d_v4f16_i32(c.x, c.y, c.z, LOAD_TSHARP(i));
}

RATTR half4
OCKL_MANGLE_T(image_loadh,CM)(TSHARP i, int2 c, int f)
{
    return __llvm_amdgcn_image_load_cube_v4f16_i32(c.x, c.y, f, LOAD_TSHARP(i));
}

RATTR half4
OCKL_MANGLE_T(image_loadh,CMa)(TSHARP i, int4 c, int f)
{
    f = LS_ARRAY_FACE(c.z, f);
    return __llvm_amdgcn_image_load_cube_v4f16_i32(c.x, c.y, f, LOAD_TSHARP(i));
}

RATTR half4
OCKL_MANGLE_T(image_loadh_lod,1D)(TSHARP i, int c, int l)
{
    return __llvm_amdgcn_image_load_mip_1d_v4f16_i32(c, l, LOAD_TSHARP(i));
}

RATTR half4
OCKL_MANGLE_T(image_loadh_lod,1Da)(TSHARP i, int2 c, int l)
{
    return __llvm_amdgcn_image_load_mip_1darray_v4f16_i32(c.x, c.y, l, LOAD_TSHARP(i));
}

RATTR half4
OCKL_MANGLE_T(image_loadh_lod,2D)(TSHARP i, int2 c, int l)
{
    return __llvm_amdgcn_image_load_mip_2d_v4f16_i32(c.x, c.y, l, LOAD_TSHARP(i));
}

RATTR half4
OCKL_MANGLE_T(image_loadh_lod,2Da)(TSHARP i, int4 c, int l)
{
    return __llvm_amdgcn_image_load_mip_2darray_v4f16_i32(c.x, c.y, c.z, l, LOAD_TSHARP(i));
}

RATTR half4
OCKL_MANGLE_T(image_loadh_lod,3D)(TSHARP i, int4 c, int l)
{
    return __llvm_amdgcn_image_load_mip_3d_v4f16_i32(c.x, c.y, c.z, l, LOAD_TSHARP(i));
}

RATTR half4
OCKL_MANGLE_T(image_loadh_lod,CM)(TSHARP i, int2 c, int f, int l)
{
    return __llvm_amdgcn_image_load_mip_cube_v4f16_i32(c.x, c.y, f, l, LOAD_TSHARP(i));
}

RATTR half4
OCKL_MANGLE_T(image_loadh_lod,CMa)(TSHARP i, int4 c, int f, int l)
{
    f = LS_ARRAY_FACE(c.z, f);
    return __llvm_amdgcn_image_load_mip_cube_v4f16_i32(c.x, c.y, f, l, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store,1D)(TSHARP i, int c, float4 p)
{
    __llvm_amdgcn_image_store_1d_v4f32_i32(p, c, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store,1Da)(TSHARP i, int2 c, float4 p)
{
    __llvm_amdgcn_image_store_1darray_v4f32_i32(p, c.x, c.y, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store,1Db)(TSHARP i, int c, float4 p)
{
    __llvm_amdgcn_buffer_store_format_v4f32(p, LOAD_VSHARP(i), c, 0, false, false);
}

WATTR void
OCKL_MANGLE_T(image_store,2D)(TSHARP i, int2 c, float4 p)
{
    __llvm_amdgcn_image_store_2d_v4f32_i32(p, c.x, c.y, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store,2Da)(TSHARP i, int4 c, float4 p)
{
    __llvm_amdgcn_image_store_2darray_v4f32_i32(p, c.x, c.y, c.z, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store,2Dad)(TSHARP i, int4 c, float p)
{
    __llvm_amdgcn_image_store_2darray_f32_i32(p, c.x, c.y, c.z, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store,2Dd)(TSHARP i, int2 c, float p)
{
    __llvm_amdgcn_image_store_2d_f32_i32(p, c.x, c.y, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store,3D)(TSHARP i, int4 c, float4 p)
{
    __llvm_amdgcn_image_store_3d_v4f32_i32(p, c.x, c.y, c.z, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store,CM)(TSHARP i, int2 c, int f, float4 p)
{
    __llvm_amdgcn_image_store_cube_v4f32_i32(p, c.x, c.y, f, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store,CMa)(TSHARP i, int4 c, int f, float4 p)
{
    f = LS_ARRAY_FACE(c.z, f);
    __llvm_amdgcn_image_store_cube_v4f32_i32(p, c.x, c.y, f, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store_lod,1D)(TSHARP i, int c, int l, float4 p)
{
    __llvm_amdgcn_image_store_mip_1d_v4f32_i32(p, c, l, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store_lod,1Da)(TSHARP i, int2 c, int l, float4 p)
{
    __llvm_amdgcn_image_store_mip_1darray_v4f32_i32(p, c.x, c.y, l, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store_lod,2D)(TSHARP i, int2 c, int l, float4 p)
{
    __llvm_amdgcn_image_store_mip_2d_v4f32_i32(p, c.x, c.y, l, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store_lod,2Da)(TSHARP i, int4 c, int l, float4 p)
{
    __llvm_amdgcn_image_store_mip_2darray_v4f32_i32(p, c.x, c.y, c.z, l, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store_lod,2Dad)(TSHARP i, int4 c, int l, float p)
{
    __llvm_amdgcn_image_store_mip_2darray_f32_i32(p, c.x, c.y, c.z, l, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store_lod,2Dd)(TSHARP i, int2 c, int l, float p)
{
    __llvm_amdgcn_image_store_mip_2d_f32_i32(p, c.x, c.y, l, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store_lod,3D)(TSHARP i, int4 c, int l, float4 p)
{
    __llvm_amdgcn_image_store_mip_3d_v4f32_i32(p, c.x, c.y, c.z, l, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store_lod,CM)(TSHARP i, int2 c, int f, int l, float4 p)
{
    __llvm_amdgcn_image_store_mip_cube_v4f32_i32(p, c.x, c.y, f, l, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_store_lod,CMa)(TSHARP i, int4 c, int f, int l, float4 p)
{
    f = LS_ARRAY_FACE(c.z, f);
    __llvm_amdgcn_image_store_mip_cube_v4f32_i32(p, c.x, c.y, f, l, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_storeh,1D)(TSHARP i, int c, half4 p)
{
    __llvm_amdgcn_image_store_1d_v4f16_i32(p, c, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_storeh,1Da)(TSHARP i, int2 c, half4 p)
{
    __llvm_amdgcn_image_store_1darray_v4f16_i32(p, c.x, c.y, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_storeh,1Db)(TSHARP i, int c, half4 p)
{
    __llvm_amdgcn_buffer_store_format_v4f16(p, LOAD_VSHARP(i), c, 0, false, false);
}

WATTR void
OCKL_MANGLE_T(image_storeh,2D)(TSHARP i, int2 c, half4 p)
{
    __llvm_amdgcn_image_store_2d_v4f16_i32(p, c.x, c.y, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_storeh,2Da)(TSHARP i, int4 c, half4 p)
{
    __llvm_amdgcn_image_store_2darray_v4f16_i32(p, c.x, c.y, c.z, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_storeh,3D)(TSHARP i, int4 c, half4 p)
{
    __llvm_amdgcn_image_store_3d_v4f16_i32(p, c.x, c.y, c.z, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_storeh,CM)(TSHARP i, int2 c, int f, half4 p)
{
    __llvm_amdgcn_image_store_cube_v4f16_i32(p, c.x, c.y, f, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_storeh,CMa)(TSHARP i, int4 c, int f, half4 p)
{
    f = LS_ARRAY_FACE(c.z, f);
    __llvm_amdgcn_image_store_cube_v4f16_i32(p, c.x, c.y, f, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_storeh_lod,1D)(TSHARP i, int c, int l, half4 p)
{
    __llvm_amdgcn_image_store_mip_1d_v4f16_i32(p, c, l, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_storeh_lod,1Da)(TSHARP i, int2 c, int l, half4 p)
{
    __llvm_amdgcn_image_store_mip_1darray_v4f16_i32(p, c.x, c.y, l, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_storeh_lod,2D)(TSHARP i, int2 c, int l, half4 p)
{
    __llvm_amdgcn_image_store_mip_2d_v4f16_i32(p, c.x, c.y, l, LOAD_TSHARP(i));
}
WATTR void
OCKL_MANGLE_T(image_storeh_lod,2Da)(TSHARP i, int4 c, int l, half4 p)
{
    __llvm_amdgcn_image_store_mip_2darray_v4f16_i32(p, c.x, c.y, c.z, l, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_storeh_lod,3D)(TSHARP i, int4 c, int l, half4 p)
{
    __llvm_amdgcn_image_store_mip_3d_v4f16_i32(p, c.x, c.y, c.z, l, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_storeh_lod,CM)(TSHARP i, int2 c, int f, int l, half4 p)
{
    __llvm_amdgcn_image_store_mip_cube_v4f16_i32(p, c.x, c.y, f, l, LOAD_TSHARP(i));
}

WATTR void
OCKL_MANGLE_T(image_storeh_lod,CMa)(TSHARP i, int4 c, int f, int l, half4 p)
{
    f = LS_ARRAY_FACE(c.z, f);
    __llvm_amdgcn_image_store_mip_cube_v4f16_i32(p, c.x, c.y, f, l, LOAD_TSHARP(i));
}

RATTR float4
OCKL_MANGLE_T(image_sample,1D)(TSHARP i, SSHARP s, float c)
{
    ADJUST_X(c, i, s);
    return __llvm_amdgcn_image_sample_lz_1d_v4f32_f32(c, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample,1Da)(TSHARP i, SSHARP s, float2 c)
{
    ADJUST_X(c.x, i, s);
    c.y = __builtin_rintf(c.y);
    return __llvm_amdgcn_image_sample_lz_1darray_v4f32_f32(c.x, c.y, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample,2D)(TSHARP i, SSHARP s, float2 c)
{
    ADJUST_XY(c, i, s);
    return __llvm_amdgcn_image_sample_lz_2d_v4f32_f32(c.x, c.y, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample,2Da)(TSHARP i, SSHARP s, float4 c)
{
    ADJUST_XY(c, i, s);
    c.z = __builtin_rintf(c.z);
    return __llvm_amdgcn_image_sample_lz_2darray_v4f32_f32(c.x, c.y, c.z, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float
OCKL_MANGLE_T(image_sample,2Dad)(TSHARP i, SSHARP s, float4 c)
{
    ADJUST_XY(c, i, s);
    c.z = __builtin_rintf(c.z);
    return __llvm_amdgcn_image_sample_lz_2darray_f32_f32(c.x, c.y, c.z, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float
OCKL_MANGLE_T(image_sample,2Dd)(TSHARP i, SSHARP s, float2 c)
{
    ADJUST_XY(c, i, s);
    return __llvm_amdgcn_image_sample_lz_2d_f32_f32(c.x, c.y, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample,3D)(TSHARP i, SSHARP s, float4 c)
{
    ADJUST_XYZ(c, i, s);
    return __llvm_amdgcn_image_sample_lz_3d_v4f32_f32(c.x, c.y, c.z, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample,CM)(TSHARP i, SSHARP s, float4 c)
{
    CUBE_PREP(c);
    return __llvm_amdgcn_image_sample_lz_cube_v4f32_f32(c.x, c.y, c.z, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample,CMa)(TSHARP i, SSHARP s, float4 c)
{
    CUBE_PREP(c);
    c.z = SAMPLE_ARRAY_FACE(c.w, c.z);
    return __llvm_amdgcn_image_sample_lz_cube_v4f32_f32(c.x, c.y, c.z, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample_grad,1D)(TSHARP i, SSHARP s, float c, float dx, float dy)
{
    ADJUST_X(c, i, s);
    return __llvm_amdgcn_image_sample_d_1d_v4f32_f32_f32(dx, dy, c, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample_grad,1Da)(TSHARP i, SSHARP s, float2 c, float dx, float dy)
{
    ADJUST_X(c.x, i, s);
    c.y = __builtin_rintf(c.y);
    return __llvm_amdgcn_image_sample_d_1darray_v4f32_f32_f32(dx, dy, c.x, c.y, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample_grad,2D)(TSHARP i, SSHARP s, float2 c, float2 dx, float2 dy)
{
    ADJUST_XY(c, i, s);
    return __llvm_amdgcn_image_sample_d_2d_v4f32_f32_f32(dx.x, dx.y, dy.x, dy.y, c.x, c.y, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample_grad,2Da)(TSHARP i, SSHARP s, float4 c, float2 dx, float2 dy)
{
    ADJUST_XY(c, i, s);
    c.z = __builtin_rintf(c.z);
    return __llvm_amdgcn_image_sample_d_2darray_v4f32_f32_f32(dx.x, dx.y, dy.x, dy.y, c.x, c.y, c.z, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float
OCKL_MANGLE_T(image_sample_grad,2Dad)(TSHARP i, SSHARP s, float4 c, float2 dx, float2 dy)
{
    ADJUST_XY(c, i, s);
    c.z = __builtin_rintf(c.z);
    return __llvm_amdgcn_image_sample_d_2darray_f32_f32_f32(dx.x, dx.y, dy.x, dy.y, c.x, c.y, c.z, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float
OCKL_MANGLE_T(image_sample_grad,2Dd)(TSHARP i, SSHARP s, float2 c, float2 dx, float2 dy)
{
    ADJUST_XY(c, i, s);
    return __llvm_amdgcn_image_sample_d_2d_f32_f32_f32(dx.x, dx.y, dy.x, dy.y, c.x, c.y, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample_grad,3D)(TSHARP i, SSHARP s, float4 c, float4 dx, float4 dy)
{
    ADJUST_XYZ(c, i, s);
    return __llvm_amdgcn_image_sample_d_3d_v4f32_f32_f32(dx.x, dx.y, dx.z, dy.x, dy.y, dy.z, c.x, c.y, c.z, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample_lod,1D)(TSHARP i, SSHARP s, float c, float l)
{
    return __llvm_amdgcn_image_sample_l_1d_v4f32_f32(c, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample_lod,1Da)(TSHARP i, SSHARP s, float2 c, float l)
{
    c.y = __builtin_rintf(c.y);
    return __llvm_amdgcn_image_sample_l_1darray_v4f32_f32(c.x, c.y, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample_lod,2D)(TSHARP i, SSHARP s, float2 c, float l)
{
    return __llvm_amdgcn_image_sample_l_2d_v4f32_f32(c.x, c.y, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample_lod,2Da)(TSHARP i, SSHARP s, float4 c, float l)
{
    c.z = __builtin_rintf(c.z);
    return __llvm_amdgcn_image_sample_l_2darray_v4f32_f32(c.x, c.y, c.z, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float
OCKL_MANGLE_T(image_sample_lod,2Dad)(TSHARP i, SSHARP s, float4 c, float l)
{
    c.z = __builtin_rintf(c.z);
    return __llvm_amdgcn_image_sample_l_2darray_f32_f32(c.x, c.y, c.z, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float
OCKL_MANGLE_T(image_sample_lod,2Dd)(TSHARP i, SSHARP s, float2 c, float l)
{
    return __llvm_amdgcn_image_sample_l_2d_f32_f32(c.x, c.y, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample_lod,3D)(TSHARP i, SSHARP s, float4 c, float l)
{
    return __llvm_amdgcn_image_sample_l_3d_v4f32_f32(c.x, c.y, c.z, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample_lod,CM)(TSHARP i, SSHARP s, float4 c, float l)
{
    CUBE_PREP(c);
    return __llvm_amdgcn_image_sample_l_cube_v4f32_f32(c.x, c.y, c.z, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_sample_lod,CMa)(TSHARP i, SSHARP s, float4 c, float l)
{
    CUBE_PREP(c);
    c.z = SAMPLE_ARRAY_FACE(c.w, c.z);
    return __llvm_amdgcn_image_sample_l_cube_v4f32_f32(c.x, c.y, c.z, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh,1D)(TSHARP i, SSHARP s, float c)
{
    ADJUST_X(c, i, s);
    return __llvm_amdgcn_image_sample_lz_1d_v4f16_f32(c, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh,1Da)(TSHARP i, SSHARP s, float2 c)
{
    ADJUST_X(c.x, i, s);
    c.y = __builtin_rintf(c.y);
    return __llvm_amdgcn_image_sample_lz_1darray_v4f16_f32(c.x, c.y, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh,2D)(TSHARP i, SSHARP s, float2 c)
{
    ADJUST_XY(c, i, s);
    return __llvm_amdgcn_image_sample_lz_2d_v4f16_f32(c.x, c.y, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh,2Da)(TSHARP i, SSHARP s, float4 c)
{
    ADJUST_XY(c, i, s);
    c.z = __builtin_rintf(c.z);
    return __llvm_amdgcn_image_sample_lz_2darray_v4f16_f32(c.x, c.y, c.z, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh,3D)(TSHARP i, SSHARP s, float4 c)
{
    ADJUST_XYZ(c, i, s);
    return __llvm_amdgcn_image_sample_lz_3d_v4f16_f32(c.x, c.y, c.z, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh,CM)(TSHARP i, SSHARP s, float4 c)
{
    CUBE_PREP(c);
    return __llvm_amdgcn_image_sample_lz_cube_v4f16_f32(c.x, c.y, c.z, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh,CMa)(TSHARP i, SSHARP s, float4 c)
{
    CUBE_PREP(c);
    c.z = SAMPLE_ARRAY_FACE(c.w, c.z);
    return __llvm_amdgcn_image_sample_lz_cube_v4f16_f32(c.x, c.y, c.z, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh_grad,1D)(TSHARP i, SSHARP s, float c, float dx, float dy)
{
    ADJUST_X(c, i, s);
    return __llvm_amdgcn_image_sample_d_1d_v4f16_f32_f32(dx, dy, c, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh_grad,1Da)(TSHARP i, SSHARP s, float2 c, float dx, float dy)
{
    ADJUST_X(c.x, i, s);
    c.y = __builtin_rintf(c.y);
    return __llvm_amdgcn_image_sample_d_1darray_v4f16_f32_f32(dx, dy, c.x, c.y, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh_grad,2D)(TSHARP i, SSHARP s, float2 c, float2 dx, float2 dy)
{
    ADJUST_XY(c, i, s);
    return __llvm_amdgcn_image_sample_d_2d_v4f16_f32_f32(dx.x, dx.y, dy.x, dy.y, c.x, c.y, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh_grad,2Da)(TSHARP i, SSHARP s, float4 c, float2 dx, float2 dy)
{
    ADJUST_XY(c, i, s);
    c.z = __builtin_rintf(c.z);
    return __llvm_amdgcn_image_sample_d_2darray_v4f16_f32_f32(dx.x, dx.y, dy.x, dy.y, c.x, c.y, c.z, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh_grad,3D)(TSHARP i, SSHARP s, float4 c, float4 dx, float4 dy)
{
    ADJUST_XYZ(c, i, s);
    return __llvm_amdgcn_image_sample_d_3d_v4f16_f32_f32(dx.x, dx.y, dx.z, dy.x, dy.y, dy.z, c.x, c.y, c.z, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh_lod,1D)(TSHARP i, SSHARP s, float c, float l)
{
    return __llvm_amdgcn_image_sample_l_1d_v4f16_f32(c, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh_lod,1Da)(TSHARP i, SSHARP s, float2 c, float l)
{
    c.y = __builtin_rintf(c.y);
    return __llvm_amdgcn_image_sample_l_1darray_v4f16_f32(c.x, c.y, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh_lod,2D)(TSHARP i, SSHARP s, float2 c, float l)
{
    return __llvm_amdgcn_image_sample_l_2d_v4f16_f32(c.x, c.y, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh_lod,2Da)(TSHARP i, SSHARP s, float4 c, float l)
{
    c.z = __builtin_rintf(c.z);
    return __llvm_amdgcn_image_sample_l_2darray_v4f16_f32(c.x, c.y, c.z, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh_lod,3D)(TSHARP i, SSHARP s, float4 c, float l)
{
    return __llvm_amdgcn_image_sample_l_3d_v4f16_f32(c.x, c.y, c.z, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh_lod,CM)(TSHARP i, SSHARP s, float4 c, float l)
{
    CUBE_PREP(c);
    return __llvm_amdgcn_image_sample_l_cube_v4f16_f32(c.x, c.y, c.z, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR half4
OCKL_MANGLE_T(image_sampleh_lod,CMa)(TSHARP i, SSHARP s, float4 c, float l)
{
    CUBE_PREP(c);
    c.z = SAMPLE_ARRAY_FACE(c.w, c.z);
    return __llvm_amdgcn_image_sample_l_cube_v4f16_f32(c.x, c.y, c.z, l, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_gather4r,2D)(TSHARP i, SSHARP s, float2 c)
{
    ADJUST_XY(c, i, s);
    return __llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_r(c.x, c.y, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_gather4g,2D)(TSHARP i, SSHARP s, float2 c)
{
    ADJUST_XY(c, i, s);
    return __llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_g(c.x, c.y, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_gather4b,2D)(TSHARP i, SSHARP s, float2 c)
{
    ADJUST_XY(c, i, s);
    return __llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_b(c.x, c.y, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

RATTR float4
OCKL_MANGLE_T(image_gather4a,2D)(TSHARP i, SSHARP s, float2 c)
{
    ADJUST_XY(c, i, s);
    return __llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_a(c.x, c.y, LOAD_TSHARP(i), LOAD_SSHARP(s));
}

// We rely on the fact that the runtime allocates 12 words for the T# or V#
// and fills words 8, 9, and 10 with the data we need to answer all of the queries

#define ARRAY_SIZE(I) \
    if (__oclc_ISA_version < 9000) { \
        return FIELD(I, 173, 13) + 1U; \
    } else { \
        return FIELD(I, 128, 13) + 1U; \
    }
GATTR int OCKL_MANGLE_T(image_array_size,1Da)(TSHARP i)  { ARRAY_SIZE(i) }
GATTR int OCKL_MANGLE_T(image_array_size,2Da)(TSHARP i)  { ARRAY_SIZE(i) }
GATTR int OCKL_MANGLE_T(image_array_size,2Dad)(TSHARP i) { ARRAY_SIZE(i) }
GATTR int OCKL_MANGLE_T(image_array_size,CMa)(TSHARP i)  { ARRAY_SIZE(i) }

GATTR int OCKL_MANGLE_T(image_channel_data_type,1D)(TSHARP i)   { return WORD(i, 8); }
GATTR int OCKL_MANGLE_T(image_channel_data_type,1Da)(TSHARP i)  { return WORD(i, 8); }
GATTR int OCKL_MANGLE_T(image_channel_data_type,1Db)(TSHARP i)  { return WORD(i, 8); }
GATTR int OCKL_MANGLE_T(image_channel_data_type,2D)(TSHARP i)   { return WORD(i, 8); }
GATTR int OCKL_MANGLE_T(image_channel_data_type,2Da)(TSHARP i)  { return WORD(i, 8); }
GATTR int OCKL_MANGLE_T(image_channel_data_type,2Dad)(TSHARP i) { return WORD(i, 8); }
GATTR int OCKL_MANGLE_T(image_channel_data_type,2Dd)(TSHARP i)  { return WORD(i, 8); }
GATTR int OCKL_MANGLE_T(image_channel_data_type,3D)(TSHARP i)   { return WORD(i, 8); }
GATTR int OCKL_MANGLE_T(image_channel_data_type,CM)(TSHARP i)   { return WORD(i, 8); }
GATTR int OCKL_MANGLE_T(image_channel_data_type,CMa)(TSHARP i)  { return WORD(i, 8); }

GATTR int OCKL_MANGLE_T(image_channel_order,1D)(TSHARP i)   { return WORD(i, 9); }
GATTR int OCKL_MANGLE_T(image_channel_order,1Da)(TSHARP i)  { return WORD(i, 9); }
GATTR int OCKL_MANGLE_T(image_channel_order,1Db)(TSHARP i)  { return WORD(i, 9); }
GATTR int OCKL_MANGLE_T(image_channel_order,2D)(TSHARP i)   { return WORD(i, 9); }
GATTR int OCKL_MANGLE_T(image_channel_order,2Da)(TSHARP i)  { return WORD(i, 9); }
GATTR int OCKL_MANGLE_T(image_channel_order,2Dad)(TSHARP i) { return WORD(i, 9); }
GATTR int OCKL_MANGLE_T(image_channel_order,2Dd)(TSHARP i)  { return WORD(i, 9); }
GATTR int OCKL_MANGLE_T(image_channel_order,3D)(TSHARP i)   { return WORD(i, 9); }
GATTR int OCKL_MANGLE_T(image_channel_order,CM)(TSHARP i)   { return WORD(i, 9); }
GATTR int OCKL_MANGLE_T(image_channel_order,CMa)(TSHARP i)  { return WORD(i, 9); }

GATTR int OCKL_MANGLE_T(image_depth,3D)(TSHARP i) { return FIELD(i, 128, 13) + 1U; }

GATTR int OCKL_MANGLE_T(image_height,2D)(TSHARP i)   { return FIELD(i, 78, 14) + 1U; }
GATTR int OCKL_MANGLE_T(image_height,2Da)(TSHARP i)  { return FIELD(i, 78, 14) + 1U; }
GATTR int OCKL_MANGLE_T(image_height,2Dad)(TSHARP i) { return FIELD(i, 78, 14) + 1U; }
GATTR int OCKL_MANGLE_T(image_height,2Dd)(TSHARP i)  { return FIELD(i, 78, 14) + 1U; }
GATTR int OCKL_MANGLE_T(image_height,3D)(TSHARP i)   { return FIELD(i, 78, 14) + 1U; }
GATTR int OCKL_MANGLE_T(image_height,CM)(TSHARP i)   { return FIELD(i, 78, 14) + 1U; }
GATTR int OCKL_MANGLE_T(image_height,CMa)(TSHARP i)  { return FIELD(i, 78, 14) + 1U; }

GATTR int OCKL_MANGLE_T(image_num_mip_levels,1D)(TSHARP i)   { return FIELD(i, 112, 4); }
GATTR int OCKL_MANGLE_T(image_num_mip_levels,1Da)(TSHARP i)  { return FIELD(i, 112, 4); }
GATTR int OCKL_MANGLE_T(image_num_mip_levels,2D)(TSHARP i)   { return FIELD(i, 112, 4); }
GATTR int OCKL_MANGLE_T(image_num_mip_levels,2Da)(TSHARP i)  { return FIELD(i, 112, 4); }
GATTR int OCKL_MANGLE_T(image_num_mip_levels,2Dad)(TSHARP i) { return FIELD(i, 112, 4); }
GATTR int OCKL_MANGLE_T(image_num_mip_levels,2Dd)(TSHARP i)  { return FIELD(i, 112, 4); }
GATTR int OCKL_MANGLE_T(image_num_mip_levels,3D)(TSHARP i)   { return FIELD(i, 112, 4); }
GATTR int OCKL_MANGLE_T(image_num_mip_levels,CM)(TSHARP i)   { return FIELD(i, 112, 4); }
GATTR int OCKL_MANGLE_T(image_num_mip_levels,CMa)(TSHARP i)  { return FIELD(i, 112, 4); }

// In FIELD(i, 64, 14) but also copied into word 11 of the 12 that are allocated
GATTR int OCKL_MANGLE_T(image_width,1D)(TSHARP i)   { return WORD(i, 10); }
GATTR int OCKL_MANGLE_T(image_width,1Da)(TSHARP i)  { return WORD(i, 10); }
GATTR int OCKL_MANGLE_T(image_width,2D)(TSHARP i)   { return WORD(i, 10); }
GATTR int OCKL_MANGLE_T(image_width,2Da)(TSHARP i)  { return WORD(i, 10); }
GATTR int OCKL_MANGLE_T(image_width,2Dad)(TSHARP i) { return WORD(i, 10); }
GATTR int OCKL_MANGLE_T(image_width,2Dd)(TSHARP i)  { return WORD(i, 10); }
GATTR int OCKL_MANGLE_T(image_width,3D)(TSHARP i)   { return WORD(i, 10); }
GATTR int OCKL_MANGLE_T(image_width,CM)(TSHARP i)   { return WORD(i, 10); }
GATTR int OCKL_MANGLE_T(image_width,CMa)(TSHARP i)  { return WORD(i, 10); }
// This would be a bit trickier since we actually have a V# here and need to look at const_num_records and const_stride
GATTR int OCKL_MANGLE_T(image_width,1Db)(TSHARP i)  { return WORD(i, 10); }
