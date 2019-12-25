/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#if MIOPEN_USE_FP16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)

#define UNUSED __attribute__((__unused__))

#ifndef MLO_POOLING_INDEX_TYPE
#error "MLO_POOLING_INDEX_TYPE not defined"
#else
typedef MLO_POOLING_INDEX_TYPE index_t;
#endif

#define MLO_POOLING_OP_AVE 0
#define MLO_POOLING_OP_MAX 1
#define MLO_POOLING_OP_STC 2
#define MLO_POOLING_OP_AVE_INCLUSIVE 3

#define MLO_POOLING_GROUP_SZ2 1

#ifndef MLO_POOLING_OP_ID
#define MLO_POOLING_OP_ID 0
#endif
// max
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
#define MLO_POOLING_OP(A, B) (fmax((A), (B)))
#elif(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || \
    (MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
#define MLO_POOLING_OP(A, B) ((A) + (B))
#endif

/*********************************************************************************

**********************************************************************************/

#define MLO_BOT_DATA_SZ0 \
    ((MLO_POOLING_N_HORIZ_OUT_PIX - 1) * MLO_POOLING_STRIDE0 + MLO_POOLING_KERNEL_SZ0)
#define MLO_BOT_DATA_SZ1 \
    ((MLO_POOLING_N_VERT_OUT_PIX - 1) * MLO_POOLING_STRIDE1 + MLO_POOLING_KERNEL_SZ1)

__attribute__((reqd_work_group_size(MLO_POOLING_GROUP_SZ0,
                                    MLO_POOLING_GROUP_SZ1,
                                    MLO_POOLING_GROUP_SZ2))) __kernel void
mloPoolingG(const __global _FLOAT* bot,
            __global _FLOAT* top,
#if !defined(MLO_POOLING_SAVE_INDEX) || MLO_POOLING_OP_ID != MLO_POOLING_OP_MAX
            UNUSED
#endif
                __global index_t* mask)
{

    uint x       = get_group_id(0) * MLO_POOLING_GROUP_SZ0 * MLO_POOLING_N_HORIZ_OUT_PIX;
    uint y       = get_group_id(1) * MLO_POOLING_GROUP_SZ1 * MLO_POOLING_N_VERT_OUT_PIX;
    uint lcl_id0 = get_local_id(0);
    uint lcl_id1 = get_local_id(1);
    //		int lcl_id = (lcl_id1 << MLO_POOLING_GROUP_LG2SZ0) + lcl_id0;
    uint ob      = get_global_id(2); // output * batch_sz
    uint b       = ob / MLO_POOLING_N_OUTPUTS;
    uint o       = ob - b * MLO_POOLING_N_OUTPUTS;
    uint bot_x   = (x + lcl_id0 * MLO_POOLING_N_HORIZ_OUT_PIX) * MLO_POOLING_STRIDE0;
    uint bot_y   = (y + lcl_id1 * MLO_POOLING_N_VERT_OUT_PIX) * MLO_POOLING_STRIDE1;
    uint bot_off = b * MLO_POOLING_BOT_BATCH_STRIDE + o * MLO_POOLING_BOT_CHANNEL_STRIDE;

    _FLOAT bot_data[MLO_BOT_DATA_SZ1][MLO_BOT_DATA_SZ0];
    _FLOAT res[MLO_POOLING_N_VERT_OUT_PIX][MLO_POOLING_N_HORIZ_OUT_PIX];
#if defined(MLO_POOLING_SAVE_INDEX) && MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
    index_t mask_private[MLO_POOLING_N_VERT_OUT_PIX][MLO_POOLING_N_HORIZ_OUT_PIX];
#endif
    for(int k = 0; k < MLO_POOLING_N_VERT_OUT_PIX; k++)
    {
        for(int l = 0; l < MLO_POOLING_N_HORIZ_OUT_PIX; l++)
        {
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
            res[k][l] = (_FLOAT)(-MAX_VAL);
#elif(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || \
    (MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
            res[k][l] = (_FLOAT)(0);
#endif
        }
    }

    for(uint j = 0; j < MLO_BOT_DATA_SZ1; ++j)
    {
        int run_y = (int)bot_y + j - MLO_POOLING_PAD1;

        for(uint i = 0; i < MLO_BOT_DATA_SZ0; ++i)
        {
            int run_x        = (int)bot_x + i - MLO_POOLING_PAD0;
            uint bot_gbl_off = bot_off + (uint)run_y * MLO_POOLING_BOT_STRIDE + (uint)run_x;
            bool vis         = ((run_y >= 0 && run_y < MLO_POOLING_BOT_HEIGHT) &&
                        (run_x >= 0 && run_x < MLO_POOLING_BOT_WIDTH))
                           ? true
                           : false;
            bot_data[j][i] = (vis) ? bot[bot_gbl_off] :
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
                                   (_FLOAT)(-MAX_VAL)
#elif(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || \
    (MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
                                   (_FLOAT)(0)
#endif
                ;
        }
    }

#pragma unroll
    for(uint k = 0; k < MLO_POOLING_N_VERT_OUT_PIX; k++)
    {
#if(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || (MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
        uint y_dst = y + lcl_id1 * MLO_POOLING_N_VERT_OUT_PIX + k;
        int hstart = (int)y_dst * MLO_POOLING_STRIDE1 - MLO_POOLING_PAD1;
        int hend   = min((hstart + MLO_POOLING_KERNEL_SZ1), (int)(MLO_POOLING_BOT_HEIGHT));
        hstart     = max(hstart, 0);
#endif
        for(uint l = 0; l < MLO_POOLING_N_HORIZ_OUT_PIX; l++)
        {

#if(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || (MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
            uint x_dst = x + lcl_id0 * MLO_POOLING_N_HORIZ_OUT_PIX + l;
            int wstart = (int)x_dst * MLO_POOLING_STRIDE0 - MLO_POOLING_PAD0;
            int wend   = min((wstart + MLO_POOLING_KERNEL_SZ0), (int)(MLO_POOLING_BOT_WIDTH));
            wstart     = max(wstart, 0);
            uint pool_size =
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE
                MLO_POOLING_KERNEL_SZ0 * MLO_POOLING_KERNEL_SZ1;
            (void)wend;
            (void)hend;
#else
                (hend - hstart) * (wend - wstart);
#endif
            pool_size = (pool_size == 0) ? 1 : pool_size;
#endif
#if defined(MLO_POOLING_SAVE_INDEX) && MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
            mask_private[k][l] = 0;
#endif

            for(uint j = 0; j < MLO_POOLING_KERNEL_SZ1; j++)
            {
                for(uint i = 0; i < MLO_POOLING_KERNEL_SZ0; i++)
                {

                    _FLOAT bot_val =
                        bot_data[j + k * MLO_POOLING_STRIDE1][i + l * MLO_POOLING_STRIDE0];

#if defined(MLO_POOLING_SAVE_INDEX) && MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
                    if(bot_val > res[k][l])
                    {
                        res[k][l]          = bot_val;
                        mask_private[k][l] = i + MLO_POOLING_KERNEL_SZ0 * j;
                    }
#else
                    res[k][l] = MLO_POOLING_OP(res[k][l], bot_val);
#endif
                }
            }

#if(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || (MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
            res[k][l] *= (_FLOAT)1.f / (_FLOAT)pool_size;
#endif
        }
    }

    uint top_y   = (y + lcl_id1 * MLO_POOLING_N_VERT_OUT_PIX);
    uint top_x   = (x + lcl_id0 * MLO_POOLING_N_HORIZ_OUT_PIX);
    uint top_off = b * MLO_POOLING_TOP_BATCH_STRIDE + o * MLO_POOLING_TOP_CHANNEL_STRIDE +
                   top_y * MLO_POOLING_TOP_STRIDE + top_x;
    for(uint k = 0; k < MLO_POOLING_N_VERT_OUT_PIX; k++)
    {
        for(uint l = 0; l < MLO_POOLING_N_HORIZ_OUT_PIX; l++)
        {
            if(top_y + k < MLO_POOLING_TOP_HEIGHT && top_x + l < MLO_POOLING_TOP_WIDTH)
            {
                top[top_off + k * MLO_POOLING_TOP_STRIDE + l] = res[k][l];
#if defined(MLO_POOLING_SAVE_INDEX) && MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
                mask[top_off + k * MLO_POOLING_TOP_STRIDE + l] = mask_private[k][l];
#endif
            }
        }
    }
}
