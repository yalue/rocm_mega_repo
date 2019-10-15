/******************************************************************************
* Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#ifndef BUTTERFLY_H
#define BUTTERFLY_H

#include "common.h"
#include <hip/hip_runtime.h>

enum StrideBin
{
    SB_UNIT,
    SB_NONUNIT,
};

#define TWIDDLE_STEP_MUL_FWD(TWFUNC, TWIDDLES, INDEX, REG) \
    {                                                      \
        T     W = TWFUNC(TWIDDLES, INDEX);                 \
        float TR, TI;                                      \
        TR    = (W.x * REG.x) - (W.y * REG.y);             \
        TI    = (W.y * REG.x) + (W.x * REG.y);             \
        REG.x = TR;                                        \
        REG.y = TI;                                        \
    }

#define TWIDDLE_STEP_MUL_INV(TWFUNC, TWIDDLES, INDEX, REG) \
    {                                                      \
        T     W = TWFUNC(TWIDDLES, INDEX);                 \
        float TR, TI;                                      \
        TR    = (W.x * REG.x) + (W.y * REG.y);             \
        TI    = -(W.y * REG.x) + (W.x * REG.y);            \
        REG.x = TR;                                        \
        REG.y = TI;                                        \
    }

#define TWIDDLE_MUL_FWD(TWIDDLES, INDEX, REG)  \
    {                                          \
        T     W = TWIDDLES[INDEX];             \
        float TR, TI;                          \
        TR    = (W.x * REG.x) - (W.y * REG.y); \
        TI    = (W.y * REG.x) + (W.x * REG.y); \
        REG.x = TR;                            \
        REG.y = TI;                            \
    }

#define TWIDDLE_MUL_INV(TWIDDLES, INDEX, REG)   \
    {                                           \
        T     W = TWIDDLES[INDEX];              \
        float TR, TI;                           \
        TR    = (W.x * REG.x) + (W.y * REG.y);  \
        TI    = -(W.y * REG.x) + (W.x * REG.y); \
        REG.x = TR;                             \
        REG.y = TI;                             \
    }

template <typename T>
__device__ void Rad2(T* R0, T* R1)
{

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
}

template <typename T>
__device__ void FwdRad4(T* R0, T* R2, T* R1, T* R3)
{

    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + lib_make_vector2<T>(-(*R3).y, (*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);

    res   = (*R1);
    (*R1) = (*R2);
    (*R2) = res;
}

template <typename T>
__device__ void InvRad4(T* R0, T* R2, T* R1, T* R3)
{

    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + lib_make_vector2<T>((*R3).y, -(*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);

    res   = (*R1);
    (*R1) = (*R2);
    (*R2) = res;
}

#define C8Q 0.707106781186547573

template <typename T>
__device__ void FwdRad8(T* R0, T* R4, T* R2, T* R6, T* R1, T* R5, T* R3, T* R7)
{

    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);
    (*R5) = (*R4) - (*R5);
    (*R4) = 2.0 * (*R4) - (*R5);
    (*R7) = (*R6) - (*R7);
    (*R6) = 2.0 * (*R6) - (*R7);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + lib_make_vector2<T>(-(*R3).y, (*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);
    (*R6) = (*R4) - (*R6);
    (*R4) = 2.0 * (*R4) - (*R6);
    (*R7) = (*R5) + lib_make_vector2<T>(-(*R7).y, (*R7).x);
    (*R5) = 2.0 * (*R5) - (*R7);

    (*R4) = (*R0) - (*R4);
    (*R0) = 2.0 * (*R0) - (*R4);
    (*R5) = ((*R1) - C8Q * (*R5)) - C8Q * lib_make_vector2<T>((*R5).y, -(*R5).x);
    (*R1) = 2.0 * (*R1) - (*R5);
    (*R6) = (*R2) + lib_make_vector2<T>(-(*R6).y, (*R6).x);
    (*R2) = 2.0 * (*R2) - (*R6);
    (*R7) = ((*R3) + C8Q * (*R7)) - C8Q * lib_make_vector2<T>((*R7).y, -(*R7).x);
    (*R3) = 2.0 * (*R3) - (*R7);

    res   = (*R1);
    (*R1) = (*R4);
    (*R4) = res;
    res   = (*R3);
    (*R3) = (*R6);
    (*R6) = res;
}

template <typename T>
__device__ void InvRad8(T* R0, T* R4, T* R2, T* R6, T* R1, T* R5, T* R3, T* R7)
{

    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);
    (*R5) = (*R4) - (*R5);
    (*R4) = 2.0 * (*R4) - (*R5);
    (*R7) = (*R6) - (*R7);
    (*R6) = 2.0 * (*R6) - (*R7);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + lib_make_vector2<T>((*R3).y, -(*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);
    (*R6) = (*R4) - (*R6);
    (*R4) = 2.0 * (*R4) - (*R6);
    (*R7) = (*R5) + lib_make_vector2<T>((*R7).y, -(*R7).x);
    (*R5) = 2.0 * (*R5) - (*R7);

    (*R4) = (*R0) - (*R4);
    (*R0) = 2.0 * (*R0) - (*R4);
    (*R5) = ((*R1) - C8Q * (*R5)) + C8Q * lib_make_vector2<T>((*R5).y, -(*R5).x);
    (*R1) = 2.0 * (*R1) - (*R5);
    (*R6) = (*R2) + lib_make_vector2<T>((*R6).y, -(*R6).x);
    (*R2) = 2.0 * (*R2) - (*R6);
    (*R7) = ((*R3) + C8Q * (*R7)) + C8Q * lib_make_vector2<T>((*R7).y, -(*R7).x);
    (*R3) = 2.0 * (*R3) - (*R7);

    res   = (*R1);
    (*R1) = (*R4);
    (*R4) = res;
    res   = (*R3);
    (*R3) = (*R6);
    (*R6) = res;
}

#define C16A 0.923879532511286738
#define C16B 0.382683432365089837

template <typename T>
__device__ void FwdRad16(T* R0,
                         T* R8,
                         T* R4,
                         T* R12,
                         T* R2,
                         T* R10,
                         T* R6,
                         T* R14,
                         T* R1,
                         T* R9,
                         T* R5,
                         T* R13,
                         T* R3,
                         T* R11,
                         T* R7,
                         T* R15)
{

    T res;

    (*R1)  = (*R0) - (*R1);
    (*R0)  = 2.0 * (*R0) - (*R1);
    (*R3)  = (*R2) - (*R3);
    (*R2)  = 2.0 * (*R2) - (*R3);
    (*R5)  = (*R4) - (*R5);
    (*R4)  = 2.0 * (*R4) - (*R5);
    (*R7)  = (*R6) - (*R7);
    (*R6)  = 2.0 * (*R6) - (*R7);
    (*R9)  = (*R8) - (*R9);
    (*R8)  = 2.0 * (*R8) - (*R9);
    (*R11) = (*R10) - (*R11);
    (*R10) = 2.0 * (*R10) - (*R11);
    (*R13) = (*R12) - (*R13);
    (*R12) = 2.0 * (*R12) - (*R13);
    (*R15) = (*R14) - (*R15);
    (*R14) = 2.0 * (*R14) - (*R15);

    (*R2)  = (*R0) - (*R2);
    (*R0)  = 2.0 * (*R0) - (*R2);
    (*R3)  = (*R1) + lib_make_vector2<T>(-(*R3).y, (*R3).x);
    (*R1)  = 2.0 * (*R1) - (*R3);
    (*R6)  = (*R4) - (*R6);
    (*R4)  = 2.0 * (*R4) - (*R6);
    (*R7)  = (*R5) + lib_make_vector2<T>(-(*R7).y, (*R7).x);
    (*R5)  = 2.0 * (*R5) - (*R7);
    (*R10) = (*R8) - (*R10);
    (*R8)  = 2.0 * (*R8) - (*R10);
    (*R11) = (*R9) + lib_make_vector2<T>(-(*R11).y, (*R11).x);
    (*R9)  = 2.0 * (*R9) - (*R11);
    (*R14) = (*R12) - (*R14);
    (*R12) = 2.0 * (*R12) - (*R14);
    (*R15) = (*R13) + lib_make_vector2<T>(-(*R15).y, (*R15).x);
    (*R13) = 2.0 * (*R13) - (*R15);

    (*R4)  = (*R0) - (*R4);
    (*R0)  = 2.0 * (*R0) - (*R4);
    (*R5)  = ((*R1) - C8Q * (*R5)) - C8Q * lib_make_vector2<T>((*R5).y, -(*R5).x);
    (*R1)  = 2.0 * (*R1) - (*R5);
    (*R6)  = (*R2) + lib_make_vector2<T>(-(*R6).y, (*R6).x);
    (*R2)  = 2.0 * (*R2) - (*R6);
    (*R7)  = ((*R3) + C8Q * (*R7)) - C8Q * lib_make_vector2<T>((*R7).y, -(*R7).x);
    (*R3)  = 2.0 * (*R3) - (*R7);
    (*R12) = (*R8) - (*R12);
    (*R8)  = 2.0 * (*R8) - (*R12);
    (*R13) = ((*R9) - C8Q * (*R13)) - C8Q * lib_make_vector2<T>((*R13).y, -(*R13).x);
    (*R9)  = 2.0 * (*R9) - (*R13);
    (*R14) = (*R10) + lib_make_vector2<T>(-(*R14).y, (*R14).x);
    (*R10) = 2.0 * (*R10) - (*R14);
    (*R15) = ((*R11) + C8Q * (*R15)) - C8Q * lib_make_vector2<T>((*R15).y, -(*R15).x);
    (*R11) = 2.0 * (*R11) - (*R15);

    (*R8) = (*R0) - (*R8);
    (*R0) = 2.0 * (*R0) - (*R8);
    (*R9) = ((*R1) - C16A * (*R9)) - C16B * lib_make_vector2<T>((*R9).y, -(*R9).x);
    res   = (*R8);
    (*R1) = 2.0 * (*R1) - (*R9);

    (*R10) = ((*R2) - C8Q * (*R10)) - C8Q * lib_make_vector2<T>((*R10).y, -(*R10).x);
    (*R2)  = 2.0 * (*R2) - (*R10);
    (*R11) = ((*R3) - C16B * (*R11)) - C16A * lib_make_vector2<T>((*R11).y, -(*R11).x);
    (*R3)  = 2.0 * (*R3) - (*R11);

    (*R12) = (*R4) + lib_make_vector2<T>(-(*R12).y, (*R12).x);
    (*R4)  = 2.0 * (*R4) - (*R12);
    (*R13) = ((*R5) + C16B * (*R13)) - C16A * lib_make_vector2<T>((*R13).y, -(*R13).x);
    (*R5)  = 2.0 * (*R5) - (*R13);

    (*R14) = ((*R6) + C8Q * (*R14)) - C8Q * lib_make_vector2<T>((*R14).y, -(*R14).x);
    (*R6)  = 2.0 * (*R6) - (*R14);
    (*R15) = ((*R7) + C16A * (*R15)) - C16B * lib_make_vector2<T>((*R15).y, -(*R15).x);
    (*R7)  = 2.0 * (*R7) - (*R15);

    res    = (*R1);
    (*R1)  = (*R8);
    (*R8)  = res;
    res    = (*R2);
    (*R2)  = (*R4);
    (*R4)  = res;
    res    = (*R3);
    (*R3)  = (*R12);
    (*R12) = res;
    res    = (*R5);
    (*R5)  = (*R10);
    (*R10) = res;
    res    = (*R7);
    (*R7)  = (*R14);
    (*R14) = res;
    res    = (*R11);
    (*R11) = (*R13);
    (*R13) = res;
}

template <typename T>
__device__ void InvRad16(T* R0,
                         T* R8,
                         T* R4,
                         T* R12,
                         T* R2,
                         T* R10,
                         T* R6,
                         T* R14,
                         T* R1,
                         T* R9,
                         T* R5,
                         T* R13,
                         T* R3,
                         T* R11,
                         T* R7,
                         T* R15)
{

    T res;

    (*R1)  = (*R0) - (*R1);
    (*R0)  = 2.0 * (*R0) - (*R1);
    (*R3)  = (*R2) - (*R3);
    (*R2)  = 2.0 * (*R2) - (*R3);
    (*R5)  = (*R4) - (*R5);
    (*R4)  = 2.0 * (*R4) - (*R5);
    (*R7)  = (*R6) - (*R7);
    (*R6)  = 2.0 * (*R6) - (*R7);
    (*R9)  = (*R8) - (*R9);
    (*R8)  = 2.0 * (*R8) - (*R9);
    (*R11) = (*R10) - (*R11);
    (*R10) = 2.0 * (*R10) - (*R11);
    (*R13) = (*R12) - (*R13);
    (*R12) = 2.0 * (*R12) - (*R13);
    (*R15) = (*R14) - (*R15);
    (*R14) = 2.0 * (*R14) - (*R15);

    (*R2)  = (*R0) - (*R2);
    (*R0)  = 2.0 * (*R0) - (*R2);
    (*R3)  = (*R1) + lib_make_vector2<T>((*R3).y, -(*R3).x);
    (*R1)  = 2.0 * (*R1) - (*R3);
    (*R6)  = (*R4) - (*R6);
    (*R4)  = 2.0 * (*R4) - (*R6);
    (*R7)  = (*R5) + lib_make_vector2<T>((*R7).y, -(*R7).x);
    (*R5)  = 2.0 * (*R5) - (*R7);
    (*R10) = (*R8) - (*R10);
    (*R8)  = 2.0 * (*R8) - (*R10);
    (*R11) = (*R9) + lib_make_vector2<T>((*R11).y, -(*R11).x);
    (*R9)  = 2.0 * (*R9) - (*R11);
    (*R14) = (*R12) - (*R14);
    (*R12) = 2.0 * (*R12) - (*R14);
    (*R15) = (*R13) + lib_make_vector2<T>((*R15).y, -(*R15).x);
    (*R13) = 2.0 * (*R13) - (*R15);

    (*R4)  = (*R0) - (*R4);
    (*R0)  = 2.0 * (*R0) - (*R4);
    (*R5)  = ((*R1) - C8Q * (*R5)) + C8Q * lib_make_vector2<T>((*R5).y, -(*R5).x);
    (*R1)  = 2.0 * (*R1) - (*R5);
    (*R6)  = (*R2) + lib_make_vector2<T>((*R6).y, -(*R6).x);
    (*R2)  = 2.0 * (*R2) - (*R6);
    (*R7)  = ((*R3) + C8Q * (*R7)) + C8Q * lib_make_vector2<T>((*R7).y, -(*R7).x);
    (*R3)  = 2.0 * (*R3) - (*R7);
    (*R12) = (*R8) - (*R12);
    (*R8)  = 2.0 * (*R8) - (*R12);
    (*R13) = ((*R9) - C8Q * (*R13)) + C8Q * lib_make_vector2<T>((*R13).y, -(*R13).x);
    (*R9)  = 2.0 * (*R9) - (*R13);
    (*R14) = (*R10) + lib_make_vector2<T>((*R14).y, -(*R14).x);
    (*R10) = 2.0 * (*R10) - (*R14);
    (*R15) = ((*R11) + C8Q * (*R15)) + C8Q * lib_make_vector2<T>((*R15).y, -(*R15).x);
    (*R11) = 2.0 * (*R11) - (*R15);

    (*R8)  = (*R0) - (*R8);
    (*R0)  = 2.0 * (*R0) - (*R8);
    (*R9)  = ((*R1) - C16A * (*R9)) + C16B * lib_make_vector2<T>((*R9).y, -(*R9).x);
    (*R1)  = 2.0 * (*R1) - (*R9);
    (*R10) = ((*R2) - C8Q * (*R10)) + C8Q * lib_make_vector2<T>((*R10).y, -(*R10).x);
    (*R2)  = 2.0 * (*R2) - (*R10);
    (*R11) = ((*R3) - C16B * (*R11)) + C16A * lib_make_vector2<T>((*R11).y, -(*R11).x);
    (*R3)  = 2.0 * (*R3) - (*R11);
    (*R12) = (*R4) + lib_make_vector2<T>((*R12).y, -(*R12).x);
    (*R4)  = 2.0 * (*R4) - (*R12);
    (*R13) = ((*R5) + C16B * (*R13)) + C16A * lib_make_vector2<T>((*R13).y, -(*R13).x);
    (*R5)  = 2.0 * (*R5) - (*R13);
    (*R14) = ((*R6) + C8Q * (*R14)) + C8Q * lib_make_vector2<T>((*R14).y, -(*R14).x);
    (*R6)  = 2.0 * (*R6) - (*R14);
    (*R15) = ((*R7) + C16A * (*R15)) + C16B * lib_make_vector2<T>((*R15).y, -(*R15).x);
    (*R7)  = 2.0 * (*R7) - (*R15);

    res    = (*R1);
    (*R1)  = (*R8);
    (*R8)  = res;
    res    = (*R2);
    (*R2)  = (*R4);
    (*R4)  = res;
    res    = (*R3);
    (*R3)  = (*R12);
    (*R12) = res;
    res    = (*R5);
    (*R5)  = (*R10);
    (*R10) = res;
    res    = (*R7);
    (*R7)  = (*R14);
    (*R14) = res;
    res    = (*R11);
    (*R11) = (*R13);
    (*R13) = res;
}

#endif // BUTTERFLY_H
