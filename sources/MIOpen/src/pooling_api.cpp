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
#include <miopen/pooling.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>

#include <numeric>
#include <algorithm>
#include <array>
#include <initializer_list>

extern "C" miopenStatus_t miopenCreatePoolingDescriptor(miopenPoolingDescriptor_t* poolDesc)
{
    MIOPEN_LOG_FUNCTION(poolDesc);
    return miopen::try_([&] { miopen::deref(poolDesc) = new miopen::PoolingDescriptor(); });
}

extern "C" miopenStatus_t miopenSetPoolingIndexType(miopenPoolingDescriptor_t poolDesc,
                                                    miopenIndexType_t index_type)
{
    MIOPEN_LOG_FUNCTION(poolDesc, index_type);
    return miopen::try_([&] { miopen::deref(poolDesc).SetIndexType(index_type); });
}

extern "C" miopenStatus_t miopenGetPoolingIndexType(miopenPoolingDescriptor_t poolDesc,
                                                    miopenIndexType_t* index_type)
{
    MIOPEN_LOG_FUNCTION(poolDesc, index_type);
    return miopen::try_([&] { *index_type = miopen::deref(poolDesc).GetIndexType(); });
}

extern "C" miopenStatus_t miopenSet2dPoolingDescriptor(miopenPoolingDescriptor_t poolDesc,
                                                       miopenPoolingMode_t mode,
                                                       int windowHeight,
                                                       int windowWidth,
                                                       int pad_h,
                                                       int pad_w,
                                                       int stride_h,
                                                       int stride_w)
{

    MIOPEN_LOG_FUNCTION(
        poolDesc, mode, windowHeight, windowWidth, pad_h, pad_w, stride_h, stride_w);
    return miopen::try_([&] {
        std::initializer_list<int> lens    = {windowHeight, windowWidth};
        std::initializer_list<int> pads    = {pad_h, pad_w};
        std::initializer_list<int> strides = {stride_h, stride_w};
        miopen::deref(poolDesc)            = miopen::PoolingDescriptor(
            mode, miopenPaddingDefault, lens.begin(), pads.begin(), strides.begin(), 2);
    });
}

extern "C" miopenStatus_t miopenGet2dPoolingDescriptor(const miopenPoolingDescriptor_t poolDesc,
                                                       miopenPoolingMode_t* mode,
                                                       int* windowHeight,
                                                       int* windowWidth,
                                                       int* pad_h,
                                                       int* pad_w,
                                                       int* stride_h,
                                                       int* stride_w)
{

    MIOPEN_LOG_FUNCTION(
        poolDesc, mode, windowHeight, windowWidth, pad_h, pad_w, stride_h, stride_w);
    return miopen::try_([&] {
        miopen::deref(mode) = miopen::deref(poolDesc).mode;
        std::tie(miopen::deref(windowHeight), miopen::deref(windowWidth)) =
            miopen::tien<2>(miopen::deref(poolDesc).GetLengths());
        std::tie(miopen::deref(stride_h), miopen::deref(stride_w)) =
            miopen::tien<2>(miopen::deref(poolDesc).GetStrides());
        std::tie(miopen::deref(pad_h), miopen::deref(pad_w)) =
            miopen::tien<2>(miopen::deref(poolDesc).GetPads());
    });
}

extern "C" miopenStatus_t miopenSetNdPoolingDescriptor(miopenPoolingDescriptor_t poolDesc,
                                                       miopenPoolingMode_t mode,
                                                       miopenPaddingMode_t pmode,
                                                       int nbDims,
                                                       int* windowDimA,
                                                       int* padA,
                                                       int* stridesA)
{

    return miopen::try_([&] {
        miopen::deref(poolDesc) =
            miopen::PoolingDescriptor(mode, pmode, windowDimA, padA, stridesA, nbDims);
    });
}

extern "C" miopenStatus_t miopenGetNdPoolingDescriptor(miopenPoolingDescriptor_t poolDesc,
                                                       miopenPoolingMode_t* mode,
                                                       miopenPaddingMode_t* pmode,
                                                       int* nbDims,
                                                       int* windowDimA,
                                                       int* padA,
                                                       int* stridesA)
{

    return miopen::try_([&] {
        if(mode != nullptr)
        {
            *mode = miopen::deref(poolDesc).mode;
        }
        if(pmode != nullptr)
        {
            *pmode = miopen::deref(poolDesc).pmode;
        }

        if(nbDims != nullptr)
        {
            *nbDims = miopen::deref(poolDesc).GetSize();
        }
        if(windowDimA != nullptr)
        {
            std::copy(miopen::deref(poolDesc).GetLengths().begin(),
                      miopen::deref(poolDesc).GetLengths().end(),
                      windowDimA);
        }
        if(stridesA != nullptr)
        {
            std::copy(miopen::deref(poolDesc).GetStrides().begin(),
                      miopen::deref(poolDesc).GetStrides().end(),
                      stridesA);
        }
        if(padA != nullptr)
        {
            std::copy(miopen::deref(poolDesc).GetPads().begin(),
                      miopen::deref(poolDesc).GetPads().end(),
                      padA);
        }

    });
}

extern "C" miopenStatus_t
miopenGetPoolingForwardOutputDim(const miopenPoolingDescriptor_t poolDesc,
                                 const miopenTensorDescriptor_t tensorDesc,
                                 int* n,
                                 int* c,
                                 int* h,
                                 int* w)
{

    MIOPEN_LOG_FUNCTION(poolDesc, tensorDesc, n, c, h, w);
    return miopen::try_([&] {
        miopen::tie_deref(n, c, h, w) =
            miopen::deref(poolDesc).GetForwardOutputDim(miopen::deref(tensorDesc));
    });
}

// this should deprecate because it assume we are always using uint8_t for max pooling indexing
extern "C" miopenStatus_t miopenPoolingGetWorkSpaceSize(const miopenTensorDescriptor_t yDesc,
                                                        size_t* workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(yDesc, workSpaceSize);
    return miopen::try_([&] {
        auto len  = miopen::deref(yDesc).GetLengths();
        size_t sz = std::accumulate(len.begin(), len.end(), 1, std::multiplies<int>());
        miopen::deref(workSpaceSize) = sz * sizeof(uint8_t);
    });
}

extern "C" miopenStatus_t miopenPoolingGetWorkSpaceSizeV2(const miopenPoolingDescriptor_t poolDesc,
                                                          const miopenTensorDescriptor_t yDesc,
                                                          size_t* workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(poolDesc, yDesc, workSpaceSize);
    return miopen::try_(
        [&] { *workSpaceSize = miopen::deref(poolDesc).GetWorkSpaceSize(miopen::deref(yDesc)); });
}

extern "C" miopenStatus_t miopenPoolingForward(miopenHandle_t handle,
                                               const miopenPoolingDescriptor_t poolDesc,
                                               const void* alpha,
                                               const miopenTensorDescriptor_t xDesc,
                                               const void* x,
                                               const void* beta,
                                               const miopenTensorDescriptor_t yDesc,
                                               void* y,
                                               bool do_backward,
                                               void* workSpace,
                                               size_t workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(
        handle, poolDesc, alpha, xDesc, x, beta, yDesc, y, do_backward, workSpace, workSpaceSize);

    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        if(miopen::deref(xDesc).GetType() == miopenHalf)
        {
            ss << "poolfp16";
        }
        else
        {
            ss << "pool";
        }
        ss << " -n " << miopen::deref(xDesc).GetLengths()[0] // clang-format off
            << " -c " << miopen::deref(xDesc).GetLengths()[1]
            << " -H " << miopen::deref(xDesc).GetLengths()[2]
            << " -W " << miopen::deref(xDesc).GetLengths()[3]
            << " -y " << miopen::deref(poolDesc).lens[0]
            << " -x " << miopen::deref(poolDesc).lens[1]
            << " -p " << miopen::deref(poolDesc).pads[0]
            << " -q " << miopen::deref(poolDesc).pads[1]
            << " -u " << miopen::deref(poolDesc).strides[0]
            << " -v " << miopen::deref(poolDesc).strides[1]
            << " -m " << (miopen::deref(poolDesc).mode == 0 ? "max" : (miopen::deref(poolDesc).mode == 1 ? "avg" : "avg_in"))
            << " -t 1"; // clang-format on
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }

    return miopen::try_([&] {
        miopen::deref(poolDesc).Forward(miopen::deref(handle),
                                        alpha,
                                        miopen::deref(xDesc),
                                        DataCast(x),
                                        beta,
                                        miopen::deref(yDesc),
                                        DataCast(y),
                                        do_backward,
                                        DataCast(workSpace),
                                        workSpaceSize);
    });
}

extern "C" miopenStatus_t miopenPoolingBackward(miopenHandle_t handle,
                                                const miopenPoolingDescriptor_t poolDesc,
                                                const void* alpha,
                                                const miopenTensorDescriptor_t yDesc,
                                                const void* y,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const void* dy,
                                                const miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                const void* beta,
                                                const miopenTensorDescriptor_t dxDesc,
                                                void* dx,
                                                const void* workSpace)
{

    MIOPEN_LOG_FUNCTION(
        handle, poolDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx, workSpace);

    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        if(miopen::deref(xDesc).GetType() == miopenHalf)
        {
            ss << "poolfp16";
        }
        else
        {
            ss << "pool";
        }
        ss << " -n " << miopen::deref(xDesc).GetLengths()[0] // clang-format off
            << " -c " << miopen::deref(xDesc).GetLengths()[1]
            << " -H " << miopen::deref(xDesc).GetLengths()[2]
            << " -W " << miopen::deref(xDesc).GetLengths()[3]
            << " -y " << miopen::deref(poolDesc).lens[0]
            << " -x " << miopen::deref(poolDesc).lens[1]
            << " -p " << miopen::deref(poolDesc).pads[0]
            << " -q " << miopen::deref(poolDesc).pads[1]
            << " -u " << miopen::deref(poolDesc).strides[0]
            << " -v " << miopen::deref(poolDesc).strides[1]
            << " -m " << (miopen::deref(poolDesc).mode == 0 ? "max" : (miopen::deref(poolDesc).mode == 1 ? "avg" : "avg_in"))
            << " -t 1"; // clang-format on
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }

    return miopen::try_([&] {
        miopen::deref(poolDesc).Backward(miopen::deref(handle),
                                         alpha,
                                         miopen::deref(yDesc),
                                         DataCast(y),
                                         miopen::deref(dyDesc),
                                         DataCast(dy),
                                         miopen::deref(xDesc),
                                         DataCast(x),
                                         beta,
                                         miopen::deref(dxDesc),
                                         DataCast(dx),
                                         DataCast(workSpace));
    });
}

extern "C" miopenStatus_t miopenDestroyPoolingDescriptor(miopenPoolingDescriptor_t poolDesc)
{
    MIOPEN_LOG_FUNCTION(poolDesc);
    return miopen::try_([&] { miopen_destroy_object(poolDesc); });
}
