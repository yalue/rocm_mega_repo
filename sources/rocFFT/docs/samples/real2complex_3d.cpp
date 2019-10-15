// Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <cassert>
#include <complex>
#include <iostream>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "rocfft.h"

int main(int argc, char* argv[])
{
    std::cout << "rocFFT real/complex 3d FFT example\n";

    // The problem size
    const size_t Nx      = (argc < 2) ? 8 : atoi(argv[1]);
    const size_t Ny      = (argc < 3) ? 8 : atoi(argv[2]);
    const size_t Nz      = (argc < 4) ? 8 : atoi(argv[3]);
    const bool   inplace = (argc < 5) ? false : atoi(argv[4]);

    std::cout << "Nx: " << Nx << "\tNy: " << Ny << "\tNz: " << Nz << "\tin-place: " << inplace
              << std::endl;

    const size_t Nzcomplex = Nz / 2 + 1;
    const size_t Nzstride  = inplace ? 2 * Nzcomplex : Nz;
    std::cout << "Nzcomplex: " << Nzcomplex << "\tNzstride: " << Nzstride << std::endl;

    std::cout << "Input:\n";
    std::vector<float> cx(Nx * Ny * Nzstride);
    std::fill(cx.begin(), cx.end(), 0.0);
    for(size_t i = 0; i < Nx; ++i)
    {
        for(size_t j = 0; j < Ny; ++j)
        {
            for(size_t k = 0; k < Nz; ++k)
            {
                const size_t pos = i * Ny * Nzstride + j * Nzstride + k;
                cx[pos]          = i + j + k;
            }
        }
    }
    for(size_t i = 0; i < Nx; ++i)
    {
        for(size_t j = 0; j < Ny; ++j)
        {
            for(size_t k = 0; k < Nzstride; ++k)
            {
                const size_t pos = i * Ny * Nzstride + j * Nzstride + k;
                std::cout << cx[pos] << "  ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Output buffer
    std::vector<std::complex<float>> cy(Nx * Ny * Nzcomplex);

    // Create HIP device objects:
    float* x = NULL;
    hipMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));
    hipMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), hipMemcpyHostToDevice);
    float2* y = inplace ? (float2*)x : NULL;
    if(!inplace)
    {
        hipMalloc(&y, cy.size() * sizeof(decltype(cy)::value_type));
    }

    // Length are in reverse order because rocfft is column-major.
    const size_t lengths[3] = {Nz, Ny, Nx};

    rocfft_status status = rocfft_status_success;

    // Create plans
    rocfft_plan forward = NULL;
    status              = rocfft_plan_create(&forward,
                                inplace ? rocfft_placement_inplace : rocfft_placement_notinplace,
                                rocfft_transform_type_real_forward,
                                rocfft_precision_single,
                                3, // Dimensions
                                lengths, // lengths
                                1, // Number of transforms
                                NULL); // Description
    assert(status == rocfft_status_success);

    // The real-to-complex transform uses work memory, which is passed
    // via a rocfft_execution_info struct.
    rocfft_execution_info forwardinfo = NULL;
    status                            = rocfft_execution_info_create(&forwardinfo);
    assert(status == rocfft_status_success);
    size_t fbuffersize = 0;
    rocfft_plan_get_work_buffer_size(forward, &fbuffersize);
    assert(status == rocfft_status_success);
    void* fbuffer = NULL;
    hipMalloc(&fbuffer, fbuffersize);
    status = rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);
    assert(status == rocfft_status_success);

    // Execute the forward transform
    status = rocfft_execute(forward, // plan
                            (void**)&x, // in_buffer
                            (void**)&y, // out_buffer
                            forwardinfo); // execution info
    assert(status == rocfft_status_success);

    hipMemcpy(cy.data(), y, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);

    std::cout << "Transformed:\n";
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; ++j)
        {
            for(size_t k = 0; k < Nzcomplex; k++)
            {
                const size_t pos = (i * Ny + j) * Nzcomplex + k;
                std::cout << cy[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Create plans
    rocfft_plan backward = NULL;
    status               = rocfft_plan_create(&backward,
                                inplace ? rocfft_placement_inplace : rocfft_placement_notinplace,
                                rocfft_transform_type_real_inverse,
                                rocfft_precision_single,
                                3, // Dimensions
                                lengths, // lengths
                                1, // Number of transforms
                                NULL); // Description
    assert(status == rocfft_status_success);

    rocfft_execution_info backwardinfo = NULL;
    status                             = rocfft_execution_info_create(&backwardinfo);
    assert(status == rocfft_status_success);
    size_t bbuffersize = 0;
    status             = rocfft_plan_get_work_buffer_size(backward, &bbuffersize);
    assert(status == rocfft_status_success);
    void* bbuffer = NULL;
    hipMalloc(&bbuffer, bbuffersize);
    status = rocfft_execution_info_set_work_buffer(backwardinfo, bbuffer, bbuffersize);
    assert(status == rocfft_status_success);

    // Execute the backward transform
    status = rocfft_execute(backward, // plan
                            (void**)&y, // in_buffer
                            (void**)&x, // out_buffer
                            backwardinfo); // execution info
    assert(status == rocfft_status_success);

    std::cout << "Transformed back:\n";
    std::vector<float> backx(cx.size());
    hipMemcpy(
        backx.data(), x, backx.size() * sizeof(decltype(backx)::value_type), hipMemcpyDeviceToHost);

    for(size_t i = 0; i < Nx; ++i)
    {
        for(size_t j = 0; j < Ny; ++j)
        {
            for(size_t k = 0; k < Nzstride; ++k)
            {
                const size_t pos = i * Ny * Nzstride + j * Nzstride + k;
                std::cout << backx[pos] << "  ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    const float overN = 1.0f / (Nx * Ny * Nz);
    float       error = 0.0f;
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            for(size_t k = 0; k < Nzstride; k++)
            {
                float diff = std::abs(backx[i] * overN - cx[i]);
                if(diff > error)
                {
                    error = diff;
                }
            }
        }
    }
    std::cout << "Maximum error: " << error << "\n";

    hipFree(x);
    if(!inplace)
    {
        hipFree(y);
    }
    hipFree(fbuffer);
    hipFree(bbuffer);

    // Destroy plans
    rocfft_plan_destroy(forward);
    rocfft_plan_destroy(backward);

    rocfft_cleanup();
}
