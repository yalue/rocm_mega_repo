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

#include "hipfft.h"
#include <fftw3.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <iostream>
#include <math.h>
#include <vector>

int main()
{
    int v;
    hipfftGetVersion(&v);
    std::cout << "hipFFT version " << v << std::endl;

    // For size N <= 4096
    const size_t N = 16;

    // FFTW reference compute
    // ==========================================

    std::vector<float2> cx(N);
    fftwf_complex *     in, *out;
    fftwf_plan          p;

    in  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N);
    out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N);
    p   = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for(size_t i = 0; i < N; i++)
    {
        cx[i].x = in[i][0] = i + (i % 3) - (i % 7);
        cx[i].y = in[i][1] = 0;
    }

    fftwf_execute(p);

    // hipfft gpu compute
    // ========================================

    size_t Nbytes = N * sizeof(float2);

    // Create HIP device object.
    hipfftComplex* x;
    hipMalloc(&x, Nbytes);

    //  Copy data to device
    hipMemcpy(x, &cx[0], Nbytes, hipMemcpyHostToDevice);

    // Create plan
    hipfftHandle plan   = NULL;
    size_t       length = N;
    hipfftPlan1d(&plan, length, HIPFFT_C2C, 1);

    // Execute plan
    hipfftExecC2C(plan, x, x, HIPFFT_FORWARD);

    // Destroy plan
    hipfftDestroy(plan);

    // Copy result back to host
    std::vector<float2> y(N);
    hipMemcpy(&y[0], x, Nbytes, hipMemcpyDeviceToHost);

    double error      = 0;
    size_t element_id = 0;
    for(size_t i = 0; i < N; i++)
    {
        printf("element %d: input %f, %f; FFTW result %f, %f; hipFFT result %f, %f \n",
               (int)i,
               cx[i].x,
               cx[i].y,
               out[i][0],
               out[i][1],
               y[i].x,
               y[i].y);
        double err = fabs(out[i][0] - y[i].x) + fabs(out[i][1] - y[i].y);
        if(err > error)
        {
            error      = err;
            element_id = i;
        }
    }

    printf("max error of FFTW and hipFFT is %e at element %d\n",
           error / (fabs(out[element_id][0]) + fabs(out[element_id][1])),
           (int)element_id);

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);

    hipFree(x);

    return 0;
}
