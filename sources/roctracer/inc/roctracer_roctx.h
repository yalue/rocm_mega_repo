/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

////////////////////////////////////////////////////////////////////////////////
//
// ROC-TX API
//
// ROC-TX library, Code Annotation API.
// The goal of the implementation is to provide functionality for annotating
// events, code ranges, and resources in applications.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef INC_ROCTRACER_ROCTX_H_
#define INC_ROCTRACER_ROCTX_H_

#include "cb_table.h"

// ROC-TX API ID enumeration
enum roctx_api_id_t {
  ROCTX_API_ID_roctxMarkA = 0,
  ROCTX_API_ID_roctxRangePushA = 1,
  ROCTX_API_ID_roctxRangePop = 2,

  ROCTX_API_ID_NUMBER,
};

// ROCTX callbacks data type
struct roctx_api_data_t {
  union {
    const char* message;
    struct {
      const char* message;
    } roctxMarkA;
    struct {
      const char* message;
    } roctxRangePushA;
    struct {
      const char* message;
    } roctxRangePop;
  } args;
};

namespace roctx {

// ROCTX callbacks table type
typedef roctracer::CbTable<ROCTX_API_ID_NUMBER> cb_table_t;

}  // namespace roctx

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Regiter ROCTX callback for given opertaion id
bool RegisterApiCallback(uint32_t op, void* callback, void* arg);

// Remove ROCTX callback for given opertaion id
bool RemoveApiCallback(uint32_t op);

// Iterate range stack to support tracing start/stop
typedef struct {
  const char* message;
  uint32_t tid;
} roctx_range_data_t;
typedef void (*roctx_range_iterate_cb_t)(const roctx_range_data_t* data, void* arg);
void RangeStackIterate(roctx_range_iterate_cb_t callback, void* arg);

#ifdef __cplusplus
}  // extern "C" block
#endif  // __cplusplus

#endif  // INC_ROCTRACER_ROCTX_H_
