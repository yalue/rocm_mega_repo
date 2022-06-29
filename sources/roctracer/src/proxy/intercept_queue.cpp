/******************************************************************************
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
*******************************************************************************/

#include "proxy/intercept_queue.h"

namespace rocprofiler {
void InterceptQueue::HsaIntercept(HsaApiTable* table) {
  table->core_->hsa_queue_create_fn = rocprofiler::InterceptQueue::QueueCreate;
  table->core_->hsa_queue_destroy_fn = rocprofiler::InterceptQueue::QueueDestroy;
}

InterceptQueue::mutex_t InterceptQueue::mutex_;
//rocprofiler_callback_t InterceptQueue::dispatch_callback_ = NULL;
//InterceptQueue::queue_callback_t InterceptQueue::create_callback_ = NULL;
//InterceptQueue::queue_callback_t InterceptQueue::destroy_callback_ = NULL;
//void* InterceptQueue::callback_data_ = NULL;
InterceptQueue::obj_map_t* InterceptQueue::obj_map_ = NULL;
const char* InterceptQueue::kernel_none_ = "";
bool InterceptQueue::in_create_call_ = false;
InterceptQueue::queue_id_t InterceptQueue::current_queue_id = 0;
bool InterceptQueue::is_enabled = false;

}  // namespace rocprofiler
