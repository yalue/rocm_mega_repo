/* Copyright (c) 2019-present Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#pragma once

/** \file Support for invoking host services from the device.
 *
 *  A hostcall is a fixed-size request generated by a kernel running
 *  on the device, for some predefined service provided by the
 *  host. The life-cycle of a hostcall is as follows:
 *
 *  1. A workitem in the some kernel dispatch submits a request as a
 *     "packet" in a "hostcall buffer". The workitem blocks until it
 *     receives a response from the host.
 *
 *  2. A host thread called the "hostcall listener" notices the packet
 *     and invokes the desired service on the host.
 *
 *  3. When the service completes, the listener copies the response
 *     into the request packet. This unblocks the workitem, and the
 *     hostcall is said to be completed.
 *
 *  The hostcall listeners and buffers are managed by the VDI
 *  runtime. The typical flow is as follows:
 *
 *  - Create and launch one or more hostcall listeners.
 *
 *  - Create and initialize a distinct hostcall buffer for each
 *    command queue in hardware (e.g., an hsa_queue_t on ROCm).
 *
 *  - Register this buffer with the appropriate listener.
 *
 *  - When a buffer is no longer used, deregister and then free
 *    it. This usually happens when the corresponding hardware queue
 *    is freed.
 *
 *  - Destroy the listener(s) when they are no longer required. This must be
 *    done before exiting the application, so that the listener
 *    threads can join() correctly.
 *
 *  A single listener is sufficient to correctly handle all hostcall
 *  buffers created in the application. The client may also launch
 *  multiple listeners, as long the same hostcall buffer is not
 *  registered with multiple listeners.
 */

/** \brief Determine the buffer size to be allocated
 *  \param num_packets Number of packets to be supported.
 *  \return Required size, including any internal padding required for
 *          the packets and their headers.
 */
size_t getHostcallBufferSize(uint32_t num_packets);

/** \brief Return the required alignment for a hostcall buffer.
 */
uint32_t getHostcallBufferAlignment(void);

bool enableHostcalls(void* buffer, uint32_t numPackets);
void disableHostcalls(void* buffer);
