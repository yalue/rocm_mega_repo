Kernel Module to Support System-Wide GPU Locking in HIP
=======================================================

This directory contains a kernel module that is used in the implementation of
suspension-based locking to enable isolation on GPUs, or parts of GPUs. This
module is intended to be used by HIP code to enforce exclusive access to the
GPU (or partitions of the GPU).


Usage
-----

First, compile the module by running `make`.  Next, load the module:
`sudo insmod hip_locking_module.ko`.

