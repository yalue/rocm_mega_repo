About
=====

This directory contains a Python module that provides functions for carrying
out low-level functions for controlling ROCm.  For now, the only function
that's provided is `rocm_control.set_default_cu_mask(...)`.

Prerequisites
-------------

This module is intended to work with the quick-and-dirty kernel hack I added to
enable a `set_default_cu_mask` ioctl to AMD's `amdkfd` driver.  The patch was
generated on version 5.4.0-rc8+ of the Linux kernel (commit c74386d50fbaf4a5).
So, apply the patch, and build and install the kernel before attempting to use
this module.

This patch does *not* constitute what I'd consider particularly clean code, as
it was only a proof-of-concept.  I will update it in the future if it appears
to offer more utility that can't be obtained more easily (for my purposes) via
modifications to HIP.

Installation
------------

From within this directory, run `python setup.py install`.

Usage
-----

In your python code, run `import rocm_control`. More information can be seen
via `help(rocm_control)`.

