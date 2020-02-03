About
=====

This directory contains a Python module that provides functions for carrying
out low-level functions for controlling ROCm.  For now, the only function
that's provided is `rocm_control.set_default_cu_mask(...)`.

Installation
------------

From within this directory, run `python setup.py install`.

Usage
-----

In your python code, run `import rocm_control`. More information can be seen
via `help(rocm_control)`.

