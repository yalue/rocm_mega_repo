Python Helper Library for ROCm
==============================

This defines a python library for providing some lower-level access to ROCm
stuff from within a python script. It requires Python 3, assumes `hipcc` is
on your path, and that `/opt/rocm` points to the correct ROCm installation.


Usage
-----

To use it, run `CC=hipcc python setup.py install` from within this directory.
Afterwards, in python:

```python
import rocm_helper

# To view specific usage options:
help(rocm_helper)

# Create a stream with a given CU mask.
hip_stream = rocm_helper.create_stream_with_cu_mask(0xffff0000, 0x0000ffff)

# etc

rocm_helper.destroy_stream(hip_stream)
```

