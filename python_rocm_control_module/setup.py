from distutils.core import setup, Extension

rocm_control_module = Extension("rocm_control", sources=["rocm_control.c"])
description = "Functions for extra control over the ROCm envieronment."
setup(name="ROCm Control", version="1.0", description=description,
    ext_modules=[rocm_control_module])

