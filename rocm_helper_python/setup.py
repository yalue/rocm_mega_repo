from distutils.core import setup, Extension

rocm_helper = Extension("rocm_helper",
    sources=["rocm_helper.cpp"],
    runtime_library_dirs=["/opt/rocm/lib", "/opt/rocm/hip/lib"],
    library_dirs=["/opt/rocm/lib", "/opt/rocm/hip/lib",
        "/opt/rocm/roctracer/lib"],
    libraries=["amdhip64", "roctracer64", "roctx64"],
    include_dirs=["/opt/rocm/include", "/opt/rocm/include/roctracer"]
)

description = "Some utilities for interacting with ROCm from python."
setup(name="ROCm Helper", version="1.0", description=description,
    ext_modules=[rocm_helper])

