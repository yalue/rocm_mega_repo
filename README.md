The ROCm Mega Repo
==================

This repository is used to track modifications to several components of the
ROCm stack.


About
-----

This repository currently contains copies of the `ROCR-Runtime`, `HIP`, and
`hcc` code, from ROCm version 3.3. However, only `ROCR-Runtime` and `HIP`
contain any modifications for now.

Prerequisites
-------------

You will first need to have the following prerequisites installed for compiling
ROCm (there may be others that I'm forgetting):
```
sudo apt install cmake pkg-config libpci-dev libelf-dev liblapack-dev \
  libboost-program-options-dev libnuma-dev
```

You will need to make sure that you are running a sufficiently recent version
of the Linux kernel, supporting the `amdgpu` and `amdkfd` modules. Anything
past kernel version 5.0 should be fine. Additionally, make sure you are running
a version of the Linux kernel with the `amdkfd` module available, that the
`video` group has RW access to `/dev/kfd`, and that your user is a member of
the `video` group.

Finally, you will need to install the following packages from AMD's pre-built
ROCm repositories. AMD's repositories currently contain version 3.3 of ROCm--
the subsequent instructions in this README may not work if the versions in
AMD's repositories have updated. If that happens, the source code versions in
this repo's `sources/` directory will need to be updated to match the upstream
versions and re-patched. Assuming that ROCm is still at version 3.3, install
the following packages from AMD's repository (see
[these instructions](https://github.com/RadeonOpenCompute/ROCm#Ubuntu) for
doing so):
```
sudo apt install rocm-utils rocm-libs rocm-dev rocm-debug-agent rocm-cmake \
    rocalution rocblas rocfft rocprim rocrand rocsparse rocthrust roctracer-dev
```


Compilation and Installation (on top of an existing ROCm installation)
----------------------------------------------------------------------

You *should* be able to build everything using the included `install.sh`
script: `sudo bash ./install.sh`.  The script requires `sudo` access in order
to be able to write to `/opt/rocm`.


Note about libhsa-runtime64.so
------------------------------

You may need to manually delete any existing versions of the libraries
overwritten by our modified version of `ROCR-Runtime`, as I don't think the
installation script in its current state overwrites them properly. To do so,
find and delete any copies of the `libhsa-runtime64.so*` you can find in any
subdirectories of `/opt/rocm`.  Next, compile and install the `ROCR-Runtime`
modifications using the `./install.sh` script mentioned above. Finally, make
sure to run `sudo ldconfig`.


Uninstallation
--------------

Restoring an "original" version of ROCm will likely require re-installing the
packages overwritten by this project.


Kernel Modification and Python Module
-------------------------------------

This repository assumes the user is already running an up-to-date version of
the Linux kernel with the `amdkfd` (and related) drivers enabled. The
`python_rocm_control_module` directory contains a kernel patch to enable a
`set_default_cu_mask` ioctl to the `/dev/kfd` character device, and a python
module that can be used to call the function from within python code (I had
used it for some early PyTorch tests). In the unlikely case that somebody wants
to use this code, more instructions can be found in the README in that
subdirectory.
