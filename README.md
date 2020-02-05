The ROCm Mega Repo
==================

This repository contains a copy of the entire ROCm stack, minus kernel
components, sufficient to run HIP programs. The objective of this is to
simplify the process of obtaining and compiling a standalone version of the
entire ROCm stack from source, with few to no changes requiring administrator
access.

For the time being, I have excluded most of the ROCm code from the repository,
keeping only the components I have modified. The commit tagged with
`full_rocm_3.0_source` should contain the latest (mostly) working version of
the repository that attempts to compile all of ROCm from source in a local
directory.

About
-----

This contains the ROCm 3.0 source code, as downloaded by the repo tool. The
source code was downloaded as follows:
```
cd sources
./repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-2.9.0
./repo sync
```

Afterwards, I deleted all of the `.git` folders in each project. If re-running
this step, make sure to do it *before* creating your own git repo!
```
find . -type d -name *.git -exec rm -rf {} \;
```

I made a few small modifications to the files. There may be more of them, but
from what I remember, they are as follows:

 - I modified the `hipcc` script to change the LLVM path. Other hardcoded
   paths in this script will be updated by `install.sh`.

 - I changed some `CMakeLists.txt` files to specify C++14 explicitly.

Prerequisites
-------------

You will first need to have the following prerequisites installed (there may be
others that I'm forgetting):
```
sudo apt install cmake pkg-config libpci-dev libelf-dev liblapack-dev \
  libboost-program-options-dev libnuma-dev
```

You will need to make sure that you are running a sufficiently recent version
of the Linux kernel, supporting the `amdgpu` and `amdkfd` modules. Anything
past kernel version 5.0 should be fine. At the time of writing, you can enable
the AMD's `kfd` driver when compiling a bleeding-edge Linux kernel by enabling
the following options using `make menuconfig`:

From the top-level menu, select "Device Drivers", then "Graphics Support".
First make sure the "AMD GPU" option is enabled, and then make sure the "HSA
kernel driver for AMD GPU devices" is enabled. Save your configuration and
rebuild/reinstall the kernel.

Additionally, make sure you are running a version of the Linux kernel with the
`amdkfd` module available, that the `video` group has RW access to `/dev/kfd`,
and that your user is a member of the `video` group.

Finally, you will need to install the following packages from AMD's pre-built
ROCm repositories. AMD's repositories currently contain version 3.0 of ROCm--
the subsequent instructions in this README may not work if the versions in
AMD's repositories have updated. If that happens, the source code versions in
this repo's `sources/` directory will need to be updated to match the upstream
versions and re-patched. Assuming that ROCm is still at version 3.0, install
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
script: `bash ./install.sh`.  Note that the script will prompt for `sudo`
access at a few points, in order to write to `/opt/rocm`.

Uninstallation
--------------

Restoring an "original" version of ROCm will likely require re-installing the
packages overwritten by this project. However, at least in the case of HIP and
hcc, you may be able to restore old versions by retaining copies of the
original `/opt/rocm/hip` and `/opt/rocm/hcc` directories.

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
