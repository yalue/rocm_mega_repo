The ROCm Mega Repo
==================

This repository is used to track modifications to several components of the
ROCm stack.


NOTE: DEPRECATION
=================

This repository was largely intended to expose an API for setting CU masks to
HIP, but this is now possible in the upstream version of HIP, as of ROCm 3.6.
[See the `hipExtStreamCreateWithCUMask` function](https://github.com/ROCm-Developer-Tools/HIP/blob/rocm-3.7.x/include/hip/hcc_detail/hip_runtime_api.h#L911).

So, this repository is unlikely to be of much use to most users, and will
instead mostly serve as a way for me to track my own experimental modifications
to HIP, ROCclr, and HSA code.


About
-----

This repository currently contains copies of the `ROCR-Runtime`, `HIP`, and
`ROCclr` code.  The source code is based on ROCm 3.7.

The main goal of this repository, for now, is to add a new function to the HIP
API: `hipStreamSetComputeUnitMask`, which enables control over the compute-unit
masking hardware feature of AMD GPUs (at least through the Polaris
architecture).  Adding this function requires a modified version of ROCclr and
HIP.

Using the `hipStreamSetComputeUnitMask` API is simple.  Its first argument is
just a `hipStream_t` instance to set the CU mask for, and the second argument
is a `uint64_t` mask (the API does not support hypothetical GPUs with more than
64 CUs, but AMD doesn't currently offer any such GPU). At least one CU must be
enabled in the mask (indicated by a '1' bit).  The `HIP_HAS_STREAM_SET_CU_MASK`
macro will be defined if this function is present, so an
`#ifdef HIP_HAS_STREAM_SET_CU_MASK` check can be used to write source code
compatible with the unmodified version of HIP, if necessary.


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
the `video` and possibly the `render` group.

Finally, you will need to install the following packages from AMD's pre-built
ROCm repositories AMD's repositories (see
[these instructions](https://github.com/RadeonOpenCompute/ROCm#Ubuntu) for
information about adding the repo):
```
sudo apt install rocm-dev3.7.0 rocm-libs3.7.0 rocm-utils3.7.0
```

After running the above command to install ROCm version 3.7.0, you will need to
create a symlink to link `/opt/rocm` to `/opt/rocm/3.7.0`.

Finally, make sure that `/opt/rocm/bin` is on your PATH, and that your system
looks for shared libraries in `/opt/rocm/lib`.

(Note that if you need to install additional ROCm libraries, always use the
package versions with names ending in `3.7.0`: mixing the `<...>3.7.0` packages
with the packages that don't have a version number EASILY leads to broken
installations.  (Speaking from experience.)


Compilation and Installation (on top of an existing ROCm installation)
----------------------------------------------------------------------

You *should* be able to build everything using the included `install.sh`
script: `bash ./install.sh`.  The script will prompt for `sudo` access at some
points in order to be able to write to `/opt/rocm`.


Uninstallation
--------------

Restoring an "original" version of ROCm will likely require re-installing the
packages overwritten by this project. (i.e. uninstalling all existing ROCm
packages, then deleting `/opt/rocm*`, and then reinstalling ROCm from AMD's
repositories.  USE THIS AT YOUR OWN RISK.  I have encountered many headaches
with AMD's installation process, though it sounds like they're working to
improve it with multi-version support.

