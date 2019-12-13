The ROCm Mega Repo
==================

This repository contains a copy of the entire ROCm stack, minus kernel
components, sufficient to run HIP programs. The objective of this is to
simplify the process of obtaining and compiling a standalone version of the
entire ROCm stack from source, with few to no changes requiring administrator
access.

About
-----

This contains all of the ROCm 2.9 source code, as downloaded by the repo tool.
The source code was downloaded as follows:
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
  libboost-program-options-dev
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

NOTE: This repository's install script assumes that you do *not* have an
existing ROCm installation. If you do, you may want to uninstall it (or
alternatively, temporarily rename the `/opt/rocm` directory so its utilities
won't be available) in case some of the newly compiled utilities end up with
incorrect paths or environment variables.

Compilation and Installation
----------------------------

You *should* be able to build everything using the included `install.sh`
script:

 1. Run `bash install.sh` and wait for it to finish (this may take hours). I'd
    recommend saving its output to a log file to help you diagnose if something
    goes wrong. I personally run the following command to log all stdout and
    stderr output to `install_log.txt` while `install.sh` is running:
    ```
    stdbuf -oL bash install.sh 2>&1 | tee install_log.txt
    ```

 2. Add several environment variables to your `.bashrc` or whatever way you use
    to set environment variables. If you use `.bashrc`, the following lines
    should be sufficient:
    ```
    # Replace <YOUR INSTALLATION DIRECTORY> with the full path to the `install`
    # directory created by install.sh.
    export ROCM_PATH=<YOUR INSTALLATION DIRECTORY>
    export HIP_PATH=$ROCM_PATH
    export HSA_PATH=$ROCM_PATH/hsa
    export HCC_HOME=$ROCM_PATH/hcc_home
    export PATH=$PATH:$ROCM_PATH/bin
    ```

 3. Make sure your system is able to find the dynamic libraries. Assuming you
    have set the above environment variables already, you can accomplish this
    by running the following commands (at least on Ubuntu 18.04):
    ```
    echo $ROCM_PATH/lib | sudo tee /etc/ld.so.conf.d/ROCm.conf
    echo $ROCM_PATH/hsa/lib | sudo tee -a /etc/ld.so.conf.d/ROCm.conf
    sudo ldconfig
    ```
    If the above steps worked correctly, you should have a file at
    `/etc/ld.so.conf.d/ROCm.conf` containing two lines--a path to the
    `<...>/install/lib` directory, and a path to the `<...>/install/hsa/lib`
    directories. Alternatively, you can add these paths to the
    `LD_LIBRARY_PATH` environment variable using the means of your choice, for
    example if you want to avoid requiring administrator access.

Uninstallation
--------------

Removing this version of ROCm just requires undoing the steps taken in the
installation setup described above:

 1. Delete the `install` directory. If you followed the steps above, this is as
    simple as `rm -rf $ROCM_PATH`.

 2. Remove these environment variable definitions from your `.bashrc` (or other
    environment configuration): `ROCM_PATH`, `HIP_PATH`, `HSA_PATH`, and
    `HCC_HOME`.

 3. If you followed the steps above, run `sudo rm /etc/ld.so.conf.d/ROCm.conf`,
    and then run `sudo ldconfig` to update the shared-library search paths.

