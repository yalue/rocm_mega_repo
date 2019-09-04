About
=====

This contains all of the ROCm 2.6 source code, as downloaded by the repo tool.
The source code was downloaded as follows:
```
cd sources
./repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-2.6.0
./repo sync
```

Afterwards, I deleted all of the `.git` folders in each project. If re-running
this step, make sure to do it *before* creating your own git repo!
```
find . -type d -name *.git -exec rm -rf {} \;
```

Other notes on obtaining the source code:

 - I replaced the version of `llvm_amd-common` downloaded by `repo` with the
   trunk version from the ROCm repos (in addition to `clang` and `lld`). I put
   `clang` and `lld` sources in `llvm_amd-common/tools`, which also is not
   where `repo` initially downloaded them.

Compilation
-----------

You *should* be able to build everything using the included `install.sh`
script:

 1. Run `bash install.sh` and wait for it to finish (this may take hours).

 2. Add the `install/bin` directory to your `PATH` using whatever means you
    normally do.

 3. Make sure your system is able to find the dynamic libraries. On my
    Ubuntu 18.04 system, the "correct" way to do this is:
    ```
    echo `pwd`/install/lib | sudo tee -a /etc/ld.so.conf.d/ROCm.conf
    echo `pwd`/install/hsa/lib | sudo tee -a /etc/ld.so.conf.d/ROCm.conf
    sudo ldconfig
    ```
    If the above steps worked correctly, you should have a file at
    `/etc/ld.so.conf.d/ROCm.conf` containing two lines--a path to the
    `install/lib` directory, and a path to the `install/hsa/lib` directory.

If you don't want to use the script I included (and I wouldn't blame you--it's
very basic!), then you'll probably want to first, set up some environment
variables for convenience:
```
mkdir install
export ROCM_INSTALL_DIR=`pwd`/install
export HCC_HOME=$ROCM_INSTALL_DIR
export PATH=$PATH:$ROCM_INSTALL_DIR/bin
```

Build and install each project in the following order, making sure to install
each one to `$ROCM_INSTALL_DIR`:

 - `ROCT-Thunk-Interface`
 - `ROCR-Runtime`
 - `ROC-smi`
 - `rocm-cmake`
 - `rocminfo`
 - `rocprofiler`
 - `llvm_amd-common`
 - `ROCm-OpenCL-Runtime`
 - `clang-ocl`
 - `hcc`
 - `HIP`
 - `ROCm-Device-Libs`
 - `atmi`
 - `ROCm-CompilerSupport`
 - `rocr_debug_agent`
 - `rocm_bandwidth_test`
