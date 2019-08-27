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

Compilation
-----------

First, set up some environment variables for convenience:
```
export ROCM_INSTALL_DIR=`pwd`/install
export HCC_HOME=$ROCM_INSTALL_DIR
export PATH=$PATH:$ROCM_INSTALL_DIR/bin
```

Build and install each project in the following order, making sure to install
each one to `ROCM_INSTALL_DIR`:

 - `ROCT-Thunk-Interface`
 - `ROCR-Runtime`
 - `ROC-smi`
 - `rocm-cmake`
 - `rocminfo`
 - `rocprofiler`
 - `ROCm-OpenCL-Runtime`
 - `clang-ocl`
 - `hcc`
 - `HIP`
 - `ROCm-Device-Libs`
 - `atmi`
 - `ROCm-CompilerSupport`
 - `rocr_debug_agent`
 - `rocm_bandwidth_test`
