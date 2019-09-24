# This script rebuilds all of ROCm from source, installing it to a pre-existing
# ./install directory. Clean the directory manually before running this script.
#
# The following prerequisites must be installed on an Ubuntu 18.04 system in
# order for this script to work:
#    sudo apt install cmake pkg-config libpci-dev
# Additionally, the following packages *may* be needed--I'm not sure.
#    sudo apt install ocaml ocaml-findlib python-z3 git-svn
# Z3 may need to be installed from source on some older systems. There also may
# be ways to disable it manually, but I don't know if I've covered all of them.


export ROCM_INSTALL_DIR=`pwd`/install
mkdir -p $ROCM_INSTALL_DIR
export HCC_HOME=$ROCM_INSTALL_DIR
export PATH=$PATH:ROCM_INSTALL_DIR/bin

cd sources/ROCT-Thunk-Interface
rm -r build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR ..
make
make install
make install-dev
cd $ROCM_INSTALL_DIR/..

cd sources/ROCR-Runtime/src
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DHSAKMT_LIB_PATH:STRING=$ROCM_INSTALL_DIR/lib \
	-DHSAKMT_INC_PATH:STRING=$ROCM_INSTALL_DIR/include \
	..
make -j8
make install
cd $ROCM_INSTALL_DIR/..

# rocm-smi doesn't seem to need anything special
mkdir -p $ROCM_INSTALL_DIR/bin
cp sources/ROC-smi/rocm-smi $ROCM_INSTALL_DIR/bin/
cp sources/ROC-smi/rocm_smi.py $ROCM_INSTALL_DIR/bin/

cd sources/rocm-cmake
rm -r build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR ..
cmake --build . --target install
cd $ROCM_INSTALL_DIR/..

cd sources/rocminfo
mkdir build
cd build
cmake \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	..
make
make install
cd $ROCM_INSTALL_DIR/..

cd sources/rocprofiler
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR:$ROCM_INSTALL_DIR/include/hsa \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	..
make -j8
make install
cd $ROCM_INSTALL_DIR/..

# I don't think this is necessary...
#cd sources/llvm_amd-common
#rm -r build
#mkdir build
#cd build
#cmake \
#	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
#	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
#	..
## It's slow, but too many threads brought about memory trouble on my system
## with 16 GB of memory... Alternatively, run make -j12, then cold reboot your
## machine when it crashes near the end with the linker, then finish up the last
## bit of work with a make -j2 :)
#make -j2
#make -j8 install
#cd $ROCM_INSTALL_DIR/..

#cd sources/ROCm-OpenCL-Runtime
#echo "Need sudo to copy a file to /etc/OpenCL/vendors"
#sudo cp api/opencl/config/amdocl64.icd /etc/OpenCL/vendors
#rm -r build
#mkdir build
#cd build
#cmake \
#	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
#	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
#	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
#	-DCMAKE_LIBRARY_PATH=$ROCM_INSTALL_DIR/lib \
#	-DCMAKE_INCLUDE_PATH=$ROCM_INSTALL_DIR/include \
#	-DCLANG_ANALYZER_ENABLE_Z3_SOLVER=OFF \
#	..
#make -j8
#make install
#cd $ROCM_INSTALL_DIR/..
## clang-ocl expects the "clang" binary, not clang-9
#ln -s -T $ROCM_INSTALL_DIR/bin/x86_64/clang-9 $ROCM_INSTALL_DIR/bin/x86_64/clang

#cd sources/clang-ocl
#rm -r build
#mkdir build
#cd build
#cmake \
#	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
#	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
#	..
#make -j8
#make install
#cd $ROCM_INSTALL_DIR/..

cd sources/hcc
rm -r build
mkdir build
cd build
# The gfx803 should include the RX 570....
cmake \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DHSA_AMDGPU_GPU_TARGET="gfx803;gfx900" \
	..
make -j4
make -j4 install
cd $ROCM_INSTALL_DIR/..

# To reinstall HIP, we first need to make sure old versions are gone, otherwise
# the headers get included somehow.
rm -r $ROCM_INSTALL_DIR/include/hip
rm $ROCM_INSTALL_DIR/src/hip*
cd sources/HIP
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DHSA_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
	..
make -j8
make -j8 install
cd $ROCM_INSTALL_DIR/..
# Fix some hardcoded paths in the hipcc script
sed -i 's@/opt/rocm@'"$ROCM_INSTALL_DIR"'@g' $ROCM_INSTALL_DIR/bin/hipcc

# Steps for installing z3 from source:
#cd sources/libz3_48_master
#python scripts/mk_make.py
#cd build
#make -j8
#sudo make install
