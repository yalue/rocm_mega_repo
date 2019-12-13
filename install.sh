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
export HCC_HOME=$ROCM_INSTALL_DIR/hcc_home
mkdir -p $HCC_HOME
export PATH=$PATH:ROCM_INSTALL_DIR/bin
export HIP_PATH=$ROCM_INSTALL_DIR
export ROCM_PATH=$ROCM_INSTALL_DIR

echo -e "\nInstalling ROCT-Thunk-Interface\n"
cd sources/ROCT-Thunk-Interface
rm -r build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR ..
make
make install
make install-dev
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling ROCR-Runtime\n"
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

echo -e "\nInstalling rocm-smi\n"
# rocm-smi doesn't seem to need anything special
mkdir -p $ROCM_INSTALL_DIR/bin
cp sources/ROC-smi/rocm-smi $ROCM_INSTALL_DIR/bin/
cp sources/ROC-smi/rocm_smi.py $ROCM_INSTALL_DIR/bin/

echo -e "\nInstalling rocm-cmake\n"
cd sources/rocm-cmake
rm -r build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR ..
cmake --build . --target install
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling rocminfo\n"
cd sources/rocminfo
mkdir build
cd build
cmake \
	-DROCM_DIR=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	..
make
make install
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling rocprofiler\n"
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

echo -e "\nInstalling LLVM_amd-common\n"
cd sources/llvm_amd-common
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DCMAKE_BUILD_TYPE=Release \
	-DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
	..
make -j6
make -j6 install
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling ROCm-Device-Libs\n"
cd sources/ROCm-Device-Libs
rm -r build
mkdir build
cd build
CC=$ROCM_INSTALL_DIR/bin/clang cmake \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DLLVM_DIR=$ROCM_INSTALL_DIR \
	-DCMAKE_BUILD_TYPE=Release \
	..
make -j8
make -j8 install
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling comgr (code object manager)\n"
cd sources/ROCm-CompilerSupport/lib/comgr
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DCMAKE_BUILD_TYPE=Release \
	..
make -j8
make -j8 install
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling hcc\n"
cd sources/hcc
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$HCC_HOME \
	-DCMAKE_BUILD_TYPE=Release \
	-DHSA_AMDGPU_GPU_TARGET="gfx803;gfx900" \
	..
make -j6
make -j6 install
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling HIP\n"
cd sources/HIP
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DHSA_PATH=$ROCM_INSTALL_DIR \
	-DHIP_CLANG_PATH=$ROCM_INSTALL_DIR/bin \
	-DDEVICE_LIB_PATH=$ROCM_INSTALL_DIR \
	-DHIP_COMPILER=clang \
	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
	..
make -j8
make -j8 install
cd $ROCM_INSTALL_DIR/..
# Fix some hardcoded paths in the hipcc script.
sed -i 's@/opt/rocm@'"$ROCM_INSTALL_DIR"'@g' $ROCM_INSTALL_DIR/bin/hipcc

echo -e "\nInstalling rocRAND\n"
cd sources/rocRAND
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DHIP_PLATFORM=hcc \
	-DBUILD_TEST=OFF \
	-DCMAKE_CXX_COMPILER=$ROCM_INSTALL_DIR/bin/hipcc \
	..
make
make install
cd $ROCM_INSTALL_DIR/..

cd sources/rocBLAS
rm -r build
mkdir -p build/release
cd build/release
# NOTE: This cmake command may take a while to complete--it takes about 10
# minutes on my machine.
CXX=$ROCM_INSTALL_DIR/hcc_home/bin/hcc \
TENSILE_ROCM_ASSEMBLER_PATH=$ROCM_INSTALL_DIR/hcc_home/bin/hcc \
	cmake \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DTensile_LOGIC=hip_lite \
	-DTensile_COMPILER=hipcc \
	-DTensile_CODE_OBJECT_VERSION=V2 \
	-DCMAKE_BUILD_TYPE=Release \
	../..
# I modified a script in rocBLAS so that it uses a ROCM_PATH environment
# variable rather than a hardcoded path to /opt/rocm. Also be aware that this
# compilation may be very memory-hungry; one of the Tensile files is auto-
# generated and includes hundreds of thousands to millions of lines of code,
# depending on if you modified the above cmake command.
ROCM_PATH=$ROCM_INSTALL_DIR make
make install
cd $ROCM_INSTALL_DIR/..

echo "Installation complete!"
echo ""
echo "After checking for errors, make sure the following environment variables"
echo "are set in your ~/.bashrc or similar configuration files."
echo ""
echo "export ROCM_PATH=$ROCM_INSTALL_DIR"
echo "export HIP_PATH=\$ROCM_PATH"
echo "export HSA_PATH=\$ROCM_PATH/hsa"
echo "export HCC_HOME=\$ROCM_PATH/hcc_home"
echo "export PATH=\$PATH:\$ROCM_PATH/bin"
echo ""
echo "Additionally, you need to add a couple dynamic library directories."
echo "On Ubuntu, run the following commands to accomplish this:"
echo ""
echo "echo \"$ROCM_INSTALL_DIR/lib\" | sudo tee /etc/ld.so.conf.d/ROCm.conf"
echo "echo \"$ROCM_INSTALL_DIR/hsa/lib\" | sudo tee -a /etc/ld.so.conf.d/ROCm.conf"
echo "sudo ldconfig"
echo ""

