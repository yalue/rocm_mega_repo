# NOTE: I no longer use this script, since I found it simpler to just install
# ROCm from AMD's repositories, then overwrite components with versions I
# compiled from source. However, I'm keeping this script around so that I don't
# forget the compilation options that (at least partially) worked at some
# point.
#
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
export LLVM_DIR=$ROCM_INSTALL_DIR/llvm
mkdir -p $ROCM_INSTALL_DIR
mkdir -p $LLVM_DIR
export HCC_HOME=$ROCM_INSTALL_DIR
export PATH=$PATH:ROCM_INSTALL_DIR/bin
export HIP_PATH=$ROCM_INSTALL_DIR
export ROCM_PATH=$ROCM_INSTALL_DIR

check_install_error() {
	if [ $? -ne 0 ];
	then
		echo "Failed installing $1"
		exit 1
	fi
}

echo -e "\nInstalling ROCT-Thunk-Interface\n"
cd sources/ROCT-Thunk-Interface
rm -r build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR ..
make
make install
check_install_error "ROCT-Thunk-Interface"
make install-dev
check_install_error "ROCT-Thunk-Interface (dev)"
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
check_install_error "ROCR-Runtime"
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
check_install_error "rocm-cmake"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling rocminfo\n"
cd sources/rocminfo
rm -r build
mkdir build
cd build
cmake \
	-DROCM_DIR=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	..
make
make install
check_install_error "rocminfo"
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
check_install_error "rocprofiler"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling LLVM_amd-stg-open\n"
cd sources/llvm_amd-stg-open
rm -r build
mkdir build
cd build
cmake \
	-DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld;compiler-rt;libclc" \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$LLVM_DIR \
	-DCMAKE_BUILD_TYPE=Release \
	-DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
	../llvm
make -j6
make -j6 install
check_install_error "LLVM_amd-stg-open"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling ROCm-Device-Libs\n"
cd sources/ROCm-Device-Libs
rm -r build
mkdir build
cd build
CC=$ROCM_INSTALL_DIR/llvm/bin/clang cmake \
	-DCMAKE_PREFIX_PATH="$ROCM_INSTALL_DIR;$ROCM_INSTALL_DIR/llvm" \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DLLVM_DIR=$ROCM_INSTALL_DIR/llvm \
	-DCMAKE_BUILD_TYPE=Release \
	..
make -j8
make -j8 install
check_install_error "ROCm-Device-Libs"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling comgr (code object manager)\n"
cd sources/ROCm-CompilerSupport/lib/comgr
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_PREFIX_PATH="$ROCM_INSTALL_DIR;$ROCM_INSTALL_DIR/llvm" \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DCMAKE_BUILD_TYPE=Release \
	..
make -j8
make -j8 install
check_install_error "comgr"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling hcc\n"
cd sources/hcc
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DCMAKE_BUILD_TYPE=Release \
	-DHSA_AMDGPU_GPU_TARGET="gfx803;gfx900" \
	-DROCM_ROOT=$ROCM_INSTALL_DIR \
	-DROCM_INSTALL_PATH=$ROCM_INSTALL_DIR \
	..
make -j6
make -j6 install
check_install_error "hcc"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling HIP\n"
cd sources/HIP
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_PREFIX_PATH="$ROCM_INSTALL_DIR" \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DHSA_PATH=$ROCM_INSTALL_DIR \
	-DHCC_HOME=$ROCM_INSTALL_DIR \
	-DDEVICE_LIB_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_BUILD_TYPE=Release \
	-DHSA_AMDGPU_GPU_TARGET="gfx803;gfx900" \
	..
make -j8
make -j8 install
check_install_error "HIP"
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
	-DCMAKE_PREFIX_PATH="$ROCM_INSTALL_DIR;$ROCM_INSTALL_DIR/llvm" \
	..
make
make install
check_install_error "rocRAND"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling rocBLAS\n"
cd sources/rocBLAS
rm -r build
mkdir -p build/release
cd build/release
# NOTE: This cmake command may take a while to complete--it takes about 10
# minutes on my machine.
CXX=hipcc cmake \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DCMAKE_PREFIX_PATH="$ROCM_INSTALL_DIR" \
	-DTensile_LOGIC=hip_lite \
	-DTensile_COMPILER=hcc \
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
check_install_error "rocBLAS"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling clang-ocl\n"
cd sources/clang-ocl
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DCMAKE_PREFIX_PATH="$ROCM_INSTALL_DIR;$ROCM_INSTALL_DIR/llvm" \
	-DOPENCL_ROOT=$ROCM_INSTALL_DIR \
	..
make
make install
check_install_error "clang-ocl"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling MIOpen\n"
cd sources/MIOpen
cmake -P install_deps.cmake --minimum --prefix $ROCM_INSTALL_DIR
rm -r build
mkdir build
cd build
CXX=$ROCM_INSTALL_DIR/bin/hcc \
	cmake \
	-DMIOPEN_BACKEND=HIP \
	-DCMAKE_PREFIX_PATH="$ROCM_INSTALL_DIR;$ROCM_INSTALL_DIR/llvm" \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	..
make -j4
make -j4 install
check_install_error "MIOpen"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling rocFFT\n"
cd sources/rocFFT
rm -r build
mkdir build
cd build
CXX=$ROCM_INSTALL_DIR/bin/hcc \
	cmake \
	-DCMAKE_PREFIX_PATH="$ROCM_INSTALL_DIR" \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DCMAKE_BUILD_TYPE=Release \
	..
make -j6
make install
check_install_error "rocFFT"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling rocPRIM\n"
cd sources/rocPRIM
rm -r build
mkdir build
cd build
cmake \
	-DHIP_PLATFORM=hcc \
	-DCMAKE_CXX_COMPILER=$ROCM_INSTALL_DIR/bin/hcc \
	-DBUILD_TEST=OFF \
	-DBUILD_BENCHMARK=OFF \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DCMAKE_BUILD_TYPE=Release \
	..
make
make install
check_install_error "rocPRIM"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling rocSPARSE\n"
cd sources/rocSPARSE
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_CXX_COMPILER=$ROCM_INSTALL_DIR/bin/hcc \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DBUILD_CLIENTS_TESTS=OFF \
	-DBUILD_CLIENTS_BENCHMARKS=OFF \
	-DBUILD_CLIENTS_SAMPLES=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	..
make -j6
make install
check_install_error "rocSPARSE"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling hipSPARSE\n"
cd sources/hipSPARSE
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DBUILD_CLIENTS_SAMPLES=OFF \
	-DBUILD_CLIENTS_TESTS=OFF \
	..
make
make install
check_install_error "hipSPARSE"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling hipCUB\n"
cd sources/hipCUB
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_CXX_COMPILER=$ROCM_INSTALL_DIR/bin/hcc \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DHIP_PLATFORM=hcc \
	-Drocprim_DIR=$ROCM_INSTALL_DIR \
	-DBUILD_TEST=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	..
make -j6
make -j6 install
check_install_error "hipCUB"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling rccl\n"
cd sources/rccl
rm -r build
mkdir build
cd build
CXX=hcc cmake \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_TESTS=OFF \
	..
make -j6
make -j6 install
check_install_error "rccl"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling rocThrust\n"
cd sources/rocThrust
rm -r build
mkdir build
cd build
CXX=hcc cmake \
	-DHIP_PLATFORM=hcc \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	-DBUILD_TEST=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DROCPRIM_ROOT=$ROCM_INSTALL_DIR \
	..
make
make install
check_install_error "rocThrust"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling roctracer\n"
cd sources/roctracer
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_MODULE_PATH=$ROCM_INSTALL_DIR/roctracer/cmake_modules \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR \
	..
make -j6
make -j6 install
check_install_error "roctracer"
cd $ROCM_INSTALL_DIR/..

echo "Installation complete!"
echo ""
echo "After checking for errors, make sure the following environment variables"
echo "are set in your ~/.bashrc or similar configuration files."
echo ""
echo "export ROCM_PATH=$ROCM_INSTALL_DIR"
echo "export HIP_PATH=\$ROCM_PATH"
echo "export HSA_PATH=\$ROCM_PATH/hsa"
echo "export HCC_HOME=\$ROCM_PATH"
echo "export PATH=\$PATH:\$ROCM_PATH/bin"
echo ""
echo "Additionally, you need to add a couple dynamic library directories."
echo "On Ubuntu, run the following commands to accomplish this:"
echo ""
echo "echo \"$ROCM_INSTALL_DIR/lib\" | sudo tee /etc/ld.so.conf.d/ROCm.conf"
echo "echo \"$ROCM_INSTALL_DIR/hsa/lib\" | sudo tee -a /etc/ld.so.conf.d/ROCm.conf"
echo "sudo ldconfig"
echo ""

