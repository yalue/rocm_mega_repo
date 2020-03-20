# This script builds the modified ROCm components from source, and installs it
# to /opt/rocm, overwriting the binaries provided by AMD. Run at your own
# risk--if it fails you may end up with a broken ROCm installation and need to
# uninstall all of AMD's prebuilt packages, delete /opt/rocm, and re-install
# AMD's packages to get back to a sane state. That shouldn't happen, though--
# my changes are pretty minor. Make sure the version (currently ROCm 3.0)
# matches, though!

export ROCM_INSTALL_DIR=/opt/rocm
export HCC_HOME=$ROCM_INSTALL_DIR/hcc
export PATH=$PATH:ROCM_INSTALL_DIR/bin
export HIP_PATH=$ROCM_INSTALL_DIR/hip
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
sudo make install
check_install_error "ROCT-Thunk-Interface"
sudo make install-dev
check_install_error "ROCT-Thunk-Interface (dev)"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling ROCR-Runtime\n"
cd sources/ROCR-Runtime/src
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR/hsa \
	..
make -j8
sudo make install
check_install_error "ROCR-Runtime"
cd $ROCM_INSTALL_DIR/..

echo -e "\nInstalling hcc\n"
cd sources/hcc
rm -r build
mkdir build
cd build
cmake \
	-DCMAKE_PREFIX_PATH=$ROCM_INSTALL_DIR \
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR/hcc \
	-DCMAKE_BUILD_TYPE=Release \
	-DHSA_AMDGPU_GPU_TARGET="gfx803;gfx900" \
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
	-DCMAKE_INSTALL_PREFIX=$ROCM_INSTALL_DIR/hip \
	-DCMAKE_BUILD_TYPE=Release \
	-DHSA_AMDGPU_GPU_TARGET="gfx803;gfx900" \
	..
make -j8
make -j8 install
check_install_error "HIP"
cd $ROCM_INSTALL_DIR/..

# Make sure the ld.so knows if we've updated any shared libraries.
sudo ldconfig
