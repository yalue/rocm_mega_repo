# This script builds and installs the modified version of HIP. Overwrites the
# existing HIP version. Will prompt for sudo access in order to install HIP.

check_install_error() {
	if [ $? -ne 0 ];
	then
		echo "Failed installing $1"
		exit 1
	fi
}

export PROJECT_TOP_DIR="$(pwd)"

cd sources
export ROCclr_DIR="$(readlink -f ROCclr)"
export OPENCL_DIR="$(readlink -f ROCm-OpenCL-Runtime)"
cd "$PROJECT_TOP_DIR"


echo -e "\nBuilding ROCR-Runtime\n"
cd sources/ROCR-Runtime/src
rm -rf build
mkdir build
cd build
cmake \
	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DCMAKE_INSTALL_PREFIX=/opt/rocm \
	..
make -j4
sudo make install
check_install_error "ROCR-Runtime"
cd "$PROJECT_TOP_DIR"


echo -e "\nBuilding ROCclr\n"
cd sources/ROCclr
rm -rf build
mkdir build
cd build
cmake \
	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DOPENCL_DIR="$OPENCL_DIR" \
	-DCMAKE_INSTALL_PREFIX=/opt/rocm/rocclr \
	..
# We don't need to "install" ROCclr, but only compile it. The HIP build will
# just look in the "ROCclr/build" directory.
make -j4
sudo make install
check_install_error "ROCclr"
cd "$PROJECT_TOP_DIR"


echo -e "\nInstalling HIP\n"
cd sources/HIP
rm -rf build
mkdir build
cd build
cmake \
	-DCMAKE_BUILD_TYPE=Release \
	-DHIP_COMPILER=clang \
	-DHIP_PLATFORM=rocclr \
	-DCMAKE_PREFIX_PATH="$ROCclr_DIR/build" \
	..
make -j4
sudo make install
check_install_error "HIP"
cd "$PROJECT_TOP_DIR"

# Make sure the ld.so knows if we've updated any shared libraries.
sudo ldconfig

