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
export HIP_PATH=$ROCM_INSTALL_DIR
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
	-DROCM_DIR=$ROCM_INSTALL_DIR \
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

