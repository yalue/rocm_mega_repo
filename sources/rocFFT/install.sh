#!/usr/bin/env bash
# #############################################################################
# Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# #############################################################################


# #################################################
# helper functions
# #################################################

function display_help()
{
    echo "rocfft build & installation helper script"
    echo "./install [-h|--help] "
    echo "    [-h|--help] prints this help message"
    echo "    [--prefix] Specify an alternate CMAKE_INSTALL_PREFIX for cmake"
    echo "    [-i|--install] install via package manger after build"
    echo "    [-d|--dependencies] install build dependencies"
    echo "    [-c|--clients] build library clients too (combines with -i & -d)"
    echo "    [-g|--debug] -DCMAKE_BUILD_TYPE=Debug (default is =Release)"
    echo "    [--cuda] build library for cuda backend"
    echo "    [--hip-clang] build library for amdgpu backend using hip-clang"
}

# This function is helpful for dockerfiles that do not have sudo installed, but the
# default user is root true is a system command that completes successfully, function
# returns success prereq: ${ID} must be defined before calling
supported_distro( )
{
    if [ -z ${ID+foo} ]; then
        printf "supported_distro(): \$ID must be set\n"
        exit 2
    fi

    case "${ID}" in
        ubuntu|centos|rhel|fedora|sles)
            true
            ;;
        *)  printf "This script is currently supported on Ubuntu, CentOS, RHEL, Fedora, and SLES\n"
            exit 2
            ;;
    esac
}

# This function is helpful for dockerfiles that do not have sudo installed, but the
# default user is root
check_exit_code( )
{
    if (( $? != 0 )); then
        exit $?
    fi
}

# Check if the given proposed directory is contained in an existing
# directory to which the user has write access.
# Returns 0 if the directory is ok.
check_install_dir() {
    dir=$1
    lasdir=""
    indir_is_ok=1
    while [[ ${dir} != ${lastdir}  ]]
    do
	#echo ${dir}
        if [ -d "${dir}" ]; then
            if [ -w ${dir} ]; then
                indir_is_ok=0
            fi
            break
        fi
        dir="$(dirname "$dir")"
    done
    return ${indir_is_ok}
}

# This function is helpful for dockerfiles that do not have sudo installed, but the
# default user is root
elevate_if_not_root( )
{
    local uid=$(id -u)

    if (( ${uid} )); then
        sudo $@
        check_exit_code
    else
        $@
        check_exit_code
    fi
}

# Take an array of packages as input, and install those packages with 'apt' if they are
# not already installed
install_apt_packages( )
{
    package_dependencies=("$@")
    for package in "${package_dependencies[@]}"; do
        if [[ $(dpkg-query --show --showformat='${db:Status-Abbrev}\n' ${package} 2> /dev/null | grep -q "ii"; echo $?) -ne 0 ]]; then
            printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
            elevate_if_not_root apt install -y --no-install-recommends ${package}
        fi
    done
}

# Take an array of packages as input, and install those packages with 'yum' if they are
# not already installed
install_yum_packages( )
{
    package_dependencies=("$@")
    for package in "${package_dependencies[@]}"; do
        if [[ $(yum list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
            printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
            elevate_if_not_root yum -y --nogpgcheck install ${package}
        fi
    done
}

# Take an array of packages as input, and install those packages with 'dnf' if they are
# not already installed
install_dnf_packages( )
{
    package_dependencies=("$@")
    for package in "${package_dependencies[@]}"; do
        if [[ $(dnf list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
            printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
            elevate_if_not_root dnf install -y ${package}
        fi
    done
}

# Take an array of packages as input, and install those packages with 'zypper' if they are
# not already installed
install_zypper_packages( )
{
    package_dependencies=("$@")
    for package in "${package_dependencies[@]}"; do
        if [[ $(yum list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
            printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
            elevate_if_not_root zypper -n --no-gpg-checks install ${package}
        fi
    done
}

# Take an array of packages as input, and delegate the work to the appropriate distro
# installer prereq: ${ID} must be defined before calling prereq: ${build_clients} must be
# defined before calling
install_packages( )
{
    if [ -z ${ID+foo} ]; then
        printf "install_packages(): \$ID must be set\n"
        exit 2
    fi

    if [ -z ${build_clients+foo} ]; then
        printf "install_packages(): \$build_clients must be set\n"
        exit 2
    fi

    # dependencies needed for rocfft and clients to build
    local library_dependencies_ubuntu=( "make" "cmake-curses-gui" "rocm-dev" "pkg-config" )
    local library_dependencies_centos=( "epel-release" "make" "cmake3" "rocm-dev" "gcc-c++" "rpm-build" )
    local library_dependencies_fedora=( "make" "cmake" "rocm-dev" "gcc-c++" "libcxx-devel" "rpm-build" )
    local library_dependencies_sles=( "make" "cmake" "rocm-dev" "gcc-c++" "gcc-fortran" "libcxxtools9" "rpm-build" )

    if [[ "${build_cuda}" == true ]]; then
        # Ideally, this could be cuda-cufft-dev, but the package name has a version number in it
        library_dependencies_ubuntu+=( "cuda" )
        library_dependencies_centos+=( "" ) # how to install cuda on centos?
        library_dependencies_fedora+=( "" ) # how to install cuda on fedora?
        library_dependencies_sles+=( "" ) # how to install cuda on fedora?
    fi

    local client_dependencies_ubuntu=( "libfftw3-dev" "libboost-program-options-dev" )
    local client_dependencies_centos=( "fftw-devel" "boost-devel" )
    local client_dependencies_fedora=( "fftw-devel" "boost-devel" )
    local client_dependencies_sles=( "fftw3-devel" "libboost_program_options1_66_0-devel" "pkg-config" "dpkg")

    case "${ID}" in
        ubuntu)
            # elevate_if_not_root apt update
            install_apt_packages "${library_dependencies_ubuntu[@]}"

            if [[ "${build_clients}" == true ]]; then
            install_apt_packages "${client_dependencies_ubuntu[@]}"
            fi
            ;;

        centos|rhel)
            # elevate_if_not_root yum -y update
            install_yum_packages "${library_dependencies_centos[@]}"

            if [[ "${build_clients}" == true ]]; then
                install_yum_packages "${client_dependencies_centos[@]}"
            fi
            ;;

        fedora)
            # elevate_if_not_root dnf -y update
            install_dnf_packages "${library_dependencies_fedora[@]}"

            if [[ "${build_clients}" == true ]]; then
                install_dnf_packages "${client_dependencies_fedora[@]}"
            fi
            ;;
        
        sles)
            # elevate_if_not_root zypper -n update
            install_zypper_packages "${library_dependencies_sles[@]}"

            if [[ "${build_clients}" == true ]]; then
                install_zypper_packages "${client_dependencies_sles[@]}"
            fi
            ;;
        *)
            echo "This script is currently supported on Ubuntu, CentOS, RHEL, Fedora, and SLES"
            exit 2
            ;;
    esac
}

# #################################################
# Pre-requisites check
# #################################################
# Exit code 0: alls well
# Exit code 1: problems with getopt
# Exit code 2: problems with supported platforms

# check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
    echo "This script uses getopt to parse arguments; try installing the util-linux package";
    exit 1
fi

# os-release file describes the system
if [[ -e "/etc/os-release" ]]; then
    source /etc/os-release
else
    echo "This script depends on the /etc/os-release file"
    exit 2
fi

# The following function exits script if an unsupported distro is detected
supported_distro

# #################################################
# global variables
# #################################################
install_package=false
install_dependencies=false
install_prefix=rocfft-install
build_clients=false
build_cuda=false
build_release=true

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
    GETOPT_PARSE=$(getopt --name "${0}" -o 'hidcg' --long 'help,install,clients,dependencies,debug,cuda,hip-clang,prefix:' --options hicgd -- "$@")
else
    echo "Need a new version of getopt"
    exit 1
fi

if [[ $? -ne 0 ]]; then
    echo "getopt invocation failed; could not parse the command line";
    exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
    case "${1}" in
        -h|--help)
            display_help
            exit 0
            ;;
        -i|--install)
            install_package=true
            shift ;;
        -d|--dependencies)
            install_dependencies=true
            shift ;;
        -c|--clients)
            build_clients=true
            shift ;;
        -g|--debug)
            build_release=false
            shift ;;
        --cuda)
            build_cuda=true
            shift ;;
        --hip-clang)
            build_hip_clang=true
            shift ;;
        --prefix)
            echo $2
            install_prefix=${2}
            shift 2 ;;
        --) shift ; break ;;
        *)  echo "Unexpected command line parameter received; aborting";
            exit 1
            ;;
    esac
done

build_dir=./build
printf "\033[32mCreating project build directory in: \033[33m${build_dir}\033[0m\n"

# #################################################
# prep
# #################################################
# ensure a clean build environment
if [[ "${build_release}" == true ]]; then
    rm -rf ${build_dir}/release
else
    rm -rf ${build_dir}/debug
fi

# Default cmake executable is called cmake
cmake_executable=cmake

case "${ID}" in
    centos|rhel)
	cmake_executable=cmake3
	;;
esac

# #################################################
# dependencies 
# #################################################
if [[ "${install_dependencies}" == true ]]; then

    install_packages

    # The following builds googletest from source
    pushd .
    printf "\033[32mBuilding \033[33mgoogletest\033[32m from source"
    mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
    ${cmake_executable} -DBUILD_BOOST=OFF ../../deps
    # ${cmake_executable} -DCMAKE_INSTALL_PREFIX=deps-install -DBUILD_BOOST=OFF ../../deps
    make -j$(nproc)
    elevate_if_not_root make install
    popd
fi


# We append customary rocm path; if user provides custom rocm path in ${path}, our
# hard-coded path has lesser priority
export PATH=${PATH}:/opt/rocm/bin

pushd .
# #################################################
# configure & build
# #################################################
cmake_common_options=""
cmake_client_options=""

# build type
if [[ "${build_release}" == true ]]; then
    mkdir -p ${build_dir}/release/clients && cd ${build_dir}/release
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Release"
else
    mkdir -p ${build_dir}/debug/clients && cd ${build_dir}/debug
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Debug"
fi

# clients
if [[ "${build_clients}" == true ]]; then
    cmake_client_options="${cmake_client_options} -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SELFTEST=ON -DBUILD_CLIENTS_RIDER=ON"
fi

compiler="hcc"
if [[ "${build_cuda}" == true || "${build_hip_clang}" == true ]]; then
    compiler="hipcc"
fi

if [[ "${build_hip_clang}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DUSE_HIP_CLANG=ON -DHIP_COMPILER=clang"
fi

# On ROCm platforms, hcc compiler can build everything
if [[ "${build_cuda}" == false ]]; then

    # Build library with AMD toolchain because of existense of device kernels
    if [[ "${build_clients}" == true ]]; then
        CXX=${compiler} ${cmake_executable} ${cmake_common_options} ${cmake_client_options} -DCPACK_SET_DESTDIR=OFF -DCMAKE_INSTALL_PREFIX=${install_prefix} -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm ../..
    else
        CXX=${compiler} ${cmake_executable} ${cmake_common_options} -DCPACK_SET_DESTDIR=OFF -DCMAKE_INSTALL_PREFIX=${install_prefix} -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm ../..
    fi
    check_exit_code
    make -j$(nproc)
    check_exit_code
    if ( check_install_dir ${install_prefix} ) ; then
        make install
    else
        elevate_if_not_root make install
    fi
    check_exit_code


else
    # The nvidia compile is a little more complicated, in that we split compiling the
    # library from the clients.  We use the hipcc compiler to build the rocfft library
    # for a cuda backend (hipcc offloads the compile to nvcc).  However, we run into a
    # compiler incompatibility compiling the clients between nvcc and fftw3.h 3.3.4
    # headers.  The incompatibility is fixed in fft v3.3.6, but that is not shipped by
    # default on Ubuntu

    # As a workaround, since clients do not contain device code, we opt to build clients
    # with the native compiler on the platform.  The compiler cmake chooses during
    # configuration time is mostly unchangeable, so we launch multiple cmake invocation
    # with a different compiler on each.

    # Build library only with hipcc as compiler
    CXX=${compiler} ${cmake_executable} ${cmake_common_options} -DCMAKE_INSTALL_PREFIX=${install_prefix} -DCPACK_PACKAGE_INSTALL_DIRECTORY=/opt/rocm ../..
    check_exit_code
    make -j$(nproc)
    check_exit_code
    if ( check_install_dir ${install_prefix} ) ; then
        make install
    else
        elevate_if_not_root make install
    fi
    check_exit_code
    
    # Build cuda clients with default host compiler
    if [[ "${build_clients}" == true ]]; then
        pushd clients
        ${cmake_executable} ${cmake_common_options} ${cmake_client_options} -DCMAKE_PREFIX_PATH="$(pwd)/../rocfft-install;$(pwd)/../deps/deps-install" ../../../clients
        make -j$(nproc)
        popd
    fi
fi

# #################################################
# install
# #################################################
# installing through package manager, which makes uninstalling easy
if [[ "${install_package}" == true ]]; then
    make package
    check_exit_code
    if [[ "${build_clients}" == true ]]; then
        make package_clients
        check_exit_code
    fi
    case "${ID}" in
        ubuntu)
            elevate_if_not_root dpkg -i rocfft-*.deb
            ;;
        centos|rhel)
            elevate_if_not_root yum -y localinstall rocfft-*.rpm
            ;;
        fedora)
            elevate_if_not_root dnf install rocfft-*.rpm
            ;;
        sles)
            elevate_if_not_root zypper -n --no-gpg-checks install rocfft-*.rpm
            ;;
    esac

fi
popd
