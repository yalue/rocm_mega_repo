The ROCm Mega Repo
==================

This repository is used to track modifications to several components of the
ROCm stack.


About
-----

This repository currently contains copies of the `ROCR-Runtime`, `HIP`, and
`ROCclr` code.  The source code is based on ROCm 4.0.

This repository is where I keep track of modifications to the ROCm stack for my
own personal research.  For example, it previously contained a separate
implementation of an API for setting compute-unit masks for HIP streams (this
is no longer necessary, as such support has been upstreamed).

It is reasonable to expect code contained here to require extra changes, i.e.
additional kernel modules or Linux kernel patches.


Prerequisites
-------------

You will first need to have the following prerequisites installed for compiling
ROCm (there may be others that I'm forgetting):
```
sudo apt install cmake pkg-config libpci-dev libelf-dev liblapack-dev \
  libboost-program-options-dev libnuma-dev
```

You will need to make sure that you are running a sufficiently recent version
of the Linux kernel, supporting not only ROCm but also my own kernel
modifications, which are tracked in a different repo. (I will add a link
later.)

Finally, you will need to install the following packages from AMD's pre-built
ROCm repositories AMD's repositories (see
[these instructions](https://github.com/RadeonOpenCompute/ROCm#Ubuntu) for
information about adding the repo):
```
sudo apt install rocm-dev4.0.0 rocm-libs4.0.0 rocm-utils4.0.0 rccl4.0.0
```

After running the above command to install ROCm version 4.0.0, you will need to
create a symlink to link `/opt/rocm` to `/opt/rocm/4.0.0`.

Finally, make sure that `/opt/rocm/bin` is on your PATH, and that your system
looks for shared libraries in `/opt/rocm/lib`.

(Note that if you need to install additional ROCm libraries, always use the
package versions with names ending in `4.0.0`: mixing the `<...>4.0.0` packages
with the packages that don't have a version number EASILY leads to broken
installations.  (Speaking from experience.)


Compilation and Installation (on top of an existing ROCm installation)
----------------------------------------------------------------------

You *should* be able to build everything using the included `install.sh`
script: `bash ./install.sh`.  The script will prompt for `sudo` access at some
points in order to be able to write to `/opt/rocm`.


Uninstallation
--------------

Restoring an "original" version of ROCm will likely require re-installing the
packages overwritten by this project. (i.e. uninstalling all existing ROCm
packages, then deleting `/opt/rocm*`, and then reinstalling ROCm from AMD's
repositories.  USE THIS AT YOUR OWN RISK.  I have encountered many headaches
with AMD's installation process, though the ability to install specific ROCM
releases has been a major improvement.

