About
=====

This contains all of the ROCm 2.6 source code, as downloaded by the repo tool.
The source code was downloaded as follows:
```
cd rocm_sources
./repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-2.6.0
./repo sync
```

Afterwards, I deleted all of the `.git` folders in each project. If re-running
this step, make sure to do it *before* creating your own git repo!
```
find . -type d -name *.git -exec rm -rf {} \;
```

