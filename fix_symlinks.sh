# Fix HIP symlinks to our locally-built version.
rm /opt/rocm/lib/libamdhip64.so
ln -s -T /opt/rocm/hip/lib/libamdhip64.so /opt/rocm/lib/libamdhip64.so
rm /opt/rocm/lib/libamdhip64.so.4
ln -s -T /opt/rocm/hip/lib/libamdhip64.so.4 /opt/rocm/lib/libamdhip64.so.4

# Fix ROCR-Runtime symlinks
rm /opt/rocm/lib/libhsa-runtime64.so.1
ln -s -T /opt/rocm/lib/libhsa-runtime64.so.1.3.0 /opt/rocm/lib/libhsa-runtime64.so.1

