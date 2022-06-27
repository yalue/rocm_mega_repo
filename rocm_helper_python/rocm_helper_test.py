# This test script simply tries creating and using a stream with a CU mask in
# conjunction with PyTorch.

#import torch
import rocm_helper
import time

# Can't figure out why this garbage doesn't work in ROCm 4.2. Or, rather, I
# don't want to spend so long figuring out the mess of tracing methods.
print("Initially stopping tracing.")
rocm_helper.roctracer_stop()

# Create a stream that uses 8 CUs only
s = rocm_helper.create_stream_with_cu_mask(0x11111111, 0)
print("In python. Stream ptr = 0x%x" % (s, ))

a = torch.rand((2, 100000, 1000), device="cuda:0")
b = torch.ones((2, 100000, 1000), device="cuda:0")
external_stream = torch.cuda.streams.ExternalStream(s)
print("Warming up")
for i in range(1000):
    a += b
torch.cuda.synchronize()
print("First value = " + str(a[0][0][0]))

print("Testing using stream")
start = time.time()
with torch.cuda.stream(external_stream):
    for i in range(1000):
        a += b
external_stream.synchronize()
end = time.time()
print("Using stream took %f seconds." % (end - start, ))
print("First value = " + str(a[0][0][0]))
print("Testing without stream")
start = time.time()
for i in range(1000):
    a += b
torch.cuda.synchronize()
end = time.time()
print("Without stream took %f seconds." % (end - start, ))
print("First value = " + str(a[0][0][0]))

# Same note as above. roctracer_start, roctracer_stop, and roctx_mark don't
# seem to work properly.
rocm_helper.fake_kernel_a()
print("Starting tracing.")
rocm_helper.roctracer_start()
rocm_helper.fake_kernel_a()
rocm_helper.fake_kernel_b()
print("Tracing done.")
rocm_helper.roctracer_stop()

external_stream = None
rocm_helper.destroy_stream(s)
print("Stream destroyed.")

