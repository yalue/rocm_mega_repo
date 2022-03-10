# This test script simply tries creating and using a stream with a CU mask in
# conjunction with PyTorch.

import rocm_helper
import torch
import time

# Create a stream that uses 8 CUs only
s = rocm_helper.create_stream_with_cu_mask(0, 0xff)
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

external_stream = None
rocm_helper.destroy_stream(s)
print("Stream destroyed.")
