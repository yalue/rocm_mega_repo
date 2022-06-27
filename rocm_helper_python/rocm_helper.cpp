#define PY_SSIZE_T_CLEAN

#include <stdio.h>
#include <hip/hip_runtime.h>
#include <roctracer/roctracer_ext.h>
#include <roctracer/roctx.h>
#include <Python.h>

extern "C" {
  PyMODINIT_FUNC PyInit_rocm_helper(void);
}

__global__ void FakeKernelA(uint64_t *dummy) {
  if (dummy) dummy[threadIdx.x] = 1337;
}

__global__ void FakeKernelB(uint64_t *dummy) {
  if (dummy) dummy[threadIdx.x] = 1338;
}

static PyObject* LaunchFakeKernelA(PyObject *self, PyObject *args) {
  hipError_t result = hipSuccess;
  hipLaunchKernelGGL(FakeKernelA, 1, 1, 0, NULL, (uint64_t *) NULL);
  result = hipDeviceSynchronize();
  if (result != hipSuccess) {
    PyErr_Format(PyExc_OSError, "Failed launching fake kernel: %d",
      (int) result);
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyObject* LaunchFakeKernelB(PyObject *self, PyObject *args) {
  hipError_t result = hipSuccess;
  hipLaunchKernelGGL(FakeKernelB, 1, 1, 0, NULL, (uint64_t *) NULL);
  result = hipDeviceSynchronize();
  if (result != hipSuccess) {
    PyErr_Format(PyExc_OSError, "Failed launching fake kernel: %d",
      (int) result);
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyObject* CreateStreamWithCUMask(PyObject *self, PyObject *args) {
  unsigned long low_cus, high_cus;
  unsigned long long to_return;
  uint32_t cu_mask[2];
  hipError_t result;
  hipStream_t stream;

  if (!PyArg_ParseTuple(args, "kk", &low_cus, &high_cus)) {
    return NULL;
  }
  if ((low_cus > 0xffffffff) || (high_cus > 0xffffffff)) {
    PyErr_Format(PyExc_ValueError, "Only the lower 32 bits of each arg to "
      "create_stream_with_mask may be used");
    return NULL;
  }
  if ((low_cus == 0) && (high_cus == 0)) {
    PyErr_Format(PyExc_ValueError, "The CU mask for a stream must not be 0");
    return NULL;
  }
  cu_mask[0] = low_cus;
  cu_mask[1] = high_cus;

  result = hipExtStreamCreateWithCUMask(&stream, 2, cu_mask);
  if (result != hipSuccess) {
    PyErr_Format(PyExc_ValueError, "Failed creating a HIP stream. Error %d",
      (int) result);
    return NULL;
  }

  to_return = (unsigned long long) ((uintptr_t) ((void *) stream));
  printf("In C: Allocated HIP stream with mask. Ptr = 0x%llx\n", to_return);

  return Py_BuildValue("K", to_return);
}

static PyObject* DestroyHIPStream(PyObject *self, PyObject *args) {
  unsigned long long arg;
  hipStream_t toDestroy;

  if (!PyArg_ParseTuple(args, "K", &arg)) {
    return NULL;
  }
  toDestroy = (hipStream_t) ((void *) ((uintptr_t) arg));
  printf("In C: Destroying HIP stream @%p\n", (void *) toDestroy);
  hipStreamDestroy(toDestroy);

  Py_RETURN_NONE;
}

static PyObject* ROCTracerStart(PyObject *self, PyObject *args) {
  roctracer_start();
  Py_RETURN_NONE;
}

static PyObject* ROCTracerStop(PyObject *self, PyObject *args) {
  roctracer_stop();
  Py_RETURN_NONE;
}

static PyObject* ROCTXMark(PyObject *self, PyObject *args) {
  const char *mark = NULL;
  if (!PyArg_ParseTuple(args, "s", &mark)) return NULL;
  roctxMark(mark);
  Py_RETURN_NONE;
}

static PyMethodDef rocm_helper_methods[] = {
  {
    "create_stream_with_cu_mask",
    CreateStreamWithCUMask,
    METH_VARARGS,
    "Creates a stream with a CU mask. Requires two arguments: the lower and "
      "upper 32 bits of the CU mask. Returns a new HIP stream. The caller "
      "must destroy the returned stream when it's no longer needed.",
  },
  {
    "destroy_stream",
    DestroyHIPStream,
    METH_VARARGS,
    "Destroys a HIP stream returned by create_stream_with_cu_mask. Requires "
      "one arg: the value returned by create_stream_with_cu_mask.",
  },
  {
    "roctracer_start",
    ROCTracerStart,
    METH_NOARGS,
    "Starts the ROCm tracer.",
  },
  {
    "roctracer_stop",
    ROCTracerStop,
    METH_NOARGS,
    "Stops the ROCm tracer.",
  },
  {
    "fake_kernel_a",
    LaunchFakeKernelA,
    METH_NOARGS,
    "Launches a fake kernel, FakeKernelA, that does nothing but can be used "
      "as a simple mark in the trace logs.",
  },
  {
    "fake_kernel_b",
    LaunchFakeKernelB,
    METH_NOARGS,
    "Same as fake_kernel_a, but launches FakeKernelB.",
  },
  {
    "roctx_mark",
    ROCTXMark,
    METH_VARARGS,
    "Inserts a mark into the roctracer log. Takes a string.",
  },
  {NULL, NULL, 0, NULL},
};

static struct PyModuleDef rocm_helper_module = {
  PyModuleDef_HEAD_INIT,
  "rocm_helper",
  "A library for interacting with ROCm from python.",
  -1,
  rocm_helper_methods,
  NULL,
  NULL,
  NULL,
  NULL,
};

PyMODINIT_FUNC PyInit_rocm_helper(void) {
  return PyModule_Create(&rocm_helper_module);
}
