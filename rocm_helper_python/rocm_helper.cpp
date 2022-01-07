#define PY_SSIZE_T_CLEAN

#include <stdio.h>
#include <hip/hip_runtime.h>
#include <Python.h>

extern "C" {
  PyMODINIT_FUNC PyInit_rocm_helper(void);
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

  Py_INCREF(Py_None);
  return Py_None;
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
