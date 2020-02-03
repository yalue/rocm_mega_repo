// This file contains definitions for a C python module for interacting with
// ROCm settings--basically just my own wrapper around the /dev/kfd ioctls.
//
// Links to useful documentation (for my reference):
//  - https://docs.python.org/3/extending/extending.html
//  - https://docs.python.org/3/extending/building.html (for setup.py stuff)

#define PY_SSIZE_T_CLEAN
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <Python.h>

// The arbitrary magic number I created for the ioctl to set the default CU
// mask for new queues. (It's not actually entirely arbitrary; we need to make
// sure it doesn't conflict with any of AMD's ioctl commands.)
#define SET_DEFAULT_CU_MASK_IOCTL_CMD (0xff1337ff)

// A file descriptor for /dev/kfd. Opened once during module initialization.
static int kfd_fd = -1;

static PyObject* rocm_control_set_default_cu_mask(PyObject *self,
    PyObject *args) {
  unsigned long long new_mask_arg;
  uint32_t new_mask = 0;
  int result;
  if (!PyArg_ParseTuple(args, "K", &new_mask_arg)) return NULL;
  new_mask = new_mask_arg;
  if (new_mask == 0) {
    PyErr_SetString(PyExc_ValueError, "The CU mask must have at least one of "
      "its bottom 32 bits set");
    return NULL;
  }
  result = ioctl(kfd_fd, SET_DEFAULT_CU_MASK_IOCTL_CMD, &new_mask);
  if (result != 0) {
    PyErr_Format(PyExc_OSError, "Set default CU mask ioctl failed: %s",
      strerror(errno));
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef rocm_control_methods[] = {
  {
    "set_default_cu_mask",
    rocm_control_set_default_cu_mask,
    METH_VARARGS,
    "Set the default CU mask for the calling process' future queues. "
      "Requires a single integer: a 32-bit number containing the CU mask.",
  },
  {NULL, NULL, 0, NULL},
};

// TODO: Maybe add a m_clear or m_free function to close kfd_fd?
static struct PyModuleDef rocm_control_module = {
    PyModuleDef_HEAD_INIT,
    "rocm_control",
    "Provides functions for modifying the ROCm environment.",
    -1,
    rocm_control_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_rocm_control(void) {
  kfd_fd = open("/dev/kfd", O_RDWR);
  if (kfd_fd < 0) {
    PyErr_Format(PyExc_ImportError, "Couldn't open /dev/kfd: %s",
      strerror(errno));
    return NULL;
  }
  return PyModule_Create(&rocm_control_module);
}
