// Contains the definition for the module responsible for locking GPU access
// in HIP.
//
// TODO:
//  1. Add an unlock ioctl, and a create-partition IOCTL, maybe? Or maybe just
//     have a fixed number of partitions, each with an associated queue?
//  2. As a proof-of-concept, just add lock/unlock calls, along with an
//     unlock-all ioctl. (Create a userspace program that can issue the unlock-
//     all ioctl to break locks if deadlock occurs during testing.)

#include <linux/device.h>
#include <linux/fs.h>
#include <linux/module.h>
#include <linux/types.h>
#include <linux/uaccess.h>
#include <linux/wait.h>
#include "gpu_locking_module.h"

// The name of the character device.
#define DEVICE_NAME "gpu_locking_module"

// Prepend this to all log messages.
#define PRINTK_TAG "GPU locking module: "

// The maximum number of waiting tasks in the queue for any single partition.
#define MAX_QUEUE_LENGTH (64)

// The number, class and device references set during initialization, and
// needed when destroying the chardev.
static int major_number;
// Keeps track of whether we've added the DEVMODE=0666 environment variable to
// the device class.
static int devmode_var_added = 0;
static struct class *gpu_lock_device_class = NULL;
static struct device *gpu_lock_chardev = NULL;

static int OpenDevice(struct inode *inode, struct file *file) {
  printk(PRINTK_TAG "Device opened\n");
  return 0;
}

static int ReleaseDevice(struct inode *inode, struct file *file) {
  printk(PRINTK_TAG "Device closed\n");
  return 0;
}

// Handles the GPULOCK_IOC_ACQUIRE_LOCK ioctl.
static long AcquireGPULockIoctl(void __user *arg) {
  GPULockArgs a;
  if (copy_from_user(&a, arg, sizeof(a)) != 0) {
    printk(PRINTK_TAG "Failed copying lock-acquire args.\n");
    return -EFAULT;
  }
  printk(PRINTK_TAG "Got request to lock partition %d, with priority %lu, "
    "from PID %d\n", a.partition_id, (unsigned long) a.priority,
    (int) current->pid);

  printk(PRINTK_TAG "Support for acquiring lock not implemented!\n");
  return -EINVAL;
}

// For GPULOCK_IOC_RELEASE_LOCK
static long ReleaseGPULockIoctl(void __user *arg) {
  GPULockArgs a;
  if (copy_from_user(&a, arg, sizeof(a)) != 0) {
    printk(PRINTK_TAG "Failed copying lock-release args.\n");
    return -EFAULT;
  }
  printk(PRINTK_TAG "Got request to release lock for partition %d from PID "
    "%d\n", a.partition_id, (int) current->pid);

  printk(PRINTK_TAG "Support for releasing lock not implemented!\n");
  return -EINVAL;
}

// For GPULOCK_IOC_RELEASE_ALL_PARTITION
static long ReleaseAllPartitionIoctl(void __user *arg) {
  GPULockArgs a;
  if (copy_from_user(&a, arg, sizeof(a)) != 0) {
    printk(PRINTK_TAG "Failed copying partition-release args.\n");
    return -EFAULT;
  }
  printk(PRINTK_TAG "Got request to release all waiters for partition %d from "
    "PID %d\n", a.partition_id, (int) current->pid);

  printk(PRINTK_TAG "Support for releasing all partition waiters not "
    "implemented!\n");
  return -EINVAL;
}

// For GPULOCK_IOC_RELEASE_ALL
static long ReleaseAllIoctl(void) {
  printk(PRINTK_TAG "Got request to release all waiters.\n");

  printk(PRINTK_TAG "Support for releasing all waiters not implemented!\n");
  return -EINVAL;
}

static long DeviceIoctl(struct file *filp, unsigned int cmd,
    unsigned long arg) {
  printk(PRINTK_TAG "Got ioctl %x with arg %lx\n", cmd, arg);
  switch (cmd) {
  case GPULOCK_IOC_ACQUIRE_LOCK:
    return AcquireGPULockIoctl((void __user *) arg);
  case GPULOCK_IOC_RELEASE_LOCK:
    return ReleaseGPULockIoctl((void __user *) arg);
  case GPULOCK_IOC_RELEASE_ALL_PARTITION:
    return ReleaseAllPartitionIoctl((void __user *) arg);
  case GPULOCK_IOC_RELEASE_ALL:
    return ReleaseAllIoctl();
  default:
    break;
  }
  printk(PRINTK_TAG "Invalid ioctl command!\n");
  return -EINVAL;
}

static struct file_operations fops = {
  .open = OpenDevice,
  .release = ReleaseDevice,
  .unlocked_ioctl = DeviceIoctl,
};

// Intended to be executed when the chardev is created. Assigns 0666
// permissions to the device.
static int LockingClassUdevEventCallback(struct device *dev,
    struct kobj_uevent_env *env) {
  if (devmode_var_added) return 0;
  devmode_var_added = 1;
  printk(PRINTK_TAG "Got first uevent, adding DEVMODE uevent var\n");
  return add_uevent_var(env, "DEVMODE=0666");
}

static int __init InitModule(void) {
  major_number = register_chrdev(0, DEVICE_NAME, &fops);
  if (major_number < 0) {
    printk(PRINTK_TAG "Failed registering chardev.\n");
    return 1;
  }
  gpu_lock_device_class = class_create(THIS_MODULE, "gpu_locking");
  if (IS_ERR(gpu_lock_device_class)) {
    printk(PRINTK_TAG "Failed to register device class.\n");
    unregister_chrdev(major_number, DEVICE_NAME);
    return 1;
  }
  // Make our chardev have 0666 permissions.
  devmode_var_added = 0;
  gpu_lock_device_class->dev_uevent = LockingClassUdevEventCallback;

  gpu_lock_chardev = device_create(gpu_lock_device_class, NULL,
    MKDEV(major_number, 0), NULL, DEVICE_NAME);
  if (IS_ERR(gpu_lock_chardev)) {
    printk(PRINTK_TAG "Failed to create chardev.\n");
    class_destroy(gpu_lock_device_class);
    unregister_chrdev(major_number, DEVICE_NAME);
    return 1;
  }
  printk(PRINTK_TAG "Loaded, chardev major number %d.\n", major_number);
  return 0;
}

static void __exit CleanupModule(void) {
  printk(PRINTK_TAG "Unloading...\n");
  device_destroy(gpu_lock_device_class, MKDEV(major_number, 0));
  class_unregister(gpu_lock_device_class);
  class_destroy(gpu_lock_device_class);
  unregister_chrdev(major_number, DEVICE_NAME);
  printk(PRINTK_TAG "Unloaded.\n");
}

module_init(InitModule);
module_exit(CleanupModule);
MODULE_LICENSE("GPL");

