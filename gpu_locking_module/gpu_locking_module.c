// Contains the definition for the module responsible for locking GPU access
// in HIP.
//
// TODO:
//  1. Fix the user test program to open the device. It probably has tons of
//     stuff to be deleted.
//  2. Add an unlock ioctl, and a create-partition IOCTL, maybe? Or maybe just
//     have a fixed number of partitions, each with an associated queue?
//  3. As a proof-of-concept, just add lock/unlock calls, along with an
//     unlock-all ioctl. (Create a userspace program that can issue the unlock-
//     all ioctl to break locks if deadlock occurs during testing.)
//  4. Figure out how to implement a priority queue, prioritized by deadline.

#include <linux/device.h>
#include <linux/fs.h>
#include <linux/module.h>
#include <linux/uaccess.h>
#include "gpu_locking_module.h"

// The name of the character device.
#define DEVICE_NAME "gpu_locking_module"

// The number, class and device references set during initialization, and
// needed when destroying the chardev.
static int major_number;
static struct class *gpu_lock_device_class = NULL;
static struct device *gpu_lock_chardev = NULL;

static int OpenDevice(struct inode *inode, struct file *file) {
  printk("GPU locking management device opened.\n");
  return 0;
}

static int ReleaseDevice(struct inode *inode, struct file *file) {
  printk("GPU locking management device closed.\n");
  return 0;
}

// Handles the GPUIOC_ACQUIRE_LOCK ioctl.
static long AcquireGPULockIoctl(void __user *arg) {
  AcquireGPULockArgs a;
  if (copy_from_user(&a, arg, sizeof(a)) != 0) {
    printk("GPU locking manager failed copying ioctl args from user.\n");
    return -EFAULT;
  }
  printk("GPU locking manager: Got request to lock partition %d, with deadline"
    " %u\n", a.partition_id, a.deadline);

  printk("GPU locking manager: support for acquiring lock not implemented!\n");
  return -EINVAL;
}

static long DeviceIoctl(struct file *filp, unsigned int cmd,
    unsigned long arg) {
  printk("GPU locking module got ioctl %x with arg %lx\n", cmd, arg);
  switch (cmd) {
  case GPULOCKIOC_ACQUIRE_LOCK:
    return AcquireGPULockIoctl((void __user *) arg);
  default:
    break;
  }
  printk("Invalid ioctl command to GPU locking module!\n");
  return -EINVAL;
}

static struct file_operations fops = {
  .open = OpenDevice,
  .release = ReleaseDevice,
  .unlocked_ioctl = DeviceIoctl,
};

static int __init InitModule(void) {
  major_number = register_chrdev(0, DEVICE_NAME, &fops);
  if (major_number < 0) {
    printk("Failed registering GPU locking module chardev.\n");
    return 1;
  }
  gpu_lock_device_class = class_create(THIS_MODULE, "gpu_locking");
  if (IS_ERR(gpu_lock_device_class)) {
    printk("Failed to register GPU locking device class.\n");
    unregister_chrdev(major_number, DEVICE_NAME);
    return 1;
  }
  gpu_lock_chardev = device_create(gpu_lock_device_class, NULL,
    MKDEV(major_number, 0), NULL, DEVICE_NAME);
  if (IS_ERR(gpu_lock_chardev)) {
    printk("Failed to create GPU locking chardev.\n");
    class_destroy(gpu_lock_device_class);
    unregister_chrdev(major_number, DEVICE_NAME);
    return 1;
  }
  printk("GPU locking module loaded, chardev major number %d\n", major_number);
  return 0;
}

static void __exit CleanupModule(void) {
  printk("Unloading GPU locking module...\n");
  device_destroy(gpu_lock_device_class, MKDEV(major_number, 0));
  class_unregister(gpu_lock_device_class);
  class_destroy(gpu_lock_device_class);
  unregister_chrdev(major_number, DEVICE_NAME);
  printk("GPU locking module unloaded.\n");
}

module_init(InitModule);
module_exit(CleanupModule);
MODULE_LICENSE("GPL");

