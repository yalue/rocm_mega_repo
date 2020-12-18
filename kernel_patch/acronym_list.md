AMDGPU Acronym List
===================

This is a non-exhaustive list of the acronyms I've seen in ROCm code, including
the AMDGPU driver. Mostly, I just remember to update this list after I've
needed to spend time tracking down the definition of non-obvious acronyms. So,
I won't include obvious things like "AMD".

 - BO: Buffer Object
 - CP: Command Processor
 - DIQ: A queue for the kernel like the HIQ, but for sending debugging commands (`drivers/gpu/drm/amd/include/kgd_kfd_interface.h`)
 - HIQ: Special queue for the kernel to send commands to the GPU (`drivers/gpu/drm/amd/include/kgd_kfd_interface.h`)
 - HQD: Hardware Queue Descriptor
 - HWS: Hardware Scheduling, a scheduling policy using the CP
 - IB: Indirect Buffer, "areas of GPU-accessible memory where commands are stored" (`drivers/gpu/drm/amd/amdgpu/amdgpu_ib.c`)
 - MQD: Memory Queue Descriptor
 - PASID: A global address space identifier that can be shared between the GPU, IOMMU, and the driver. (See `drivers/gpu/drm/amd/amdgpu/amdgpu_ids.c`)

