About
=====

This directory contains code and notes for my modifications to the AMDGPU
Linux driver.

Usage
-----

The current `amdgpu_evict_restore_try1.patch` is simply part of a quick
experimental attempt to investigate behavior when evicting queues of GPU work
on a process-wide basis.  It adds queue-eviction and restoration IOCTLs to the
`/dev/kfd` chardev.  In the very unlikely chance that anybody wants to use it,
it was based on Linux 5.10.0, commit 2c85ebc57b3e1817b6ce1a6b703928e113a90442.

Acronyms
--------

The kernel driver, and related ROCm code, contains lots of acronyms. I've been
compiling a list of acronyms and their meanings as I encounter them. This list
is contained in [`acronym_list.md`](acronym_list.md).

