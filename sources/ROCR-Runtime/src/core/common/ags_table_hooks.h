// This file defines the functions used by AGS to modify the HSA API function
// tables and the modified functions that are inserted into the table.
//
// Some notes for my own benefit:
//
//  1. The core/common/hsa_table_interface.cpp contains the *actual* HSA API
//     entry points--the first points of contact when a C program calls the HSA
//     API.
//  2. The core/runtime/hsa_api_trace.cpp contains the functions that
//     initialize the table with its default entries, which will be function
//     pointers to code in core/runtime/hsa.cpp, etc.
#ifndef AGS_TABLE_HOOKS_H
#define AGS_TABLE_HOOKS_H
#include <stdio.h>

// This must match the definitions used by AGS.
#define AGS_PIPE_DIR "/tmp/ags_pipes/"
#define AGS_MAIN_PIPE "main_pipe"

// Returns a handle to AGS' main pipe, used for notifying AGS of process
// creation. Returns NULL if the pipe can't be opened (typically meaning that
// AGS is not running).
FILE* GetAGSMainPipeHandle(void);

#endif  // AGS_TABLE_HOOKS_H
