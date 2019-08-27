
#include "irif.h"
#include "ockl.h"


uint
OCKL_MANGLE_U32(wfbcast)(uint a, uint i)
{
    uint j = __builtin_amdgcn_readfirstlane(i);
    return __builtin_amdgcn_readlane(a, j);
}

ulong
OCKL_MANGLE_U64(wfbcast)(ulong a, uint i)
{
    uint j = __builtin_amdgcn_readfirstlane(i);
    return ((ulong)__builtin_amdgcn_readlane((uint)(a >> 32), j) << 32) |
            (ulong)__builtin_amdgcn_readlane((uint)a, j);
}

