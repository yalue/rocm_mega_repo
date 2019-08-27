/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

UGEN(cos)

half
MATH_MANGLE(cos)(half x)
{
    struct redret r = MATH_PRIVATE(trigred)(BUILTIN_ABS_F16(x));
    struct scret sc = MATH_PRIVATE(sincosred)(r.hi);
    sc.s = -sc.s;

    short c =  AS_SHORT((r.i & 1) == (short)0 ? sc.c : sc.s);
    c ^= r.i > 1 ? (short)0x8000 : (short)0;

    if (!FINITE_ONLY_OPT()) {
        c = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? (short)QNANBITPATT_HP16 : c;
    }

    return AS_HALF(c);
}

