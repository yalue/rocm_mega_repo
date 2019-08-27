//==============================================================================
// Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
/// \author AMD Developer Tools Team
/// \file
/// \brief The implementation for the Profiler Timer class.
//==============================================================================

#include "ProfilerTimer.h"

void ProfilerTimer::SetTimerFinishHandler(TimerEndHandler timerEndHandler)
{
    m_timerHandler = timerEndHandler;
}

ProfilerTimerType ProfilerTimer::GetTimerType() const
{
    return m_timerType;
}

void ProfilerTimer::SetTimerType(ProfilerTimerType timerType)
{
    m_timerType = timerType;
}

ProfilerTimer::ProfilerTimer(long timerInterval) :
    osTimer(timerInterval),
    m_timerHandler(nullptr),
    m_timerType(NONE)
{

}

void ProfilerTimer::onTimerNotification()
{
    m_timerHandler(m_timerType);
}

