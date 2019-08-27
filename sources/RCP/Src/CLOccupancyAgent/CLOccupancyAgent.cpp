//==============================================================================
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
/// \author AMD Developer Tools Team
/// \file
/// \brief Entry point for computing occupancy
//==============================================================================

#include <CL/opencl.h>
#include <CL/internal/cl_agent_amd.h>
#include <string>
#include <sstream>
#include <iostream>
#include <cstring>
#include "../Common/Logger.h"
#include "../Common/Version.h"
#include "../Common/FileUtils.h"
#include "../Common/GlobalSettings.h"
#include "../Common/StringUtils.h"
#include "../CLCommon/CLFunctionDefs.h"
#include "AMDTActivityLogger/CXLActivityLogger.h"
#include "CLIntercept.h"
#include "CLOccupancyInfoManager.h"

static cl_icd_dispatch_table dispatch;
static cl_icd_dispatch_table agentDispatch;

using namespace std;
using namespace GPULogger;

void DumpOccupancy()
{
    static bool alreadyDumped = false;

    if (!alreadyDumped)
    {
        alreadyDumped = true;

        if (!OccupancyInfoManager::Instance()->IsTimeOutMode())
        {
            OccupancyInfoManager::Instance()->SaveToOccupancyFile();
        }

        OccupancyInfoManager::Instance()->Release();
    }
}

void TimerThread(void* params)
{
    SP_UNREFERENCED_PARAMETER(params);

    unsigned int interval = OccupancyInfoManager::Instance()->GetInterval();

    if (interval == 0)
    {
        interval = 1; // safety net in case interval is zero (it shouldn't be...)
    }

    const unsigned int sleepInterval = interval < 10 ? interval : 10; // sleep at most 10 ms at a time
    const unsigned int sleepsBeforeFlush = sleepInterval == 0 ? 1 : interval / sleepInterval;

    unsigned int iterationNum = 1;

    while (OccupancyInfoManager::Instance()->IsRunning())
    {
        OSUtils::Instance()->SleepMillisecond(sleepInterval);

        if (iterationNum == sleepsBeforeFlush)
        {
            iterationNum = 1;
            OccupancyInfoManager::Instance()->TrySwapBuffer();
            OccupancyInfoManager::Instance()->FlushTraceData();
        }
        else
        {
            iterationNum++;
        }
    }
}

extern "C" DLL_PUBLIC void amdtCodeXLStopProfiling(amdtProfilingControlMode mode)
{
    bool isCLTraceProfiling = GlobalSettings::GetInstance()->m_params.m_bTrace;
    bool isCLPerfCounterProfiling = GlobalSettings::GetInstance()->m_params.m_bPerfCounter;

    bool shouldStop = false;

    if (isCLTraceProfiling)
    {
        // stop if user asked to stop tracing, and we're also tracing
        shouldStop = (mode & AMDT_TRACE_PROFILING) == AMDT_TRACE_PROFILING;
    }
    else if (isCLPerfCounterProfiling)
    {
        // stop if user asked to stop perf counting, and we're also perf counting
        shouldStop = (mode & AMDT_PERF_COUNTER_PROFILING) == AMDT_PERF_COUNTER_PROFILING;
    }
    else
    {
        // stop if user asked to stop all profiling if we're not tracing or perf counting
        shouldStop = mode == AMDT_ALL_PROFILING;
    }

    if (shouldStop)
    {
        OccupancyInfoManager::Instance()->EnableProfiling(false);
    }
}

extern "C" DLL_PUBLIC void amdtCodeXLResumeProfiling(amdtProfilingControlMode mode)
{
    bool isCLTraceProfiling = GlobalSettings::GetInstance()->m_params.m_bTrace;
    bool isCLPerfCounterProfiling = GlobalSettings::GetInstance()->m_params.m_bPerfCounter;

    bool shouldStart = false;

    if (isCLTraceProfiling)
    {
        // start if user asked to start tracing, and we're also tracing
        shouldStart = (mode & AMDT_TRACE_PROFILING) == AMDT_TRACE_PROFILING;
    }
    else if (isCLPerfCounterProfiling)
    {
        // start if user asked to start perf counting, and we're also perf counting
        shouldStart = (mode & AMDT_PERF_COUNTER_PROFILING) == AMDT_PERF_COUNTER_PROFILING;
    }
    else
    {
        // start if user asked to start all profiling if we're not tracing or perf counting
        shouldStart = mode == AMDT_ALL_PROFILING;
    }

    if (shouldStart)
    {
        OccupancyInfoManager::Instance()->EnableProfiling(true);
    }
}

cl_int CL_CALLBACK
clAgent_OnLoad(cl_agent* agent)
{
#ifdef _DEBUG
    FileUtils::CheckForDebuggerAttach();
#endif

    std::string strLogFile = FileUtils::GetDefaultOutputPath() + "cloccupancyagent.log";
    LogFileInitialize(strLogFile.c_str());

    cl_icd_dispatch_table nextTable;
    cl_icd_dispatch_table realTable;
    cl_int status = InitAgent(agent, CL_OCCUPANCY_AGENT_DLL, &nextTable, &realTable);

    if (CL_SUCCESS != status)
    {
        return CL_SUCCESS;
    }

    memcpy(&dispatch, &nextTable, sizeof(cl_icd_dispatch_table));
    memcpy(&agentDispatch, &dispatch, sizeof(cl_icd_dispatch_table));

    InitNextCLFunctions(&nextTable, &realTable);

    std::cout << RCP_PRODUCT_NAME " Kernel occupancy module is enabled" << std::endl;

    agentDispatch.EnqueueNDRangeKernel = CL_OCCUPANCY_API_ENTRY_EnqueueNDRangeKernel;
    agentDispatch.ReleaseContext = CL_OCCUPANCY_API_ENTRY_ReleaseContext;
    agentDispatch.GetPlatformInfo = CL_OCCUPANCY_API_ENTRY_GetPlatformInfo;
    agentDispatch.GetDeviceIDs = CL_OCCUPANCY_API_ENTRY_GetDeviceIDs;

    status = agent->SetICDDispatchTable(agent, &agentDispatch, sizeof(cl_icd_dispatch_table));

    Parameters params;
    FileUtils::GetParametersFromFile(params);

    if (!params.m_bTrace)
    {
        FileUtils::ReadKernelListFile(params);
    }

    std::string occupancyFile = params.m_strOutputFile;

    size_t passStringPosition = occupancyFile.find("_pass");

    if (passStringPosition != std::string::npos)
    {
        //Remove the appended "_pass"" string and the extension
        occupancyFile = occupancyFile.substr(0, passStringPosition);
    }

    OccupancyInfoManager::Instance()->SetOutputFile(occupancyFile);

    GlobalSettings::GetInstance()->m_params = params;
    OccupancyInfoEntry::m_cListSeparator = params.m_cOutputSeparator;

    if (params.m_bStartDisabled)
    {
        OccupancyInfoManager::Instance()->StopTracing();
    }
    else
    {
        OccupancyInfoManager::Instance()->EnableProfileDelayStart(params.m_bDelayStartEnabled, params.m_delayInMilliseconds);
        OccupancyInfoManager::Instance()->EnableProfileDuration(params.m_bProfilerDurationEnabled, params.m_durationInMilliseconds);

        if (params.m_bDelayStartEnabled)
        {
            OccupancyInfoManager::Instance()->CreateTimer(PROFILEDELAYTIMER, params.m_delayInMilliseconds);
            OccupancyInfoManager::Instance()->SetTimerFinishHandler(PROFILEDELAYTIMER, CLOccupancyAgentTimerEndResponse);
            OccupancyInfoManager::Instance()->StopTracing();
            OccupancyInfoManager::Instance()->EnableProfiling(false);
            OccupancyInfoManager::Instance()->startTimer(PROFILEDELAYTIMER);
        }
        else if (params.m_bProfilerDurationEnabled)
        {
            OccupancyInfoManager::Instance()->CreateTimer(PROFILEDURATIONTIMER, params.m_durationInMilliseconds);
            OccupancyInfoManager::Instance()->SetTimerFinishHandler(PROFILEDURATIONTIMER, CLOccupancyAgentTimerEndResponse);
            OccupancyInfoManager::Instance()->startTimer(PROFILEDURATIONTIMER);
        }
    }

    if (params.m_bTimeOutBasedOutput)
    {
        OccupancyInfoManager::Instance()->SetInterval(params.m_uiTimeOutInterval);

        if (!OccupancyInfoManager::Instance()->StartTimer(TimerThread))
        {
            std::cout << "Failed to initialize CLOccupancyAgent." << std::endl;
        }
    }

    return status;
}


#ifdef _WIN32

extern "C" DLL_PUBLIC void OnExitProcess()
{
    DumpOccupancy();
}

// On Windows, we can't dump data in OnUnload() because of ocl runtime bug
BOOL APIENTRY DllMain(HMODULE,
                      DWORD   ul_reason_for_call,
                      LPVOID)
{
    switch (ul_reason_for_call)
    {
        case DLL_PROCESS_ATTACH:
            break;

        case DLL_THREAD_ATTACH:
            break;

        case DLL_PROCESS_DETACH:
        {
            // Dump all data out
            DumpOccupancy();
        }
        break;

        case DLL_THREAD_DETACH:
            break;
    }

    return TRUE;
}
#endif
