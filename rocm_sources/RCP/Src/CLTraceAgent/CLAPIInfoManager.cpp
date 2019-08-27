//==============================================================================
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
/// \author AMD Developer Tools Team
/// \file APITraceUtils.cpp
/// \brief This class manages pointers to each saved API object for API tracing.
//==============================================================================

#include <fstream>
#include <ostream>
#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <mutex>
#include <math.h>

#include <AMDTOSWrappers/Include/osProcess.h>
#include <AMDTOSWrappers/Include/osThread.h>

#include "CLAPIInfoManager.h"
#include "CLTraceAgent.h"
#include "PMCSamplerManager.h"
#include "../Common/Defs.h"
#include "../Common/FileUtils.h"
#include "../Common/GlobalSettings.h"
#include "../Common/Logger.h"
#include "../Common/Version.h"
#include "../Common/OSUtils.h"
#include "../Common/StackTracer.h"
#include "../CLCommon/CLFunctionEnumDefs.h"
#include "../CLCommon/CLUtils.h"
#include <ProfilerOutputFileDefs.h>

#ifdef USE_TEXT_WRITER
    #include "../Common/Windows/TextWriter.h"
#endif

using namespace std;
using namespace GPULogger;

void TimerThread(void* param)
{
    SP_UNREFERENCED_PARAMETER(param);

    unsigned int interval = CLAPIInfoManager::Instance()->GetInterval();

    if (interval == 0)
    {
        interval = 1; // safety net in case interval is zero (it shouldn't be...)
    }

    const unsigned int sleepInterval = interval < 10 ? interval : 10; // sleep at most 10 ms at a time
    const unsigned int sleepsBeforeFlush = sleepInterval == 0 ? 1 : interval / sleepInterval;

    unsigned int iterationNum = 1;

    while (CLAPIInfoManager::Instance()->IsRunning())
    {
        OSUtils::Instance()->SleepMillisecond(sleepInterval);

        if (iterationNum == sleepsBeforeFlush)
        {
            iterationNum = 1;
            CLAPIInfoManager::Instance()->TrySwapBuffer();
            CLAPIInfoManager::Instance()->FlushTraceData();
#ifdef NON_BLOCKING_TIMEOUT
            CLEventManager::Instance()->TrySwapBuffer();
            CLEventManager::Instance()->FlushTraceData();
#endif
        }
        else
        {
            iterationNum++;
        }
    }
}

CLAPIInfoManager::CLAPIInfoManager(void) :
    APIInfoManagerBase()
{
    m_mustInterceptAPIs.insert(CL_FUNC_TYPE_clCreateContext);
    m_mustInterceptAPIs.insert(CL_FUNC_TYPE_clCreateContextFromType);
    m_mustInterceptAPIs.insert(CL_FUNC_TYPE_clCreateCommandQueue);
    m_mustInterceptAPIs.insert(CL_FUNC_TYPE_clCreateCommandQueueWithProperties);
    m_mustInterceptAPIs.insert(CL_FUNC_TYPE_clCreateKernelsInProgram);
    m_mustInterceptAPIs.insert(CL_FUNC_TYPE_clCreateKernel);
    m_mustInterceptAPIs.insert(CL_FUNC_TYPE_clReleaseContext);
    m_uiLineNum = 0;
    m_strTraceModuleName = "ocl";
    m_bDelayStartEnabled = false;
    m_bProfilerDurationEnabled = false;
    m_delayInMilliseconds = 0ul;
    m_durationInMilliseconds = 0ul;
    m_pDurationTimer = nullptr;
    m_pDelayTimer = nullptr;
}

CLAPIInfoManager::~CLAPIInfoManager(void)
{
    if (nullptr != m_pDelayTimer)
    {
        m_pDelayTimer->stopTimer();
        delete m_pDelayTimer;
    }

    if (nullptr != m_pDurationTimer)
    {
        m_pDurationTimer->stopTimer();
        delete m_pDurationTimer;
    }
}

void CLAPIInfoManager::FlushTraceData(bool bForceFlush)
{
    SP_UNREFERENCED_PARAMETER(bForceFlush);
    m_mtxFlush.lock();
    osProcessId pid = osGetCurrentProcessId();
    TraceInfoMap& nonActiveMap = m_TraceInfoMap[ 1 - m_iActiveMap ];

    for (TraceInfoMap::iterator mapIt = nonActiveMap.begin(); mapIt != nonActiveMap.end(); mapIt++)
    {
        osThreadId tid = mapIt->first;
        stringstream ss;
        string path;

        if (GlobalSettings::GetInstance()->m_params.m_strOutputFile.empty())
        {
            path = FileUtils::GetDefaultOutputPath();
        }
        else
        {
            path = FileUtils::GetTempFragFilePath();
        }

        // File name: pid_tid.ocl.apitrace

        ss << path << pid << "_" << tid << "." << m_strTraceModuleName << TMP_TRACE_EXT;
        string tmpTraceFile = ss.str();
        ss.str("");
        ss << path << pid << "_" << tid << "." << m_strTraceModuleName << TMP_TIME_STAMP_EXT;
        string tmpTimestampFile = ss.str();

        bool bEnableStackTrace = GlobalSettings::GetInstance()->m_params.m_bStackTrace && StackTracer::Instance()->IsInitialized();
        string tmpStackTraceFile;
        ofstream foutST;

        if (bEnableStackTrace)
        {
            ss.str("");
            ss << path << pid << "_" << tid << "." << m_strTraceModuleName << TMP_TRACE_STACK_EXT;
            tmpStackTraceFile = ss.str();
            foutST.open(tmpStackTraceFile.c_str(), fstream::out | fstream::app);
            bEnableStackTrace = !foutST.fail();
        }

        // Open file for append
        ofstream foutTrace(tmpTraceFile.c_str(), fstream::out | fstream::app);
        ofstream foutTS(tmpTimestampFile.c_str(), fstream::out | fstream::app);

        if (foutTrace.fail() || foutTS.fail())
        {
            continue;
        }

        while (!mapIt->second.empty())
        {
            CLAPIBase* item = dynamic_cast<CLAPIBase*>(mapIt->second.front());

            m_mtxPreviousGEI.lock();

            // don't flush an item if it is the previous clGetEventInfo item for this thread
            // this is necessary so that consecutive api collapsing/accumulating works
            PreviousGEIMap::iterator iter = m_previousGEIMap.find(tid);

            if (iter != m_previousGEIMap.end() && (iter->second == item))
            {
                m_mtxPreviousGEI.unlock();
                break;
            }

            m_mtxPreviousGEI.unlock();

#ifdef NON_BLOCKING_TIMEOUT
            item->WriteTimestampEntry(foutTS, m_bTimeOutMode);

            // append event handle to the line
            if ((item->m_apiType & CL_ENQUEUE_BASE_API) > 1)
            {
                CLEnqueueAPIBase* pEnqAPI = dynamic_cast<CLEnqueueAPIBase*>(item);

                if (pEnqAPI->GetAPISucceeded() && pEnqAPI->GetEvent() != NULL)
                {
                    foutTS << setw(25) << pEnqAPI->GetEvent()->m_clEventString;
                }
            }

            foutTS << endl;
#else
            bool isReady = item->WriteTimestampEntry(foutTS, m_bTimeOutMode);

            if (!isReady)
            {
                // encountered not ready enqueue command
                break;
            }
            else
            {
                foutTS << endl;
            }

#endif

            // If isReady, write API entry as well
            item->WriteAPIEntry(foutTrace);
            foutTrace << endl;

            if (bEnableStackTrace)
            {
                item->WriteStackEntry(foutST);
                foutST << endl;
            }

            mapIt->second.pop_front();

            // don't remove clCreateCommandQueue, clCreateCommandQueueWithProperties, clCreateContext, clCreateContextFromType
            // In time out mode, we keep clCreateCommandQueue API Object so that we can retrieve device name and etc
            if (nullptr != item &&
                CL_FUNC_TYPE_clCreateCommandQueue != item->m_type &&
                CL_FUNC_TYPE_clCreateCommandQueueWithProperties != item->m_type &&
                CL_FUNC_TYPE_clCreateContext != item->m_type &&
                CL_FUNC_TYPE_clCreateContextFromType != item->m_type)
            {
                delete item;
            }
        }

        foutTrace.close();
        foutTS.close();

        if (bEnableStackTrace)
        {
            foutST.close();
        }
    }

    m_mtxFlush.unlock();
}

void CLAPIInfoManager::AddAPIInfoEntry(APIBase* api)
{
    CLAPIBase* en = dynamic_cast<CLAPIBase*>(api);

    if (IsInFilterList(en->m_type))
    {
        return;
    }

    if (IsCapReached())
    {
        // don't free clCreateCommandQueue, clCreateCommandQueueWithProperties, clCreateContext, and clCreateContextFromType since they may be referenced by future enqueue commands (in CLEnqueueAPIBase::GetContextInfo)
        if (nullptr != en &&
            CL_FUNC_TYPE_clCreateCommandQueue != en->m_type &&
            CL_FUNC_TYPE_clCreateCommandQueueWithProperties != en->m_type &&
            CL_FUNC_TYPE_clCreateContext != en->m_type &&
            CL_FUNC_TYPE_clCreateContextFromType != en->m_type)
        {
            delete en;
        }

        return;
    }

    en->m_tid = osGetUniqueCurrentThreadId();

    // if the user has asked to collapse clGetEventInfo calls, track the previous API called and special case the clGetEventInfo calls
    if (CLAPI_clGetEventInfo::ms_collapseCalls)
    {
        std::lock_guard<std::mutex> lock(m_mtxPreviousGEI);
        PreviousGEIMap::iterator iter = m_previousGEIMap.find(en->m_tid);
        bool bThreadFound = iter != m_previousGEIMap.end();

        if (CL_FUNC_TYPE_clGetEventInfo == en->m_type)
        {
            CLAPI_clGetEventInfo* clGetEventInfoAPI = static_cast<CLAPI_clGetEventInfo*>(en);

            if (bThreadFound)
            {
                // do the parameters of the current call match the parameters of the previous call?
                CLAPI_clGetEventInfo* prevEntry = iter->second;

                SP_TODO("if in test mode, then collapse ALL consecutive calls, regardless of their parameters");

                if (prevEntry->SameParameters(clGetEventInfoAPI))
                {
                    // increment the consecutive call count
                    prevEntry->m_consecutiveCount++;
                    // update the previous entry's end timestamp to match this entry's end timestamp
                    prevEntry->m_ullEnd = en->m_ullEnd;
                    // throw out this entry
                    delete en;


                    return;
                }

                // a clGetEventInfo call was made with different parameters -- update the previous call item in m_PreviousGEIMap
                iter->second = clGetEventInfoAPI;
            }
            else
            {
                m_previousGEIMap.insert(PreviousGEIMapPair(en->m_tid, clGetEventInfoAPI));
            }
        }
        else if (bThreadFound)
        {
            // a non-clGetEventInfo call was made -- erase the previous call item in the m_PreviousGEIMap
            m_previousGEIMap.erase(iter->first);
        }

    }

    // if we're tracing, add the api
    if (IsTracing())
    {
        APIInfoManagerBase::AddTraceInfoEntry(en);
        m_uiLineNum++;
    }
    else
    {
        // if we're not tracing, either add the api to the must-intercept api list (if it is a must-intercept api) or just delete it
        if (m_mustInterceptAPIs.find(en->m_type) != m_mustInterceptAPIs.end())
        {
            m_mustInterceptAPIList.push_back(en);
        }
        else
        {
            SAFE_DELETE(en);
        }
    }
}

void CLAPIInfoManager::AddToCommandQueueMap(const cl_command_queue cmdQueue, const CLAPI_clCreateCommandQueueBase* cmdQueueAPIObj)
{
    SP_TODO("add scope lock")
    CLCommandQueueMap::iterator it = m_clCommandQueueMap.find(cmdQueue);

    if (it != m_clCommandQueueMap.end())
    {
        it->second.push_back(cmdQueueAPIObj);
        return;
    }

    list<const CLAPI_clCreateCommandQueueBase*> queueList;
    queueList.push_back(cmdQueueAPIObj);
    m_clCommandQueueMap.insert(CLCommandQueueMapPair(cmdQueue, queueList));
}

void CLAPIInfoManager::AddToContextMap(const cl_context context, const CLAPI_clCreateContextBase* contextAPIObj)
{
    //-----------------
    CLContextMap::iterator it = m_clContextMap.find(context);

    if (it != m_clContextMap.end())
    {
        it->second.push_back(contextAPIObj);
        return;
    }

    list<const CLAPI_clCreateContextBase*> l;
    l.push_back(contextAPIObj);
    m_clContextMap.insert(CLContextMapPair(context, l));
}

const CLAPI_clCreateContextBase* CLAPIInfoManager::GetCreateContextAPIObj(const cl_context context)
{
    CLContextMap::iterator it = m_clContextMap.find(context);

    if (it != m_clContextMap.end() && it->second.size() > 0)
    {
        return it->second.back();
    }
    else
    {
        Log(logERROR, "Context pair not found\n");
        return NULL;
    }
}

const CLAPI_clCreateCommandQueueBase* CLAPIInfoManager::GetCreateCommandQueueAPIObj(const cl_command_queue cmdQueue)
{
    CLCommandQueueMap::iterator it = m_clCommandQueueMap.find(cmdQueue);

    if (it != m_clCommandQueueMap.end() && it->second.size() > 0)
    {
        return it->second.back();
    }
    else
    {
        Log(logERROR, "Command queue pair not found\n");
        return NULL;
    }
}

void CLAPIInfoManager::AddToKernelMap(const cl_kernel kernel, const char* szName)
{
    CLKernelMap::iterator it = m_clKernelMap.find(kernel);

    if (it != m_clKernelMap.end())
    {
        // Old kernel got deleted, new one shares the same pointer, replace the old one
        m_clKernelMap[ kernel ] = szName;
        return;
    }

    m_clKernelMap.insert(CLKernelMapPair(kernel, szName));
}


std::string& CLAPIInfoManager::GetKernelName(const cl_kernel kernel)
{
    CLKernelMap::iterator it = m_clKernelMap.find(kernel);

    if (it != m_clKernelMap.end())
    {
        return it->second;
    }
    else
    {
        Log(logERROR, "Kernel pair not found\n");
        static std::string emptyStr = "";
        return emptyStr;
    }
}

void CLAPIInfoManager::SaveToOutputFile()
{
    //*********************Atp file format*************************
    // TraceFileVersion=*.*
    // Application=
    // ApplicationArgs=
    // =====AMD APP Profiler Trace Output=====
    // [Thread ID]
    // [Num of Items]
    // [API trace item 0]
    // ...
    // [Another Thread block]
    // =====AMD APP Profiler Timestamp Output=====
    // [Thread ID]
    // [Num of Items]
    // [Timestamp item 0]
    // ...
    // [Another Thread block]
    //*********************End Atp file format**********************
    ofstream fout(m_strOutputFile.c_str());

    if (fout.fail())
    {
        Log(logWARNING, "Failed to open file: %s.\n", m_strOutputFile.c_str());
        cout << "Failed to generate .atp file: " << m_strOutputFile << ". Make sure you have permission to write to the path you specified." << endl;
        return;
    }
    else
    {
        WriteAPITraceDataToStream(fout);
        WriteTimestampToStream(fout);
        fout.close();
    }

    if (GlobalSettings::GetInstance()->m_params.m_bStackTrace)
    {
        string stackFile = FileUtils::GetBaseFileName(m_strOutputFile) + TRACE_STACK_EXT;

        ofstream fout1(stackFile.c_str());

        if (fout1.fail())
        {
            Log(logWARNING, "Failed to open file: %s.\n", stackFile.c_str());
            cout << "Failed to generate .atp file: " << stackFile << ". Make sure you have permission to write to the path you specified." << endl;
            return;
        }
        else
        {
            WriteStackTraceDataToStream(fout1);
            fout1.close();
        }
    }
}

void CLAPIInfoManager::Release()
{
    for (int i = 0; i < 2; i++)
    {
        for (TraceInfoMap::iterator mapIt = m_TraceInfoMap[i].begin(); mapIt != m_TraceInfoMap[i].end(); mapIt++)
        {
            for (list<ITraceEntry*>::iterator listIt = mapIt->second.begin(); listIt != mapIt->second.end(); listIt++)
            {
                CLAPIBase* item = dynamic_cast<CLAPIBase*>(*listIt);

                if ((!m_bTimeOutMode) ||
                    (nullptr != item &&
                     item->m_type != CL_FUNC_TYPE_clCreateCommandQueue &&
                     item->m_type != CL_FUNC_TYPE_clCreateCommandQueueWithProperties &&
                     item->m_type != CL_FUNC_TYPE_clCreateContext &&
                     item->m_type != CL_FUNC_TYPE_clCreateContextFromType))
                {
                    // if in timeout mode, don't delete clCreateCommandQueue* and clCreateContext* here
                    SAFE_DELETE(item);
                }
            }
        }
    }

    if (m_bTimeOutMode)
    {
        // In time out mode, we keep clCreateCommandQueue API Object so that we can retrieve device name and etc
        // we need to remove all clCreateCommandQueue* API Object
        for (CLCommandQueueMap::iterator it = m_clCommandQueueMap.begin(); it != m_clCommandQueueMap.end(); it++)
        {
            for (list<const CLAPI_clCreateCommandQueueBase*>::iterator lit = it->second.begin(); lit != it->second.end(); lit++)
            {
                SAFE_DELETE(*lit);
            }
        }

        m_clCommandQueueMap.clear();

        // In time out mode, we keep clCreateContext API Object so that we can retrieve device name and etc
        // we need to remove all clCreateContext API Object
        for (CLContextMap::iterator it = m_clContextMap.begin(); it != m_clContextMap.end(); it++)
        {
            for (list<const CLAPI_clCreateContextBase*>::iterator lit = it->second.begin(); lit != it->second.end(); lit++)
            {
                SAFE_DELETE(*lit);
            }
        }

        m_clContextMap.clear();
    }
    else
    {
        for (list<ITraceEntry*>::iterator it = m_mustInterceptAPIList.begin(); it != m_mustInterceptAPIList.end(); ++it)
        {
            SAFE_DELETE(*it);
        }
    }

    m_TraceInfoMap[0].clear();
    m_TraceInfoMap[1].clear();
}

void CLAPIInfoManager::AddAPIToFilter(const std::string& strAPIName)
{
    CL_FUNC_TYPE type = ToCLFuncType(strAPIName);

    if (type != CL_FUNC_TYPE_Unknown)
    {
        m_filterAPIs.insert(type);
    }
    else
    {
        Log(logWARNING, "Unknown API name = %s\n", strAPIName.c_str());
    }
}

bool CLAPIInfoManager::IsInFilterList(CL_FUNC_TYPE type)
{
    // If API is in Disabled API list but not in the MustIntercept API list
    return m_filterAPIs.find(type) != m_filterAPIs.end();
}

bool CLAPIInfoManager::ShouldIntercept(const char* szAPIName)
{
    string strName = szAPIName;
    CL_FUNC_TYPE type = ToCLFuncType(strName);
    SpAssert(type != CL_FUNC_TYPE_Unknown);

    if (type != CL_FUNC_TYPE_Unknown)
    {
        return !IsInFilterList(type) || m_mustInterceptAPIs.find(type) != m_mustInterceptAPIs.end();
    }
    else
    {
        return false;
    }
}

ULONGLONG CLAPIInfoManager::GetTimeNanosStart(CLAPIBase* pEntry)
{
#ifdef AMDT_INTERNAL

    if (GlobalSettings::GetInstance()->m_params.m_bUserPMC && pEntry != NULL)
    {
        PMCSamplerManager::Instance()->Sample(pEntry->m_PrePMCs);
    }

#else
    SP_UNREFERENCED_PARAMETER(pEntry);
#endif

    return OSUtils::Instance()->GetTimeNanos();
}

ULONGLONG CLAPIInfoManager::GetTimeNanosEnd(CLAPIBase* pEntry)
{
    ULONGLONG ret = OSUtils::Instance()->GetTimeNanos();
#ifdef AMDT_INTERNAL

    if (GlobalSettings::GetInstance()->m_params.m_bUserPMC && pEntry != NULL)
    {
        PMCSamplerManager::Instance()->Sample(pEntry->m_PostPMCs);
    }

#else
    SP_UNREFERENCED_PARAMETER(pEntry);
#endif

    return ret;
}

void CLAPIInfoManager::AddEnqueuedTask(const cl_kernel kernel)
{
    std::lock_guard<std::mutex> lock(m_mtxEnqueuedTask);
    m_enqueuedTasks.push_back(kernel);
}

bool CLAPIInfoManager::CheckEnqueuedTask(const cl_kernel kernel)
{
    std::lock_guard<std::mutex> lock(m_mtxEnqueuedTask);
    bool retVal = false;
    EnqueuedTaskList::iterator iter = std::find(m_enqueuedTasks.begin(), m_enqueuedTasks.end(), kernel);

    if (m_enqueuedTasks.end() != iter)
    {
        m_enqueuedTasks.erase(iter);
        retVal = true;
    }

    return retVal;
}

void CLAPITraceAgentTimerEndResponse(ProfilerTimerType timerType)
{
    switch (timerType)
    {
        case PROFILEDELAYTIMER:
            CLAPIInfoManager::Instance()->ResumeTracing();
            unsigned long profilerDuration;

            if (CLAPIInfoManager::Instance()->IsProfilerDurationEnabled(profilerDuration))
            {
                CLAPIInfoManager::Instance()->CreateTimer(PROFILEDURATIONTIMER, profilerDuration);
                CLAPIInfoManager::Instance()->SetTimerFinishHandler(PROFILEDURATIONTIMER, CLAPITraceAgentTimerEndResponse);
                CLAPIInfoManager::Instance()->startTimer(PROFILEDURATIONTIMER);
            }

            break;

        case PROFILEDURATIONTIMER:
            CLAPIInfoManager::Instance()->StopTracing();
            break;

        default:
            break;
    }
}

void CLAPIInfoManager::EnableProfileDelayStart(bool doEnable, unsigned long delayInMilliseconds)
{
    m_bDelayStartEnabled = doEnable;
    m_delayInMilliseconds = doEnable ? delayInMilliseconds : 0;
}

void CLAPIInfoManager::EnableProfileDuration(bool doEnable, unsigned long durationInMilliseconds)
{
    m_bProfilerDurationEnabled = doEnable;
    m_durationInMilliseconds = doEnable ? durationInMilliseconds : 0;
}

bool CLAPIInfoManager::IsProfilerDelayEnabled(unsigned long& delayInMilliseconds) const
{
    delayInMilliseconds = m_delayInMilliseconds;
    return m_bDelayStartEnabled;
}

bool CLAPIInfoManager::IsProfilerDurationEnabled(unsigned long& durationInMilliseconds) const
{
    durationInMilliseconds = m_durationInMilliseconds;
    return m_bProfilerDurationEnabled;
}

void CLAPIInfoManager::SetTimerFinishHandler(ProfilerTimerType timerType, TimerEndHandler timerEndHandler)
{
    switch (timerType)
    {
        case PROFILEDELAYTIMER:
            if (nullptr != m_pDelayTimer)
            {
                m_pDelayTimer->SetTimerFinishHandler(timerEndHandler);
            }

            break;

        case PROFILEDURATIONTIMER:
            if (nullptr != m_pDurationTimer)
            {
                m_pDurationTimer->SetTimerFinishHandler(timerEndHandler);
            }

            break;

        default:
            break;
    }
}

void CLAPIInfoManager::CreateTimer(ProfilerTimerType timerType, unsigned long timeIntervalInMilliseconds)
{
    switch (timerType)
    {
        case PROFILEDELAYTIMER:
            if (m_pDelayTimer == nullptr && timeIntervalInMilliseconds > 0)
            {
                m_pDelayTimer = new(std::nothrow) ProfilerTimer(timeIntervalInMilliseconds);

                if (nullptr == m_pDelayTimer)
                {
                    Log(logERROR, "CreateTimer: unable to allocate memory for delay timer\n");
                }
                else
                {
                    m_pDelayTimer->SetTimerType(PROFILEDELAYTIMER);
                    m_bDelayStartEnabled = true;
                    m_delayInMilliseconds = timeIntervalInMilliseconds;
                }
            }

            break;

        case PROFILEDURATIONTIMER:
            if (m_pDurationTimer == nullptr && timeIntervalInMilliseconds > 0)
            {
                m_pDurationTimer = new(std::nothrow) ProfilerTimer(timeIntervalInMilliseconds);

                if (nullptr == m_pDurationTimer)
                {
                    Log(logERROR, "CreateTimer: unable to allocate memory for duration timer\n");
                }
                else
                {
                    m_pDurationTimer->SetTimerType(PROFILEDURATIONTIMER);
                    m_bProfilerDurationEnabled = true;
                    m_durationInMilliseconds = timeIntervalInMilliseconds;
                }
            }

            break;

        default:
            break;
    }
}


void CLAPIInfoManager::startTimer(ProfilerTimerType timerType) const
{
    switch (timerType)
    {
        case PROFILEDELAYTIMER:
            if (nullptr != m_pDelayTimer)
            {
                m_pDelayTimer->startTimer(true);
            }

            break;

        case PROFILEDURATIONTIMER:
            if (nullptr != m_pDurationTimer)
            {
                m_pDurationTimer->startTimer(true);
            }

            break;

        default:
            break;
    }
}
