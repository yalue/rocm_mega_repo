//==============================================================================
// Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
/// \author AMD Developer Tools Team
/// \file
/// \brief  This file contains code to collect timestamps from AQL packets.
///         NOTE: the code here is a direct port of the libhsa-runtime-tools
///               code that performs the same function.  In discussions with
///               the runtime team, we've decided that this code belongs in
///               the profiler rather than in the runtime.  This code can
///               probably be cleaned up a bit from its current state, but that
///               can be done as a future step, after we've validated that
///               everything is working correctly.
//==============================================================================

#ifndef _HSA_AQL_PACKET_TIME_COLLECTOR_H_
#define _HSA_AQL_PACKET_TIME_COLLECTOR_H_


#include <stack>
#include <queue>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <map>
#include <vector>
#include <limits>

#include "hsa.h"

#include <AMDTOSWrappers/Include/osThread.h>

#if defined (_LINUX) || defined (LINUX)
    #include <AMDTOSWrappers/Include/osCondition.h>
#endif

#include <TSingleton.h>

#include "HSAAqlPacketInfo.h"

/// Struct that handles replacement signals for AQL packets
///
/// We need to replace signals to allow us to collect timestamps from the signals
/// from a thread.  By replacing the signal, we can manage when the user application
/// sees completion.  This allows us to get the timestamps from the signal without
/// needing to worry about the user application destroying the signal out from under
/// us.
struct HSAPacketSignalReplacer
{
    HSAAqlKernelDispatchPacket* m_pAqlPacket;      ///< the AQL packet
    hsa_signal_t                m_originalSignal;  ///< the original signal provided by the user application
    hsa_signal_t                m_profilerSignal;  ///< the replacement signal created by us that is actually given to the runtime
    hsa_agent_t                 m_agent;           ///< the agent on which the packet is dispatched
    const hsa_queue_t*          m_pQueue;          ///< the queue on which the packet is dispatched

    /// Constructor
    HSAPacketSignalReplacer(HSAAqlKernelDispatchPacket* pAqlPacket,
                            hsa_signal_t originalSignal,
                            hsa_signal_t profilerSignal,
                            hsa_agent_t agent,
                            const hsa_queue_t* pQueue) :
        m_pAqlPacket(pAqlPacket),
        m_originalSignal(originalSignal),
        m_profilerSignal(profilerSignal),
        m_agent(agent),
        m_pQueue(pQueue)
    {
    }

    /// Default Constructor
    HSAPacketSignalReplacer() {}
};

/// Singleton class for global vars used by the SignalCollector thread and supporting code
class HSATimeCollectorGlobals: public TSingleton<HSATimeCollectorGlobals>
{
    friend class TSingleton<HSATimeCollectorGlobals>;

public:
    hsa_signal_t              m_forceSignalCollection;  ///< signal used to indicate that the collector thread should collect timestamps for all remaining signals
    bool                      m_doQuit;                 ///< flag to indicate that the signal collector should finish
    std::mutex                m_signalCollectorMtx;     ///< mutex protecting signal collection phase
#if defined (_LINUX) || defined (LINUX)
    osCondition               m_dispatchesInFlight;     ///< condition used to wake up the signal collector thread
#endif

private:
    /// Constructor
    HSATimeCollectorGlobals();
};

/// Singleton class that holds list of kernel dispatch packets with replacement
/// signals that are in flight
class HSASignalQueue : public TSingleton<HSASignalQueue>
{
    friend class TSingleton<HSASignalQueue>;

public:
    /// Add a signal to the queue
    /// \param signal the replacer signal to add to the back of the queue
    /// \return true if the queue entry of the queue map is empty
    bool AddSignalToBack(const HSAPacketSignalReplacer& signal);

    /// Get the signal from the queue
    /// \param[out] outSignals the replacer signal pointer array that holds all front entries of each queue in the queue map
    void GetSignalFromFront(std::vector<const HSAPacketSignalReplacer*>& outSignals);

    /// Pop the signal from the queue
    /// \param queue the queue pointer that need to be popped out from the queue map
    void PopSignalFromQueue(const hsa_queue_t* queue);

    /// Test whether the queue map is empty
    /// \return empty status
    bool IsEmpty();

    /// Gets the size of the queue
    /// \return the size of the queue
    size_t GetSize() const;

    /// Clears the queue
    void Clear();

private:
    std::map<const hsa_queue_t*, std::queue<HSAPacketSignalReplacer>> m_queueSignalsMap; ///< Queue map holding the signal replacers
    std::mutex                m_signalQueueMtx;                                          ///< Mutex protecting access to m_signalList
};

/// Thread to collect timestamps for kernel dispatch packets.
class HSASignalCollectorThread : public osThread
{
public:
    /// Constructor
    HSASignalCollectorThread();

protected:
    /// overridden function that is the main thread entry point
    virtual int entryPoint();

private:
    static const unsigned int m_deferLen = 10;          ///< number of deferred signals to track
    HSAPacketSignalReplacer   m_deferList[m_deferLen];  ///< deferred signal list -- these are signals that are done, but we don't collect their timestamps until later
    unsigned int              m_index;                  ///< index used to track which signals we've already collected timestamps for
};

#endif // _HSA_AQL_PACKET_TIME_COLLECTOR_H_
