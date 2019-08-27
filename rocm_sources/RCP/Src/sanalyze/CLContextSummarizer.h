//==============================================================================
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
/// \author AMD Developer Tools Team
/// \file
/// \brief  This file provides a CL Context Summarizer
//==============================================================================

#ifndef _CL_CONTEXT_SUMMARIZER_H_
#define _CL_CONTEXT_SUMMARIZER_H_

#include <map>
#include <vector>
#include "../CLTraceAgent/CLAPIInfo.h"
#include "../Common/IParserListener.h"
#include "../Common/OSUtils.h"
#include "CLKernelSummarizer.h"

//------------------------------------------------------------------------------------
/// Context Summary table header
//------------------------------------------------------------------------------------
class ContextSummaryItems
{
public:
    unsigned int uiContextID;           ///< Context ID

    unsigned int uiNumMemOp;            ///< Number of memory transfers
    ULONGLONG ullTotalMemDuration;      ///< Total time on data transfer

    KernelSumMap KernelMap;          ///< Per context, per device kernel summary

    //unsigned int uiDeviceID;

    unsigned int uiNumCopy;             ///< Number of EnqueueCopy*
    unsigned int uiNumMap;              ///< Number of EnqueueMap*
    unsigned int uiNumWrite;            ///< Number of EnqueueWrite*
    unsigned int uiNumRead;             ///< Number of EnqueueRead*

    ULONGLONG ullByteCopy;            ///< size of data transfer from EnqueueCopy*
    ULONGLONG ullByteMap;             ///< size of data transfer from EnqueueMap*
    ULONGLONG ullByteWrite;           ///< size of data transfer from EnqueueWrite*
    ULONGLONG ullByteRead;            ///< size of data transfer from EnqueueRead*

    ULONGLONG ullDurationCopy;          ///< Ttoal duration for EnqueueCopy*
    ULONGLONG ullDurationMap;           ///< Ttoal duration for EnqueueMap*
    ULONGLONG ullDurationWrite;         ///< Ttoal duration for EnqueueWrite*
    ULONGLONG ullDurationRead;          ///< Ttoal duration for EnqueueRead*

    unsigned int uiNumBuffer;           ///< Number of Buffers created on this context
    unsigned int uiNumImage;            ///< Number of Images created on this context
    unsigned int uiNumQueue;            ///< Number of Queues created on this context

    /// Constructor
    ContextSummaryItems() :
        uiContextID(static_cast<unsigned int>(-1)),
        uiNumMemOp(0u),
        ullTotalMemDuration(0u),
        uiNumCopy(0u),
        uiNumMap(0u),
        uiNumWrite(0u),
        uiNumRead(0u),
        ullByteCopy(0u),
        ullByteMap(0u),
        ullByteWrite(0u),
        ullByteRead(0u),
        ullDurationCopy(0ull),
        ullDurationMap(0ull),
        ullDurationWrite(0ull),
        ullDurationRead(0ull),
        uiNumBuffer(0u),
        uiNumImage(0u),
        uiNumQueue(0u)
    {}

    /// Copy constructor
    /// \param obj object
    ContextSummaryItems(const ContextSummaryItems& obj)
    {
        uiNumCopy      =  obj.uiNumCopy;
        uiNumMap       =  obj.uiNumMap;
        uiNumWrite     =  obj.uiNumWrite;
        uiNumRead      =  obj.uiNumRead;

        ullByteCopy    =  obj.ullByteCopy;
        ullByteMap     =  obj.ullByteMap;
        ullByteWrite   =  obj.ullByteWrite;
        ullByteRead    =  obj.ullByteRead;

        ullDurationCopy =  obj.ullDurationCopy;
        ullDurationMap  =  obj.ullDurationMap;
        ullDurationWrite =  obj.ullDurationWrite;
        ullDurationRead =  obj.ullDurationRead;

        uiContextID    =  obj.uiContextID;

        uiNumMemOp = obj.uiNumMemOp;

        ullTotalMemDuration = obj.ullTotalMemDuration;

        // Use std::map's assignment operator, shallow copy is enough here
        KernelMap = obj.KernelMap;

        uiNumBuffer = obj.uiNumBuffer;
        uiNumImage = obj.uiNumImage;
        uiNumQueue = obj.uiNumQueue;
    }

    /// Assignment operator
    /// \param obj object
    /// \return ref to itself
    const ContextSummaryItems& operator=(const ContextSummaryItems& obj)
    {
        if (this != &obj)
        {
            uiNumCopy      =  obj.uiNumCopy;
            uiNumMap       =  obj.uiNumMap;
            uiNumWrite     =  obj.uiNumWrite;
            uiNumRead      =  obj.uiNumRead;

            ullByteCopy    =  obj.ullByteCopy;
            ullByteMap     =  obj.ullByteMap;
            ullByteWrite   =  obj.ullByteWrite;
            ullByteRead    =  obj.ullByteRead;

            ullDurationCopy =  obj.ullDurationCopy;
            ullDurationMap  =  obj.ullDurationMap;
            ullDurationWrite =  obj.ullDurationWrite;
            ullDurationRead =  obj.ullDurationRead;

            uiContextID    =  obj.uiContextID;

            uiNumMemOp = obj.uiNumMemOp;

            ullTotalMemDuration = obj.ullTotalMemDuration;

            // Use std::map's assignment operator, shallow copy is enough here
            KernelMap = obj.KernelMap;

            uiNumBuffer = obj.uiNumBuffer;
            uiNumImage = obj.uiNumImage;
            uiNumQueue = obj.uiNumQueue;
        }

        return *this;
    }

    /// Plus operator
    /// \param obj object
    /// \return ref to itself
    const ContextSummaryItems& operator+=(const ContextSummaryItems& obj)
    {
        uiNumCopy      +=  obj.uiNumCopy;
        uiNumMap       +=  obj.uiNumMap;
        uiNumWrite     +=  obj.uiNumWrite;
        uiNumRead      +=  obj.uiNumRead;

        ullByteCopy     +=  obj.ullByteCopy;
        ullByteMap      +=  obj.ullByteMap;
        ullByteWrite    +=  obj.ullByteWrite;
        ullByteRead     +=  obj.ullByteRead;

        ullDurationCopy +=  obj.ullDurationCopy;
        ullDurationMap  +=  obj.ullDurationMap;
        ullDurationWrite +=  obj.ullDurationWrite;
        ullDurationRead +=  obj.ullDurationRead;

        uiNumMemOp += obj.uiNumMemOp;
        ullTotalMemDuration += obj.ullTotalMemDuration;

        // Search device type, if existed, update it, if not, copy and add to kernel map.
        for (KernelSumMap::const_iterator it = obj.KernelMap.begin(); it != obj.KernelMap.end(); it++)
        {
            KernelSumMap::iterator thisIt = KernelMap.find(it->first);

            if (thisIt != KernelMap.end())
            {
                thisIt->second.ullTotalTime += it->second.ullTotalTime;
                thisIt->second.uiNumCalls += it->second.uiNumCalls;
            }
            else
            {
                KernelSummaryItems kitem;
                kitem = it->second;
                KernelMap.insert(std::pair< std::string, KernelSummaryItems>(it->first, kitem));
            }
        }

        uiNumBuffer += obj.uiNumBuffer;
        uiNumImage += obj.uiNumImage;
        uiNumQueue += obj.uiNumQueue;

        return *this;
    }
};

//------------------------------------------------------------------------------------
/// Temporary CLObject counter per context
/// When we handle clCreate* APIs, we don't know what context ID but context handle the object was created on.
/// m_tmpCLObjCounter maintains a map from context handle to number of cl objects that were created on the context.
/// Context handle could be reused, therefore, we flush tmp counter everytime we handle an enqueueCmd
//------------------------------------------------------------------------------------
struct CLObjectCounter
{
    unsigned int uiImageCount;    /// Image count
    unsigned int uiBufferCount;   /// Buffer count
    unsigned int uiQueueCount;    /// Queue count

    /// Constructor
    CLObjectCounter() :
        uiImageCount(0),
        uiBufferCount(0),
        uiQueueCount(0)
    {
    }
};

typedef std::map<unsigned int, ContextSummaryItems> ContextSumMap;


//------------------------------------------------------------------------------------
/// OpenCL Context Summarizer
//------------------------------------------------------------------------------------
class CLContextSummarizer
    : public IParserListener<CLAPIInfo>
{
public:
    /// Constructor
    CLContextSummarizer(void);

    /// Destructor
    ~CLContextSummarizer(void);

    /// Listener function
    /// \param pAPIInfo API Info object
    /// \param[out] stopParsing flag indicating if parsing should stop after this item
    void OnParse(CLAPIInfo* pAPIInfo, bool& stopParsing);

    /// Generate HTML table from statistic data and write to std::ostream
    /// \param sout output stream
    void GenerateHTMLTable(std::ostream& sout);

    /// Generate simple HTML page
    /// \param szFileName file name
    /// \return true if the page was generated, false otherwise
    bool GenerateHTMLPage(const char* szFileName);

    /// When we handle clCreate* APIs, we don't know what context ID but context handle the object was created on.
    /// m_tmpCLObjCounter maintains a map from context handle to number of cl objects that were created on the context.
    /// Context handle could be reused, therefore, we flush tmp counter everytime we handle an enqueueCmd
    void FlushTmpCounters(std::string& strCntx, ContextSummaryItems* pItems);

protected:
    ContextSumMap m_ContextSumMap;                                 ///< Context summary map (ContextID to ContextSummaryItems )
    std::map< std::string, CLObjectCounter > m_tmpCLObjCounter;    ///< Temp cl object counter: map from context handle string to CLObjectCounter
    std::vector< std::string > m_vecDevices;                       ///< Global Devices (as oppose to devices per context) created, this is used generate table header
private:
    /// Copy constructor
    /// \param obj object
    CLContextSummarizer(const CLContextSummarizer& obj);

    /// Assignment operator
    /// \param obj object
    /// \return ref to itself
    const CLContextSummarizer& operator = (const CLContextSummarizer& obj);
};

#endif //_CL_CONTEXT_SUMMARIZER_H_
