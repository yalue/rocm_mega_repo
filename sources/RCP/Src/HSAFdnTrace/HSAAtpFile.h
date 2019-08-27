//==============================================================================
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
/// \author AMD Developer Tools Team
/// \file
/// \brief  HSA Atp File writer and parser
//==============================================================================

#ifndef _HSA_ATP_FILE_H_
#define _HSA_ATP_FILE_H_

#include "../sprofile/AtpFile.h"
#include "../Common/FileUtils.h"
#include "HSAAPIInfo.h"

//------------------------------------------------------------------------------------
/// HSA API trace result
//------------------------------------------------------------------------------------
class HSAAtpFilePart : public IAtpFilePart, public IAtpFilePartParser, public BaseParser<HSAAPIInfo>
{
public:
    /// Constructor
    /// \param config Config object
    HSAAtpFilePart(const Config& config, bool shouldReleaseMemory = true);

    /// Destructor
    ~HSAAtpFilePart(void);

    /// Write header section
    /// If a AptFilePart wants to output to header section, implement this method
    /// \param sout Output stream
    void WriteHeaderSection(SP_fileStream& sout) override;

    /// Write content section
    /// \param sout Output stream
    /// \param strTmpFilePath Output fragment files path
    /// \param strPID child process ID
    /// \return true if any contents were written, false otherwise
    bool WriteContentSection(SP_fileStream& sout, const std::string& strTmpFilePath, const std::string& strPID) override;

    /// Parse input stream
    /// \param in Input stream
    /// \return True if succeeded
    bool Parse(std::istream& in, std::string& outErrorMsg) override;

    /// Parse header
    /// \param strKey Key name
    /// \param strVal Value
    /// \return True if succeeded
    bool ParseHeader(const std::string& strKey, const std::string& strVal) override;

protected:
    /// Parse host side timestamp
    /// \param buf Input string
    /// \param[out] pAPIInfo API info object
    /// \param bTimeoutMode Timeout mode flag
    /// \return true if no error occurs
    bool ParseHostTimestamp(const char* buf, HSAAPIInfo* pAPIInfo, bool bTimeoutMode = false);

    /// Parse device side timestamp
    /// \param buf Input string
    /// \param[out] pDispatchInfo Dispatch info object
    /// \return true if no error occurs
    bool ParseDeviceTimestamp(const char* buf, HSADispatchInfo* pDispatchInfo);

    /// Create API info object from API name
    /// \param strAPIName API name
    /// \return API info object
    HSAAPIInfo* CreateAPIInfo(const std::string& strAPIName);

private:
    /// struct used to match data from the temp async copy timestamp file to the api timestamps
    struct AsyncCopyItem
    {
        std::string m_strSignalHandle;     ///< signal handle string from a async copy timestamp
        uint64_t    m_start;               ///< start timestamp from a async copy
        uint64_t    m_end;                 ///< end timestamp from a async copy
        uint32_t    m_apiIndex;            ///< api index of the hsa_amd_memory_async_copy api that initiated this data transfer
        uint64_t    m_asyncCopyIdentifier; ///< unique identifier for a certain async copy call
    };

    /// Update tmp timestamp file
    /// \param strTmpFilePath Tmp file path
    /// \param strFilePrefix File prefix
    /// \return true on success
    bool UpdateTmpTimestampFiles(const std::string& strTmpFilePath, const std::string& strFilePrefix);

    /// typedef for a list of AsyncCopyItem
    typedef std::vector<AsyncCopyItem> AsyncCopyItemList;

    /// typedef for a map from thread id to AsyncCopyItemList
    typedef std::unordered_map<osThreadId, AsyncCopyItemList> ThreadCopyItemMap;

    /// Loads Async Copy timestamp info from the specified temp file into the specified ThreadCopyItemMap
    /// \param strFile the .copytstamp temp file to load
    /// \param threadCopyInfoMap the map to fill with async copy timestamp info
    /// \return true on success
    bool LoadAsyncCopyTimestamps(const std::string& strFile, ThreadCopyItemMap& threadCopyInfoMap);

    /// Updates the .tstamp file to include the async copy timestamps in the specified threadCopyInfoMap
    /// \param strFile the .tstamp file
    /// \param threadCopyInfoMap the map containing the loaded async copy timestamps
    /// \return true on success
    bool UpdateAsyncCopyTimestamps(const std::string strFile, ThreadCopyItemMap threadCopyInfoMap);

    /// Checks the thread id encoded in the temp file name to see if that thread is found in the
    /// specified threadCopyInfoMap
    /// \param strFile the temp file name to check
    /// \param threadCopyInfoMap the map containing the loaded async copy timestamps
    /// \param[out] threadId thread id that is found
    /// \return if the strFile is from a thread that can be found in the threadCopyInfoMap
    bool IsCorrectTidFile(const std::string strFile, ThreadCopyItemMap threadCopyInfoMap, osThreadId& threadId);

    HSAAPIInfoMap       m_HSAAPIInfoMap;       ///< HSA API info map
    HSADispatchInfoList m_HSADispatchInfoList; ///< HSA Dispatch Info list

    unsigned int        m_dispatchIndex;       ///< dispatch index, incremented while parsing kernel dispatch timestamps
    unsigned int        m_atpMajorVer;         ///< major version of the .atp file
    unsigned int        m_atpMinorVer;         ///< minor version of the .atp file
};

#endif //_HSA_ATP_FILE_H_
