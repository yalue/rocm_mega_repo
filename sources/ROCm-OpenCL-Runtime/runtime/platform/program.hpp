//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

/*! \file program.hpp
 *  \brief  Declarations for Program and ProgramBinary objects.
 *
 *  \author Laurent Morichetti (laurent.morichetti@amd.com)
 *  \date   October 2008
 */

#ifndef PROGRAM_HPP_
#define PROGRAM_HPP_

#include "top.hpp"
#include "device/device.hpp"
#include "platform/object.hpp"
#include "platform/kernel.hpp"

#include <set>
#include <string>
#include <vector>
#include <map>
#include <utility>

namespace amd {

/*! \addtogroup Runtime
 *  @{
 *
 *  \addtogroup Program Programs and Kernel functions
 *  @{
 */

//! A kernel function symbol
class Symbol : public HeapObject {
 public:
  typedef std::unordered_map<const Device*, const device::Kernel*> devicekernels_t;

 private:
  devicekernels_t deviceKernels_;    //! All device kernels objects.
  KernelSignature signature_;        //! Kernel signature.

 public:
  //! Default constructor
  Symbol() {}

  //! Set the entry point and check or set the signature.
  bool setDeviceKernel(const Device& device,        //!< Device object.
                       const device::Kernel* func   //!< Device kernel object.
                       );

  //! Return the device kernel.
  const device::Kernel* getDeviceKernel(const Device& device //!< Device object.
                                        ) const;

  //! Return this Symbol's signature.
  const KernelSignature& signature() const { return signature_; }
};

class Context;

//! A collection of binaries for devices in the associated context.
class Program : public RuntimeObject {
 public:
  typedef std::pair<uint8_t*, size_t> binary_t;
  typedef std::set<Device const*> devicelist_t;
  typedef std::unordered_map<Device const*, binary_t> devicebinary_t;
  typedef std::unordered_map<Device const*, device::Program*> deviceprograms_t;
  typedef std::unordered_map<std::string, Symbol> symbols_t;

  enum Language {
    Binary = 0,
    OpenCL_C,
    SPIRV,
    Assembly,
    HIP
  };

  typedef bool(CL_CALLBACK* VarInfoCallback)(cl_program, std::string, void**, size_t*);
  VarInfoCallback varcallback;

 private:
  //! Replaces the compiled program with the new version from HD
  void StubProgramSource(const std::string& app_name);

  //! The context this program is part of.
  SharedReference<Context> context_;

  std::string sourceCode_;   //!< Strings that make up the source code
  Language language_;        //!< Input source language
  devicebinary_t binary_;    //!< The binary image, provided by the app
  symbols_t* symbolTable_;   //!< The program's kernels symbol table
  std::string kernelNames_;  //!< The program kernel names

  //! The device program objects included in this program
  deviceprograms_t devicePrograms_;
  devicelist_t deviceList_;

  std::string programLog_;  //!< Log for parsing options, etc.

 protected:
  //! Destroy this program.
  ~Program();

  //! Clears the program object if the app attempts to rebuild the program
  void clear();

  //! Global build lock (remove when LLVM is thread-safe).
  static Monitor buildLock_;

 public:
  //! Construct a new program to be compiled from the given source code.
  Program(Context& context, const std::string& sourceCode, Language language)
      : context_(context),
        sourceCode_(sourceCode),
        language_(language),
        symbolTable_(NULL),
        programLog_() {}

  //! Construct a new program associated with a context.
  Program(Context& context, Language language = Binary)
      : context_(context), language_(language), symbolTable_(NULL) {}

  //! Returns context, associated with the current program.
  const Context& context() const { return context_(); }

  //! Return the sections for this program.
  const deviceprograms_t& devicePrograms() const { return devicePrograms_; }

  //! Return the associated devices.
  const devicelist_t& deviceList() const { return deviceList_; }

  //! Return the symbols for this program.
  const symbols_t& symbols() const { return *symbolTable_; }

  //! Return the pointer to symbols for this program.
  const symbols_t* symbolsPtr() const { return symbolTable_; }

  //! Return the program source code.
  const std::string& sourceCode() const { return sourceCode_; }

  //! Return the program language.
  const Language language() const { return language_; }

  //! Append to source code.
  void appendToSource(const char* newCode) { sourceCode_.append(newCode); }

  //! Return the program log.
  const std::string& programLog() const { return programLog_; }

  //! Add a new device program with or without binary image and options.
  cl_int addDeviceProgram(Device&, const void* image = NULL, size_t len = 0,
                          amd::option::Options* options = NULL);

  //! Find the section for the given device. Return NULL if not found.
  device::Program* getDeviceProgram(const Device& device) const;

  //! Return the symbol for the given kernel name.
  const Symbol* findSymbol(const char* name) const;

  //! Return the binary image.
  const binary_t& binary(const Device& device) { return binary_[&device]; }

  //! Return the program kernel names
  const std::string& kernelNames() const { return kernelNames_; }

  //! Compile the program for the given devices.
  cl_int compile(const std::vector<Device*>& devices, size_t numHeaders,
                 const std::vector<const Program*>& headerPrograms, const char** headerIncludeNames,
                 const char* options = NULL,
                 void(CL_CALLBACK* notifyFptr)(cl_program, void*) = NULL, void* data = NULL,
                 bool optionChangable = true);

  //! Link the programs for the given devices.
  cl_int link(const std::vector<Device*>& devices, size_t numInputs,
              const std::vector<Program*>& inputPrograms, const char* options = NULL,
              void(CL_CALLBACK* notifyFptr)(cl_program, void*) = NULL, void* data = NULL,
              bool optionChangable = true);

  //! Build the program for the given devices.
  cl_int build(const std::vector<Device*>& devices, const char* options = NULL,
               void(CL_CALLBACK* notifyFptr)(cl_program, void*) = NULL, void* data = NULL,
               bool optionChangable = true);

  //! RTTI internal implementation
  virtual ObjectType objectType() const { return ObjectTypeProgram; }

  static int GetOclCVersion(const char* clVer);

  bool static ParseAllOptions(const std::string& options, option::Options& parsedOptions,
                              bool optionChangable, bool linkOptsOnly, bool isLC);

  void setVarInfoCallBack(VarInfoCallback callback) {
    varcallback = callback;
  }
};

/*! @}
 *  @}
 */

}  // namespace amd

#endif /*PROGRAM_HPP_*/
