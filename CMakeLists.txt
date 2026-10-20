cmake_minimum_required(VERSION 3.17)
project(QDP-JIT VERSION 0.1 LANGUAGES CXX;C)

include(CheckSymbolExists)
include(CheckFunctionExists)
include(GNUInstallDirs)

include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckTypeSize)


set(QDP_ENABLE_BACKEND "CUDA" CACHE STRING "The backend, can be either CUDA, ROCM, or L0")
set_property(CACHE QDP_ENABLE_BACKEND PROPERTY STRINGS CUDA ROCM L0)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" CACHE STRING "Root dir for CUDA")

option(QDP_ENABLE_PROFILING "Enable profiling" OFF)
option(QDP_ENABLE_ROCM_STATS "Enable ROCM statistics" OFF)
option(QDP_JIT_ROCM_COMGR_LINK "ROCm: link JIT code objects in-process with amd_comgr instead of ld.lld" ON)


set(QDP_USE_PROFILING ${QDP_ENABLE_PROFILING})

option(QDP_ENABLE_DEEP_LOG "Enable Deep logging" OFF)
set(QDP_DEEP_LOG ${QDP_ENABLE_DEEP_LOG})

option(QDP_ENABLE_HOST_COMPILER "Enable the host compiler" ON)
set(QDP_HOST_COMPILER ${QDP_ENABLE_HOST_COMPILER})


#set(QDP_ENABLE_CUDA_NVCC "nvcc" CACHE STRING "Path to cuda NVCC")
#set(CUDA_NVCC ${QDP_ENABLE_CUDA_NVCCC})

# qdp-jit requires LLVM 22 or newer; this only asserts that LLVM 22 is found
option(QDP_ENABLE_LLVM22 "Require LLVM 22 exactly (the tested version)" OFF)

option(QDP_BUILD_EXAMPLES "Build examples" ON)




option(QDP_ENABLE_COMM_SPLIT_DEVICEINIT "Enable using MPI_Comm_split when initializing devices" OFF)
if(QDP_ENABLE_COMM_SPLIT_DEVICEINIT)
  set(QDP_USE_COMM_SPLIT_INIT "1")
endif()

##
## NB: Substituting Incflags and link variables will need
##     to be done differently for CUDA and ROCm
##     CUda has nice modules: CUDA::cuda CUDA::cudart
##     There is no find_package(for ROCm so we may need
##     To explicitly use -I flags and -L and -l flags...
##
if( ${QDP_ENABLE_BACKEND} STREQUAL "L0")
  list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

  message(STATUS "Building For Level Zero")
  set(QDP_BACKEND_L0  "1")
  find_package(PkgConfig REQUIRED)
  pkg_search_module(PKG_LLVMSPIRV REQUIRED LLVMSPIRVLib)
  # Remember where LLVMSPIRVLib.pc lives so QDPXXConfig.cmake can find it again
  pkg_get_variable(QDP_LLVMSPIRV_PCFILEDIR LLVMSPIRVLib pcfiledir)
  find_package(LevelZero REQUIRED)
endif()


if( ${QDP_ENABLE_BACKEND} STREQUAL "CUDA")
  message(STATUS "Building For CUDA")
  find_package(CUDAToolkit REQUIRED)
  set(QDP_BACKEND_CUDA  "1")
  set(GPU_LIBS "CUDA::cuda_driver")
endif()

 
set(QDP_IS_QDPJIT "1")
set(QDP_IS_QDPJIT2 "1")

set(QDP_PRECISION "double" CACHE STRING "THe Base Precision: single or double (default is double)")
set_property(CACHE QDP_PRECISION PROPERTY
		   STRINGS single double) # For GUIs

set(QDP_LAYOUT "cb2" CACHE STRING "The QDP layout: lexico, cb2, cb32, cb3d (default is cb2)")
set_property(CACHE QDP_LAYOUT PROPERTY 
		  STRINGS  lexico cb2 cb32 cb3d) # For GUIs

# NC, ND, NS
set(QDP_ND 4 CACHE STRING "Number of Spacetime Dimension (default 4)")
set(QDP_NC 3 CACHE STRING "Number of colors (default 3)")
set(QDP_NS 4 CACHE STRING "Number of spins (default 4)")



# LibXML on by default
option(QDP_USE_LIBXML2 "Use Libxml2 library" ON)

# FileDB on by default
option(QDP_USE_FILEDB  "Enable FileDB Usage" ON)

# HDF5 from Thorsten
option(QDP_USE_HDF5 "Enable HDF5 Code" OFF)

# Threaded Building Block (needs TBB installed, and their CMake Location on the Module path
option(QDP_USE_TBBPOOL_ALLOCATOR "Enable TBB based Pool Allocator" OFF)

# From our friends in Bonn - terse compilation
option(QDP_EXTRA_MESSAGES "Enable Extra diagnostic messages during compile time" OFF)


option(QDP_ENABLE_SANITIZERS "Enable Address and Undefined Behaviour Sanitizers" OFF)

# These options will only apperar in ccmake if you toggle advanced features
mark_as_advanced(QDP_NC, QDP_ND, QDP_NS, QDP_USE_PROFILING, QDP_USE_HDF5)

string(TOLOWER ${QDP_PRECISION} qdpPrecision)
string(TOLOWER ${QDP_LAYOUT} qdpLayout)

# Some status
message(STATUS "QDP++:  Configuring System")
message(STATUS "QDP++:  Nc=${QDP_NC}")
message(STATUS "QDP++:  Nd=${QDP_ND}")
message(STATUS "QDP++:  Ns=${QDP_NS}")

if( ${QDP_NS} EQUAL 1 ) 
  set(BUILD_STAGGERED_EXAMPLES ON)
else()
  set(BUILD_WILSON_EXAMPLES ON)
endif()



# The only architecture is parscalar (QMP/MPI); QIO is built parallel
set(QIO_ENABLE_PARALLEL_BUILD ON CACHE BOOL "Parscalar build: force QIO Parallel build" FORCE)


if( ${qdpLayout} STREQUAL "lexico") 
  message( STATUS "QDP++: Enabling Lexicographic Layout" )
  set(QDP_USE_LEXICO_LAYOUT 1)
  
elseif( ${qdpLayout} STREQUAL "cb2")
  message( STATUS "QDP++: Enabling CB2 (4D Checkerboard) Layout" )
  set(QDP_USE_CB2_LAYOUT 1)
  
elseif( ${qdpLayout} STREQUAL "cb32") 
  message( STATUS "QDP++: Enabling CB32 (32 colored ) Layout" )
  set(QDP_USE_CB32_LAYOUT 1)

elseif( ${qdpLayout} STREQUAL "cb3d")
  message( STATUS "QDP++: Enabling CB3D (3D checkerboarded ) Layout" )
  set(QDP_USE_CB3D_LAYOUT 1)

else()
  message( FATAL_ERROR "QDP++: Unknown layout ${QDP_LAYOUT}. Please set QDP_LAYOUT to lexico, cb2, cb32 or cb3d")
endif()

if( ${qdpPrecision} STREQUAL "single")
  set(BASE_PRECISION 32)
else()
  set(BASE_PRECISION 64)
endif()

message( STATUS "QDP++: Setting Base Precision to ${BASE_PRECISION}")


## These are higher up in the original confiure. They define CUDA_DIR and ROCM_DIR
#


if( QDP_ENABLE_EXTRA_MESSAGES )
  set(QPHIX_EMIT_MESSAGES ON) # FIXME: Refactor this later to use QDP
  message( STATUS "QDP++: Enabling pragma messages" )
endif()	


# CHeck for Strnlen which seems to be needed

if(QDP_ENABLE_BACKEND STREQUAL "ROCM")
  message(STATUS "Building for ROCM")
  set(QDP_BACKEND_ROCM "1")

  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm installation prefix")
    else()
      set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation prefix")
    endif()
  endif()

  set(ROCM_DIR "${ROCM_PATH}")

  if(NOT DEFINED LLVM_DIR)
    set(LLVM_DIR "${ROCM_PATH}/lib/llvm/lib/cmake/llvm" CACHE PATH "LLVMConfig.cmake from ROCm LLVM")
  endif()

  if(NOT DEFINED hip_DIR)
    set(hip_DIR "${ROCM_PATH}/lib/cmake/hip" CACHE PATH "HIP CMake package")
  endif()

  list(PREPEND CMAKE_PREFIX_PATH
    "${ROCM_PATH}"
    "${ROCM_PATH}/lib/llvm"
  )

  # hip-config overwrites ROCM_PATH with $ENV{ROCM_PATH}; keep ours so ld.lld
  # and device bitcode come from the same ROCm as HIP.
  set(_qdp_rocm_path "${ROCM_PATH}")
  find_package(hip REQUIRED)
  set(ROCM_PATH "${_qdp_rocm_path}")

if(NOT QDP_JIT_ROCM_COMGR_LINK)
find_program(QDP_ROCM_LD_LLD
  NAMES ld.lld
  PATHS
    "${ROCM_PATH}/lib/llvm/bin"
    "${ROCM_PATH}/llvm/bin"
    "${ROCM_PATH}/bin"
  NO_DEFAULT_PATH
)

if(NOT QDP_ROCM_LD_LLD)
  message(FATAL_ERROR
    "Could not find ROCm ld.lld under ${ROCM_PATH}. "
    "Expected something like ${ROCM_PATH}/lib/llvm/bin/ld.lld")
endif()

message(STATUS "Using ROCm ld.lld: ${QDP_ROCM_LD_LLD}")
endif()


  set(GPU_LIBS hip::amdhip64)

  if(QDP_JIT_ROCM_COMGR_LINK OR QDP_ENABLE_ROCM_STATS)
    find_package(amd_comgr CONFIG REQUIRED)
    message(STATUS "Using amd_comgr: ${amd_comgr_DIR}")
  endif()

  if(QDP_ENABLE_ROCM_STATS)
    set(QDP_USE_ROCM_STATS "1")
    set(GPU_LIBS ${GPU_LIBS} amd_comgr)
  endif()

  set(QDPJIT_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}")
endif()


find_package(LLVM CONFIG REQUIRED)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in ${LLVM_DIR}")

set(QDP_LLVM_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
set(QDP_LLVM_VERSION_MINOR ${LLVM_VERSION_MINOR})

# LLVM 22 or newer is required; 22 is the tested version.
if(QDP_ENABLE_BACKEND STREQUAL "ROCM")
  set(_qdp_llvm_hint " With the ROCM backend, LLVM_DIR defaults to ROCm's bundled LLVM (${ROCM_PATH}/lib/llvm) unless set explicitly.")
endif()

if(LLVM_VERSION_MAJOR LESS 22)
  message(FATAL_ERROR
    "qdp-jit requires LLVM 22 or newer, but found LLVM ${LLVM_PACKAGE_VERSION} "
    "in ${LLVM_DIR}. Point LLVM_DIR at an LLVM 22 installation.${_qdp_llvm_hint}")
endif()

if(QDP_ENABLE_LLVM22 AND NOT LLVM_VERSION_MAJOR EQUAL 22)
  message(FATAL_ERROR
    "QDP_ENABLE_LLVM22=ON but found LLVM ${LLVM_PACKAGE_VERSION} "
    "in ${LLVM_DIR}. Point LLVM_DIR at an LLVM 22 installation "
    "or drop QDP_ENABLE_LLVM22.${_qdp_llvm_hint}")
endif()

if(LLVM_VERSION_MAJOR GREATER 22)
  message(WARNING "Found LLVM ${LLVM_PACKAGE_VERSION}; qdp-jit is tested with LLVM 22.")
endif()




if(QDP_ENABLE_BACKEND STREQUAL "ROCM")
  file(REAL_PATH "${ROCM_PATH}" ROCM_REAL)
  file(REAL_PATH "${LLVM_DIR}" LLVM_DIR_REAL)

  string(FIND "${LLVM_DIR_REAL}" "${ROCM_REAL}" LLVM_DIR_INSIDE_ROCM)

  if(LLVM_DIR_INSIDE_ROCM EQUAL -1)
    message(WARNING
      "ROCm backend is enabled, but LLVM_DIR does not appear to come from ROCM_PATH.\n"
      "  ROCM_PATH = ${ROCM_PATH}\n"
      "  LLVM_DIR  = ${LLVM_DIR}\n"
      "This may fail when reading ROCm device bitcode. Prefer:\n"
      "  -DLLVM_DIR=${ROCM_PATH}/lib/llvm/lib/cmake/llvm")
  endif()
endif()


if(QDP_ENABLE_BACKEND STREQUAL "ROCM")
  set(QDP_ROCM_BITCODE_DIR "${ROCM_PATH}/amdgcn/bitcode" CACHE PATH "ROCm AMDGCN bitcode directory")

  set(QDP_ROCM_REQUIRED_BC
    ocml.bc
    oclc_finite_only_off.bc
    oclc_unsafe_math_off.bc
    oclc_daz_opt_off.bc
    oclc_correctly_rounded_sqrt_on.bc
    oclc_abi_version_500.bc
  )

  set(QDP_ROCM_OPTIONAL_BC
    oclc_daz_opt_off.bc
    oclc_correctly_rounded_sqrt_on.bc
  )

  foreach(bc IN LISTS QDP_ROCM_OPTIONAL_BC)
    if(NOT EXISTS "${QDP_ROCM_BITCODE_DIR}/${bc}")
      message(STATUS "ROCm device bitcode ${bc} is absent; this ROCm has retired that control")
    endif()
  endforeach()

  foreach(bc IN LISTS QDP_ROCM_REQUIRED_BC)
    if(NOT EXISTS "${QDP_ROCM_BITCODE_DIR}/${bc}")
      message(FATAL_ERROR "Required ROCm device bitcode not found: ${QDP_ROCM_BITCODE_DIR}/${bc}")
    endif()
  endforeach()
endif()



find_package(QMP REQUIRED)

# If we use HDF5 we need HDF5
if ( QDP_USE_HDF5 ) 
  find_package(HDF5 REQUIRED)
  message(STATUS "QDP++: Building with HDF5")
endif()

# If we want filedb mark it on 
if ( QDP_USE_FILEDB ) 
   set(BUILD_FILEDB ON)
endif()

# If we use Threaded building blocks Find TBB
if( QDP_USE_TBBPOOL_ALLOCATOR ) 
   find_package(TBB REQUIRED )
   message(STATUS "QDP++: Using Threader Building Blocks Pool Allocator")
endif()

# Write config file with currently set values
configure_file(include/qdp_config_internal.h.cmake.in include/qdp_config_internal.h)

# Install generated file (when make install is called)
# I cannot mark the configured file as a 'target' but I can 
# Use CMAKE_INSTALL_INCLUDEDIR from the GNUInstallDirs to select the standard
# Include dir. On Linux/UNIX it should be CMAKE_INSTALL_PREFIX/include 
#
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/qdp_config_internal.h 
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Deal with Sanitizer
if( QDP_ENABLE_SANITIZERS )
  include(cmake/CheckSanitizeOpts.cmake)
  check_sanitizer_options( "${QDP_ENABLE_SANITIZERS}" QDP_SANITIZER_OPTS )
  message(STATUS "QDPXX: Setting Sanitizer options: ${QDP_SANITIZER_OPTS}")
  set(XPathReader_ENABLE_SANITIZERS ${QDP_ENABLE_SANITIZERS} )
  set(QIO_ENABLE_SANITIZERS ${QDP_ENABLE_SANITIZERS})
  set(FileDB_ENABLE_SANITIZERS ${QDP_ENABLE_SANITIZERS} )
endif()  


# Configure subdirectories
# Include just does installing
add_subdirectory(include)

# In practice this is always taken, but
# in principle we can make QDP++ not depend on LibXML
# 
if (QDP_USE_LIBXML2 ) 
  add_subdirectory(other_libs/xpath_reader)
endif()

# FileDB nearly always gets built
if (QDP_USE_FILEDB )
  add_subdirectory(other_libs/filedb)
endif()

# We always build QIO
add_subdirectory(other_libs/qio)
add_subdirectory(lib)


# The examples and nersc2ildg etc
if( QDP_BUILD_EXAMPLES )
  add_subdirectory(examples)
endif()

#Install config files.
# Make the config file
include(CMakePackageConfigHelpers)
configure_package_config_file(QDPXXConfig.cmake.in QDPXXConfig.cmake INSTALL_DESTINATION lib/cmake/QDPXX)

# Make the version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    QDPXXConfigVersion.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
    )

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/QDPXXConfigVersion.cmake ${CMAKE_CURRENT_BINARY_DIR}/QDPXXConfig.cmake DESTINATION lib/cmake/QDPXX)
