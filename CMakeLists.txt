cmake_minimum_required(VERSION 3.17)
project(QDP-JIT VERSION 0.1 LANGUAGES CXX;C)

include(CheckSymbolExists)
include(CheckFunctionExists)
include(GNUInstallDirs)

include(FindOpenMP)          # Gives us find_package() for OpenMP

# I have not actually used these -- I dont know if they are nHAVE_STRNLENHAVE_STRNLENeeded
#
include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckTypeSize)


set(QDP_ENABLE_BACKEND "CUDA" CACHE STRING "The backend, can be either CUDA or ROCM")
set_property(CACHE QDP_ENABLE_BACKEND PROPERTY
  STRINGS CUDA ROCM)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" CACHE STRING "Root dir for CUDA")

option(QDP_ENABLE_PROFILING "Enable profiling" OFF)
option(QDP_ENABLE_ROCM_STATS "Enable ROCM statistics" OFF)
option(QDP_ENABLE_CUDA_MANAGED_MEMORY "Enable CUDA Managed Memory" OFF)

option(QDP_ENABLE_CUSTOM_KERNELS "Enable Custom Kernels" OFF)
option(QDP_ENABLE_CUDA_SPECIAL "Enable CUDA Special" OFF)
option(QDP_ENABLE_DEEP_LOG "Enable Deep logging" OFF)
set(QDP_DEEP_LOG ${QDP_ENABLE_DEEP_LOG})


set(QDP_ENABLE_CUDA_NVCC "nvcc" CACHE STRING "Path to cuda NVCC")
set(CUDA_NVCC ${QDP_ENABLE_CUDA_NVCCC})

option(QDP_ENABLE_LLVM12 "Enable changes to compile with LLVM12 release" ON)

option(QDP_BUILD_EXAMPLES "Build examples" ON)

if( QDP_ENABLE_LLVM12 )
  set(QDP_LLVM12 "1")
endif()

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
if( ${QDP_ENABLE_BACKEND} STREQUAL "CUDA")
  message(STATUS "Building For CUDA")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CUDA_GCC ${CMAKE_CXX_COMPILER})
  set(QDP_BACKEND_CUDA  "1")
  set(GPU_LIBS "CUDA::cuda_driver")
  if( QDP_ENABLE_CUDA_SPECIAL )
    set( QDP_CUDA_SPECIAL "yes" )
    set( GPU_LIBS ${GPU_LIBS} "CUDA::cudart")
  endif()
endif()

if( ${QDP_ENABLE_BACKEND} STREQUAL "ROCM")
  message(STATUS "Building for ROCM")
  set(QDP_BACKEND_ROCM "1")

  if( NOT DEFINED ROCM_PATH )
    if( NOT DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "/opt/rocm" CACHE_PATH "ROCm PATH")
    else()
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "ROCm PATH")
    endif()
  endif()

  if( NOT DEFINED  )
    if( NOT DEFINED ENV{GPU_TARGETS} )
	    set( "gfx908" CACHE STRING "THe AMD GPU TARGET")
    else()
	    set( $ENV{GPU_TARGETS} CACHE STRING "THe AMD GPU TARGET")
    endif()
  endif()
  message(STATUS "Setting  to ${GPU_TARGETS}") 
  set(CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake" ${CMAKE_MODULE_PATH})
  find_package(hip REQUIRED)

  set(GPU_LIBS hip::host;hip::device;hip::amdhip64)
  if( QDP_ENABLE_ROCM_STATS )
    set(QDP_USE_ROCM_STATS "1")
    find_package(amd_comgr REQUIRED)
    set(GPU_LIBS ${GPU_LIBS};amd_comgr)
  endif()
  set(ROCM_DIR ${ROCM_PATH})
endif()
 
set(QDP_IS_QDPJIT "1")

if ( QDP_ENABLE_CUSTOM_KERNELS )
  set(QDP_CUSTOM_KERNELS "yes" )
endif()

# Config parameters: ARCHs - one can turn them on/off
set(QDP_PARALLEL_ARCH "parscalar" CACHE STRING "The Parallel ARCH. One of scalar, parscalar")
set_property(CACHE QDP_PARALLEL_ARCH PROPERTY
		STRINGS scalar parscalar
) # FOR GUIs

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


# Alignment for OLattice stuff (64=512 bits e.g. KNL, Xeon)
set(QDP_ALIGNMENT_SIZE "64" CACHE STRING "OLattice memory alignment size in bytes (default 512)")


# Profiling and memory debugging
option(QDP_USE_PROFILING "Enable Profiling interface" OFF)
option(QDP_DEBUG_MEMORY "Enable Memory Debugging" OFF)

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
mark_as_advanced(QDP_NC, QDP_ND, QDP_NS, QDP_ALIGNMENT_SIZE, QDP_USE_PROFILING, QDP_DEBUG_MEMORY, QDP_USE_HDF5)

string(TOLOWER ${QDP_PARALLEL_ARCH} qdpParArch)
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



if ( ${qdpParArch} STREQUAL "parscalar" )
	message(STATUS "QDP++: Configuring for Parscalar build" )
	
	# Turn on Parallel QIO
	set(QIO_ENABLE_PARALLEL_BUILD ON CACHE BOOL "Parscalar build: force QIO Parallel build" FORCE)
		
	set(QDP_ARCH_PARSCALAR ON)  # Build System
	set(ARCH_PARSCALAR 1)      # Configured header file (should rationalize these)

elseif( ${qdpParArch} STREQUAL "scalar" )
  message(STATUS "QDP++: Configuring for Scalar build" ) 
     
  # Set QIO Parallel build to false
	set(QIO_ENABLE_PARALLEL_BUILD OFF CACHE BOOL "Scalar buld: Force QIO Scalar build" FORCE )
        
  set(QDP_ARCH_SCALAR ON)		# Build system 
	set(ARCH_SCALAR 1)   		# Configured header file (should rationalize this
else()
    message(ERROR "QDP++: Unknown architecture. Please set QDP_PARALLEL_ARCH to either scalar, parscalar, scalarvec or parscalarvec")
endif()


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
  message( ERROR "QDP++: Unknown layout. Please set either QDP_LAYOUT_LEXICO, QDP_LAYOUT_CB2, QDP_LAYOUT_CB3D or QDP_LAYOUT_CB32")
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

# Autoconf variable name backard compatibility 
set(QDP_AC_ALIGNMENT_SIZE ${QDP_ALIGNMENT_SIZE})
message( STATUS "QDP++: Setting alignment size to ${QDP_ALIGNMENT_SIZE}")

# CHeck for Strnlen which seems to be needed
check_function_exists("gethostname" HAVE_GETHOSTNAME)
check_function_exists("strnlen" HAVE_STRNLEN)

find_package(LLVM "12.0" REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in ${LLVM_DIR}")

# If we are parallel we need QMP
if ( QDP_ARCH_PARSCALAR  OR  QDP_ARCH_PARSCALARVEC )
  find_package(QMP REQUIRED)
endif()

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
  if( ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 3.1)
     OR ( CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 4.8) )
      message(STATUS "QDPXX: Setting Sanitizer options")
      set(QDP_SANITIZER_OPTS "-fsanitize=address,undefined")
      
      # Turn on sanitizers for module below me
      set(XPathReader_ENABLE_SANITIZERS ON)
      set(LibIntrin_ENABLE_SANITIZERS ON)
      set(QIO_ENABLE_SANITIZERS ON)
      set(FileDB_ENABLE_SANITIZERS ON)
  else()
    message(STATUS "Don't know how to enable sanitizers for ${CMAKE_CXX_COMPILER_ID}... Ignoring")
  endif()
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

# The tests -- test primarily whether SSE implementations 
# are correct so we can do that

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
