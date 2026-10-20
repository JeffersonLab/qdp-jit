

add_library(qdp qdp_map.cc 
	qdp_subset.cc 
	qdp_random.cc
	qdp.cc
	qdp_layout.cc 
	qdp_layout_util.cc 
	qdp_layout_stack.cc 
	qdp_shape.cc 
	qdp_io.cc
	qdp_byteorder.cc
	qdp_util.cc
	qdp_stdio.cc
	qdp_jit_function.cc
	qdp_profile.cc 
 	qdp_crc32.cc
 	qdp_stopwatch.cc
	qdp_rannyu.cc
	qdp_mapresource.cc
	qdp_cache.cc
	qdp_pool_allocator.cc
	qdp_mastermap.cc
	qdp_masterset.cc
	qdp_kernel_reduce.cc
	qdp_kernel_blocksum.cc
	qdp_jit_translation.cc
	qdp_wordjit.cc
	qdp_datalayout.cc
	qdp_jit_util.cc
	qdp_jit_config.cc
	qdp_deep_log.cc
        qdp_llvm.cc
	qdp_alloc_cache.cc
)

add_library(jit qdp_jit.cc qdp_jit_factory.cc)


if( QDP_ENABLE_BACKEND STREQUAL "CUDA" )
  target_sources(qdp PRIVATE qdp_gpu_cuda.cc)
  target_sources(jit PRIVATE qdp_jit_cuda.cc)
endif()

if( QDP_ENABLE_BACKEND STREQUAL "ROCM" )
  target_sources(qdp PRIVATE qdp_gpu_rocm.cc)
  target_sources(jit PRIVATE qdp_jit_rocm.cc)
endif()

if( QDP_ENABLE_BACKEND STREQUAL "L0" )
  target_sources(qdp PRIVATE qdp_gpu_l0.cc)
  target_sources(jit PRIVATE qdp_jit_l0.cc)
  target_link_libraries(qdp PUBLIC ${LevelZero_LIBRARIES})
  target_include_directories(qdp PUBLIC ${LevelZero_INCLUDE_DIRS})
endif()

if (QDP_ENABLE_HOST_COMPILER)
  target_sources(jit PRIVATE qdp_jit_host.cc)
endif()



if(QDP_USE_LIBXML2)
	target_sources(qdp PRIVATE
		qdp_xmlio.cc
		qdp_iogauge.cc
		qdp_qdpio.cc
		qdp_qio_strings.cc
		qdp_map_obj_disk.cc
	)
endif()

if(QDP_USE_HDF5)
	target_sources(qdp PRIVATE qdp_hdf5.cc)
endif()


target_sources(qdp PRIVATE 
	qdp_parscalar_init.cc 
	qdp_parscalar_layout.cc
	qdp_parscalar_specific.cc
	qdp_parscalar_mapresource.cc
	qdp_scalarsite_specific.cc
)


target_include_directories( jit PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
        $<INSTALL_INTERFACE:include> )
target_include_directories(jit PUBLIC ${LLVM_INCLUDE_DIRS} )

set(QDPJIT_LLVM_COMPONENTS
  Core
  Support
  Analysis
  TransformUtils
  ScalarOpts
  InstCombine
  Vectorize
  ipo
  Passes
  IRReader
  BitReader
  BitWriter
  AsmParser
  AsmPrinter
  Target
  CodeGen
  MC
  MCParser
  BinaryFormat
  Object
  ExecutionEngine
  RuntimeDyld
  JITLink
  OrcJIT
  OrcShared
  OrcTargetProcess
  X86CodeGen
  X86AsmParser
  X86Desc
  X86Info
  Linker
  Demangle
  MCJIT
  TargetParser
)

if(QDP_ENABLE_BACKEND STREQUAL "CUDA")
  list(APPEND QDPJIT_LLVM_COMPONENTS
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
  )
endif()

if( QDP_ENABLE_BACKEND STREQUAL "ROCM" )
  target_include_directories(jit PUBLIC ${LLD_INCLUDE_DIRS} )
  target_link_libraries( jit PUBLIC ${LLD_EXPORTED_TARGETS} )
  list(APPEND QDPJIT_LLVM_COMPONENTS 
	AMDGPU
	)
  target_compile_definitions(jit PUBLIC
    ROCM_DIR="${ROCM_PATH}"
  )
  if(QDP_JIT_ROCM_COMGR_LINK)
    target_compile_definitions(jit PRIVATE QDP_JIT_ROCM_COMGR_LINK)
    target_link_libraries(jit PUBLIC amd_comgr)
  else()
    target_compile_definitions(jit PUBLIC
      QDP_ROCM_LD_LLD="${QDP_ROCM_LD_LLD}"
    )
  endif()
endif()



if( QDP_ENABLE_BACKEND STREQUAL "L0" )
#  target_include_directories(jit PUBLIC SPIRV-Headers:: )
#  target_link_libraries( jit PUBLIC PkgConfig::PKG_LLVMSPIRV )
#  target_link_libraries(jit PUBLIC ${PKG_LLVMSPIRV_LIBRARIES})
#  target_include_directories(jit PUBLIC ${PKG_LLVMSPIRV_INCLUDE_DIRS})
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(PKG_LLVMSPIRV REQUIRED IMPORTED_TARGET LLVMSPIRVLib)
  target_link_libraries(jit PUBLIC PkgConfig::PKG_LLVMSPIRV)
endif()



llvm_map_components_to_libnames(QDPJIT_LLVM_LIBS
  ${QDPJIT_LLVM_COMPONENTS}
)

target_link_libraries(jit PUBLIC ${QDPJIT_LLVM_LIBS})


target_link_libraries( jit PUBLIC QMP::qmp)






set_target_properties( jit PROPERTIES CXX_STANDARD 17 )
#  CXX_EXTENSIONS OFF
#  PROPERTIES C_STANDARD 99
#  PROPERTIES CXX_EXTENSIONS OFF
#)


target_include_directories( qdp PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
        $<INSTALL_INTERFACE:include> )

target_link_libraries( qdp PUBLIC ${GPU_LIBS} )

set_target_properties( qdp PROPERTIES 
	CXX_STANDARD 20
	CXX_EXTENSIONS OFF
	PROPERTIES C_STANDARD 99 
	PROPERTIES CXX_EXTENSIONS OFF
)

target_compile_features( qdp PUBLIC cxx_std_20 )


# Link QIO from subtree
target_link_libraries( qdp PUBLIC jit QIO::qio )

# jit uses qdp (StopWatch, QDPIO, jit_config, QDP_abort, ...): a cycle
# of static libraries, which CMake repeats on the link line. Without it
# a program links only what qdp happens to reference itself.
target_link_libraries( jit INTERFACE qdp )

# If using libxml link the XPathReader and xmlWriter libraries
if( QDP_USE_LIBXML2 )
        target_link_libraries( qdp PUBLIC XPathReader::XPathReader XPathReader::xmlWriter )
endif()

# If we want filedb (always?) link that
if( QDP_USE_FILEDB )
        target_link_libraries( qdp PUBLIC FileDB::filedb )
endif()

if( QDP_SANITIZER_OPTS )
  target_compile_options(qdp PRIVATE ${QDP_SANITIZER_OPTS})
  target_link_options(qdp PUBLIC ${QDP_SANITIZER_OPTS})
endif()

install(TARGETS qdp jit
  EXPORT QDPXXTargets
  ARCHIVE DESTINATION lib
  LIBRARY  DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
#install the export description of the targets
install(EXPORT QDPXXTargets
	FILE QDPXXTargets.cmake
        NAMESPACE QDPXX::
        DESTINATION lib/cmake/QDPXX)

add_library(QDPXX::qdp ALIAS qdp)
add_library(QDPXX::jit ALIAS jit)
