

add_library(qdp qdp_map.cc 
	qdp_subset.cc 
	qdp_random.cc
	qdp.cc
	qdp_layout.cc 
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
	qdp_mastermap.cc
	qdp_masterset.cc
	qdp_jitf_sum.cc
	qdp_wordreg.cc
	qdp_datalayout.cc
	qdp_jit_util.cc
	qdp_jit_config.cc
	qdp_pete_visitors.cc
	qdp_default_allocator.cc
	qdp_deep_log.cc
)

add_library(jit qdp_llvm.cc)

if(LLVM_LIBRARY_DIRS)
  list(GET LLVM_LIBRARY_DIRS 0 QDP_LLVM_LIBRARY_DIR)
else()
  get_filename_component(QDP_LLVM_LIBRARY_DIR "${LLVM_DIR}/../../../lib" ABSOLUTE)
endif()

message(STATUS "QDP-JIT: LLVM library directory: ${QDP_LLVM_LIBRARY_DIR}")

target_link_options(jit INTERFACE
  "LINKER:-rpath,${QDP_LLVM_LIBRARY_DIR}"
)

target_link_options(qdp INTERFACE
  "LINKER:-rpath,${QDP_LLVM_LIBRARY_DIR}"
)


if (QDP_ENABLE_MANAGED_MEMORY)
  target_sources(qdp PRIVATE qdp_alloc_cache.cc)
endif()

if( QDP_ENABLE_BACKEND STREQUAL "CUDA" )
  target_sources(qdp PRIVATE qdp_gpu_cuda.cc)
endif()

if( QDP_ENABLE_BACKEND STREQUAL "ROCM" )
  target_sources(qdp PRIVATE qdp_gpu_rocm.cc)
endif()

if( QDP_ENABLE_BACKEND STREQUAL "AVX" )
  target_sources(qdp PRIVATE qdp_gpu_avx.cc)
endif()

if( QDP_ENABLE_BACKEND STREQUAL "L0" )
  target_sources(qdp PRIVATE qdp_gpu_l0.cc)
  target_link_libraries(qdp PUBLIC ${LevelZero_LIBRARIES})
  target_include_directories(qdp PUBLIC ${LevelZero_INCLUDE_DIRS})
endif()



if(QDP_ENABLE_BACKEND STREQUAL "ROCM")
  target_link_options(jit INTERFACE
    "LINKER:-rpath,${QDP_LLVM_LIBRARY_DIR}"
    "LINKER:-rpath,${ROCM_PATH}/lib"
  )

  target_link_options(qdp INTERFACE
    "LINKER:-rpath,${QDP_LLVM_LIBRARY_DIR}"
    "LINKER:-rpath,${ROCM_PATH}/lib"
  )
else()
  target_link_options(jit INTERFACE
    "LINKER:-rpath,${QDP_LLVM_LIBRARY_DIR}"
  )

  target_link_options(qdp INTERFACE
    "LINKER:-rpath,${QDP_LLVM_LIBRARY_DIR}"
  )
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

if(QDP_ENABLE_CUSTOM_KERNELS)
	target_sources(qdp PRIVATE stream_local_inner_product.cc)
endif()

if( QDP_ARCH_SCALAR )
   target_sources(qdp PRIVATE
				qdp_scalar_init.cc 
				qdp_scalar_layout.cc
				qdp_scalar_specific.cc
				qdp_scalar_mapresource.cc
				qdp_scalarsite_specific.cc
   )
endif()

if( QDP_ARCH_PARSCALAR ) 
	target_sources(qdp PRIVATE 
		qdp_parscalar_init.cc 
		qdp_parscalar_layout.cc
		qdp_parscalar_specific.cc
		qdp_parscalar_mapresource.cc
		qdp_scalarsite_specific.cc
	)	
endif()


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
  list(APPEND QDPJIT_LLVM_COMPONENTS 
	AMDGPU
	)
  target_compile_definitions(jit PUBLIC
    ROCM_DIR="${ROCM_PATH}"
    QDP_ROCM_LD_LLD="${QDP_ROCM_LD_LLD}"
  )
endif()

if( QDP_ENABLE_BACKEND STREQUAL "L0" )
#  target_include_directories(jit PUBLIC SPIRV-Headers:: )
#  target_link_libraries( jit PUBLIC PkgConfig::PKG_LLVMSPIRV )
  target_link_libraries(jit PUBLIC ${PKG_LLVMSPIRV_LIBRARIES})
  target_include_directories(jit PUBLIC ${PKG_LLVMSPIRV_INCLUDE_DIRS})
endif()


llvm_map_components_to_libnames(QDPJIT_LLVM_LIBS
  ${QDPJIT_LLVM_COMPONENTS}
)

target_link_libraries(jit PUBLIC ${QDPJIT_LLVM_LIBS})

# LLVM arrives as static archives, so without this libjit.so re-exports ~29k
# unversioned llvm:: symbols and the loader binds ROCm's own libLLVM.so into
# this copy. Two LLVMs then share statics and glibc aborts at exit.
if(NOT APPLE)
  target_link_options(jit PRIVATE "LINKER:--exclude-libs,ALL")
endif()

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


# Link QIO from subtree
target_link_libraries( qdp PUBLIC jit QIO::qio )

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
