

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
)

add_library(jit qdp_llvm.cc)

if( QDP_ENABLE_BACKEND STREQUAL "CUDA" )
	target_sources(qdp PRIVATE qdp_gpu_cuda.cc)
endif()

if( QDP_ENABLE_BACKEND STREQUAL "ROCM" )
	target_sources(qdp PRIVATE qdp_gpu_rocm.cc)
endif()

if( QDP_ENABLE_CUDA_SPECIAL )
	target_sources(qdp PRIVATE
		qdp_functions_special.cc
		cuda_special.cc
    cuda_special_quarkContract13.cu
		cuda_special_quarkContract14.cu
		cuda_special_quarkContract23.cu
		cuda_special_quarkContract24.cu
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
target_link_libraries( jit PUBLIC ${GPU_LIBS} ${LLVM_AVAILABLE_LIBS} )
set_target_properties( jit PROPERTIES 
	CXX_STANDARD 14
	CXX_EXTENSIONS OFF
	PROPERTIES C_STANDARD 99 
	PROPERTIES CXX_EXTENSIONS OFF
)


target_include_directories( qdp PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
        $<INSTALL_INTERFACE:include> )
target_include_directories(qdp PUBLIC ${LLVM_INCLUDE_DIRS} )
target_link_libraries( qdp PUBLIC ${GPU_LIBS} ${LLVM_AVAILABLE_LIBS} jit )
set_target_properties( qdp PROPERTIES 
	CXX_STANDARD 20
	CXX_EXTENSIONS OFF
	PROPERTIES C_STANDARD 99 
	PROPERTIES CXX_EXTENSIONS OFF
)





# Link QIO from subtree
target_link_libraries( qdp PUBLIC jit QIO::qio )
target_link_libraries( jit PUBLIC qdp QIO::qio )

# If using libxml link the XPathReader and xmlWriter libraries
if( QDP_USE_LIBXML2 )
        target_link_libraries( qdp PUBLIC XPathReader::XPathReader XPathReader::xmlWriter )
        target_link_libraries( jit PUBLIC XPathReader::XPathReader XPathReader::xmlWriter )
endif()

# If we want filedb (always?) link that
if( QDP_USE_FILEDB )
        target_link_libraries( qdp PUBLIC FileDB::filedb )
        target_link_libraries( jit PUBLIC FileDB::filedb )
endif()

if( QDP_SANITIZER_OPTS )
  target_compile_options(qdp PUBLIC ${QDP_SANITIZER_OPTS})
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
