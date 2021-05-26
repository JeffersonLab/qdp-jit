cmake_minimum_required(VERSION 3.17)
PROJECT( QDPXXExamples VERSION 1.0.0 LANGUAGES C;CXX)

get_directory_property(SubBuild PARENT_DIRECTORY)

if(NOT SubBuild)
        find_package(QDPXX REQUIRED)
endif()

add_library(examples mesplq.cc reunit.cc dslashm_w.cc rgauge.cc taproj.cc 
		expm12.cc baryon_w.cc mesons_w.cc formfac_w.cc linalg1.cc linalg.h 
		blas1.cc blas1.h cblas1.cc cblas1.h unittest.h testOpenPartFile.h)
			
target_link_libraries(examples PUBLIC QDPXX::qdp)

set(TESTS t_skeleton 
	t_io
	t_mesplq
	t_db
	t_xml
	t_entry
	t_nersc
	t_shift
	t_exotic
	t_basic
	t_qio
	t_cugauge
	t_transpose_spin
	t_partfile
	t_map_obj_disk
	t_map_obj_memory
	t_iprod	
	)

if( BUILD_WILSON_EXAMPLES )
  set(TESTS ${TESTS} t_dslashm t_formfac t_spectrum t_qdp t_linalg)
endif()

set(TESTS ${TESTS} nersc2ildg test_ildglat lhpc2ildg)


foreach(TEST ${TESTS})
 add_executable(${TEST} ${TEST}.cc)
 target_link_libraries(${TEST} examples)
endforeach()

install(TARGETS nersc2ildg test_ildglat lhpc2ildg)
