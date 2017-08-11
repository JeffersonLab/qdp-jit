# Have to add top_builddir/include to pick up made qdp_config_internal.h

INCFLAGS = -I@top_srcdir@/include -I@top_builddir@/include \
           -I@top_srcdir@/other_libs/qio/include \
           -I@top_builddir@/other_libs/qio/include \
           -I@top_srcdir@/other_libs/qio/other_libs/c-lime/include \
           -I@top_builddir@/other_libs/qio/other_libs/c-lime/include

if QDP_USE_LIBXML2
INCFLAGS += -I@top_srcdir@/other_libs/xpath_reader/include
endif

AM_CXXFLAGS = $(INCFLAGS) @LIBXML2_CXXFLAGS@ @QMP_CFLAGS@  @QMT_CXXFLAGS@  @LLVM_CXXFLAGS@ @CUDA_CXXFLAGS@

lib_LIBRARIES = libqdp.a


## Base set of sources and conditionally included sources
libqdp_a_SOURCES = qdp_map.cc qdp_subset.cc qdp_random.cc qdp.cc \
	qdp_layout.cc qdp_io.cc qdp_byteorder.cc qdp_util.cc \
	qdp_stdio.cc \
        qdp_profile.cc qdp_strnlen.cc qdp_crc32.cc \
        qdp_stopwatch.cc \
        qdp_rannyu.cc \
	qdp_mapresource.cc qdp_autotuning.cc qdp_deviceparams.cc\
	qdp_llvm.cc qdp_cuda.cc qdp_cache.cc qdp_mastermap.cc qdp_masterset.cc \
        qdp_jitf_sum.cc qdp_wordreg.cc qdp_datalayout.cc qdp_jit_util.cc qdp_libdevice.cc


if QDP_USE_LIBXML2
libqdp_a_SOURCES += qdp_xmlio.cc qdp_iogauge.cc qdp_qdpio.cc qdp_qio_strings.cc qdp_map_obj_disk.cc
endif

libqdp_a_SOURCES += qdp_parscalar_init.cc qdp_parscalar_layout.cc \
	qdp_parscalar_specific.cc \
	qdp_scalarsite_specific.cc


libqdp_a_SOURCES += qdp_default_allocator.cc



