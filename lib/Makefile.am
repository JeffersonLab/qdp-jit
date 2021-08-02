# Have to add top_builddir/include to pick up made qdp_config_internal.h

INCFLAGS = -I@top_srcdir@/include -I@top_builddir@/include \
           -I@top_srcdir@/other_libs/qio/include \
           -I@top_builddir@/other_libs/qio/include \
           -I@top_srcdir@/other_libs/qio/other_libs/c-lime/include \
           -I@top_builddir@/other_libs/qio/other_libs/c-lime/include

if QDP_USE_LIBXML2
INCFLAGS += -I@top_srcdir@/other_libs/xpath_reader/include
endif

#
# need the HDF5 cxx flags with and without /include
#
#AM_CXXFLAGS = $(INCFLAGS) @HDF5_CXXFLAGS@/include @HDF5_CXXFLAGS@ @LIBXML2_CXXFLAGS@ @QMP_CFLAGS@ @LLVM_CXXFLAGS@ @CUDA_CXXFLAGS@
AM_CXXFLAGS = $(INCFLAGS) @HDF5_CXXFLAGS@ @LIBXML2_CXXFLAGS@ @QMP_CFLAGS@ @LLVM_CXXFLAGS@ @GPU_CXXFLAGS@

lib_LIBRARIES = libqdp.a


## Base set of sources and conditionally included sources
libqdp_a_SOURCES = qdp_map.cc qdp_subset.cc qdp_random.cc qdp.cc \
	qdp_layout.cc qdp_io.cc qdp_byteorder.cc qdp_util.cc \
	qdp_stdio.cc qdp_jit_function.cc \
        qdp_profile.cc qdp_strnlen.cc qdp_crc32.cc \
        qdp_stopwatch.cc \
        qdp_rannyu.cc \
	qdp_mapresource.cc \
	qdp_llvm.cc qdp_cache.cc qdp_mastermap.cc qdp_masterset.cc \
        qdp_jitf_sum.cc qdp_wordreg.cc qdp_datalayout.cc qdp_jit_util.cc \
	qdp_jit_config.cc qdp_pete_visitors.cc qdp_deep_log.cc

#

if QDP_BACKEND_CUDA
  libqdp_a_SOURCES += qdp_gpu_cuda.cc
else
  libqdp_a_SOURCES += qdp_gpu_rocm.cc
endif


if QDP_CUDA_SPECIAL
libqdp_a_SOURCES += \
	qdp_functions_special.cc \
	cuda_special.cc \
        cuda_special_quarkContract13.cu \
        cuda_special_quarkContract14.cu \
        cuda_special_quarkContract23.cu \
        cuda_special_quarkContract24.cu
endif


.cu.o:
	@CUDA_NVCC@ -arch=@CUDA_SM@ -I@top_srcdir@/include -dw -o $@ $<




if QDP_USE_LIBXML2
libqdp_a_SOURCES += qdp_xmlio.cc qdp_iogauge.cc qdp_qdpio.cc qdp_qio_strings.cc qdp_map_obj_disk.cc
endif

if QDP_USE_HDF5
libqdp_a_SOURCES += qdp_hdf5.cc
endif

if QDP_CUSTOM_KERNELS
libqdp_a_SOURCES += stream_local_inner_product.cc
endif


# Scalar	
if ARCH_SCALAR
libqdp_a_SOURCES += \
	qdp_scalar_init.cc \
	qdp_scalar_layout.cc \
	qdp_scalar_specific.cc \
	qdp_scalar_mapresource.cc	\
	qdp_scalarsite_specific.cc
endif

# Parallel-scalar
if ARCH_PARSCALAR
libqdp_a_SOURCES += \
	qdp_parscalar_init.cc \
	qdp_parscalar_layout.cc \
	qdp_parscalar_specific.cc \
	qdp_parscalar_mapresource.cc	\
	qdp_scalarsite_specific.cc
endif

#libqdp_a_SOURCES += qdp_parscalar_init.cc qdp_parscalar_layout.cc \
#	qdp_parscalar_specific.cc \
#	qdp_scalarsite_specific.cc


libqdp_a_SOURCES += qdp_default_allocator.cc



