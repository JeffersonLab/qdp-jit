AC_INIT([qdp-jit],[0.1],[fwinter@jlab.org])

AC_CONFIG_AUX_DIR(config)

AC_LANG([C++])
AC_PROG_CXX(g++ cxx CC)
AC_PROG_RANLIB
AC_CHECK_TOOL(AR, ar, [ar])
AC_CHECK_TOOL(XXD, xxd, [xxd])

dnl Initialise automake
AM_INIT_AUTOMAKE

dnl defines...
AC_CONFIG_HEADERS([include/qdp_config_internal.h])


AC_ARG_WITH(cuda,
  AC_HELP_STRING([--with-cuda=CUDADIR], [ Specify CUDA installation directory (default: /usr/local/cuda)]),
  [ cuda_home="${withval}" ],
  [ cuda_home="/usr/local/cuda" ]
)

AC_ARG_ENABLE(cuda-managed-memory,
   AC_HELP_STRING([--enable-cuda-managed-memory], [ Enable memory to be allocated with Cuda Managed Memory ] ),
   [ cuda_managed_memory="${enableval}" ],
   [ cuda_managed_memory="no" ]
)

AC_MSG_NOTICE([Configuring with CUDA installed in = ${cuda_home}])
AC_DEFINE_UNQUOTED(CUDA_DIR,"${cuda_home}",[CUDA install path])
AC_SUBST(SUBST_CUDA_DIR,"${cuda_home}")

if test "X${cuda_managed_memory}X" == "XyesX";
then
  AC_MSG_NOTICE([Will Use CUDA Managed Memory for Allocations])
  AC_DEFINE(QDP_USE_CUDA_MANAGED_MEMORY, [1], [ CUDA use managed memory ])
fi


dnl
dnl
dnl --with-xxxx and --enable-xxxx switches 
dnl



dnl --enable-Nd
AC_ARG_ENABLE(Nd,
  AC_HELP_STRING([--enable-Nd=N ],
    [Set No of Spacetime Dimensions, default is Nd=4 ]),
  [ ac_Nd=${enableval} ],
  [ ac_Nd=4 ]
)
    
AC_DEFINE_UNQUOTED(QDP_ND, ${ac_Nd}, [Number of dimensions])        dnl config.h file
AC_SUBST(CONFIG_ND, [${ac_Nd}] )  dnl qdp++_config

dnl Tell the user about Nd
AC_MSG_NOTICE([Configuring QDP++ for Nd = ${ac_Nd}])


dnl --enable-Nc
AC_ARG_ENABLE(Nc,
  AC_HELP_STRING([--enable-Nc=N ],
    [Set Number of Colors, ie N for SU(N) default is SU(3)]),
  [ac_Nc=${enableval}],
  [ac_Nc=3]
)

AC_DEFINE_UNQUOTED(QDP_NC, ${ac_Nc}, [Number of color components])
AC_SUBST(CONFIG_NC, [${ac_Nc}] )

dnl Tell the user about Nc
AC_MSG_NOTICE([Configuring QDP++ for Nc = ${ac_Nc}])


dnl --enable-Ns
AC_ARG_ENABLE(Ns,
  AC_HELP_STRING([--enable-Ns=N],
    [Set No of Spin Components, Default is Ns=4]),

  [ac_Ns=${enableval}],
  [ac_Ns=4]
)

AC_DEFINE_UNQUOTED(QDP_NS, ${ac_Ns}, [Number of spin components])
AC_SUBST(CONFIG_NS, [${ac_Ns}] )
dnl Tell the user about Ns
AC_MSG_NOTICE([Configuring QDP++ for Ns = ${ac_Ns}])

dnl --enable-alignment
AC_ARG_ENABLE(alignment,
	AC_HELP_STRING([--enable-alignment=N],
	[Set alignment to N bytes for OLattice]),
	[ac_alignment=${enableval}],
	[ac_alignment=16]
)

AC_DEFINE_UNQUOTED(QDP_AC_ALIGNMENT_SIZE, ${ac_alignment}, [Alignment for OLattice])
AC_SUBST(CONFIG_ALIGNMENT, [${ac_alignment}])

dnl tell the user about alignment	
AC_MSG_NOTICE([Configuring QDP++ Alignment size=${ac_alignment}])





dnl --enable-custom-kernels
AC_ARG_ENABLE(custom-kernels,
  AC_HELP_STRING([--enable-custom-kernels],
    [Includes custom kernels]),
  [ac_custom="yes"],
  [ac_custom="no"]
)


dnl --enable-custom-kernels
AC_ARG_ENABLE(cuda-special,
  AC_HELP_STRING([--enable-cuda-special],
    [Build CUDA functions. Use --enable-cuda-sm and possibly --enable-cuda-nvcc]),
  [ac_cuda_special="yes"],
  [ac_cuda_special="no"]
)


AC_ARG_ENABLE(cuda-sm,
  AC_HELP_STRING([--enable-cuda-sm=<arch>],
    [Build special CUDA function for compute capability <sm_50/...>]),
    [ac_cuda_sm=${enableval}],
    [ac_cuda_sm="sm_50"]
)

AC_SUBST(CUDA_SM, "${ac_cuda_sm}")


AC_ARG_ENABLE(cuda-nvcc,
  AC_HELP_STRING([--enable-cuda-nvcc=<arch>],
    [Path to CUDA NVCC]),
    [ac_cuda_nvcc=${enableval}],
    [ac_cuda_nvcc="nvcc"]
)

AC_SUBST(CUDA_NVCC, "${ac_cuda_nvcc}")







dnl --enable-parallel-arch argument
AC_ARG_ENABLE(parallel-arch,
  AC_HELP_STRING([--enable-parallel-arch=<arch>],
    [Build QDP++ for parallel architecture <arch>]),
  [PARALLEL_ARCH=${enableval}],
  AC_MSG_ERROR([Require architecture to be set])
)

dnl --enable-memory-debugging
AC_ARG_ENABLE(memory-debugging,
   AC_HELP_STRING([--enable-memory-debugging], 
     [Enable memory debugging code]),
     [ac_memdebug_enabled=1
      AC_MSG_NOTICE([Enabling memory debugging code])],
     [ac_memdebug_enabled=0]
)

dnl database wrapper
AC_ARG_ENABLE(filedb,
  AC_HELP_STRING([--enable-filedb],
    [Build routines using FileDB]),
  [filedb_enabled="${enableval}"],
  [filedb_enabled="yes"]
)


AC_ARG_ENABLE(llvm12,
  AC_HELP_STRING(
    [--enable-llvm12],
    [Enable changes to compile with LLVM 12.0 release]
  ),
  [LLVM12_ENABLED=$enableval],
  [LLVM12_ENABLED="no"]
)

if test "X${LLVM12_ENABLED}X" = "XyesX"; then
  AC_MSG_NOTICE([Enabling changes for LLVM 12.0 release ])
  AC_DEFINE(QDP_LLVM12, [1], [ Compile for LLVM 12.0 release  ])
fi



dnl comm-split to find device id
AC_ARG_ENABLE(comm-split-deviceinit,
  AC_HELP_STRING([--enable-comm-split-deviceinit],
     [Select Device by identifying Node Local Ranks using MPI_Comm_split ]),
  [qdp_comm_split_init_enabled="${enableval}"],
  [qdp_comm_split_init_enabled="no"]
)

if test "X${qdp_comm_split_init_enabled}X" == "XyesX";
then
 AC_MSG_NOTICE([Local Rank identification with MPI_Comm_split])
 AC_DEFINE([QDP_USE_COMM_SPLIT_INIT], [1],[Comm split for local ranks])
fi

ac_gcc="$CXX"     				      
AC_DEFINE_UNQUOTED([CUDA_GCC],"${ac_gcc}",[CUDA C/C++ Compiler])
AC_DEFINE_UNQUOTED(QDP_IS_QDPJIT, ${gpu_enabled}, [Enable GPU/JIT])
AC_DEFINE_UNQUOTED(QDPJIT_IS_QDPJITNVVM, ${gpu_enabled}, [Enable GPU/JIT])
#AC_SUBST(CUDA_LIBS, [-lcuda])
AC_SUBST(CUDA_CXXFLAGS, [-I${cuda_home}/include])
AC_SUBST(CUDA_LDFLAGS, [-L${cuda_home}/lib64])





dnl Include custom kernels
if test "X${ac_custom}X" = "XyesX"; then
   AC_DEFINE_UNQUOTED(QDP_CUSTOM_KERNELS, ${ac_custom}, [Include custom kernels])
   AC_MSG_NOTICE([Include custom kernels])
fi


dnl Include custom kernels
if test "X${ac_cuda_special}X" = "XyesX"; then
   AC_DEFINE_UNQUOTED(QDP_CUDA_SPECIAL, ${ac_cuda_special}, [Include CUDA special kernels])
   AC_MSG_NOTICE([Include CUDA special kernels])
   tmp_cuda_libs="-lcuda -lcudart"
else
   tmp_cuda_libs="-lcuda"
fi

AC_SUBST(CUDA_LIBS, "${tmp_cuda_libs}" )


dnl Check if memory debugging is enabled
if test ${ac_memdebug_enabled} -eq 1; then 
   AC_DEFINE_UNQUOTED(QDP_DEBUG_MEMORY, ${ac_memdebug_1}, [Enable memory debugging])
   AC_MSG_NOTICE([Enable memory debugging])
fi

dnl Precision munging
AC_ARG_ENABLE(precision,
  AC_HELP_STRING([--enable-precision=single|double],
                 [Sets the base precision to be single or double precision (default is single) ]),
  [ac_precision=${enableval}],
  [ac_precision="single"])

case ${ac_precision} in 
single)
	AC_MSG_NOTICE( [Configuring QDP++ for single precision ] )
	AC_DEFINE(BASE_PRECISION,[32], [Single precision])
	AC_SUBST(CONFIG_PRECISION, [32])
	;;
double)
	AC_MSG_NOTICE( [Configuring for double precision ] )
	AC_DEFINE(BASE_PRECISION, [64], [Double precision] )
	AC_SUBST(CONFIG_PRECISION, [64])
	;;
*)
	AC_MSG_ERROR( [Unsupported value of precision. Check --enable-precision ] )
	;;
esac

AC_ARG_ENABLE(layout,
  AC_HELP_STRING([--enable-layout=lexico|cb2|cb3d|cb32],
                 [Sets the layout. lexico=lexicographic, cb2=even odd checkerboard, cb3d=even odd checkerboard in 3d, cb32=hypercubic checkerboard. (default is cb2)]),
  [ ac_layout=${enableval} ],
  [ ac_layout="cb2" ]
)

case "${ac_layout}" in
lexico|lexicographic)
	AC_SUBST(CONFIG_LAYOUT,[lexicographic])
	AC_DEFINE(QDP_USE_LEXICO_LAYOUT, [1], [Use lexicographic layout])
	;;
cb2|checkerboard)
	AC_SUBST(CONFIG_LAYOUT,[cb2])
	AC_DEFINE(QDP_USE_CB2_LAYOUT, [1], [Use checkerboarded layout])
	;;
cb3d|checkerboard-3d)
	AC_SUBST(CONFIG_LAYOUT,[cb3d])
	AC_DEFINE(QDP_USE_CB3D_LAYOUT,[1], [Use 3d checkerboarded layout])
	;;
cb32|hyper|hypercube)
	AC_SUBST(CONFIG_LAYOUT,[cb32])
	AC_DEFINE(QDP_USE_CB32_LAYOUT, [1], [Use hypercube checkerboard layout])
	;;
*)
	AC_MSG_ERROR([Unsupported Layout. Check --enable-layout])
	;;
esac
AC_MSG_NOTICE([Configuring QDP++ with ${CONFIG_LAYOUT} layout])

dnl George T. Fleming 03/03/2003
dnl This is a total reworking of Balint Joo's implementation
dnl of the qmp_build_env.sh script.  First of all, since the plan seems
dnl to have changed from having QMP live in a subdirectory of QDP++
dnl to having QMP configured, built and installed separately, we only
dnl need to know the flags necessary to compile and link user apps
dnl against QMP and not the flags necessary to compile QMP itself.
dnl
dnl Also, QMP is not needed for ac_parallel_arch={scalar,scalarvec}, so don't bother
dnl looking for it in that case.
AC_ARG_WITH(qmp,
  AC_HELP_STRING(
    [--with-qmp=DIR],
    [Build QDP++ on top of QMP, where QMP is installed in DIR.]
  ),
  [QMP_HOME="${with_qmp}"]
)

case ${PARALLEL_ARCH} in
  scalar)
    AC_DEFINE(ARCH_SCALAR, [1], [Scalar architecture])
    AC_SUBST(ARCH_SITE, "scalarsite")
    AC_MSG_NOTICE([ Scalar Build! Not checking for QMP ])
    AC_SUBST(QMP_HOME)
    AC_SUBST(QMP_CFLAGS)
    AC_SUBST(QMP_LDFLAGS)
    AC_SUBST(QMP_LIBS)
    ;;
  scalarvec)
    AC_DEFINE(ARCH_SCALARVEC, [1], [Scalar with vector extensions architecture])
    AC_SUBST(ARCH_SITE, "scalarvecsite")
    AC_MSG_NOTICE([ Scalar with Vector extensions Build! Not checking for QMP ])
    AC_SUBST(QMP_HOME)
    AC_SUBST(QMP_CFLAGS)
    AC_SUBST(QMP_LDFLAGS)
    AC_SUBST(QMP_LIBS)
    ;;
  parscalar)
    AC_DEFINE(ARCH_PARSCALAR, [1], [Parallel-scalar architecture])
    AC_SUBST(ARCH_SITE, "scalarsite")
    AC_MSG_NOTICE([ Parscalar build! Checking for QMP ])

    if test "X${QMP_HOME}X" = "XX" ; then
      AC_PATH_PROG([QMP_CONFIG], [qmp-config], [])
    else
      AC_PATH_PROG([QMP_CONFIG], [qmp-config], [], [${QMP_HOME}/bin:${PATH}])
    fi

    if test "X${QMP_CONFIG}X" != "XX" ; then
      AC_MSG_NOTICE([Found QMP configuration program ${QMP_CONFIG}])
      AC_SUBST(QMP_HOME,    "${QMP_HOME}")
      AC_SUBST(QMP_CFLAGS,  "`${QMP_CONFIG} --cflags`")
      AC_MSG_NOTICE([QMP compile flags: ${QMP_CFLAGS}])
      AC_SUBST(QMP_LDFLAGS, "`${QMP_CONFIG} --ldflags`")
      AC_MSG_NOTICE([QMP linking flags: ${QMP_LDFLAGS}])
      AC_SUBST(QMP_LIBS,    "`${QMP_CONFIG} --libs`")
      AC_MSG_NOTICE([QMP libraries flags: ${QMP_LIBS}])
    else 
      AC_MSG_WARN([QMP configuration program qmp-config not found.])
      AC_MSG_WARN([Set environment variables QMP_CFLAGS QMP_LDFAGS QMP_LIBS
        before configure])
    fi
    ;;
  parscalarvec)
    AC_DEFINE(ARCH_PARSCALARVEC, [1], [Parallel-scalar with vector extensions layout])
    AC_SUBST(ARCH_SITE, "scalarvecsite")
    AC_MSG_NOTICE([ Parscalar with Vector extensions build! Checking for QMP ])

    if test "X${QMP_HOME}X" = "XX" ; then
      AC_PATH_PROG([QMP_CONFIG], [qmp-config], [])
    else
      AC_PATH_PROG([QMP_CONFIG], [qmp-config], [], [${QMP_HOME}/bin:${PATH}])
    fi

    if test "X${QMP_CONFIG}X" != "XX" ; then
      AC_MSG_NOTICE([Found QMP configuration program ${QMP_CONFIG}])
      AC_SUBST(QMP_HOME,    "${QMP_HOME}")
      AC_SUBST(QMP_CFLAGS,  "`${QMP_CONFIG} --cflags`")
      AC_MSG_NOTICE([QMP compile flags: ${QMP_CFLAGS}])
      AC_SUBST(QMP_LDFLAGS, "`${QMP_CONFIG} --ldflags`")
      AC_MSG_NOTICE([QMP linking flags: ${QMP_LDFLAGS}])
      AC_SUBST(QMP_LIBS,    "`${QMP_CONFIG} --libs`")
      AC_MSG_NOTICE([QMP libraries flags: ${QMP_LIBS}])
    else 
      AC_MSG_WARN([QMP configuration program qmp-config not found.])
      AC_MSG_WARN([Set environment variables QMP_CFLAGS QMP_LDFAGS QMP_LIBS
        before configure])
    fi
    ;;
  *)
    AC_MSG_ERROR([ Unknown value for --enable-parallel-arch ])
    ;;
esac
AC_SUBST(CONFIG_PARALLEL_ARCH, [${PARALLEL_ARCH}])

dnl verify we can compile and link against QMP, if needed
PAC_QMP_LINK_CXX_FUNC(
  [${QMP_CFLAGS}],
  [${QMP_LDFLAGS}],
  [${QMP_LIBS}],
  ,
  ,
  [qmp_link_ok=yes],
  [qmp_link_ok=no]
)

AC_MSG_CHECKING([if we can compile/link of a simple QMP C++ program])
case ${PARALLEL_ARCH} in
  parscalar)
    if test "X${qmp_link_ok}X" = "XyesX" ; then
      AC_MSG_RESULT(yes)
    else
      AC_MSG_RESULT(no)
      AC_MSG_ERROR([Cannot compile/link a basic QMP C++ program!
        Check QMP_CFLAGS, QMP_LDFLAGS, QMP_LIBS.])
    fi
    ;;
  parscalarvec)
    if test "X${qmp_link_ok}X" = "XyesX" ; then
      AC_MSG_RESULT(yes)
    else
      AC_MSG_RESULT(no)
      AC_MSG_ERROR([Cannot compile/link a basic QMP C++ program!
        Check QMP_CFLAGS, QMP_LDFLAGS, QMP_LIBS.])
    fi
    ;;
  *)
    AC_MSG_RESULT(ignored)
    ;;
esac



AC_ARG_WITH(llvm,
  AC_HELP_STRING(
    [--with-llvm=DIR],
    [Build QDP-JIT with LLVM installed in DIR]
  ),
  [LLVM_HOME="$with_llvm"]
)

#AC_ARG_ENABLE(llvm6-trunk,
#  AC_HELP_STRING(
#    [--enable-llvm6-trunk],
#    [Enable changes to compile with LLVM6 Trunk]
#  ),
#  [LLVM6_ENABLED=${enableval}],
#  [LLVM6_ENABLED="no"]
#)

if test "X${LLVM_HOME}X" = "XX" ; then
  AC_PATH_PROG(LLVM_CONFIG, [llvm-config], [])
else
  AC_PATH_PROG(LLVM_CONFIG, [llvm-config], [], [${LLVM_HOME}/bin:${PATH}])
fi

if test "X${LLVM_CONFIG}X" = "XX" ; then
  AC_MSG_ERROR([LLVM configuration program llvm-config not found.])
fi

#if test "X${LLVM6_ENABLED}X" = "XyesX"; then
#  AC_MSG_NOTICE([Enabling changes for LLVM6 ])
#  AC_DEFINE(QDP_LLVM6_TRUNK, [1], [ Compile for LLVM6  ])
#fi


AC_MSG_NOTICE([Found LLVM configuration program ${LLVM_CONFIG}])
AC_SUBST(LLVM_CXXFLAGS, "`${LLVM_CONFIG} --cxxflags| sed -e 's/-fno-exceptions//'| sed -e 's/-fno-rtti//'| sed -e 's/-fvisibility-inlines-hidden//'| sed -e 's/-fPIC//'| sed -e 's/-Woverloaded-virtual//'| sed -e 's/-Wcast-qual//'   `")
#AC_SUBST(LLVM_CXXFLAGS, "`${LLVM_CONFIG} --cxxflags| sed -e 's/-fno-exceptions//'| sed -e 's/-std=c++11//'`")
AC_MSG_NOTICE([LLVM compile flags: ${LLVM_CXXFLAGS}])
AC_SUBST(LLVM_LDFLAGS,  ["`${LLVM_CONFIG} --ldflags` -Wl,-rpath,`${LLVM_CONFIG} --libdir`"])
AC_MSG_NOTICE([LLVM linking flags: ${LLVM_LDFLAGS}])
AC_SUBST(LLVM_LIBS,     "`${LLVM_CONFIG} --libs`")
AC_MSG_NOTICE([LLVM libraries flags: ${LLVM_LIBS}])





dnl
dnl
dnl now libxml2
dnl
dnl
AC_ARG_WITH(libxml2,
  AC_HELP_STRING(
    [--with-libxml2=DIR],
    [Build on top of libxml2 where libxml2 is installed in DIR]
  ),
  [LIBXML2_HOME="$with_libxml2"]
)
echo Value of with_libxml2 is X${with_libxml2}X

if test "X${with_libxml2}X" = "XnoX" ; then
  AC_MSG_NOTICE([Disabling Libxml])
  AC_SUBST(LIBXML2_CXXFLAGS, "")
  AC_SUBST(LIBXML2_LIBS,"")

else

  if test "X${LIBXML2_HOME}X" = "XX" ; then
    AC_PATH_PROG(LIBXML2_CONFIG, [xml2-config], [])
  else
    AC_PATH_PROG(LIBXML2_CONFIG, [xml2-config], [], [${LIBXML2_HOME}/bin:${PATH}])
  fi  

  if test "X${LIBXML2_CONFIG}X" = "XX" ; then
    AC_MSG_ERROR([libxml2 configuration program xml2-config not found.])
  fi

  AC_MSG_NOTICE([Found libxml2 configuration program ${LIBXML2XX_CONFIG}])
  AC_SUBST(LIBXML2_CXXFLAGS, "`${LIBXML2_CONFIG} --cflags`")
  AC_MSG_NOTICE([libxml2 compile flags: ${LIBXML2_CXXFLAGS}])
  AC_SUBST(LIBXML2_LIBS,     "`${LIBXML2_CONFIG} --libs`")
  AC_MSG_NOTICE([libxml2 libraries flags: ${LIBXML2_LIBS}])
  AC_DEFINE_UNQUOTED(QDP_USE_LIBXML2, [1], [Enable LIBXML2 support])
  
  dnl This function is defined in acinclude.m4
  dnl it tries to try to compile the program 
  AC_MSG_CHECKING([if we can compile/link a simple libxml2 program])
    PAC_LIBXML2_LINK_CXX_FUNC(
    ${LIBXML2_CXXFLAGS},
    ${LIBXML2_LIBS},
    ,
    ,
    [libxml2_link_ok=yes],
    [libxml2_link_ok=no]
  )

  if test "X${libxml2_link_ok}X" = "XyesX" ; then 
    AC_MSG_RESULT(yes)
  else 
    AC_MSG_RESULT(no)
    AC_MSG_ERROR([Cannot compile/link a program with libxml2.
    Use --with-libxml2=<dir> to select a working version.])
  fi 

fi



dnl Abhinav Sarje, 10/09/2013
dnl Andre Walker-Loud, 10/09/2013
dnl
dnl adding hdf5
dnl
dnl
dnl --with-hdf5
AC_ARG_WITH(hdf5,
  AC_HELP_STRING(
    [--with-hdf5=DIR],
    [Build on top of hdf5 where hdf5 is installed in DIR]
  ),
  [ac_build_hdf5="yes";
  HDF5_HOME="$with_hdf5"],
  [ac_build_hdf5="no"]
)

if test "X${ac_build_hdf5}X" = "XnoX"; then
  AC_MSG_NOTICE([Disabling HDF5])
  AC_SUBST(HDF5_CFLAGS, "")
  AC_SUBST(HDF5_CXXFLAGS, "")
  AC_SUBST(HDF5_LDFLAGS,"")
  AC_SUBST(HDF5_LIBS,"")
   
else
  AC_DEFINE_UNQUOTED(QDP_USE_HDF5, [1], [Enable HDF5 support])
  if test "X${HDF5_HOME}X" = "XX" ; then
    AC_PATH_PROG(HDF5_BIN, [h5dump], [])
  else
    AC_PATH_PROG(HDF5_BIN, [h5dump], [], [${HDF5_HOME}/bin:${PATH}])
  fi  
  if test "X${HDF5_BIN}X" = "XX" ; then
    AC_MSG_ERROR([hdf5 utility h5dump not found.])
  fi
  AC_MSG_NOTICE([Found hdf5 utility ${HDF5_BIN}])
  AC_SUBST(HDF5_CFLAGS, "-I${HDF5_HOME}/include")
  AC_SUBST(HDF5_CXXFLAGS, "-I${HDF5_HOME}/include")
#  AC_SUBST(HDF5_CFLAGS, "-I${HDF5_HOME}")
#  AC_SUBST(HDF5_CXXFLAGS, "-I${HDF5_HOME}")
  AC_MSG_NOTICE([hdf5 c-compile flags: ${HDF5_CFLAGS}])
  AC_MSG_NOTICE([hdf5 c++-compile flags: ${HDF5_CXXFLAGS}])
  AC_SUBST(HDF5_LDFLAGS,     "-L${HDF5_HOME}/lib")
  AC_MSG_NOTICE([hdf5 ldflags: ${HDF5_LDFLAGS}])
  AC_SUBST(HDF5_LIBS,     "-lhdf5 -lz -lm")
  AC_MSG_NOTICE([hdf5 libraries: ${HDF5_LIBS}])
  AC_SUBST(CONFIG_HDF5,[1])
  AC_MSG_NOTICE([Configuring QDP++ with HDF5 enabled]);
fi



      
dnl Build filedb
if test "X${filedb_enabled}X" == "XyesX"; 
then
     AC_MSG_NOTICE([Configuring to use FileDB routines])
     AC_CONFIG_SUBDIRS(other_libs/filedb)
     AC_DEFINE([BUILD_FILEDB],[],[Build filedb])
     AC_SUBST(FILEDB_DIR, [filedb])
else 
     AC_MSG_NOTICE([Not building filedb])
     AC_SUBST(FILEDB_DIR,"")
fi

AM_CONDITIONAL(BUILD_FILEDB,
  [test "x${filedb_enabled}x" = "xyesx" ])

dnl
dnl
dnl Now have all the options... Do some configuration 
dnl
dnl

AM_CONDITIONAL(ARCH_SCALAR,       [test "X${PARALLEL_ARCH}X" = "XscalarX"])
AM_CONDITIONAL(ARCH_PARSCALAR,    [test "X${PARALLEL_ARCH}X" = "XparscalarX"])
AM_CONDITIONAL(ARCH_SCALARVEC,    [test "X${PARALLEL_ARCH}X" = "XscalarvecX"])
AM_CONDITIONAL(ARCH_PARSCALARVEC, [test "X${PARALLEL_ARCH}X" = "XparscalarvecX"])


#
# For the examples directory, conditionally compile some files.
# These conditionals are *only* used there.
#
AM_CONDITIONAL(BUILD_STAGGERED_EXAMPLES,
  [test "X${ac_Ns}X" = "X1X"])
AM_CONDITIONAL(BUILD_WILSON_EXAMPLES,
  [test "X${ac_Ns}X" != "X1X"])

AM_CONDITIONAL(BUILD_DEFAULT_ALLOCATOR,
  [test "X${ac_qcdoc_enabled}X" = "X0X"])

#
# Conditional for use of libxml
if test "x${with_libxml2}x" != "xnox"; then
 echo "True, we use libxml2"
else
 echo "False, we don't use libxml2"
fi

AM_CONDITIONAL(QDP_USE_LIBXML2, [test "x${with_libxml2}x" != "xnox" ])

AM_CONDITIONAL(QDP_USE_HDF5, [test "x${ac_build_hdf5}x" = "xyesx" ])

AM_CONDITIONAL(QDP_CUSTOM_KERNELS, [test "x${ac_custom}x" = "xyesx" ])

AM_CONDITIONAL(QDP_CUDA_SPECIAL, [test "x${ac_cuda_special}x" = "xyesx" ])



##################################
# Check for programs to build docs
##################################
AC_CHECK_PROG(LATEX, latex, latex, true)
AC_CHECK_PROG(DVIPS, dvips, dvips, true)
AC_CHECK_PROG(DVIPDF, dvipdf, dvipdf, true)
AC_CHECK_PROG(DOXYGEN, doxygen, doxygen, true)

##################################
# Check for various functions
##################################
AC_CHECK_FUNCS(gethostname)
AC_CHECK_FUNCS(strnlen)

case ${PARALLEL_ARCH} in 
parscalar|parscalarvec)
        QMP_BKUP_CXXFLAGS="${CXXFLAGS}"
        QMP_BKUP_LDFLAGS="${LDFLAGS}"
 	QMP_BKUP_LIBS="${LIBS}"
        AC_LANG_SAVE
        CXXFLAGS="${CXXFLAGS} ${QMP_CFLAGS}"
	LDFLAGS="${LDFLAGS} ${QMP_LDFLAGS}"
	LIBS="${LIBS} ${QMP_LIBS}"
	AC_CHECK_FUNCS(QMP_abort QMP_error_exit QMP_verbose)
	CXXFLAGS="${QMP_BKUP_CXXFLAGS}"
	LDFLAGS="${QMP_BKUP_LDFLAGS}"
	LIBS="${LIBS}"
	AC_LANG_RESTORE
	;;
esac

#######################
# Produce output
#######################

#
# Configure subdir packages
#


if test "x${with_libxml2}x" != "xnox";
then
  AC_CONFIG_SUBDIRS(other_libs/qio)
  AC_CONFIG_SUBDIRS(other_libs/xpath_reader)
fi
#
# Set up the doxygen file
# Substitute the right directories
#
# AC_CONFIG_FILES(docs/qdpdoc)

#
# Generate makefiles
#
AC_CONFIG_FILES(Makefile)
AC_CONFIG_FILES(qdp++-config)
AC_CONFIG_FILES(lib/Makefile)
AC_CONFIG_FILES(other_libs/Makefile)
AC_CONFIG_FILES(include/Makefile)
#AC_CONFIG_FILES(docs/Makefile)
AC_CONFIG_FILES(examples/Makefile)
AC_CONFIG_FILES(tests/Makefile)
#
# Finish
#
AC_OUTPUT

