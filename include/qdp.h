// -*- C++ -*-

/*! \file
 * \brief Primary include file for QDP
 *
 * No other file should be included by the user
 */

/*! \mainpage  QDP
 *
 * \section Description
 *
 * QDP is a C++ data-parallel interface for Lattice field theory.
 * The QDP interface provides an environment somewhat similar to 
 * Fortran 90 - namely data-parallel operations (operator/infix form)
 * which can be applied on lattice wide objects. The interface provides a
 * level of abstraction such that high-level user code written using the
 * API can be run unchanged on a single processor workstation or a
 * collection of multiprocessor nodes with parallel communications.
 * Architectural dependencies are hidden below the interface. A variety
 * of types for the site elements are provided. To achieve good
 * performance, overlapping communication and computation primitives are
 * provided.
 */

/*! \namespace QDP
 * \brief Primary namespace holding all QDP types, operations and objects
 */

#ifndef QDP_INCLUDE
#define QDP_INCLUDE

//#define NDEBUG
#include<cassert>


/* Get local configuration options (ARCH_SCALAR/PARSCALAR, Nd, Nc, Ns) */
#include <qdp_config.h>
#include "qdp_precision.h"


// Under gcc, set some attributes
#if defined(__GNUC__)
// gcc
#define QDP_ALIGN8   __attribute__ ((aligned (8)))
#define QDP_ALIGN16  __attribute__ ((aligned (16)))
#define QDP_INLINE   __attribute__ ((always_inline))
// The attributes in QDP_CONST is buggering g++-3.4 
//#define QDP_CONST    __attribute__ ((const,pure))
#define QDP_CONST
#define QDP_CINLINE  __attribute__ ((always_inline,const,pure))
#else
// default
#define QDP_ALIGN8
#define QDP_ALIGN16
#define QDP_INLINE
#define QDP_CONST
#define QDP_CINLINE
#endif

// Commented this out and set QDP_ALIGNMENT_SIZE to be 16 all the time
// This is a minimal waste of space and should allow an SSE dslash
// to be used even if the QDP itself is not compiled with SSE.
#ifdef QDP_AC_ALIGNMENT_SIZE
#define QDP_ALIGNMENT_SIZE QDP_AC_ALIGNMENT_SIZE
#else
#define QDP_ALIGNMENT_SIZE 16
#endif

// YUKKY - Eventually get rid of these includes
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <iostream>
#include <utility>
#include <memory>
#include <vector>

#include <string>
#include <fstream>
#include <streambuf>


using namespace std;   // I do not like this - fix later

using std::iostream;
using std::ostream;
// END OF YUKKINESS


#include "cuda.h"

#include "qdp_dispatch.h"

#include "qdp_forward.h"
#include "qdp_llvm.h"
#include "qdp_datalayout.h"


// Basic includes
#define PETE_USER_DEFINED_EXPRESSION
namespace QDP {
#include <PETE/PETE.h>
}

#include "qdp_init.h"

#include "qdp_deviceparams.h"
#include "qdp_cuda.h"
#include "qdp_cuda_allocator.h"

#include "qdp_multi.h"
#include "qdp_stdio.h"
#include "qdp_pool_allocator.h"

#include "qdp_jit_util.h"

//#include "qdp_multi.h"
#include "qdp_cache.h"

//

#include "qdp_arrays.h"

#include "qdp_params.h"
#include "qdp_layout.h"
#include "qdp_io.h"
//#include "qdp_stdio.h"


#ifdef QDP_NO_LIBXML2
#warning not using libxml2
#else
#include "qdp_xmlio.h"
#include "qdp_qdpio.h"
#endif

#ifndef QDP_USE_HDF5
//#warning not using hdf5
#else
#include "qdp_hdf5.h"
#endif


#include "qdp_masterset.h"

#include "qdp_subset.h"
#include "qdp_stopwatch.h"

#include "qdp_pete_visitors.h"

#include "qdp_traits.h"
#include "qdp_qdpexpr.h"
#include "qdp_qdptypejit.h"
#include "qdp_qdptype.h"
#include "qdp_qdpsubtypejit.h"
#include "qdp_qdpsubtype.h"

namespace QDP {
#include "QDPOperators.h"
}


#include "qdp_basejit.h"

#include "qdp_allocator.h"

#include "qdp_newops.h"
#include "qdp_newopsreg.h"

#include "qdp_optops.h"
#include "qdp_mastermap.h"

#include "qdp_profile.h"

#if defined(ARCH_SCALAR)
#include "qdp_scalar_mapresource.h"
#elif defined(ARCH_PARSCALAR)
#include "qdp_parscalar_mapresource.h"
#else
#error "Unknown architecture ARCH"
#endif

#include "qdp_mapresource.h"
#include "qdp_handle.h"
#include "qdp_map.h"
#include "qdp_autotuning.h"

#include "qdp_multi_outer.h"


#include "qdp_simpleword.h"
#include "qdp_wordjit.h"
#include "qdp_wordreg.h"
#include "qdp_word.h"
#include "qdp_realityjit.h"
#include "qdp_realityreg.h"
#include "qdp_reality.h"
#include "qdp_inner.h"
#include "qdp_primitive.h"
#include "qdp_outerjit.h"
#include "qdp_outersubjit.h"
#include "qdp_outer.h"
#include "qdp_outersubtype.h"

#include "qdp_viewleaf.h"

// Replaces previous ifdef structure. Structure moved into the header file
#include "qdp_defs.h"
#include "qdp_globalfuncs.h"
#include "qdp_specializations.h"

#include "qdp_random.h"
#include "qdp_poolbisect.h"

#if defined(ARCH_SCALAR)
#include "qdp_scalar_internal.h"
#elif defined(ARCH_PARSCALAR)
#include "qdp_parscalar_internal.h"
#else
#error "Unknown architecture ARCH"
#endif


#include "qdp_internal.h"
#include "qdp_jitfunction.h"
#include "qdp_jitf_copymask.h"
#include "qdp_jitf_sum.h"
#include "qdp_jitf_summulti.h"
#include "qdp_jitf_globalmax.h"
#include "qdp_jitf_gaussian.h"

#include "qdp_sum.h"
#include "qdp_functions.h"

#if defined(ARCH_SCALAR)
#include "qdp_scalar_specific.h"
#elif defined(ARCH_PARSCALAR)
#include "qdp_parscalar_specific.h"
#else
#error "Unknown architecture ARCH"
#endif


#include "qdp_flopcount.h"
#include "qdp_globalfuncs_subtype.h"

#endif  // QDP_INCLUDE
