#ifndef QDP_QDP_H
#define QDP_QDP_H


#define QDP_USE_RUNTIME_NC_NS 1

#include<cassert>


#include <qdp_config.h>
#include "qdp_precision.h"

#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <iostream>
#include <utility>
#include <memory>
#include <vector>
#include <map>
#include <deque>
#include <array>
#include <set>
#include <unistd.h>

#include <string>
#include <fstream>
#include <streambuf>

using namespace std;

//using std::iostream;
//using std::ostream;

#include "qdp_forward_jit.h"
#include "qdp_jit_config.h"

template <typename T> concept Arithmetic = std::is_arithmetic_v<T>;

#include "qdp_dispatch.h"
#include "qdp_forward.h"
#include "qdp_llvm.h"
#include "qdp_datalayout.h"


#define PETE_USER_DEFINED_EXPRESSION
namespace QDP {
#include <PETE/PETE.h>
}

#include "qdp_init.h"
#include "qdp_shape.h"



#include "qdp_cache.h"
#include "qdp_alloc_cache.h"
#include "qdp_gpu.h"
#include "qdp_multi.h"
#include "qdp_stdio.h"
#include "qdp_pool_allocator.h"
#include "qdp_arrays.h"
#include "qdp_params.h"
#include "qdp_lattice_layout.h"
#include "qdp_jit_translation.h"
#include "qdp_pete_visitors.h"
#include "qdp_layout.h"
#include "qdp_io.h"

#ifdef QDP_NO_LIBXML2
#warning not using libxml2
#else
#include "qdp_xmlio.h"
#include "qdp_qdpio.h"
#endif

#include "qdp_jit_function.h"
#include "qdp_jit_plan.h"

#ifndef QDP_USE_HDF5
#else
#include "qdp_hdf5.h"
#endif

#include "qdp_masterset.h"
#include "qdp_subset.h"
#include "qdp_stopwatch.h"
#include "qdp_jit_util.h"
#include "qdp_deep_log.h"
#include "qdp_traits.h"
#include "qdp_qdpexpr.h"
#include "qdp_qdptype.h"
#include "qdp_qdpsubtype.h"

namespace QDP {
#include "QDPOperators.h"
}

#include "qdp_basejit.h"
#include "qdp_allocator.h"
#include "qdp_primgamma.h"
#include "qdp_newops.h"
#include "qdp_newopsjit.h"
#include "qdp_optops.h"
#include "qdp_mastermap.h"
#include "qdp_expr_writer.h"
#include "qdp_profile.h"
#include "qdp_parscalar_mapresource.h"
#include "qdp_mapresource.h"
// handle
#include "qdp_map.h"
#include "qdp_baseview.h"
#include "qdp_simpleword.h"
#include "qdp_wordjit.h"
#include "qdp_word.h"
#include "qdp_realityjit.h"
#include "qdp_reality.h"
#include "qdp_primitive.h"
#include "qdp_shape_traits.h"
#include "qdp_outerjit.h"
#include "qdp_outersubjit.h"
#include "qdp_outer.h"
#include "qdp_outersubtype.h"
#include "qdp_defs.h"
#include "qdp_globalfuncs.h"
#include "qdp_specializations.h"
#include "qdp_random.h"
#include "qdp_parscalar_internal.h"
#include "qdp_internal.h"
#include "qdp_jit_plan_leaves.h"
#include "qdp_kernels.h"
#include "qdp_dyn_functions.h"
#include "qdp_functions.h"
#include "qdp_parscalar_specific.h"
#include "qdp_flopcount.h"
#include "qdp_globalfuncs_subtype.h"
#include "qdp_scalarsite_defs2.h"

using namespace QDP;

#endif
