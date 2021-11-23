#include "qdp.h"

namespace QDP {

  llvm::Value * datalayout( JitDeviceLayout lay , IndexDomainVector a ) {
    assert(a.size() > 0);

    // In case of a coalesced layout (OLattice)
    // We reverse the data layout given by the nesting order
    // of aggregates, i.e. reality slowest, lattice fastest
    // In case of a scalar layout (sums,comms buffers,OScalar)
    // We actually use the index order/data layout given by the
    // nesting order of aggregates
    //#if defined (QDP_BACKEND_CUDA) || defined (QDP_BACKEND_ROCM)
    if ( lay == JitDeviceLayout::Coalesced ) {
      std::reverse( a.begin() , a.end() );
    }
    //#endif

    // TODO: need to handle JitDeviceLayout::Coalesced_scalar_idx
    
    llvm::Value * offset = llvm_create_value(0);
    for( auto x = a.begin() ; x != a.end() ; x++ ) {
      int         Index;
      llvm::Value * index;
      std::tie(Index,index) = *x;
      llvm::Value * Index_jit = llvm_create_value(Index);
      offset = llvm_add( llvm_mul( offset , Index_jit ) , index );
    }
    return offset;
  }

  /*
    Latt: Sites, idx
    Spin: 1, sp
    Colr: 1, co
    Cmpl: 2, z

    offset = (( idx * 1 + sp ) * 1 + co ) * 2 + z

    re im re im re im
  */

  


} // namespace
