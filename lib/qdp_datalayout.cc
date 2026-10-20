#include "qdp.h"

namespace QDP {

  llvm::Value * datalayout( JitDataOrder lay , NestedIndex a )
  {
    if ( lay == JitDataOrder::SiteFast )
      {
	std::reverse( a.begin() , a.end() );
      }

    llvm::Value * offset = llvm_create_value(0);

    for( auto x = a.begin() ; x != a.end() ; x++ )
      {
	offset = llvm_add( llvm_mul( offset , x->extent.get_val() ) , x->index );
      }

    return offset;
  }


} // namespace
