#include "qdp.h"
#include "custom_kernels/custom_kernels.h"

namespace QDP
{
  void function_multi_localInnerProduct_sum_convert_exec( JitFunction& function,
							  int size, 
							  int threads, 
							  int blocks, 
							  int shared_mem_usage,
							  multi1d<QDPCache::ArgKey>& in_ids, 
							  int out_id,
							  int v_id,
							  int N,
							  const multi1d<int>& sizes,
							  const multi1d<QDPCache::ArgKey>& table_ids)
  {
    if ( (threads & (threads - 1)) != 0 )
      {
	QDPIO::cerr << "internal error: function_multi_localInnerProduct_sum_convert_exec not power of 2\n";
	QDP_abort(1);
      }

    int sizes_id = QDP_get_global_cache().add( sizes.size()*sizeof(int) , QDPCache::Flags::OwnHostMemory , QDPCache::Status::host , sizes.slice() , NULL , NULL );

    JitParam jit_numsubsets( QDP_get_global_cache().addJitParamInt( N ) );
    JitParam jit_tables(     QDP_get_global_cache().addMulti(       table_ids  ) );
    JitParam jit_in_ids(     QDP_get_global_cache().addMulti(       in_ids ) );
	      
    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_numsubsets.get_id() );
    ids.push_back( sizes_id );
    ids.push_back( jit_tables.get_id() );
    ids.push_back( jit_in_ids.get_id() );
    ids.push_back( out_id );
    ids.push_back( v_id );

    jit_launch_explicit_geom( function , ids , getGeom( size , threads ) , shared_mem_usage );

    QDP_get_global_cache().signoff(sizes_id);
  }
}
