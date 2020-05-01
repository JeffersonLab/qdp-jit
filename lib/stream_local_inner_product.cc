#include "qdp.h"
#include "custom_kernels/custom_kernels.h"

namespace QDP
{
  void function_multi_localInnerProduct_sum_convert_exec( JitFunction function,
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
    QDP_error_exit("fixme function_multi_localInnerProduct_sum_convert_exec");
#if 0

    assert( (threads & (threads - 1)) == 0 );

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

    JIT_AMD_add_workgroup_sizes( ids );

    std::vector<void*> args( QDP_get_global_cache().get_kernel_args(ids) );
    kernel_geom_t now = getGeom( size , threads );

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threads,1,1,    shared_mem_usage, 0, &args[0] , 0);

    QDP_get_global_cache().signoff(sizes_id);
#endif
  }
}
