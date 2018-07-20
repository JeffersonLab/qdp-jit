#include "qdp.h"


namespace QDP {

  void
  function_sum_convert_ind_exec( CUfunction function, 
				 int size, int threads, int blocks, int shared_mem_usage,
				 int in_id, int out_id, int siteTableId )
  {
    int lo = 0;
    int hi = size;

    JitParam jit_lo( QDP_get_global_cache().addJitParamInt( lo ) );
    JitParam jit_hi( QDP_get_global_cache().addJitParamInt( hi ) );
  
    std::vector<int> ids;
    ids.push_back( jit_lo.get_id() );
    ids.push_back( jit_hi.get_id() );
    ids.push_back( siteTableId );
    ids.push_back( in_id );
    ids.push_back( out_id );
 
    std::vector<void*> args( QDP_get_global_cache().get_kernel_args(ids) );
    kernel_geom_t now = getGeom( hi-lo , threads );

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threads,1,1,    shared_mem_usage, 0, &args[0] , 0);
  }



  void
  function_sum_convert_exec( CUfunction function, 
			     int size, int threads, int blocks, int shared_mem_usage,
			     int in_id, int out_id)
  {
    int lo = 0;
    int hi = size;

    JitParam jit_lo( QDP_get_global_cache().addJitParamInt( lo ) );
    JitParam jit_hi( QDP_get_global_cache().addJitParamInt( hi ) );
  
    std::vector<int> ids;
    ids.push_back( jit_lo.get_id() );
    ids.push_back( jit_hi.get_id() );
    ids.push_back( in_id );
    ids.push_back( out_id );
 
    std::vector<void*> args( QDP_get_global_cache().get_kernel_args(ids) );
    kernel_geom_t now = getGeom( hi-lo , threads );

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threads,1,1,    shared_mem_usage, 0, &args[0] , 0);
  }



  void
  function_sum_exec( CUfunction function, 
		     int size, int threads, int blocks, int shared_mem_usage,
		     int in_id, int out_id)
  {

    int lo = 0;
    int hi = size;

    JitParam jit_lo( QDP_get_global_cache().addJitParamInt( lo ) );
    JitParam jit_hi( QDP_get_global_cache().addJitParamInt( hi ) );
  
    std::vector<int> ids;
    ids.push_back( jit_lo.get_id() );
    ids.push_back( jit_hi.get_id() );
    ids.push_back( in_id );
    ids.push_back( out_id );
 
    std::vector<void*> args( QDP_get_global_cache().get_kernel_args(ids) );
    kernel_geom_t now = getGeom( hi-lo , threads );

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threads,1,1,    shared_mem_usage, 0, &args[0] , 0);
  }



  void function_global_max_exec( CUfunction function, 
				 int size, int threads, int blocks, int shared_mem_usage,
				 int in_id, int out_id)
  {
    int lo = 0;
    int hi = size;

    JitParam jit_lo( QDP_get_global_cache().addJitParamInt( lo ) );
    JitParam jit_hi( QDP_get_global_cache().addJitParamInt( hi ) );
  
    std::vector<int> ids;
    ids.push_back( jit_lo.get_id() );
    ids.push_back( jit_hi.get_id() );
    ids.push_back( in_id );
    ids.push_back( out_id );
 
    std::vector<void*> args( QDP_get_global_cache().get_kernel_args(ids) );
    kernel_geom_t now = getGeom( hi-lo , threads );

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threads,1,1,    shared_mem_usage, 0, &args[0] , 0);
  }

  
}

