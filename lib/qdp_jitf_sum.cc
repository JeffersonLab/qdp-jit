#include "qdp.h"


namespace QDP {

  void
  function_sum_convert_ind_exec( CUfunction function, 
				 int size, int threads, int blocks, int shared_mem_usage,
				 int in_id, int out_id, int siteTableId )
  {
    // Make sure 'threads' is a power of two (the jit kernel make this assumption)
    assert( (threads & (threads - 1)) == 0 );
    
    int lo = 0;
    int hi = size;
    

    JitParam jit_lo( QDP_get_global_cache().addJitParamInt( lo ) );
    JitParam jit_hi( QDP_get_global_cache().addJitParamInt( hi ) );
  
    std::vector<QDPCache::ArgKey> ids;
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
  function_summulti_convert_ind_exec( CUfunction function, 
				      int size, int threads, int blocks, int shared_mem_usage,
				      int in_id, int out_id,
				      int numsubsets,
				      const multi1d<int>& sizes,
				      const multi1d<QDPCache::ArgKey>& table_ids )
  {
    // Make sure 'threads' is a power of two (the jit kernel make this assumption)
    assert( (threads & (threads - 1)) == 0 );

    int sizes_id = QDP_get_global_cache().add( sizes.size()*sizeof(int) , QDPCache::Flags::OwnHostMemory , QDPCache::Status::host , sizes.slice() , NULL , NULL );

    JitParam jit_numsubsets( QDP_get_global_cache().addJitParamInt( numsubsets ) );
    JitParam jit_tables(     QDP_get_global_cache().addMulti(       table_ids  ) );
						      
    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_numsubsets.get_id() );
    ids.push_back( sizes_id );
    ids.push_back( jit_tables.get_id() );
    ids.push_back( in_id );
    ids.push_back( out_id );
 
    std::vector<void*> args( QDP_get_global_cache().get_kernel_args(ids) );
    kernel_geom_t now = getGeom( size , threads );

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threads,1,1,    shared_mem_usage, 0, &args[0] , 0);

    QDP_get_global_cache().signoff(sizes_id);
  }



  void
  function_summulti_exec( CUfunction function, 
			  int size, int threads, int blocks, int shared_mem_usage,
			  int in_id, int out_id,
			  int numsubsets,
			  const multi1d<int>& sizes )
  {
    // Make sure 'threads' is a power of two (the jit kernel make this assumption)
    assert( (threads & (threads - 1)) == 0 );

    int sizes_id = QDP_get_global_cache().add( sizes.size()*sizeof(int) , QDPCache::Flags::OwnHostMemory , QDPCache::Status::host , sizes.slice() , NULL , NULL );

    JitParam jit_numsubsets( QDP_get_global_cache().addJitParamInt( numsubsets ) );
						      
    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_numsubsets.get_id() );
    ids.push_back( sizes_id );
    ids.push_back( in_id );
    ids.push_back( out_id );
 
    std::vector<void*> args( QDP_get_global_cache().get_kernel_args(ids) );
    kernel_geom_t now = getGeom( size , threads );

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threads,1,1,    shared_mem_usage, 0, &args[0] , 0);

    QDP_get_global_cache().signoff(sizes_id);
  }



  void
  function_sum_convert_exec( CUfunction function, 
			     int size, int threads, int blocks, int shared_mem_usage,
			     int in_id, int out_id)
  {
    // Make sure 'threads' is a power of two (the jit kernel make this assumption)
    assert( (threads & (threads - 1)) == 0 );

    int lo = 0;
    int hi = size;

    JitParam jit_lo( QDP_get_global_cache().addJitParamInt( lo ) );
    JitParam jit_hi( QDP_get_global_cache().addJitParamInt( hi ) );
  
    std::vector<QDPCache::ArgKey> ids;
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
    // Make sure 'threads' is a power of two (the jit kernel make this assumption)
    assert( (threads & (threads - 1)) == 0 );

    int lo = 0;
    int hi = size;

    JitParam jit_lo( QDP_get_global_cache().addJitParamInt( lo ) );
    JitParam jit_hi( QDP_get_global_cache().addJitParamInt( hi ) );
  
    std::vector<QDPCache::ArgKey> ids;
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
    // Make sure 'threads' is a power of two (the jit kernel make this assumption)
    assert( (threads & (threads - 1)) == 0 );

    int lo = 0;
    int hi = size;

    JitParam jit_lo( QDP_get_global_cache().addJitParamInt( lo ) );
    JitParam jit_hi( QDP_get_global_cache().addJitParamInt( hi ) );
  
    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_lo.get_id() );
    ids.push_back( jit_hi.get_id() );
    ids.push_back( in_id );
    ids.push_back( out_id );
 
    std::vector<void*> args( QDP_get_global_cache().get_kernel_args(ids) );
    kernel_geom_t now = getGeom( hi-lo , threads );

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threads,1,1,    shared_mem_usage, 0, &args[0] , 0);
  }



  
  void
  function_isfinite_convert_exec( CUfunction function, 
				  int size, int threads, int blocks, int shared_mem_usage,
				  int in_id, int out_id)
  {
    // Make sure 'threads' is a power of two (the jit kernel make this assumption)
    assert( (threads & (threads - 1)) == 0 );

    int lo = 0;
    int hi = size;

    JitParam jit_lo( QDP_get_global_cache().addJitParamInt( lo ) );
    JitParam jit_hi( QDP_get_global_cache().addJitParamInt( hi ) );
  
    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_lo.get_id() );
    ids.push_back( jit_hi.get_id() );
    ids.push_back( in_id );
    ids.push_back( out_id );
 
    std::vector<void*> args( QDP_get_global_cache().get_kernel_args(ids) );
    kernel_geom_t now = getGeom( hi-lo , threads );

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threads,1,1,    shared_mem_usage, 0, &args[0] , 0);
  }



  void
  function_isfinite_exec( CUfunction function, 
			  int size, int threads, int blocks, int shared_mem_usage,
			  int in_id, int out_id)
  {
    // Make sure 'threads' is a power of two (the jit kernel make this assumption)
    assert( (threads & (threads - 1)) == 0 );

    int lo = 0;
    int hi = size;

    JitParam jit_lo( QDP_get_global_cache().addJitParamInt( lo ) );
    JitParam jit_hi( QDP_get_global_cache().addJitParamInt( hi ) );
  
    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_lo.get_id() );
    ids.push_back( jit_hi.get_id() );
    ids.push_back( in_id );
    ids.push_back( out_id );
 
    std::vector<void*> args( QDP_get_global_cache().get_kernel_args(ids) );
    kernel_geom_t now = getGeom( hi-lo , threads );

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threads,1,1,    shared_mem_usage, 0, &args[0] , 0);
  }

  
  
}

