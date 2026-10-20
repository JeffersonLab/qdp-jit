#include "qdp.h"


namespace QDP {

  namespace {
    void *sync_hst_ptr;
    void *sync_dev_ptr;
  }

  namespace {
    RingBuffer* __global_ring_buffer;
  }


  RingBuffer& QDP_get_global_ring_buffer()
  {
    if (!__global_ring_buffer) {
      __global_ring_buffer = new RingBuffer();
    }
    return *__global_ring_buffer;
  }


  RingBuffer::RingBuffer()
  {
    ringBufferOScalar.resize( jit_config_get_oscalar_ringbuffer_size() , -1 );
    ringBufferNext = 0;
  }
  

  int RingBuffer::allocate( size_t size , const void *hstPtr )
  {
    int& buf_elem = ringBufferOScalar.at( ringBufferNext );

    if ( buf_elem >= 0)
      {
      	QDP_get_global_cache().signoff( buf_elem );
      }
    
    buf_elem = QDP_get_global_cache().addOwnHostMemNoPage( size, hstPtr );

    ringBufferNext = ( ringBufferNext + 1 ) % ringBufferOScalar.size();

    return buf_elem;
  }

  
  void RingBuffer::done()
  {
    for( auto i : ringBufferOScalar )
      {
	if (i >=0 )
	  QDP_get_global_cache().signoff( i );
      }
  }



  DynKey get_dyn_key( const Shape& shape )
  {
    DynKey key;
    key.add( shape );
    return key;
  }

  

  void jit_util_sync_init()
  {
    gpu_host_alloc( &sync_hst_ptr , sizeof(int) );
    gpu_malloc( &sync_dev_ptr , sizeof(int) );
    *(int*)sync_hst_ptr = 0;
  }

  void jit_util_sync_done()
  {
    gpu_host_free( sync_hst_ptr );
    gpu_free( sync_dev_ptr );
  }

  void jit_util_sync_copy()
  {
    gpu_memcpy_d2h( sync_hst_ptr , sync_dev_ptr , sizeof(int) );
  }




  void jit_check_num_args( JitFunction& function , const std::vector<QDPCache::ArgKey>& ids )
  {
    if ( function.get_num_params() >= 0 && function.get_num_params() != (int)ids.size() )
      {
	QDPIO::cerr << "JIT launch: kernel " << function.get_kernel_name()
		    << " expects " << function.get_num_params()
		    << " arguments, but " << ids.size() << " were given" << std::endl;
	QDPIO::cerr << function.get_pretty() << std::endl;
	QDP_abort(1);
      }
  }


  void jit_check_workgroup_bound( JitFunction& function , int threads_per_block )
  {
    if ( function.get_workgroup_bound() > 0 && threads_per_block > function.get_workgroup_bound() )
      {
	QDPIO::cerr << "JIT launch: kernel " << function.get_kernel_name()
		    << " was compiled for at most " << function.get_workgroup_bound()
		    << " threads per block, but launched with " << threads_per_block
		    << " (build functions of kernels launched with explicit geometry must call set_explicit_geom)" << std::endl;
	QDPIO::cerr << function.get_pretty() << std::endl;
	QDP_abort(1);
      }
  }


  void jit_get_function(JitFunction& f)
  {
    llvm_exit();

    llvm_build_function(f);
  }



  llvm::Value* llvm_epsilon_1st( int p1 , llvm::Value* j )
  {
    return llvm_rem( llvm_add( j , llvm_create_value( p1 ) ) , llvm_create_value( 3 ) );

  }
  
  llvm::Value* llvm_epsilon_2nd( int p2 , llvm::Value* i )
  {
    return llvm_rem( llvm_add( i , llvm_create_value( p2 ) ) , llvm_create_value( 3 ) );
  }


  void f1(int l,int r)
  {
    int i = (r + 1) % 3;
    int j = (l + 1) % 3;
    cout << "s1.elem(" << i << "," << j << ") * ";
  }



  


  void jit_report_launch_out_of_memory( JitFunction& function )
  {
    const size_t stack    = function.get_stack();
    const size_t resident = gpu_get_max_resident_threads();
    const size_t free_mem = gpu_mem_free();

    QDPIO::cerr << "Kernel launch failed: out of device memory.\n"
		<< "  The kernel uses " << stack << " bytes of stack (local memory) per thread.\n";
    if ( resident > 0 && stack > 0 )
      QDPIO::cerr << "  The driver reserves it for all " << resident << " threads that can be resident on the device: "
		  << stack * resident / 1024 / 1024 << " MB.\n";
    QDPIO::cerr << "  Free device memory outside the memory pool: " << free_mem / 1024 / 1024 << " MB";
    if ( jit_config_get_pool_reserve() > 0 )
      QDPIO::cerr << " (-pool-reserve " << jit_config_get_pool_reserve() / 1024 / 1024 << " MB)";
    QDPIO::cerr << ".\n";
    if ( resident > 0 && stack > 0 )
      QDPIO::cerr << "  Keep at least " << ( stack * resident / 1024 / 1024 + 256 )
		  << " MB outside the pool: -pool-reserve " << ( stack * resident / 1024 / 1024 + 256 ) << "m, or reduce -poolsize.\n";
    else
      QDPIO::cerr << "  Keep more memory outside the pool: -pool-reserve, or reduce -poolsize.\n";
    QDPIO::cerr << "  Kernel: " << function.get_kernel_name() << std::endl;
  }


  void jit_launch_gpu( JitFunction& function , int th_count , std::vector<QDPCache::ArgKey>& ids , bool set_l0_event )
  {
    jit_check_num_args( function , ids );
    
    QDPCache::KernelArgs_t args( QDP_get_global_cache().get_kernel_args(ids) );

    // Check for no-op
    if ( th_count == 0 )
      return;

    // Increment the call counter
    function.inc_call_counter();

    int threads_per_block = std::min( jit_config_get_threads_per_block() , th_count );

    kernel_geom_t geom = getGeom( th_count , threads_per_block );

    jit_check_workgroup_bound( function , geom.threads_per_block );

    JitResult result = gpu_launch_kernel( function,
					  geom.Nblock_x,geom.Nblock_y,1,
					  geom.threads_per_block,1,1,
					  0, // shared mem
					  args ,
					  set_l0_event );
    
    if (result == JitResult::JitOutOfMemory)
      jit_report_launch_out_of_memory( function );

    if (result != JitResult::JitSuccess) {
      QDPIO::cerr << "jit launch error, grid=(" << geom.Nblock_x << "," << geom.Nblock_y << ",1), block=(" << geom.threads_per_block << ",1,1)\n";
      QDPIO::cerr << "kernel: regs = " << function.get_regs() << ", stack = " << function.get_stack() << ", cmem = " << function.get_cmem() << std::endl;      
      QDP_abort(1);
    }
  }


  void jit_launch_host( JitFunction& function , std::vector<QDPCache::ArgKey>& ids )
  {
    jit_check_num_args( function , ids );

    QDPCache::HostArgs_t args( QDP_get_global_cache().get_host_args(ids) );

    void (*FP)(void*) = (void (*)(void*))(intptr_t)function.get_function();

    FP( args.data() );

    function.inc_call_counter();
  }


  
  void jit_launch( JitFunction& function , int th_count , std::vector<QDPCache::ArgKey>& ids , bool set_l0_event )
  {
    if (function.isGPU())
      {
	jit_launch_gpu( function , th_count , ids , set_l0_event );
      }
    else
      {
	jit_launch_host( function , ids );
      }
  }





  
  bool jit_launch_explicit_geom( JitFunction& function , std::vector<QDPCache::ArgKey>& ids , const kernel_geom_t& geom , unsigned int shared )
  {
    jit_check_num_args( function , ids );
    jit_check_workgroup_bound( function , geom.threads_per_block );

    QDPCache::KernelArgs_t args( QDP_get_global_cache().get_kernel_args(ids) );

    // Increment the call counter
    function.inc_call_counter();

#if defined (QDP_BACKEND_CUDA)
    if ( geom.threads_per_block * function.get_regs() > jit_config_get_max_regs_per_block() )
      {
	// since this case is appropiately handled, don't verbose print
	// cout << "jit launch explicit, grid=(" << geom.Nblock_x << "," << geom.Nblock_y << ",1), block=(" << geom.threads_per_block << ",1,1)" << std::endl;
	// cout << "kernel: regs = " << function.get_regs() << ", stack = " << function.get_stack() << ", cmem = " << function.get_cmem() << std::endl;
	// cout << "resulting in " << geom.threads_per_block * function.get_regs() << " registers, exceeding hardware limit " << jit_config_get_max_regs_per_block() << std::endl;
	return false;
      }
#endif
    
    JitResult result = gpu_launch_kernel( function,
					  geom.Nblock_x,geom.Nblock_y,1,
					  geom.threads_per_block,1,1,
					  shared,
					  args );

    if (result == JitResult::JitOutOfMemory)
      jit_report_launch_out_of_memory( function );

    if (result != JitResult::JitSuccess) {
      QDPIO::cerr << "jit launch explicit geom error, grid=(" << geom.Nblock_x << "," << geom.Nblock_y << ",1), block=(" << geom.threads_per_block << ",1,1)\n";
      QDP_abort(1);
    }

    return true;
  }



  llvm::Value* jit_ternary( llvm::Value* cond , llvm::Value* val_true , llvm::Value* val_false )
  {
    JitIf ifCond(cond);
    {
      // empty
    }
    ifCond.els();
    {
      // empty
    }
    ifCond.end();
    
    llvm::Value* r = llvm_phi( llvm_val_type( val_true ) , 2 );
    llvm_add_incoming( r , val_true  , ifCond.get_block_true()  );
    llvm_add_incoming( r , val_false , ifCond.get_block_false() );

    return r;
  }
  
  

  namespace {
    const int small_trip_count = 4;
  }

  JitForLoop::JitForLoop( llvm::Value* start , llvm::Value* end ) : r_start(start), r_end(end)
  {
    block_outer = llvm_get_insert_point();
    block_loop_cond = llvm_new_basic_block();
    block_loop_body = llvm_new_basic_block();
    block_loop_exit = llvm_new_basic_block();

    llvm_branch( block_loop_cond );
    llvm_set_insert_point(block_loop_cond);
  
    r_i = llvm_phi( llvm_get_type<int>() , 2 );

    llvm_add_incoming( r_i , start , block_outer );

    llvm_cond_branch( llvm_lt( r_i , end ) , block_loop_body , block_loop_exit );

    llvm_set_insert_point( block_loop_body );
  }
  llvm::Value * JitForLoop::index() const
  {
    return r_i;
  }
  JitValue JitForLoop::jit() const
  {
    return JitValue(r_i);
  }
  void JitForLoop::end()
  {
    llvm::Value * r_i_plus = llvm_add( r_i , llvm_create_value(1) );

    llvm_add_incoming( r_i , r_i_plus , llvm_get_insert_point() );
  
    llvm_branch( block_loop_cond );

    if (!full_unroll)
      {
	int n = llvm_const_trip_count( r_start , r_end );
	if ( n > 0 && n <= small_trip_count )
	  full_unroll = true;
      }

    if (full_unroll)
      llvm_loop_unroll_full_latch();

    llvm_set_insert_point(block_loop_exit);

    closed = true;
  }



  llvm::Value* operator+( const JitForLoop& loop , int i )
  {
    return llvm_add( loop.index() , llvm_create_value(i) );
  }

  llvm::Value* operator-( const JitForLoop& loop , int i )
  {
    return llvm_sub( loop.index() , llvm_create_value(i) );
  }

  llvm::Value* operator*( const JitForLoop& loop , int i )
  {
    return llvm_mul( loop.index() , llvm_create_value(i) );
  }


  llvm::Value* operator+( int i , const JitForLoop& loop )
  {
    return llvm_add( llvm_create_value(i) , loop.index() );
  }

  llvm::Value* operator-( int i , const JitForLoop& loop )
  {
    return llvm_sub( llvm_create_value(i) , loop.index() );
  }

  llvm::Value* operator*( int i , const JitForLoop& loop )
  {
    return llvm_mul( llvm_create_value(i) , loop.index() );
  }


  

  JitForLoopPower::JitForLoopPower( llvm::Value* i_start  )
  {
    block_outer = llvm_get_insert_point();
    block_loop_cond = llvm_new_basic_block();
    block_loop_body = llvm_new_basic_block();
    block_loop_exit = llvm_new_basic_block();

    llvm_branch( block_loop_cond );
    llvm_set_insert_point(block_loop_cond);
  
    r_i = llvm_phi( llvm_get_type<int>() , 2 );

    llvm_add_incoming( r_i , i_start , block_outer );

    llvm_cond_branch( llvm_gt( r_i , llvm_create_value( 0 ) ) , block_loop_body , block_loop_exit );

    llvm_set_insert_point( block_loop_body );
  }
  llvm::Value * JitForLoopPower::index()
  {
    return r_i;
  }
  void JitForLoopPower::end()
  {
    llvm::Value * r_i_plus = llvm_shr( r_i , llvm_create_value(1) );
    llvm_add_incoming( r_i , r_i_plus , llvm_get_insert_point() );
  
    llvm_branch( block_loop_cond );

    llvm_set_insert_point(block_loop_exit);
  }



  JitValue operator+( const JitValue& l , const JitValue& r )
  {
    return llvm_add(l,r);
  }
  
  JitValue operator-( const JitValue& l , const JitValue& r )
  {
    return llvm_sub(l,r);
  }
  
  JitValue operator*( const JitValue& l , const JitValue& r )
  {
    return llvm_mul(l,r);
  }
  
  JitValue operator/( const JitValue& l , const JitValue& r )
  {
    return llvm_div(l,r);
  }

  JitValue operator%( const JitValue& l , const JitValue& r )
  {
    return llvm_rem(l,r);
  }

  

} //namespace
