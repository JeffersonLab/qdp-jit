#include "qdp.h"

#if defined(QDP_BACKEND_AVX)
#include <omp.h>
#endif

namespace QDP {


  namespace {
    void *sync_hst_ptr;
    void *sync_dev_ptr;
  }

  namespace
  {
    typedef std::map< std::string , int > DBTuneType;
    DBTuneType db_tune;
    bool db_tune_modified = false;
    int db_tune_count = 0;
  }

  namespace
  {
    std::vector<int> ringBufferOScalar;
    int ringBufferNext;
  }


  void jit_util_ringBuffer_init()
  {
    ringBufferOScalar.resize( jit_config_get_oscalar_ringbuffer_size() , -1 );
    ringBufferNext = 0;
  }
  

  int jit_util_ringBuffer_allocate( size_t size , const void *hstPtr )
  {
    int& buf_elem = ringBufferOScalar.at( ringBufferNext );

    if ( buf_elem >= 0)
      {
	//QDPIO::cout << "ringBuffer: sign off " << buf_elem << std::endl;
      	QDP_get_global_cache().signoff( buf_elem );
      }
    
    buf_elem = QDP_get_global_cache().registrateOwnHostMemNoPage( size, hstPtr );
    //QDPIO::cout << "ringBuffer: allocated " << buf_elem << " (" << size << ")" << std::endl;

    ringBufferNext = ( ringBufferNext + 1 ) % ringBufferOScalar.size();
    //QDPIO::cout << "ringBufferNext = " << ringBufferNext << std::endl;

    return buf_elem;
  }
  

  int jit_util_get_tune_count()
  {
    return db_tune_count;
  }
  
  void db_tune_write( std::string filename )
  {
    if (!db_tune_modified)
      return;
    
    BinaryFileWriter db(filename);

    write( db , (int)db_tune.size() );
    for ( DBTuneType::iterator it = db_tune.begin() ; it != db_tune.end() ; ++it ) 
      {
	write(db , it->first);
	write(db , it->second);
      }
    db.close();
    QDPIO::cout << "  tuning file updated" << std::endl;
  }

  
  void db_tune_read( std::string filename )
  {
    QDPIO::cout << "Tuning DB" << std::endl;
    QDPIO::cout << "  checking file                       : " << filename << std::endl;
    {
      ifstream f(filename.c_str());
      if (! f.good() )
	{
	  QDPIO::cout << "  creating new tuning file" << std::endl;
	  return;
	}
    }
    
    QDPIO::cout << "  opening file" << std::endl;

    BinaryFileReader db(filename);

    int n;
    read( db , n );
    
    QDPIO::cout << "  number of tuning records            : " << n << std::endl;

    for( int i = 0 ; i < n ; ++i )
      {
	std::string key;
	int config;

	read( db , key , 10*1024 );
	read( db , config );

	db_tune[ key ] = config;

	//QDPIO::cout << "read: " << config << "\t" << key << std::endl;
      }
    db.close();

    QDPIO::cout << "  done reading tuning file" << std::endl;
  }

  
  bool db_tune_find_info( JitFunction& function )
  {
    std::string key = jit_util_get_static_dynamic_string( function.get_pretty() );
    
    DBTuneType::iterator it = db_tune.find( key );

    if ( it != db_tune.end() )
      {
	function.set_threads_per_block( it->second );
	return true;
      }

    return false;
  }

  
  void db_tune_insert_info( JitFunction& function , int config )
  {
    std::string key = jit_util_get_static_dynamic_string( function.get_pretty() );
    db_tune[key] = config;
    db_tune_modified = true;
    db_tune_count++;
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

  // For AMD the workgroup sizes are passed as kernel parameters
  // For now as placeholder values
  //
  void JIT_AMD_add_workgroup_sizes( std::vector<QDPCache::ArgKey>& ids )
  {
    JitParam jit_ntidx( QDP_get_global_cache().addJitParamInt( -1 ) );
    JitParam jit_nctaidx( QDP_get_global_cache().addJitParamInt( -1 ) );

    ids.insert(ids.begin(), jit_nctaidx.get_id() );
    ids.insert(ids.begin(), jit_ntidx.get_id() );
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



  

  
  
  

#if defined(QDP_BACKEND_CUDA) || defined(QDP_BACKEND_ROCM)
  void jit_tune( JitFunction& function , int th_count , QDPCache::KernelArgs_t& args)
  {
    if ( ! function.get_enable_tuning() )
      {
	QDPIO::cout << "Tuning disabled for: " << function.get_kernel_name() << std::endl;
	function.set_threads_per_block( jit_config_get_threads_per_block() );
	return;
      }


    if (db_tune_find_info( function ))
      {
	return;
      }
    

    size_t field_size      = QDP_get_global_cache().getSize       ( function.get_dest_id() );

    std::vector<QDPCache::ArgKey> vec_id;
    vec_id.push_back( function.get_dest_id() );
    std::vector<void*> vec_ptrs = QDP_get_global_cache().get_dev_ptrs( vec_id );
    void* dev_ptr = vec_ptrs.at(0);
  
    void* host_ptr;

    if ( ! (host_ptr = malloc( field_size )) )
      {
	QDPIO::cout << "Tuning: Cannot allocate host memory!" << endl;
	QDP_abort(1);
      }
    
    if (jit_config_get_tuning_verbose())
      {
	QDPIO::cout << "Starting tuning of: " << function.get_kernel_name() << std::endl;
	QDPIO::cout << function.get_pretty() << std::endl;
      }


    //QDPIO::cout << "d2h: start = " << f.start << "  count = " << f.count << "  size_T = " << f.size_T << "   \t";
    
    gpu_memcpy_d2h( host_ptr , dev_ptr , field_size );

    // --------------------
    // Main tuning loop

    int config = -1;
    double best_time;


    for( int threads_per_block = jit_config_get_threads_per_block_min() ;
	 threads_per_block <= jit_config_get_threads_per_block_max() ;
	 threads_per_block += jit_config_get_threads_per_block_step() )
      {
	kernel_geom_t geom = getGeom( th_count , threads_per_block );

	StopWatch w;
	int loops = jit_config_get_threads_per_block_loops();

	
	for( int i = 0 ; i < loops+5 ; ++i )
	  {
	    if (i==5)
	      gpu_record_start();

	    JitResult result = gpu_launch_kernel( function,
						  geom.Nblock_x,geom.Nblock_y,1,
						  geom.threads_per_block,1,1,
						  0, // shared mem
						  args );
	  
	    if (result != JitResult::JitSuccess) {
	      QDPIO::cerr << "Tuning: jit launch error, grid=(" << geom.Nblock_x << "," << geom.Nblock_y << "1), block=(" << threads_per_block << ",1,1)\n";
	      QDP_abort(1);
	    }

	  }
	
	//w.stop();
	gpu_record_stop();
	gpu_event_sync();
	float ms = gpu_get_time();

	//double ms = w.getTimeInMicroseconds();

	if (jit_config_get_tuning_verbose())
	  {
	    QDPIO::cout << "blocksize\t" << threads_per_block << "\ttime = \t" << ms << std::endl;
	  }
	
	if (config == -1 || best_time > ms)
	  {
	    best_time = ms;

	    config = threads_per_block;
	  }
      }

    if (jit_config_get_tuning_verbose())
      {
	QDPIO::cout << "best   \t" << config << "\ttime = \t" << best_time << std::endl;
      }
    
    function.set_threads_per_block( config );

    // DB tune
    db_tune_insert_info( function , config );

    // -------------------

    // Restore memory for 'dest'
    gpu_memcpy_h2d( dev_ptr , host_ptr , field_size );
    
    free( host_ptr );
  }
#endif  


#if defined(QDP_BACKEND_CUDA) || defined(QDP_BACKEND_ROCM)
  void jit_launch(JitFunction& function,int th_count,std::vector<QDPCache::ArgKey>& ids)
  {
    // For ROCm we add the __threads_per_group and 
    // __grid_size_x as parameters to the kernel.
#ifdef QDP_BACKEND_ROCM
    JIT_AMD_add_workgroup_sizes( ids );
#endif
    
    QDPCache::KernelArgs_t args( QDP_get_global_cache().get_kernel_args(ids) );

    // Check for no-op
    if ( th_count == 0 )
      return;

    // Increment the call counter
    function.inc_call_counter();

    
    if (  jit_config_get_tuning()  &&  function.get_threads_per_block() == -1  )
      {
	jit_tune( function , th_count , args );
      }
    
    int threads_per_block;
    
    if ( function.get_threads_per_block() == -1 )
      {
	threads_per_block = jit_config_get_threads_per_block();
      }
    else
      {
	threads_per_block = function.get_threads_per_block();
      }

    //QDPIO::cout << "jit_launch using block size = " << threads_per_block << std::endl;
    

    kernel_geom_t geom = getGeom( th_count , threads_per_block );

    JitResult result = gpu_launch_kernel( function,
					  geom.Nblock_x,geom.Nblock_y,1,
					  geom.threads_per_block,1,1,
					  0, // shared mem
					  args );

    if (result != JitResult::JitSuccess) {
      QDPIO::cerr << "jit launch error, grid=(" << geom.Nblock_x << "," << geom.Nblock_y << "1), block=(" << threads_per_block << ",1,1)\n";
      QDP_abort(1);
    }
  }
#elif defined(QDP_BACKEND_AVX)
  void jit_launch(JitFunction& function,int th_count,std::vector<QDPCache::ArgKey>& ids)
  {
    QDPCache::KernelArgs_t args( QDP_get_global_cache().get_kernel_args(ids) );

    if ( th_count == 0 )
      return;

    void (*FP)(int,void*) = (void (*)(int,void*))(intptr_t)function.get_function();

    // std::cout << args.size() << std::endl;
    // std::cout << args[0].i32 << std::endl;
    // std::cout << args[1].ptr << " " << ((size_t)args[1].ptr)%64 << std::endl;
    // std::cout << args[2].ptr << " " << ((size_t)args[2].ptr)%64 << std::endl;
    // std::cout << args[3].ptr << " " << ((size_t)args[3].ptr)%64 << std::endl;

    QDPIO::cout << "dispatch th_count = " << th_count << std::endl;
    
    for ( int i = 0 ; i < th_count ; i++ )
      FP( i , args.data() );
    
#if 0
    //#pragma omp parallel
#pragma omp for
    for ( int i = 0 ; i < th_count ; i++ )
      {
	FP( i , args.data() );
      }
#endif

    // Increment the call counter
    function.inc_call_counter();
  }
#else
#error "No LLVM backend specified."
#endif



  
  void jit_launch_explicit_geom( JitFunction& function , std::vector<QDPCache::ArgKey>& ids , const kernel_geom_t& geom , unsigned int shared )
  {
    // For ROCm we add the __threads_per_group and 
    // __grid_size_x as parameters to the kernel.
#ifdef QDP_BACKEND_ROCM
    JIT_AMD_add_workgroup_sizes( ids );
#endif

    QDPCache::KernelArgs_t args( QDP_get_global_cache().get_kernel_args(ids) );

    // Increment the call counter
    function.inc_call_counter();

    JitResult result = gpu_launch_kernel( function,
					  geom.Nblock_x,geom.Nblock_y,1,
					  geom.threads_per_block,1,1,
					  shared,
					  args );

    if (result != JitResult::JitSuccess) {
      QDPIO::cerr << "jit launch explicit geom error, grid=(" << geom.Nblock_x << "," << geom.Nblock_y << "1), block=(" << geom.threads_per_block << ",1,1)\n";
      QDP_abort(1);
    }
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
  
  

  JitForLoop::JitForLoop( llvm::Value* start , llvm::Value* end )
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
  llvm::Value * JitForLoop::index()
  {
    return r_i;
  }
  void JitForLoop::end()
  {
    llvm::Value * r_i_plus = llvm_add( r_i , llvm_create_value(1) );

    llvm_add_incoming( r_i , r_i_plus , llvm_get_insert_point() );
  
    llvm_branch( block_loop_cond );

    llvm_set_insert_point(block_loop_exit);
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



  
  

} //namespace
