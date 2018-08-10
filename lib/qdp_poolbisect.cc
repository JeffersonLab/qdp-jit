/*! @file
 * @brief Bisection routine
 * 
 * Bisection routine to find maximum memory pool size
 */

#include "qdp.h"


namespace QDP {

  namespace {
    void* jit_param_null_dummy_ptr = NULL;
  }

  namespace {
    bool __poolbisect = false;
    size_t __poolbisectmax = 0;
  }

  bool   qdp_cache_get_pool_bisect() { return __poolbisect; }
  size_t qdp_cache_get_pool_bisect_max() { return __poolbisectmax; }
  
  void qdp_cache_set_pool_bisect(bool b) {
    std::cout << "Pool bisect run\n";
    __poolbisect = b;
  }
  
  void qdp_cache_set_pool_bisect_max(size_t val) {
    std::cout << "Pool bisect max. " << val << "\n";
    __poolbisectmax = val;
  }



  namespace 
  {
    template<class Allocator>
    std::vector<void*> get_backed_kernel_args( Allocator& pool_allocator )
    {
      assert( QDP_get_global_cache().get__vec_backed().size() > 0 );

      //QDPIO::cout << "get backed kernel args with " << __vec_backed.size() << " elements\n";

      const bool print_param = false;

      if (print_param)
	QDPIO::cout << "Jit function param: ";
    
      std::vector<void*> ret;

      for ( auto e : QDP_get_global_cache().get__vec_backed() )
	{
	  //printInfo(e);
	  //QDPIO::cout << "elem " << cnt++ << "\n";
	
	  if (e.Id >= 0)
	    {
	      if (e.flags & QDPCache::Flags::JitParam)
		{
		  if (print_param)
		    {
		      switch(e.param_type) {
		      case QDPCache::JitParamType::float_: QDPIO::cout << (float)e.param.float_ << ", "; break;
		      case QDPCache::JitParamType::double_: QDPIO::cout << (double)e.param.double_ << ", "; break;
		      case QDPCache::JitParamType::int_: QDPIO::cout << (int)e.param.int_ << ", "; break;
		      case QDPCache::JitParamType::int64_: QDPIO::cout << (int64_t)e.param.int64_ << ", "; break;
		      case QDPCache::JitParamType::bool_:
			if (e.param.bool_)
			  QDPIO::cout << "true, ";
			else
			  QDPIO::cout << "false, ";
			break;
		      default:
			QDPIO::cout << "(unkown jit param type)\n"; break;
			assert(0);
		      }
		    }
		  ret.push_back( &e.param );
		}
	      else
		{
		  // We need to copy from host memory
		  //QDPIO::cout << "allocate " << e.size << " bytes\n";
		  if ( !pool_allocator.allocate( &e.devPtr , e.size ) )
		    {
		      QDPIO::cout << "could not allocate memory\n";
		      QDP_error_exit("giving up");
		    }

		  // QDPIO::cout << "copy H2D " << e.size
		  // 	    << " bytes, from = " << (size_t)e.hstPtr
		  // 	    << " bytes, to = " << (size_t)e.devPtr
		  // 	    << "\n";

		  CudaMemcpyH2D( e.devPtr , e.hstPtr , e.size );

		  if (print_param)
		    {
		      //QDPIO::cout << (size_t)e.devPtr << ", ";
		    }

		  if (e.flags & QDPCache::Flags::Array)
		    {
		      // We store the elem access field in the parameter field
		      // This is safe since it's unused for an array.
		      if (e.param.int_ == -1)
			{
			  // Could be a whole array view
			  ret.push_back( &e.devPtr );
			}
		      else
			{
			  // .. or an element view
			  assert( e.karg_vec.size() > e.param.int_ );
			  e.karg_vec[ e.param.int_ ] = (void*)((size_t)e.devPtr + e.elem_size * e.param.int_ );
			  ret.push_back( &e.karg_vec[ e.param.int_ ] );
			}
		    }
		  else
		    {
		      ret.push_back( &e.devPtr );
		    }
		}
	    }
	  else
	    {
	      if (print_param)
		{
		  QDPIO::cout << "NULL(), ";
		}

	      ret.push_back( &jit_param_null_dummy_ptr );
	
	    }
	}
    
      if (print_param)
	QDPIO::cout << "\n";
    
      return ret;
    }
  }  // ns


  void qdp_pool_bisect()
  {
    qdp_cache_set_pool_bisect(false);

    if (! CudaCtxSynchronize() )
      {
	QDP_error_exit("device not okay");
      }
    else
      {
	QDPIO::cout << "device okay!\n";
      }

    size_t now_free;
    size_t now_total;

    CudaMemGetInfo( &now_free , &now_total );

    QDPIO::cout << "available CUDA memory at program startup: " << CudaGetInitialFreeMemory() << "\n";
    QDPIO::cout << "available CUDA memory now:                " << now_free << "\n";
    //assert( CudaGetInitialFreeMemory() == now_free );

    size_t mem_min = now_free < CudaGetInitialFreeMemory() ? now_free : CudaGetInitialFreeMemory();
    
    size_t max  = mem_min - 2 * QDP_ALIGNMENT_SIZE;
    size_t min  = max/2;
    size_t step = max - min;
    size_t cur  = max;

    if ( ( qdp_cache_get_pool_bisect_max() > 0 ) &&
	 ( qdp_cache_get_pool_bisect_max() + 2 * QDP_ALIGNMENT_SIZE ) <= now_free )
      {
	max = qdp_cache_get_pool_bisect_max();
	min = max/2;
	step = max - min;
	cur  = max;
      }
    

    CUfunction f = get_backed_kernel_ptr();
    auto geom = get_backed_kernel_geom();

    if (!geom.size())
      {
	QDPIO::cout << "No jit kernel used any local memory. As a result a bisection test of the memory pool would not be meaningful.\n";
	QDPIO::cout << "Suggested poolsize: -poolsize " << mem_min/1024/1024<< "m\n";
	return;
      }
	
    assert( geom.size() == 7 );
    unsigned  gridDimX = geom[0];
    unsigned  gridDimY = geom[1];
    unsigned  gridDimZ = geom[2];
    unsigned  blockDimX = geom[3];
    unsigned  blockDimY = geom[4];
    unsigned  blockDimZ = geom[5];
    unsigned  sharedMemBytes = geom[6];
    
    QDPIO::cout << "Starting pool bisecting with intersection interval: max = " << max << ", min = " << min << "\n";

    size_t last_success = 0;
    
    while ( step > 1024*1024   &&   cur <= max )
      {
	QDPIO::cout << "current pool size = " << cur << ", step size = " << step << "\n";

	QDPPoolAllocator<QDPCUDAAllocator> allocator;
	allocator.setPoolSize( cur );

	bool success = true;
	  
	if (allocator.allocateInternalBuffer())
	  {
	    std::vector<void*> args( get_backed_kernel_args( allocator ) );

	    success = CudaLaunchKernelNoSync( f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, 0, &args[0], NULL ) == CUDA_SUCCESS;

	    success = success && CudaCtxSynchronize();

	  }
	else
	  {
	    success = false;
	  }
	
	if (success)
	  {
	    last_success = cur;
	    
	    cur = cur + step;
	    step >>= 1;
	  }
	else
	  {
	    cur = cur - step;
	    step >>= 1;
	  }
	
	//
	//CudaDeviceSynchronize(); // checks the state
      }

    if (last_success > 0)
      {
	QDPIO::cout << "Memory pool bisection resulted in " << last_success << " bytes (-poolsize " << last_success/1024/1024 << "m)\n";
      }
    else
      {
	QDPIO::cout << "Memory pool bisection unsuccessful\n";
      }
    
  }

  
} // QDP
