/*! @file
 * @brief Bisection routine
 * 
 * Bisection routine to find maximum memory pool size
 */

#include "qdp.h"


namespace QDP {


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
    
    QDPIO::cout << "Intersection interval: max = " << max << ", min = " << min << "\n";


    CUfunction f = get_backed_kernel_ptr();
    auto geom = get_backed_kernel_geom();
	
    assert( geom.size() == 7 );
    unsigned  gridDimX = geom[0];
    unsigned  gridDimY = geom[1];
    unsigned  gridDimZ = geom[2];
    unsigned  blockDimX = geom[3];
    unsigned  blockDimY = geom[4];
    unsigned  blockDimZ = geom[5];
    unsigned  sharedMemBytes = geom[6];
    
    QDPIO::cout << "Starting pool bisecting\n";
    
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
	    //QDPIO::cout << "success\n";
	    
	    if (cur + step > max)
	      break;
	    
	    cur = cur + step;
	    step >>= 1;
	  }
	else
	  {
	    //QDPIO::cout << "failed\n";
	    cur = cur - step;
	    step >>= 1;
	  }
	
	//
	//CudaDeviceSynchronize(); // checks the state
      }

    QDPIO::cout << "Memory pool bisection resulted in " << cur << " bytes (-poolsize " << cur/1024/1024 << "m)\n";
    
  }

  
} // QDP
