#include "qdp.h"

namespace QDP {

  struct tune_t {
    tune_t(): cfg(0),best(0),best_time(0.0) {}
    tune_t(int cfg,int best,double best_time): cfg(cfg),best(best),best_time(best_time) {}
    int    cfg;
    int    best;
    double best_time;
  };

  std::map< CUfunction , tune_t > mapTune;


  void LaunchPrintArgs( std::vector<void*>& args )
  {
    QDP_info("Number of kernel arguments: %d",(int)args.size());
    int i=0;
    QDP_info("            bool          int     pointer");
    for (void *addr : args)	
      QDP_info("%2d: %12d %12d %p",i++,*(bool*)addr,*(int*)addr,*(void**)addr);
    QDP_info("Device pool info:");
    //CUDADevicePoolAllocator::Instance().printPoolInfo();
    //QDP_get_global_cache().get_allocator().printPoolInfo();
  }


  void jit_launch(CUfunction function,int th_count,std::vector<QDPCache::ArgKey>& ids)
  {
    //QDP_get_global_cache().printLockSet();
    //QDP_get_global_cache().newLockSet();

    // for (auto i : ids)
    //   QDPIO::cout << i << ", ";
    // QDPIO::cout << "\n";
    
     std::vector<void*> args( QDP_get_global_cache().get_kernel_args(ids) );

     
    // Check for thread count equals zero
    // This can happen, when inner count is zero
    if ( th_count == 0 )
      return;

    if (mapTune.count(function) == 0) {
      mapTune[function] = tune_t( DeviceParams::Instance().getMaxBlockX() , 0 , 0.0 );
    }


    tune_t& tune = mapTune[function];


    if (tune.cfg == -1) {
      kernel_geom_t now = getGeom( th_count , tune.best );

      //QDP_info("CUDA launch (settled): grid=(%u,%u,%u), block=(%d,%u,%u) ",now.Nblock_x,now.Nblock_y,1,    tune.best,1,1 );
	
      CUresult result = CudaLaunchKernelNoSync(function,   now.Nblock_x,now.Nblock_y,1,    tune.best,1,1,    0, 0, &args[0] , 0);

      if (result != CUDA_SUCCESS) {
	CudaCheckResult(result);
	LaunchPrintArgs(args);
	QDPIO::cout << getPTXfromCUFunc(function);
	QDP_error_exit("CUDA launch error (after successful tuning): grid=(%u,%u,%u), block=(%d,%u,%u) ",
		       now.Nblock_x,now.Nblock_y,1,    tune.best,1,1 );
      }

      //QDP_get_global_cache().releasePrevLockSet();
      //QDP_get_global_cache().newLockSet();

      result = cuCtxSynchronize();
      if (result != CUDA_SUCCESS) {
	CudaCheckResult(result);
	LaunchPrintArgs(args);
	QDPIO::cout << getPTXfromCUFunc(function) << "\n";
	QDP_error_exit("CUDA launch error (after successful autotune, on sync): grid=(%u,%u,%u), block=(%d,%u,%u) ",
		       now.Nblock_x,now.Nblock_y,1,    tune.best,1,1 );
      }
    } else {

      CUresult result = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
      CUresult result_sync;
      double time;

      while (result != CUDA_SUCCESS && tune.cfg > 0) {
	kernel_geom_t now = getGeom( th_count , tune.cfg );
	StopWatch w;

	w.start();

	//QDP_info("CUDA launch (trying): grid=(%u,%u,%u), block=(%d,%u,%u) ",now.Nblock_x,now.Nblock_y,1,    tune.cfg,1,1 );

	result = CudaLaunchKernelNoSync(function,   now.Nblock_x,now.Nblock_y,1,    tune.cfg,1,1,    0, 0, &args[0] , 0);

	if (result != CUDA_SUCCESS && result != CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES) {
	  CudaCheckResult(result);
	  LaunchPrintArgs(args);
	  QDPIO::cout << getPTXfromCUFunc(function);
	  QDP_error_exit("CUDA launch error: grid=(%u,%u,%u), block=(%d,%u,%u) ",
			 now.Nblock_x,now.Nblock_y,1,    tune.cfg,1,1 );
  	}

	if (result == CUDA_SUCCESS) {

	  //QDP_get_global_cache().releasePrevLockSet();
	  //QDP_get_global_cache().newLockSet();

	  result_sync = cuCtxSynchronize();
	  if (result_sync != CUDA_SUCCESS) {
	    CudaCheckResult(result_sync);
	    LaunchPrintArgs(args);
	    QDPIO::cout << getPTXfromCUFunc(function);
	    QDP_error_exit("CUDA launch error (during autotune, on sync): grid=(%u,%u,%u), block=(%d,%u,%u) ",
			   now.Nblock_x,now.Nblock_y,1,    tune.cfg,1,1 );
	  }
	}

	w.stop();

	time = w.getTimeInMicroseconds();
	if (result != CUDA_SUCCESS)
	  tune.cfg >>= 1;
      }

      if (tune.cfg == 0) {
	CudaCheckResult(result);
	QDP_error_exit("Kernel launch failed even for block size 1. Giving up.");
      }

      if (time < tune.best_time || tune.best_time == 0.0) {
	tune.best_time = time;
	tune.best = tune.cfg;
      }

      // If time is much greater than our best time
      // we are in the rising part of the performance 
      // profile and stop searching any further
      tune.cfg = time > 1.33 * tune.best_time || tune.cfg == 1 ? -1 : tune.cfg >> 1;

      //QDP_info("time = %f,  cfg = %d,  best = %d,  best_time = %f ", time,tune.cfg,tune.best,tune.best_time );
    }
  }




}
