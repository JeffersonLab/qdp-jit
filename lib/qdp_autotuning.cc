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
    QDP_get_global_cache().get_allocator().printPoolInfo();
  }


  void jit_launch(CUfunction function,int th_count,std::vector<void*>& args)
  {
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
	
      CUresult result = cuLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    tune.best,1,1,    0, 0, &args[0] , 0);

      if (result != CUDA_SUCCESS) {
	CudaCheckResult(result);
	LaunchPrintArgs(args);
	QDPIO::cout << getPTXfromCUFunc(function);
	QDP_error_exit("CUDA launch error: grid=(%u,%u,%u), block=(%d,%u,%u) ",
		       now.Nblock_x,now.Nblock_y,1,    tune.cfg,1,1 );
      }

      QDP_get_global_cache().releasePrevLockSet();
      QDP_get_global_cache().beginNewLockSet();

      result = cuCtxSynchronize();
      if (result != CUDA_SUCCESS) {
	CudaCheckResult(result);
	LaunchPrintArgs(args);
	QDPIO::cout << getPTXfromCUFunc(function);
	QDP_error_exit("CUDA launch error (on sync): grid=(%u,%u,%u), block=(%d,%u,%u) ",
		       now.Nblock_x,now.Nblock_y,1,    tune.cfg,1,1 );
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

	result = cuLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    tune.cfg,1,1,    0, 0, &args[0] , 0);

	if (result != CUDA_SUCCESS && result != CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES) {
	  CudaCheckResult(result);
	  LaunchPrintArgs(args);
	  QDPIO::cout << getPTXfromCUFunc(function);
	  QDP_error_exit("CUDA launch error: grid=(%u,%u,%u), block=(%d,%u,%u) ",
			 now.Nblock_x,now.Nblock_y,1,    tune.cfg,1,1 );
  	}

	if (result == CUDA_SUCCESS) {

	  QDP_get_global_cache().releasePrevLockSet();
	  QDP_get_global_cache().beginNewLockSet();

	  result_sync = cuCtxSynchronize();
	  if (result_sync != CUDA_SUCCESS) {
	    CudaCheckResult(result_sync);
	    LaunchPrintArgs(args);
	    QDPIO::cout << getPTXfromCUFunc(function);
	    QDP_error_exit("CUDA launch error (on sync): grid=(%u,%u,%u), block=(%d,%u,%u) ",
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




  int jit_autotuning(CUfunction function,int lo,int hi,void ** param)
  {
    // Check for thread count equals zero
    // This can happen, when inner count is zero
    if ( hi-lo == 0 )
      return 0;

    // Auto tuning

    double best_time;
    int best_cfg=-1;
    bool first=true;
    for ( int cfg = 1 ; cfg <= DeviceParams::Instance().getMaxBlockX(); cfg *= 2 ) {
      kernel_geom_t now = getGeom( hi-lo , cfg );

      StopWatch w;
      CUresult result = CUDA_SUCCESS;

      for (int i=0 ; i < 10 && result == CUDA_SUCCESS; i++) {
	if (i==1) w.start();
	result = cuLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    cfg,1,1,    0, 0, param , 0);
	CudaDeviceSynchronize();
      }

      if (result == CUDA_SUCCESS) {
	w.stop();
	double time = w.getTimeInMicroseconds();
	QDP_info_primary("launched threads per block = %d grid = (%d,%d) (time=%f micro secs)",cfg,now.Nblock_x,now.Nblock_y,time);
	if (first) {
	  best_time = time;
	  best_cfg = cfg;
	  first = false;
	} else {
	  if (time < best_time) {
	    best_time = time;
	    best_cfg = cfg;
	  }
	}
      } else {
	QDP_info_primary("tried threads per block = %d, failed, code = %d ",cfg,result);
      }
    }

    if (best_cfg < 0)
      QDP_error_exit("Auto-tuning failed!");

    QDP_info_primary("Threads per block favored = %d  (time=%f micro secs)",best_cfg,best_time);
    return best_cfg;
  }




}
