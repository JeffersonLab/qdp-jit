#include "qdp.h"

namespace QDP {

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
