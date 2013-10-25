#include "qdp.h"
#include "omp.h"

namespace QDP {

  struct tune_t {
    tune_t(): cfg(0),best(0),best_time(0.0) {}
    tune_t(int cfg,int best,double best_time): cfg(cfg),best(best),best_time(best_time) {}
    int    cfg;
    int    best;
    double best_time;
  };

  std::map< void* , tune_t > mapTune;




  void jit_dispatch( void* function , int site_count, int th_count, const AddressLeaf& args)
  {
    void (*FP)(void*) = (void (*)(void*))(intptr_t)function;

    int threads_num;
    int myId;
    int lo = 0;
    int hi = site_count;
    AddressLeaf my_args;

    omp_set_num_threads(th_count);
   
#pragma omp parallel shared(site_count, threads_num, args) private(myId, lo, hi, my_args) default(shared)
    {
      threads_num = omp_get_num_threads();
      myId = omp_get_thread_num();
      lo = site_count*myId/threads_num;
      hi = site_count*(myId+1)/threads_num;

      my_args = args;
      my_args.addr[0] = lo;
      my_args.addr[1] = hi;

      FP( my_args.addr.data() );
#pragma omp barrier
    }
  }


  void jit_call(void* function,int site_count,const AddressLeaf& args)
  {
    // Check for thread count equals zero
    // This can happen, when inner count is zero
    if ( site_count == 0 )
      return;

    if (mapTune.count(function) == 0) {
      mapTune[function] = tune_t( 32 , 0 , 0.0 );
    }


    tune_t& tune = mapTune[function];

    if (tune.cfg == -1) {
      jit_dispatch( function , site_count, tune.best, args);
    } else {

      double time;
      StopWatch w;

      w.start();
      jit_dispatch( function , site_count, tune.cfg, args);
      w.stop();

      time = w.getTimeInMicroseconds();

      if (time < tune.best_time || tune.best_time == 0.0) {
      	tune.best_time = time;
      	tune.best = tune.cfg;
      }

      QDPIO::cerr << "time = " << time 
		  << ",  cfg = " << tune.cfg 
		  << ",  best = " << tune.best 
		  << ",  best_time = " << tune.best_time << "\n";

      // If time is much greater than our best time
      // we are in the rising part of the performance 
      // profile and stop searching any further
      tune.cfg = time > 1.33 * tune.best_time || tune.cfg == 1 ? -1 : tune.cfg >> 1;

    }

  }







}
