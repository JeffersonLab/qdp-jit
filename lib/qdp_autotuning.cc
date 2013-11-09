#include "qdp.h"
#include "omp.h"

namespace QDP {

  struct tune_t {
    tune_t(): cfg(0),best(0),best_time(0.0) {}
    tune_t(int cfg,int best,double best_time): cfg(cfg),best(best),best_time(best_time) {}
    int    cfg;
    int    best;
    double best_time;
    std::vector< std::pair<int,double> > tune;
  };

  std::map< void* , tune_t > mapTune;



  void jit_call_autotune( void* function , int site_count , const AddressLeaf& args)
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

      tune.tune.push_back( make_pair(tune.cfg,time) );

      if (time < tune.best_time || tune.best_time == 0.0) {
      	tune.best_time = time;
      	tune.best = tune.cfg;
      }


      // If time is much greater than our best time
      // we are in the rising part of the performance 
      // profile and stop searching any further
      tune.cfg = time > 1.33 * tune.best_time || tune.cfg == 1 ? -1 : tune.cfg >> 1;

      if (tune.cfg == -1) {
	for (auto& t : tune.tune )
	  QDPIO::cerr << t.first << "\t" << t.second << "\n";
	QDPIO::cerr << "time = " << time 
		    << ",  cfg = " << tune.cfg 
		    << ",  best = " << tune.best 
		    << ",  best_time = " << tune.best_time << "\n";
      }

    }

  }







}
