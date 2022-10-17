// $Id: t_dslashm.cc,v 1.20 2007-02-24 01:00:29 bjoo Exp $
/*! \file
 *  \brief Test the Wilson-Dirac operator (dslash)
 */

#include "qdp.h"
#include "sftmom.h"
//#include "examples.h"
//#include <iostream>
//#include <cstdio>
//#include <sys/time.h>

using namespace QDP;






void mesons2(const LatticePropagator& quark_prop_1,
	     const LatticePropagator& quark_prop_2,
	     const SftMom& phases,
	     int t0 )
{
  StopWatch wall;
  wall.start();

  // Length of lattice in decay direction
  int length = phases.numSubsets();

  // Construct the anti-quark propagator from quark_prop_2
  int G5 = Ns*Ns-1;
  LatticePropagator anti_quark_prop =  Gamma(G5) * quark_prop_2 * Gamma(G5);

  // This variant uses the function SftMom::sft() to do all the work
  // computing the Fourier transform of the meson correlation function
  // inside the class SftMom where all the of the Fourier phases and
  // momenta are stored.  It's primary disadvantage is that it
  // requires more memory because it does all of the Fourier transforms
  // at the same time.


  
  for (int gamma_value=0; gamma_value < 1; ++gamma_value)
  {
    QDPIO::cout << "gamma value = " << gamma_value << std::endl;

    LatticeComplex corr_fn;

    corr_fn = trace(adj(anti_quark_prop) * (Gamma(gamma_value) * quark_prop_1 * Gamma(gamma_value))); // JIT

    StopWatch w;
    w.start();

#if 1
    corr_fn = trace(adj(anti_quark_prop) * (Gamma(gamma_value) *
					    quark_prop_1 * Gamma(gamma_value)));
#else
    LatticePropagator tmp = Gamma(gamma_value) * quark_prop_1;
    LatticePropagator tmp2 = tmp * Gamma(gamma_value);
    corr_fn = trace(adj(anti_quark_prop) * tmp2);
#endif
    
    //LatticePropagator tmp = Gamma(2) * anti_quark_prop;
    
    w.stop();
    double t_corr = w.getTimeInSeconds();
    QDPIO::cout << "time trace = " << t_corr << std::endl;

    multi2d<DComplex> hsum;
    hsum = phases.sft(corr_fn); // JIT

    w.reset();
    w.start();
    
    hsum = phases.sft(corr_fn);

    w.stop();
    double t_phases = w.getTimeInSeconds();
    QDPIO::cout << "time phases = " << t_phases << std::endl;

#if 0
    for (int sink_mom_num=0; sink_mom_num < phases.numMom(); ++sink_mom_num) 
    {
      multi1d<DComplex> mesprop(length);
      for (int t=0; t < length; ++t) 
      {
        int t_eff = (t - t0 + length) % length;
	mesprop[t_eff] = hsum[sink_mom_num][t];
      }
    } // end for(sink_mom_num)
#endif

    
  } // end for(gamma_value)

  wall.stop();
  double t_wall = wall.getTimeInSeconds();
  QDPIO::cout << "time overall = " << t_wall << std::endl;

}





int main(int argc, char **argv)
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  // Setup the layout
  const int foo[] = {16,16,16,16};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements

  int mom2_max = 1;
  int j_decay = Nd-1;
  
  for (int i=1; i<argc; i++) 
    {
      if (strcmp((argv)[i], "-lat")==0) 
	{
	  int lat;
	  sscanf((argv)[++i], "%d", &lat);
	  nrow[0]=nrow[1]=nrow[2]=nrow[3]=lat;
	}
      if (strcmp((argv)[i], "-latz")==0) 
	{
	  int lat;
	  sscanf((argv)[++i], "%d", &lat);
	  nrow[3]=lat;
	}
      if (strcmp((argv)[i], "-mom2_max")==0) 
	{
	  int lat;
	  sscanf((argv)[++i], "%d", &lat);
	  mom2_max=lat;
	}

    }

  Layout::setLattSize(nrow);
  Layout::create();

  //! Test out propagators
  multi1d<LatticeColorMatrix> u(Nd);
  for(int m=0; m < u.size(); ++m)
    gaussian(u[m]);

  SftMom sf(mom2_max, false, j_decay);

  QDPIO::cout << "number of subsets: " << sf.numSubsets() << std::endl;
  QDPIO::cout << "number of momenta: " << sf.numMom() << std::endl;

  LatticePropagator quark_prop_1;
  LatticePropagator quark_prop_2;

  QDPIO::cout << "Gaussian noise prop 1" << std::endl;
  gaussian(quark_prop_1);
  QDPIO::cout << "Gaussian noise prop 2" << std::endl;
  gaussian(quark_prop_2);

  mesons2( quark_prop_1 , quark_prop_2 , sf , 0 );
  
  // Time to bolt
  QDP_finalize();
}
