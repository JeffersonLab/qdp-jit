// $Id: t_dslashm.cc,v 1.20 2007-02-24 01:00:29 bjoo Exp $
/*! \file
 *  \brief Test the Wilson-Dirac operator (dslash)
 */

#include "qdp.h"
#include "examples.h"
#include <iostream>
#include <cstdio>
#include <cmath>


#include <sys/time.h>

using namespace QDP;


int main(int argc, char **argv)
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  // Setup the layout
  const int foo[] = {16,16,16,16};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements

  bool range_all = false;
  bool verify_dslash = false;
  int iter = 1000;

  for (int i=1; i<argc; i++) 
    {
      if (strcmp((argv)[i], "-all")==0) 
	{
	  range_all = true;
	}
      if (strcmp((argv)[i], "-verify-dslash")==0)
	{
	  verify_dslash = true;
	}
      if (strcmp((argv)[i], "-iters")==0)
	{
	  if ((i + 1 >= argc) || (sscanf((argv)[++i], "%d", &iter) != 1) || (iter <= 0))
	    QDP_error_exit("-iters requires a positive integer");
	}
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
    }

  Layout::setLattSize(nrow);
  Layout::create();

  //! Test out propagators
  multi1d<LatticeColorMatrix> u(Nd);
  for(int m=0; m < u.size(); ++m)
    gaussian(u[m]);

  LatticeFermion psi, chi;
  random(psi);
  chi = zero;

  const int warmup_iters = 10;

  QDPIO::cout << "Untimed warm-up iterations per variant: " << warmup_iters << endl;
  QDPIO::cout << "Timed iterations per variant: " << iter << endl;

  if (range_all)
    {
      for (int isign=-1 ; isign <= +1 ; isign += 2 )
	{
	  QDPIO::cout << "Applying D" << endl;

	  for (int i=0; i < warmup_iters; i++)
	    dslash(chi, u, psi, isign, all );
    
	  StopWatch w;
	  w.start();
	  for(int i=0; i < iter; i++)
	    dslash(chi, u, psi, isign, all );
	  w.stop();

	  double gflops = 1392.0 * ((double)((((double)Layout::vol())) * ((double)iter))) * 1.0e-9 / w.getTimeInSeconds();

	  QDPIO::cout << "sign=" << isign << "  performance = " << gflops << " GFlops\n";
	}
    }
  else
    {
      for (int isign=-1 ; isign <= +1 ; isign += 2 )
	for (int cb=0 ; cb <= 1 ; cb++ )
	  {
	    QDPIO::cout << "Applying D" << endl;

	    for (int i=0; i < warmup_iters; i++)
	      dslash(chi, u, psi, isign, rb[cb] );
    
	    StopWatch w;
	    w.start();
	    for(int i=0; i < iter; i++)
	      dslash(chi, u, psi, isign, rb[cb] );
	    w.stop();

	    double gflops = 1392.0 * ((double)((((double)Layout::vol())/2.) * ((double)iter))) * 1.0e-9 / w.getTimeInSeconds();

	    QDPIO::cout << "cb=" << cb << "  sign=" << isign << "  performance = " << gflops << " GFlops\n";
      
	  }
    }

  if (verify_dslash)
    {
      const double tolerance = 1.0e-5;
      LatticeFermion chi2, diff;
      random(psi);
      for(int isign=-1; isign < 2; isign+=2)
	for(int cb=0; cb<2; cb++)
	  {
	    int otherCB = cb == 0 ? 1 : 0;
	    chi = zero;
	    chi2 = zero;
	    diff = zero;
	    dslash(chi, u, psi, isign, rb[cb]);
	    dslash2(chi2, u, psi, isign, cb);
	    diff[rb[cb]] = chi2 - chi;
	    double relative_error = toDouble(sqrt(norm2(diff, rb[cb]) /
					          norm2(psi, rb[otherCB])));
	    QDPIO::cout << "VERIFY_DSLASH isign=" << isign
			  << " cb=" << cb
			  << " relative_error=" << relative_error << endl;
	    if (!std::isfinite(relative_error) || relative_error > tolerance)
	      {
		QDPIO::cerr << "VERIFY_DSLASH FAILED tolerance=" << tolerance << endl;
		QDP_abort(2);
	      }
	  }
      QDPIO::cout << "VERIFY_DSLASH PASSED tolerance=" << tolerance << endl;
    }

#if 0
  chi = zero;

  {
    int isign = +1;
    int cb = 0;
    QDPIO::cout << "Applying D" << endl;
      
    dslash2(chi, u, psi, isign, cb);
    clock_t myt1=clock();
    for(int i=0; i < iter; i++)
      dslash2(chi, u, psi, isign, cb);
    clock_t myt2=clock();
      
    double mydt=(double)(myt2-myt1)/((double)(CLOCKS_PER_SEC));
    mydt=1.0e6*mydt/((double)(iter*(Layout::vol()/2)));
      
    QDPIO::cout << "cb = " << cb << " isign = " << isign << endl;
    QDPIO::cout << "The time per lattice point is "<< mydt << " micro sec" 
		<< " (" <<  (double)(1392.0f/mydt) << ") Mflops " << endl;
  }


#if 0
  XMLFileWriter xml("t_dslashm.xml");
  push(xml,"t_dslashm");
  write(xml,"Nd", Nd);
  write(xml,"Nc", Nc);
  write(xml,"Ns", Ns);
  write(xml,"nrow", nrow);
  write(xml,"psi", psi);
  write(xml,"chi", chi);
  pop(xml);
  xml.close();
#endif

  LatticeFermion chi2;
  random(psi);
  chi = zero;
  chi2 = zero;
  for(int isign=-1; isign < 2; isign+=2) { 
    for(int cb=0; cb<2; cb++) {
      int otherCB= cb == 0 ? 1 : 0;
      dslash(chi, u, psi, isign, cb);
      dslash2(chi2, u, psi, isign, cb);
      LatticeFermion diff;
      diff[rb[cb]]= chi2 - chi;
      QDPIO::cout << "isign="<<isign<<" cb=" << cb << " Diff = " << sqrt( norm2(diff,rb[cb]) / norm2(psi, rb[otherCB]))<< endl;
    }
  }
#endif

  // Time to bolt
  QDP_finalize();
}
