#include "qdp.h"
#include <map>

using namespace QDP;
using namespace std;

//   N   ,     L
map< int , map<int,double> > timing;



void print_timings_csv()
{
  QDPIO::cout << "\n\nN,L,time\n";

  for (const auto& [N, inner] : timing)
    {
      for (const auto& [L, t] : inner)
        {
	  QDPIO::cout << N << "," << L << "," << t << "\n";
        }
    }
  QDPIO::cout << "\n\n";
}


//void matvec(int Lmin , int Lmax, int Lstep , int Nmin, int Nmax, int Nstep)
void matvec(int L , int N, int iter)
{
  
  multi1d<int> nrow(Nd);
  nrow = L;

  Layout::push(Layout::find_or_create(nrow));
  //Layout::create();

  ShapeScope qv_scope(Shape(Shape::current().ns(), N));

  LatticeColorMatrix u;
  LatticeFermion psi,chi;

  // do once for jit
  psi = u * chi;

  StopWatch w;
  w.start();

  for (int i = 0 ; i < iter ; ++i )
    {
      psi = u * chi;      
    }

  w.stop();

  double flops_per_site = N * (N * 6 + (N-1) * 2);
  
  double gflops = flops_per_site * ((double)((((double)Layout::vol())) * ((double)iter))) * 1.0e-9 / w.getTimeInSeconds();

  QDPIO::cout << gflops << std::endl;

  timing[N][L] = gflops;
  
  qv_scope.close();
  Layout::pop();
  
}


int main(int argc, char **argv)
{
  QDP_initialize(&argc, &argv);


  // multi1d<int> nrow(Nd);
  // nrow = foo;  // Use only Nd elements

  bool range_all = false;

  int Lmin = 4;
  int Lmax = 12;
  int Lstp = 2;

  int Nmin = 2;
  int Nmax = 8;
  int Nstp = 1;
  
  for (int i=1; i<argc; i++) 
    {
      if (strcmp((argv)[i], "-Lmin")==0) 
	sscanf((argv)[++i], "%d", &Lmin);
      if (strcmp((argv)[i], "-Lmax")==0) 
	sscanf((argv)[++i], "%d", &Lmax);
      if (strcmp((argv)[i], "-Lstp")==0) 
	sscanf((argv)[++i], "%d", &Lstp);

      if (strcmp((argv)[i], "-Nmin")==0) 
	sscanf((argv)[++i], "%d", &Nmin);
      if (strcmp((argv)[i], "-Nmax")==0) 
	sscanf((argv)[++i], "%d", &Nmax);
      if (strcmp((argv)[i], "-Nstp")==0) 
	sscanf((argv)[++i], "%d", &Nstp);
    }


  for (int L=Lmin ; L <= Lmax ; L += Lstp )
    {
      for (int N=Nmin ; N <= Nmax ; N += Nstp )
	{
	  matvec(L,N,1000);
	}
    }

  print_timings_csv();
  
  QDP_finalize();
}
