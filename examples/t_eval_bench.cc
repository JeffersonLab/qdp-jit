
#include "qdp.h"

#include <functional>
#include <string>

using namespace QDP;


template<class T>
double bytes_of(const OLattice<T>& a)
{
  double site = WordSize<T>::value(a.shape()) * sizeof(typename WordType<T>::Type_t);
  return site * a.layout().sitesOnNode();
}


void bench(const std::string& name, int iter, double bytes, const std::function<void()>& f)
{
  f();

  StopWatch w;
  w.start();
  for (int i = 0; i < iter; ++i)
    f();
  w.stop();

  double us_per_call = w.getTimeInMicroseconds() / iter;
  double gbs = bytes / (us_per_call * 1.0e-6) * 1.0e-9;

  QDPIO::cout << name << ":  " << us_per_call << " us/call,  "
	      << gbs << " GB/s (effective)" << std::endl;
}


int main(int argc, char **argv)
{
  QDP_initialize(&argc, &argv);

  multi1d<int> nrow(Nd);
  nrow = 16;
  int iter = 1000;

  for (int i = 1; i < argc; i++)
    {
      if (strcmp(argv[i], "-lat") == 0)
	{
	  int lat;
	  sscanf(argv[++i], "%d", &lat);
	  nrow = lat;
	}
      else if (strcmp(argv[i], "-nrow") == 0)
	for (int mu = 0; mu < Nd; ++mu)
	  sscanf(argv[++i], "%d", &nrow[mu]);
      else if (strcmp(argv[i], "-iter") == 0)
	sscanf(argv[++i], "%d", &iter);
    }

  Layout::setLattSize(nrow);
  Layout::create();

  QDPIO::cout << "lattice =";
  for (int mu = 0; mu < Nd; ++mu)
    QDPIO::cout << " " << nrow[mu];
  QDPIO::cout << ",  sites on node = " << Layout::sitesOnNode()
	      << ",  iterations per measurement = " << iter << std::endl;

  Real a(2.0);
  Real b(3.0);

  LatticeFermion x, y, z, psi, chi;
  LatticeColorMatrix u;
  random(x);
  random(y);
  random(psi);
  gaussian(u);

  const double F = bytes_of(x);

  bench("y = a*x + y        ", iter, 3 * F, [&]() { y = a*x + y; });

  bench("z = a*x + b*y      ", iter, 3 * F, [&]() { z = a*x + b*y; });

  bench("chi = u * psi      ", iter, bytes_of(u) + 2 * F, [&]() { chi = u * psi; });

  multi1d<int> blocking(Nd);
  blocking = 2;

  Layout::push(Layout::current().coarsen(blocking));
  LatticeFermion coarse;
  random(coarse);
  Layout::pop();

  LatticeFermion fine;
  random(fine);

  bench("fine += coarse (2^4)", iter, 2 * bytes_of(fine) + bytes_of(coarse), [&]() { fine += coarse; });

  QDP_finalize();
}
