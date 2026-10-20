#include "qdp.h"

using namespace QDP;





void compound_assign()
{
  multi1d<int> blocking{2,2,2,2};

  Layout::push(Layout::current().coarsen(blocking));

  LatticeInteger coarse_x = Layout::latticeCoordinate(0);
  LatticeReal coarse;
  coarse = Real(10.0) + coarse_x;

  Layout::pop();

  LatticeInteger fine_x =
      Layout::latticeCoordinate(0);
  LatticeReal fine;
  fine = Real(10.0) + fine_x;

  fine += coarse;
}




int main(int argc, char *argv[])
{  
  QDP_initialize(&argc, &argv);

  const int L = 8;
  
  multi1d<int> nrow(Nd);
  nrow = L;

  Layout::setLattSize( nrow );
  Layout::create();

  compound_assign();
  
  QDP_finalize();
}
