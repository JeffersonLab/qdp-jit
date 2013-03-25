#include "qdp.h"

#define SKIP_DOUBLE

namespace QDP {

  // Fully specialised strings -- can live in the .cc file - defined only once
  
  // This is for the QIO type strings
  template<>
  char* QIOStringTraits< multi1d<LatticeColorMatrixF3> >::tname = (char *)"QDP_F3_ColorMatrix";

#ifndef SKIP_DOUBLE
  template<>
  char* QIOStringTraits< multi1d<LatticeColorMatrixD3> >::tname = (char *)"QDP_D3_ColorMatrix";
#endif

  template<>
  char* QIOStringTraits< multi1d<LatticeDiracFermionF3> >::tname = (char *)"USQCD_F3_DiracFermion";

#ifndef SKIP_DOUBLE
  template<>
  char* QIOStringTraits< multi1d<LatticeDiracFermionD3> >::tname = (char *)"USQCD_D3_DiracFermion";
#endif

  // This is for the QIO precision strings
  template<>
  char* QIOStringTraits<float>::tprec = (char *)"F";
  
  template<>
  char* QIOStringTraits<double>::tprec = (char *)"D";
  
  template<>
  char* QIOStringTraits<int>::tprec = (char *)"I";


}
