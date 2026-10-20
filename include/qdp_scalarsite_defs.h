#ifndef QDP_SCALARSITE_DEFS_H
#define QDP_SCALARSITE_DEFS_H

namespace QDP {

#include <qdp_config.h>
#include "qdp_precision.h"

  typedef GammaType Gamma;
  typedef GammaTypeDP GammaDP;
 
  typedef OLattice< PScalar< PColorVector< RComplex<Word<REAL> > > > > LatticeColorVector;
  typedef OLattice< PScalar< PColorMatrix< RComplex<Word<REAL> > > > > LatticeColorMatrix;
  typedef OLattice< PSpinVector< PScalar< RComplex<Word<REAL> > > > > LatticeSpinVector;
  typedef OLattice< PSpinMatrix< PScalar< RComplex<Word<REAL> > > > > LatticeSpinMatrix;
  typedef OLattice< PSpinVector< PColorVector< RComplex<Word<REAL> > > > > LatticeFermion;
  typedef OLattice< PSpinMatrix< PColorMatrix< RComplex<Word<REAL> > > > > LatticePropagator;
  typedef OLattice< PScalar< PScalar< RComplex<Word<REAL> > > > > LatticeComplex;
  typedef OLattice< PScalar< PScalar< RScalar<Word<INTEGER32> > > > > LatticeInteger;
  typedef OLattice< PScalar< PScalar< RScalar<Word<REAL> > > > > LatticeReal;
  typedef OLattice< PScalar< PScalar< RScalar<Word<REAL64> > > > > LatticeDouble;
  typedef OLattice< PScalar< PScalar< RScalar<Word<LOGICAL> > > > > LatticeBoolean;
  typedef OScalar< PScalar< PColorVector< RComplex<Word<REAL> > > > > ColorVector;
  typedef OScalar< PScalar< PColorMatrix< RComplex<Word<REAL> > > > > ColorMatrix;
  typedef OScalar< PSpinVector< PScalar< RComplex<Word<REAL> > > > > SpinVector;
  typedef OScalar< PSpinMatrix< PScalar< RComplex<Word<REAL> > > > > SpinMatrix;
  typedef OScalar< PSpinVector< PColorVector< RComplex<Word<REAL> > > > > Fermion;
  typedef OScalar< PSpinMatrix< PColorMatrix< RComplex<Word<REAL> > > > > Propagator;
  typedef OScalar< PScalar< PScalar< RComplex<Word<REAL> > > > > Complex;
  typedef OScalar< PScalar< PSeed< RScalar<Word<INTEGER32> > > > > Seed;
  typedef OScalar< PScalar< PScalar< RScalar<Word<INTEGER32> > > > > Integer;
  typedef OScalar< PScalar< PScalar< RScalar<Word<REAL> > > > > Real;
  typedef OScalar< PScalar< PScalar< RScalar<Word<REAL64> > > > > Double;
  typedef OScalar< PScalar< PScalar< RScalar<Word<LOGICAL> > > > > Boolean;
  typedef OScalar< PScalar< PColorVector< RComplex<Word<REAL64> > > > > DColorVector;
  typedef OScalar< PScalar< PColorMatrix< RComplex<Word<REAL64> > > > > DColorMatrix;
  typedef OScalar< PSpinVector< PColorVector< RComplex<Word<REAL64> > > > > DFermion;
  typedef OScalar< PSpinMatrix< PColorMatrix< RComplex<Word<REAL64> > > > > DPropagator;
  typedef OScalar< PScalar< PScalar< RComplex<Word<REAL64> > > > > DComplex;
  typedef OLattice< PSpinVector< PColorVector< RComplex<Word<REAL32 > > > > > LatticeFermionF;
  typedef OLattice< PSpinMatrix< PColorMatrix< RComplex<Word<REAL32 > > > > > LatticePropagatorF;
  typedef OLattice< PScalar< PColorMatrix< RComplex<Word<REAL32 > > > > > LatticeColorMatrixF;
  typedef OLattice< PScalar< PColorVector< RComplex<Word<REAL32 > > > > > LatticeColorVectorF;
  typedef OLattice< PScalar< PScalar< RComplex<Word<REAL32 > > > > > LatticeComplexF;
  typedef OLattice< PScalar< PScalar< RScalar<Word<REAL32 > > > > > LatticeRealF;
  typedef OScalar< PScalar< PColorMatrix< RComplex<Word<REAL32 > > > > > ColorMatrixF;
  typedef OScalar< PScalar< PScalar< RComplex<Word<REAL32 > > > > > ComplexF;
  typedef OScalar< PScalar< PScalar< RScalar<Word<REAL32 > > > > > RealF;
  typedef OLattice< PSpinVector< PColorVector< RComplex<Word<REAL64> > > > > LatticeFermionD;
  typedef OLattice< PSpinMatrix< PColorMatrix< RComplex<Word<REAL64> > > > > LatticePropagatorD;
  typedef OLattice< PScalar< PColorMatrix< RComplex<Word<REAL64> > > > > LatticeColorMatrixD;
  typedef OLattice< PScalar< PColorVector< RComplex<Word<REAL64> > > > > LatticeColorVectorD;
  typedef OLattice< PScalar< PScalar< RComplex<Word<REAL64> > > > > LatticeComplexD;
  typedef OLattice< PScalar< PScalar< RScalar<Word<REAL64> > > > > LatticeRealD;
  typedef OScalar< PSpinMatrix< PScalar< RComplex<Word<REAL64> > > > > SpinMatrixD;
  typedef OScalar< PScalar< PColorMatrix< RComplex<Word<REAL64> > > > > ColorMatrixD;
  typedef OScalar< PScalar< PScalar< RComplex<Word<REAL64> > > > > ComplexD;
  typedef OScalar< PScalar< PScalar< RScalar<Word<REAL64> > > > > RealD;
  typedef OSubLattice< PScalar< PColorVector< RComplex<Word<REAL32 > > > > > SubLatticeColorVectorF;
  typedef OLattice< PSpinMatrix< PColorVector< RComplex<Word<REAL> > > > > LatticeColorVectorSpinMatrix;
  typedef OLattice< PSpinMatrix< PColorVector< RComplex<Word<REAL64> > > > > LatticeColorVectorSpinMatrixD;
  typedef OLattice< PSpinMatrix< PColorVector< RComplex<Word<REAL32> > > > > LatticeColorVectorSpinMatrixF;

  typedef OLattice< PScalar< PSeed < RScalar<Word<INTEGER32> > > > > LatticeSeed;

  typedef PScalar< PScalar< RScalar<Word<REAL> > > > IntReal;
  typedef PScalar< PScalar< RScalar<Word<REAL32 > > > > IntReal32;
  typedef PScalar< PScalar< RScalar<Word<INTEGER32> > > > IntInteger;
  typedef PScalar< PScalar< RScalar<Word<REAL64> > > > IntReal64;
  typedef PScalar< PScalar< RScalar<Word<REAL64> > > > IntDouble;
  typedef PScalar< PScalar< RScalar<Word<LOGICAL> > > > IntBoolean;

  typedef LatticeInteger  LatticeInt;
  typedef Integer  Int;
  typedef ComplexF  Complex32;
  typedef RealF  Real32;
  typedef ComplexD  Complex64;
  typedef RealD  Real64;
  typedef Integer  Int;
  typedef RealF  Real32;
  typedef RealD  Real64;
  typedef ComplexF  Complex32;
  typedef ComplexD  Complex64;
  typedef LatticeInteger  LatticeInt;
  

} // namespace QDP

#endif
