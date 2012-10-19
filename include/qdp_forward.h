// -*- C++ -*-

/*! @file
 * @brief Forward declarations for QDP
 */

namespace QDP
{

  // IO
  class TextReader;
  class TextWriter;
  class BinaryReader;
  class BinaryWriter;


  // Forward declarations
  //! dest  = random
  template<class T1, class T2>
  inline void
  fill_random(float& d, T1& seed, T2& skewed_seed, const T1& seed_mult);

  //! dest  = random
  template<class T1, class T2>
  inline void
  fill_random(double& d, T1& seed, T2& skewed_seed, const T1& seed_mult);

  //! dest  = random
  template<class T1, class T2, int N>
  inline void
  fill_random(float* d, T1& seed, T2& skewed_seed, const T1& seed_mult);

  //! dest  = random
  template<class T1, class T2, int N>
  inline void
  fill_random(double* d, T1& seed, T2& skewed_seed, const T1& seed_mult);


  namespace RNG 
  {
//  float sranf(Seed&, Seed&, const Seed&);
  }

  
  // Inner
  template<class T> class IScalar;
  template<class T, int N> class ILattice;

  // Word
  template<class T> class Word;
  template<class T> class WordJIT;

  // Reality
  template<class T> class RScalar;
  template<class T> class RComplex;
  template<class T> class RScalarJIT;
  template<class T> class RComplexJIT;

  // Primitives
  template<class T> class PScalar;
  template <class T, int N, template<class,int> class C> class PMatrix;
  template <class T, int N, template<class,int> class C> class PVector;
  template <class T, int N> class PColorVector;
  template <class T, int N> class PSpinVector;
  template <class T, int N> class PColorMatrix;
  template <class T, int N> class PSpinMatrix;
  template <class T> class PSeed;
  template<class T> class PScalarJIT;
  template <class T, int N, template<class,int> class C> class PMatrixJIT;
  template <class T, int N, template<class,int> class C> class PVectorJIT;
  template <class T, int N> class PColorVectorJIT;
  template <class T, int N> class PSpinVectorJIT;
  template <class T, int N> class PColorMatrixJIT;
  template <class T, int N> class PSpinMatrixJIT;
  template <class T> class PSeedJIT;

  template<int N> class GammaType;
  template<int N, int m> class GammaConst;

  template<int N> class GammaTypeDP;
  template<int N, int m> class GammaConstDP;

  // Outer
  template<class T> class OScalar;
  template<class T> class OLattice;
  template<class T> class OScalarJIT;
  template<class T> class OLatticeJIT;

  // Outer types narrowed to a subset
  template<class T> class OSubScalar;
  template<class T> class OSubLattice;

  // Main type
  template<class T, class C> class QDPType;
  template<class T, class C> class QDPTypeJIT;

  // Expression class for QDP
  template<class T, class C> class QDPExpr;

  // Main type narrowed to a subset
  template<class T, class C> class QDPSubType;

  // Simple scalar trait class
  template<class T> struct SimpleScalar;
  template<class T> struct InternalScalar;
  template<class T> struct LatticeScalar;
  template<class T> struct PrimitiveScalar;
  template<class T> struct RealScalar;
  template<class T> struct WordType;
  template<class T> struct SinglePrecType;
  template<class T> struct DoublePrecType;

  // Empty leaf functor tag
  struct ElemLeaf;

  // Empty print tag
  struct PrintTag;

  struct ShiftPhase1;
  struct ShiftPhase2;
  class Map;
  struct FnMap;
  class ArrayBiDirectionalMap;


} // namespace QDP

  
