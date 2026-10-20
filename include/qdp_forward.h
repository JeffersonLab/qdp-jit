#ifndef QDP_FORWARD_H
#define QDP_FORWARD_H

namespace llvm
{
  struct Value;
}

namespace JIT
{
  class Compiler;
}

namespace QDP
{
  typedef int ParamRef; // this repeated in qdp_llvm.h

  class Shape;

  template<class T> struct ShapeAxes;

  template<class Op, class A>
  struct UnaryShapeTransform;

  template<class Op, class LHS, class RHS>
  struct BinaryShapeTransform;

  
  template<class T, int N> 
  struct GetLimit
  {
    static int limit(const Shape& shape)
    {
      return GetLimit<typename T::Sub_t,N-1>::limit(shape);
    }
  };


      
  
  


  
  
  //enum class JitDataOrder { SiteFast , SiteSlow };

  class FnMapRsrc;

  // IO
  class TextReader;
  class TextWriter;
  class BinaryReader;
  class BinaryWriter;


  

  template<class T> class JITType;
  template<class T> class BaseJIT;

  template<class T> class WordSize;

  
  // Inner
  template<class T> class IScalar;
  template<class T, int N> class ILattice;

  // Word
  template<class T> class Word;
  template<class T> class WordJIT;

  // WordVec
  template<class T> class WordVec;
  template<class T> class WordVecJIT;

  
  // Reality
  template<class T> class RScalar;
  template<class T> class RComplex;
  template<class T> class RScalarJIT;
  template<class T> class RComplexJIT;

  // Primitives
  template<class T> class PScalar;
  template <class T, template<class> class C> class PMatrix;
  template <class T, template<class> class C> class PVector;
  template <class T> class PColorVector;
  template <class T> class PSpinVector;
  template <class T> class PColorMatrix;
  template <class T> class PSpinMatrix;
  template <class T> class PSeed;

  template<class T> class PScalarJIT;
  template <class T, template<class> class C> class PMatrixJIT;
  template <class T, template<class> class C> class PVectorJIT;
  template <class T> class PColorVectorJIT;
  template <class T> class PSpinVectorJIT;
  template <class T> class PColorMatrixJIT;
  template <class T> class PSpinMatrixJIT;
  template <class T> class PSeedJIT;

  
  class GammaType;
  class GammaTypeDP;

  struct OpMultiplyGammaType;
  struct OpGammaTypeMultiply;
  
  // Outer
  template<class T> class OScalar;
  template<class T> class OLattice;
  template<class T> class OScalarJIT;
  template<class T> class OLatticeJIT;

  template<class T> class OScalarStaggeredPropagator;
  template<class T1> class OLatticeDiracFermionF3;

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
  template<class T> struct ScalarType;
  template<class T> struct IsWordVec;

#if 0
template<>
struct RealScalar<int> {
  typedef REAL32  Type_t;
};

template<>
struct RealScalar<float> {
  typedef REAL32  Type_t;
};

template<>
struct RealScalar<double> {
  typedef REAL64  Type_t;
};
#endif
  
  // Empty leaf functor tag
  struct ElemLeaf;

  // Empty print tag
  struct PrintTag;

  struct ShiftPhase1;
  struct ShiftPhase2;
  //struct EvalQV {};
  class Map;
  struct FnMap;
  class ArrayBiDirectionalMap;

  class QDPCached;

  class Set;
  class Subset;

  class JitFunction;
  class DynKey;


  struct CollectLayouts;



  template<typename T>
  int jit_sizeof(const T& t)
  {
    int w = WordSize<T>::value(t.shape());
    return w * sizeof( typename WordType<T>::Type_t );
  }

  template<typename T>
  int jit_sizeof_words(const T& t)
  {
    return WordSize<T>::value(t.shape());
  }

  

  namespace RNG 
  {
    template<class T>
    void sranf(WordJIT<T>& dest,
	       PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > & seed, 
	       PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > >& skewed_seed, 
	       const OScalarJIT<PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > >& seed_mult);
  }


} // namespace QDP

  
#endif
