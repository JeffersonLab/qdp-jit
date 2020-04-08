#ifndef QDP_FORWARD
#define QDP_FORWARD

namespace QDP
{
  typedef void* CUfunction;


  enum class JitDeviceLayout { Coalesced , Scalar };

  class FnMapRsrc;

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


  
  // Inner
  template<class T> class IScalar;
  template<class T, int N> class ILattice;

  // Word
  template<class T> class Word;
  template<class T> class WordJIT;
  template<class T> class WordREG;
  class BASEWordJIT;
  class BASEWordREG;

  // Reality
  template<class T> class RScalar;
  template<class T> class RComplex;
  template<class T> class RScalarJIT;
  template<class T> class RComplexJIT;
  template<class T> class RScalarREG;
  template<class T> class RComplexREG;

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

  template<class T> class PScalarREG;
  template <class T, int N, template<class,int> class C> class PMatrixREG;
  template <class T, int N, template<class,int> class C> class PVectorREG;
  template <class T, int N> class PColorVectorREG;
  template <class T, int N> class PSpinVectorREG;
  template <class T, int N> class PColorMatrixREG;
  template <class T, int N> class PSpinMatrixREG;
  template <class T> class PSeedREG;

  template<int N> class GammaType;
  template<int N, int m> class GammaConst;

  template<int N> class GammaTypeDP;
  template<int N, int m> class GammaConstDP;

  // Outer
  template<class T> class OScalar;
  template<class T> class OLattice;
  template<class T> class OScalarJIT;
  template<class T> class OLatticeJIT;
  template<class T> class OScalarREG;
  template<class T> class OLatticeREG;

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

  class QDPCached;

  class Set;
  class Subset;

#if 0
  template<class T> void *function_layout_to_jit_build( const OLattice<T>& dest );
  template<class T> void  function_layout_to_jit_exec(void * function, T *dest, T *src );

  template<class T> void *function_layout_to_native_build( const OLattice<T>& dest );
  template<class T> void  function_layout_to_native_exec(void * function, T *dest, T *src );

  typedef std::pair< int , llvm::Value * > IndexDomain;
  typedef std::vector< IndexDomain >     IndexDomainVector;

  llvm::Value * datalayout( JitDeviceLayout::LayoutEnum  lay , IndexDomainVector a );

  llvm::Value      *get_index_from_index_vector( const IndexDomainVector& idx );
  IndexDomainVector get_index_vector_from_index( llvm::Value *index );
  IndexDomainVector get_scalar_index_vector_from_index( llvm::Value *index );
  void setDataLayoutInnerSize( int64_t i );
  int64_t getDataLayoutInnerSize();
#endif

  template<class T, class T1, class RHS>
  CUfunction
  function_gather_build( void* send_buf , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs );

  namespace RNG 
  {
//  float sranf(Seed&, Seed&, const Seed&);

  template<class T>
  void sranf(WordJIT<T>& dest,
	     PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > & seed, 
	     PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > >& skewed_seed, 
	     const OScalarJIT<PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > >& seed_mult);

#if 0
  template<class T>
  void sranf(WordJIT<T>& dest,
	     OLatticeJIT<PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > >& seed, 
	     OLatticeJIT<PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > >& skewed_seed, 
	     const OLatticeJIT<PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > >& seed_mult);
#endif
#if 0
  template<class T>
  WordJIT<T> sranf(OScalarJIT<PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > >& seed, 
		   OScalarJIT<PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > >& skewed_seed, 
		   const OScalarJIT<PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > >& seed_mult);
#endif
  }



} // namespace QDP

  
#endif
