#ifndef QDP_PRIMSPINMAT_H
#define QDP_PRIMSPINMAT_H


namespace QDP {

template <class T> class PSpinMatrix : public PMatrix<T, PSpinMatrix>
{
public:

  PSpinMatrix( Shape shape , typename WordType<T>::Type_t* f): PMatrix<T, PSpinMatrix>(shape,f) {}

  Shape shape() const { return PMatrix<T, PSpinMatrix>::shape(); }

  Shape::Axis axis() const { return Shape::Axis::Spin; }

  int getN() const
  {
    return shape().extent(axis());
  }

  
  T elem(int i, int j) const
  {
    return PMatrix<T, PSpinMatrix>::elem(i,j);
  }

};


  template<class T>
  struct GetLimit<PSpinMatrix<T>,0>
  {
    static int limit(const Shape& shape)
    {
      return shape.ns() * shape.ns();
    }
  };






  
template<class T1>
struct WordType<PSpinMatrix<T1> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};


template<class T> 
struct JITType<PSpinMatrix<T> >
{
  typedef PSpinMatrixJIT<typename JITType<T>::Type_t>  Type_t;
};


template<class T>
class WordSize< PSpinMatrix<T> >
{
public:
  static int value( const Shape& ret )
  {
    return ret.ns() * ret.ns() * WordSize<T>::value( ret );
  }
};

  
  

template<class T1>
struct ScalarType<PSpinMatrix<T1> >
{
  typedef PSpinMatrix<typename ScalarType<T1>::Type_t> Type_t;
};


template<class T1>
struct SinglePrecType<PSpinMatrix<T1> >
{
  typedef PSpinMatrix<typename SinglePrecType<T1>::Type_t> Type_t;
};

template<class T1>
struct DoublePrecType<PSpinMatrix<T1> >
{
  typedef PSpinMatrix<typename DoublePrecType<T1>::Type_t> Type_t;
};



template<class T>
struct InternalScalar<PSpinMatrix<T> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive into a scalar leaving grid along
template<class T>
struct PrimitiveScalar<PSpinMatrix<T> > {
  typedef PScalar<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices along
template<class T>
struct LatticeScalar<PSpinMatrix<T> > {
  typedef PSpinMatrix<typename LatticeScalar<T>::Type_t>  Type_t;
};





template<class T1, class Op>
struct UnaryReturn<PSpinMatrix<T1>, Op> {
  typedef PSpinMatrix<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

template<class T1, class T2, class Op>
struct BinaryReturn<PScalar<T1>, PSpinMatrix<T2>, Op> {
  typedef PSpinMatrix<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

template<class T1, class T2, class Op>
struct BinaryReturn<PSpinMatrix<T1>, PSpinMatrix<T2>, Op> {
  typedef PSpinMatrix<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

template<class T1, class T2, class Op>
struct BinaryReturn<PSpinMatrix<T1>, PScalar<T2>, Op> {
  typedef PSpinMatrix<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};



 


template<class T>
struct UnaryReturn<PSpinMatrix<T>, FnTrace > {
  typedef PScalar<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PSpinMatrix<T>, FnRealTrace > {
  typedef PScalar<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PSpinMatrix<T>, FnImagTrace > {
  typedef PScalar<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PSpinMatrix<T>, FnNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PSpinMatrix<T>, FnLocalNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrix<T1>, PSpinMatrix<T2>, FnTraceMultiply> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrix<T1>, PScalar<T2>, FnTraceMultiply> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PSpinMatrix<T2>, FnTraceMultiply> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrix<T1>, PSpinMatrix<T2>, FnInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrix<T1>, PScalar<T2>, FnInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PSpinMatrix<T2>, FnInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrix<T1>, PSpinMatrix<T2>, FnLocalInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};





  
template<class T1, class T2>
struct BinaryReturn<PSpinMatrix<T1>, PScalar<T2>, FnLocalInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PSpinMatrix<T2>, FnLocalInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrix<T1>, PSpinMatrix<T2>, FnInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrix<T1>, PScalar<T2>, FnInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PSpinMatrix<T2>, FnInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrix<T1>, PSpinMatrix<T2>, FnLocalInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrix<T1>, PScalar<T2>, FnLocalInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PSpinMatrix<T2>, FnLocalInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};



template<class T2>
struct BinaryReturn<GammaType, PSpinMatrix<T2>, OpGammaTypeMultiply> {
  typedef PSpinMatrix<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2>
struct BinaryReturn<PSpinMatrix<T2>, GammaType, OpMultiplyGammaType> {
  typedef PSpinMatrix<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};


template<class T2>
struct BinaryReturn<GammaTypeDP, PSpinMatrix<T2>, OpGammaTypeMultiply> {
  typedef PSpinMatrix<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2>
struct BinaryReturn<PSpinMatrix<T2>, GammaTypeDP, OpMultiplyGammaType> {
  typedef PSpinMatrix<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};




template<class T>
struct UnaryReturn<PSpinMatrix<T>, FnTraceSpin > {
  typedef PScalar<typename UnaryReturn<T, FnTraceSpin>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<PSpinMatrix<T1>, PSpinMatrix<T2>, FnTraceSpinMultiply> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<PSpinMatrix<T1>, PScalar<T2>, FnTraceSpinMultiply> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PSpinMatrix<T2>, FnTraceSpinMultiply> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t>  Type_t;
};


template <class T>
struct UnaryReturn<PSpinMatrix<T>, FnTransposeSpin > {
  typedef PSpinMatrix<typename UnaryReturn<T, FnTransposeSpin>::Type_t> Type_t;
};


template<class T1, class T2>
struct BinaryReturn<PSpinVector<T1>, PSpinVector<T2>, FnOuterProduct> {
  typedef PSpinMatrix<typename BinaryReturn<T1, T2, FnOuterProduct>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<PSpinVector<T1>, PSpinVector<T2>, FnTraceSpinOuterProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnOuterProduct>::Type_t>  Type_t;
};


template<class T>
struct UnaryReturn<PSpinMatrix<T>, FnPeekSpinMatrix > {
  typedef PScalar<typename UnaryReturn<T, FnPeekSpinMatrix >::Type_t>  Type_t;
};




} // namespace QDP

#endif
