#ifndef QDP_PRIMCOLORMAT_H
#define QDP_PRIMCOLORMAT_H


namespace QDP {


template <class T> class PColorMatrix : public PMatrix<T, PColorMatrix>
{
public:
  PColorMatrix( Shape shape , typename WordType<T>::Type_t* f): PMatrix<T, PColorMatrix>(shape,f) {}

  Shape shape() const { return PMatrix<T, PColorMatrix>::shape(); }

  Shape::Axis axis() const { return Shape::Axis::Color; }

  int getN() const
  {
    return shape().extent(axis());
  }

  
  T elem(int i, int j) const
  {
    return PMatrix<T, PColorMatrix>::elem(i,j);
  }

};




  template<class T>
  struct GetLimit<PColorMatrix<T>,0>
  {
    static int limit(const Shape& shape)
    {
      return shape.nc() * shape.nc();
    }
  };






  
  
template<class T1>
struct WordType<PColorMatrix<T1> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};


template<class T> 
struct JITType<PColorMatrix<T> >
{
  typedef PColorMatrixJIT<typename JITType<T>::Type_t>  Type_t;
};


template<class T>
class WordSize< PColorMatrix<T> >
{
public:
  static int value( const Shape& ret )
  {
    return ret.nc() * ret.nc() * WordSize<T>::value( ret );
  }
};

  
  

template<class T1>
struct ScalarType<PColorMatrix<T1> >
{
  typedef PColorMatrix<typename ScalarType<T1>::Type_t> Type_t;
};


template<class T1>
struct SinglePrecType<PColorMatrix<T1> >
{
  typedef PColorMatrix<typename SinglePrecType<T1>::Type_t> Type_t;
};

template<class T1>
struct DoublePrecType<PColorMatrix<T1> >
{
  typedef PColorMatrix<typename DoublePrecType<T1>::Type_t> Type_t;
};



template<class T>
struct InternalScalar<PColorMatrix<T> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

template<class T>
struct PrimitiveScalar<PColorMatrix<T> > {
  typedef PScalar<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

template<class T>
struct LatticeScalar<PColorMatrix<T> > {
  typedef PColorMatrix<typename LatticeScalar<T>::Type_t>  Type_t;
};




  
  
template<class T1, class Op>
struct UnaryReturn<PColorMatrix<T1>, Op> {
  typedef PColorMatrix<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

template<class T1, class T2, class Op>
struct BinaryReturn<PScalar<T1>, PColorMatrix<T2>, Op> {
  typedef PColorMatrix<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

template<class T1, class T2, class Op>
struct BinaryReturn<PColorMatrix<T1>, PColorMatrix<T2>, Op> {
  typedef PColorMatrix<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

template<class T1, class T2, class Op>
struct BinaryReturn<PColorMatrix<T1>, PScalar<T2>, Op> {
  typedef PColorMatrix<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


 


// ColorMatrix
template<class T>
struct UnaryReturn<PColorMatrix<T>, FnTrace > {
  typedef PScalar<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PColorMatrix<T>, FnRealTrace > {
  typedef PScalar<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PColorMatrix<T>, FnImagTrace > {
  typedef PScalar<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PColorMatrix<T>, FnNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PColorMatrix<T>, FnLocalNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrix<T1>, PColorMatrix<T2>, FnTraceMultiply> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrix<T1>, PScalar<T2>, FnTraceMultiply> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PColorMatrix<T2>, FnTraceMultiply> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrix<T1>, PColorMatrix<T2>, FnInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrix<T1>, PScalar<T2>, FnInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PColorMatrix<T2>, FnInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrix<T1>, PColorMatrix<T2>, FnLocalInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrix<T1>, PScalar<T2>, FnLocalInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PColorMatrix<T2>, FnLocalInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrix<T1>, PColorMatrix<T2>, FnInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrix<T1>, PScalar<T2>, FnInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PColorMatrix<T2>, FnInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrix<T1>, PColorMatrix<T2>, FnLocalInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrix<T1>, PScalar<T2>, FnLocalInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PColorMatrix<T2>, FnLocalInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};





template<class T1>
struct UnaryReturn<PColorMatrix<T1>, FnIsFinite> {
  typedef PScalar< typename UnaryReturn<T1, FnIsFinite >::Type_t > Type_t;
};


template<class T1>
inline typename UnaryReturn<PColorMatrix<T1>, FnIsFinite>::Type_t
isfinite(const PColorMatrix<T1>& l)
{
  typedef typename UnaryReturn<PColorMatrix<T1>, FnIsFinite>::Type_t Ret_t;
  Ret_t d(true);

  for(int i=0; i < l.getN(); ++i)
    for(int j=0; j < l.getN(); ++j)
      d.elem() &= isfinite(l.elem(i,j));

  return d;
}





template<class T>
struct UnaryReturn<PColorMatrix<T>, FnTraceColor > {
  typedef PScalar<typename UnaryReturn<T, FnTraceColor>::Type_t>  Type_t;
};



template<class T1, class T2>
struct BinaryReturn<PColorMatrix<T1>, PColorMatrix<T2>, FnTraceColorMultiply> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<PColorMatrix<T1>, PScalar<T2>, FnTraceColorMultiply> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PColorMatrix<T2>, FnTraceColorMultiply> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};



template <class T>
struct UnaryReturn<PColorMatrix<T>, FnTransposeColor > {
  typedef PColorMatrix<typename UnaryReturn<T, FnTransposeColor>::Type_t> Type_t;
};



template<class T1, class T2>
struct BinaryReturn<PColorVector<T1>, PColorVector<T2>, FnOuterProduct> {
  typedef PColorMatrix<typename BinaryReturn<T1, T2, FnOuterProduct>::Type_t>  Type_t;
};



template<class T>
struct UnaryReturn<PColorMatrix<T>, FnPeekColorMatrix > {
  typedef PScalar<typename UnaryReturn<T, FnPeekColorMatrix>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
struct TrinaryReturn<PColorMatrix<T1>, PColorMatrix<T2>, PColorMatrix<T3>, FnColorContract> {
  typedef PScalar<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};


} // namespace QDP

#endif
