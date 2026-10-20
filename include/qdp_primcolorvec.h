#ifndef QDP_PRIMCOLORVEC_H
#define QDP_PRIMCOLORVEC_H


namespace QDP {

template <class T> class PColorVector : public PVector<T, PColorVector>
{
public:

  Shape shape() const
  {
    return PVector<T, PColorVector>::shape();
  }
  
  int getN() const
  {
    return shape().at('c');
  }

  Shape::Axis axis() const { return Shape::Axis::Color; }

  
  PColorVector( Shape shape , typename WordType<T>::Type_t *base_ ): PVector<T, PColorVector>(shape,base_) {}

};



  

  template<class T>
  struct GetLimit<PColorVector<T>,0>
  {
    static int limit(const Shape& shape)
    {
      return shape.nc();
    }
  };

  
template<class T>
class WordSize< PColorVector<T> >
{
public:
  static int value( const Shape& ret )
  {
    return ret.nc() * WordSize<T>::value( ret );
  }
};




  

  template<class T> 
  struct JITType<PColorVector<T> >
  {
    typedef PColorVectorJIT<typename JITType<T>::Type_t>  Type_t;
  };


template<class T1>
struct WordType<PColorVector<T1> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};


template<class T1>
struct ScalarType< PColorVector<T1> >
{
  typedef PColorVector< typename ScalarType<T1>::Type_t> Type_t;
};



template<class T1>
struct SinglePrecType< PColorVector<T1> >
{
  typedef PColorVector< typename SinglePrecType<T1>::Type_t> Type_t;
};


template<class T1>
struct DoublePrecType< PColorVector<T1> >
{
  typedef PColorVector< typename DoublePrecType<T1>::Type_t> Type_t;
};


template<class T>
struct InternalScalar<PColorVector<T> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

template<class T>
struct PrimitiveScalar<PColorVector<T> > {
  typedef PScalar<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

template<class T>
struct LatticeScalar<PColorVector<T> > {
  typedef PColorVector<typename LatticeScalar<T>::Type_t>  Type_t;
};



template<class T1, class Op>
struct UnaryReturn<PColorVector<T1>, Op> {
  typedef PColorVector<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

template<class T1, class T2, class Op>
struct BinaryReturn<PScalar<T1>, PColorVector<T2>, Op> {
  typedef PColorVector<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

template<class T1, class T2, class Op>
struct BinaryReturn<PColorMatrix<T1>, PColorVector<T2>, Op> {
  typedef PColorVector<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};
  
template<class T1, class T2, class Op>
struct BinaryReturn<PColorVector<T1>, PScalar<T2>, Op> {
  typedef PColorVector<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

template<class T1, class T2, class Op>
struct BinaryReturn<PColorVector<T1>, PColorVector<T2>, Op> {
  typedef PColorVector<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};



template<class T>
struct UnaryReturn<PColorVector<T>, FnNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PColorVector<T>, FnLocalNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorVector<T1>, PColorVector<T2>, FnInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorVector<T1>, PColorVector<T2>, FnLocalInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorVector<T1>, PColorVector<T2>, FnLocalColorInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalColorInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorVector<T1>, PColorVector<T2>, FnInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorVector<T1>, PColorVector<T2>, FnLocalInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};




template<class T>
struct UnaryReturn<PColorVector<T>, FnPeekColorVector > {
  typedef PScalar<typename UnaryReturn<T, FnPeekColorVector>::Type_t>  Type_t;
};


template<class T>
inline typename UnaryReturn<PColorVector<T>, FnPeekColorVector>::Type_t
peekColor(const PColorVector<T>& l, int row)
{
  typename UnaryReturn<PColorVector<T>, FnPeekColorVector>::Type_t  d;

  // Note, do not need to propagate down since the function is eaten at this level
  d.elem() = l.elem(row);
  return d;
}


template<class T1, class T2>
inline PColorVector<T1>&
pokeColor(PColorVector<T1>& l, const PScalar<T2>& r, int row)
{
  // Note, do not need to propagate down since the function is eaten at this level
  l.elem(row) = r.elem();
  return l;
}


template<class T1, class T2, class T3>
struct TrinaryReturn<PColorVector<T1>, PColorVector<T2>, PColorVector<T3>, FnColorContract> {
  typedef PScalar<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};

//! dest  = colorContract(Qvec1,Qvec2,Qvec3)
/*!
 * Performs:
 *  \f$dest = \sum_{i,j,k} \epsilon^{i,j,k} V1^{i} V2^{j} V3^{k}\f$
 *
 * This routine is completely unrolled for 3 colors
 */
template<class T1, class T2, class T3>
inline typename TrinaryReturn<PColorVector<T1>, PColorVector<T2>, PColorVector<T3>, FnColorContract>::Type_t
colorContract(const PColorVector<T1>& s1, const PColorVector<T2>& s2, const PColorVector<T3>& s3)
{
  typename TrinaryReturn<PColorVector<T1>, PColorVector<T2>, PColorVector<T3>, FnColorContract>::Type_t  d;

  // Permutations: +(0,1,2)+(1,2,0)+(2,0,1)-(1,0,2)-(0,2,1)-(2,1,0)

  // d = \epsilon^{i,j,k} V1^{i} V2^{j} V3^{k}
  d.elem() = (s1.elem(0)*s2.elem(1)
           -  s1.elem(1)*s2.elem(0))*s3.elem(2)
           + (s1.elem(1)*s2.elem(2)
           -  s1.elem(2)*s2.elem(1))*s3.elem(0)
           + (s1.elem(2)*s2.elem(0)
           -  s1.elem(0)*s2.elem(2))*s3.elem(1);

  return d;
}


//-----------------------------------------------------------------------------
// Contraction for color vectors
// colorContract 
template<class T1, class T2>
struct BinaryReturn<PColorVector<T1>, PColorVector<T2>, FnColorVectorContract> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnColorVectorContract>::Type_t>  Type_t;
};

//! dest  = colorVectorContract(Qvec1,Qvec2)
/*!
 * Performs:
 *  \f$dest = \sum_{i} V1^{i} V2^{i}\f$
 */
template<class T1, class T2>
inline typename BinaryReturn<PColorVector<T1>, PColorVector<T2>, FnColorVectorContract>::Type_t
colorVectorContract(const PColorVector<T1>& s1, const PColorVector<T2>& s2)
{
  typename BinaryReturn<PColorVector<T1>, PColorVector<T2>, FnColorVectorContract>::Type_t  d;

  // d = V1^{i} V2^{i}
  d.elem() = s1.elem(0)*s2.elem(0);
  for(int i=1; i < s1.getN(); ++i)
    d.elem() += s1.elem(i)*s2.elem(i);

  return d;
}


//-----------------------------------------------------------------------------
// diquark color cross product   s1 X s2
//! Contraction for color vectors
template<class T1, class T2>
struct BinaryReturn<PColorVector<T1>, PColorVector<T2>, FnColorCrossProduct> {
  typedef PColorVector<typename BinaryReturn<T1, T2, FnColorCrossProduct>::Type_t>  Type_t;
};

//! dest  = colorCrossProduct(Qvec1,Qvec2)
/*!
 * Performs:
 *  \f$dest^{i} = \sum_{j,k} \epsilon^{i,j,k} V1^{j} V2^{k}\f$
 *
 * This routine is completely unrolled for 3 colors
 */
template<class T1, class T2>
inline typename BinaryReturn<PColorVector<T1>, PColorVector<T2>, FnColorCrossProduct>::Type_t
colorCrossProduct(const PColorVector<T1>& s1, const PColorVector<T2>& s2)
{
  typename BinaryReturn<PColorVector<T1>, PColorVector<T2>, FnColorCrossProduct>::Type_t  d;
  
  d.elem(0) = s1.elem(1)*s2.elem(2) - s1.elem(2)*s2.elem(1);
  d.elem(1) = s1.elem(2)*s2.elem(0) - s1.elem(0)*s2.elem(2);
  d.elem(2) = s1.elem(0)*s2.elem(1) - s1.elem(1)*s2.elem(0);

 return d;
}



} // namespace QDP

#endif

