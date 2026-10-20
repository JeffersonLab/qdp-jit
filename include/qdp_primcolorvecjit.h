#ifndef QDP_PRIMCOLORVECJIT_H
#define QDP_PRIMCOLORVECJIT_H

namespace QDP {


template <class T> class PColorVectorJIT : public PVectorJIT<T, PColorVectorJIT>
{
public:
  Shape::Axis axis() const { return Shape::Axis::Color; }

  PColorVectorJIT( llvm::Value * base , JitDataFormat fmt , NestedIndex args , Shape shape ) : PVectorJIT<T, PColorVectorJIT>(base,fmt,args,shape) {}

  // for eval Shape
  PColorVectorJIT( Shape shape ) : PVectorJIT<T, PColorVectorJIT>(shape) {}

  
  Shape shape() const { return PVectorJIT<T, PColorVectorJIT>::shape(); }

  int getN() const
  {
    return shape().extent(Shape::Axis::Color);
  }

  int this_size() const
  {
    return getN();
  }


  template<class T1>
  PColorVectorJIT operator=(PColorVectorJIT<T1> rhs)
  {
    this->assign(rhs);
    return *this;
  }


  PColorVectorJIT operator=(PColorVectorJIT rhs)
  {
    this->assign(rhs);
    return *this;
  }
  
};



  template <class T>
  struct IsWordVec< PColorVectorJIT<T > >
  {
    constexpr static bool value = IsWordVec<T>::value;
  };


  template<class T>
class WordSize< PColorVectorJIT<T> >
{
public:
  static int value( const Shape& ret )
  {
    return ret.nc() * WordSize<T>::value( ret );
  }
};


template<class T1>
struct BASEType<PColorVectorJIT<T1> > 
{
  typedef PColorVector<typename BASEType<T1>::Type_t>  Type_t;
};

  


template<class T1>
struct WordType<PColorVectorJIT<T1> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};



template<class T1>
struct SinglePrecType< PColorVectorJIT<T1> >
{
  typedef PColorVectorJIT< typename SinglePrecType<T1>::Type_t> Type_t;
};


template<class T1>
struct DoublePrecType< PColorVectorJIT<T1> >
{
  typedef PColorVectorJIT< typename DoublePrecType<T1>::Type_t> Type_t;
};



template<class T>
struct InternalScalar<PColorVectorJIT<T> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};


template<class T>
struct PrimitiveScalar<PColorVectorJIT<T> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};


template<class T>
struct LatticeScalar<PColorVectorJIT<T> > {
  typedef PColorVectorJIT<typename LatticeScalar<T>::Type_t>  Type_t;
};






template<class T1, class Op>
struct UnaryReturn<PColorVectorJIT<T1>, Op> {
  typedef PColorVectorJIT<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};


template<class T1>
struct UnaryReturn<PColorVectorJIT<T1>, FnIsFinite> {
  typedef PScalarJIT<  typename UnaryReturn<T1, FnIsFinite>::Type_t > Type_t;
};

  

template<class T1, class T2, class Op>
struct BinaryReturn<PScalarJIT<T1>, PColorVectorJIT<T2>, Op> {
  typedef PColorVectorJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};



template<class T1, class T2, class Op>
struct BinaryReturn<PColorMatrixJIT<T1>, PColorVectorJIT<T2>, Op> {
  typedef PColorVectorJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

  

template<class T1, class T2, class Op>
struct BinaryReturn<PColorVectorJIT<T1>, PScalarJIT<T2>, Op> {
  typedef PColorVectorJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<PColorVectorJIT<T1>, PColorVectorJIT<T2>, Op> {
  typedef PColorVectorJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};









template<class T>
struct UnaryReturn<PColorVectorJIT<T>, FnNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PColorVectorJIT<T>, FnLocalNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorVectorJIT<T1>, PColorVectorJIT<T2>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorVectorJIT<T1>, PColorVectorJIT<T2>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorVectorJIT<T1>, PColorVectorJIT<T2>, FnLocalColorInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalColorInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorVectorJIT<T1>, PColorVectorJIT<T2>, FnInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorVectorJIT<T1>, PColorVectorJIT<T2>, FnLocalInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};




template<class T1, class T2>
PScalarJIT<typename BinaryReturn<T1, T2, FnLocalColorInnerProduct>::Type_t>
localColorInnerProduct(PColorVectorJIT<T1> s1, PColorVectorJIT<T2> s2)
{
  assert(s1.getN() == s2.getN());

  auto d = binary_return<FnLocalColorInnerProduct>( s1 , s2 );

  d.elem() = adj( s1.elem(0) ) * s2.elem(0);
  JitForLoop i( 1 , s1.getN() );
  {
    d.elem() += adj( s1.elem(i) ) * s2.elem(i);
  }
  i.end();
  
  return d;
}



template<class T>
struct UnaryReturn<PColorVectorJIT<T>, FnPeekColorVectorJIT > {
  typedef PScalarJIT<T>  Type_t;
};


template<class T>
struct UnaryReturn<PColorVectorJIT<T>, FnPeekColorVector > {
  typedef PScalarJIT<T>  Type_t;
};



template<class T>
typename UnaryReturn<PColorVectorJIT<T>, FnPeekColorVectorJIT >::Type_t
peekColor(PColorVectorJIT<T> l, llvm::Value * row)
{
  auto d = unary_return<FnPeekColorVectorJIT>(l);
  d.elem() = l.elem(row);
  return d;
}





template<class T1, class T2>
void
pokeColor(PColorVectorJIT<T1> l, PScalarJIT<T2> r, llvm::Value * row)
{
  l.elem(row) = r.elem();
}



template<class T1, class T2, class T3>
struct TrinaryReturn<PColorVectorJIT<T1>, PColorVectorJIT<T2>, PColorVectorJIT<T3>, FnColorContract> {
  typedef PScalarJIT<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};


template<class T1, class T2, class T3>
typename TrinaryReturn<PColorVectorJIT<T1>, PColorVectorJIT<T2>, PColorVectorJIT<T3>, FnColorContract>::Type_t
colorContract(PColorVectorJIT<T1> s1, PColorVectorJIT<T2> s2, PColorVectorJIT<T3> s3)
{
  auto d = trinary_return<FnColorContract>( s1 , s2 , s3 );

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



template<class T1, class T2>
struct BinaryReturn<PColorVectorJIT<T1>, PColorVectorJIT<T2>, FnColorVectorContract> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnColorVectorContract>::Type_t>  Type_t;
};



template<class T1, class T2>
typename BinaryReturn<PColorVectorJIT<T1>, PColorVectorJIT<T2>, FnColorVectorContract>::Type_t
colorVectorContract(PColorVectorJIT<T1> s1, PColorVectorJIT<T2> s2)
{
  assert(s1.getN() == s2.getN());

  auto d = binary_return<FnColorVectorContract>( s1 , s2 );

  d.elem() = s1.elem(0)*s2.elem(0);
  JitForLoop i( 1 , s1.getN() );
  {
    d.elem() += s1.elem(i)*s2.elem(i);
  }
  i.end();
  
  return d;
}



template<class T1, class T2>
struct BinaryReturn<PColorVectorJIT<T1>, PColorVectorJIT<T2>, FnColorCrossProduct> {
  typedef PColorVectorJIT<typename BinaryReturn<T1, T2, FnColorCrossProduct>::Type_t>  Type_t;
};


template<class T1, class T2>
typename BinaryReturn<PColorVectorJIT<T1>, PColorVectorJIT<T2>, FnColorCrossProduct>::Type_t
colorCrossProduct(PColorVectorJIT<T1> s1, PColorVectorJIT<T2> s2)
{
  assert(s1.getN() == s2.getN());

  auto d = binary_return<FnColorCrossProduct>( s1 , s2 );
  
  d.elem(0) = s1.elem(1)*s2.elem(2) - s1.elem(2)*s2.elem(1);
  d.elem(1) = s1.elem(2)*s2.elem(0) - s1.elem(0)*s2.elem(2);
  d.elem(2) = s1.elem(0)*s2.elem(1) - s1.elem(1)*s2.elem(0);

 return d;
}


template<class T1, class T2, class T3>
struct TrinaryReturn<PScalarJIT<T1>, PColorVectorJIT<T2>, PColorVectorJIT<T3>, FnWhere> {
  typedef PColorVectorJIT<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};



} // namespace QDP

#endif

