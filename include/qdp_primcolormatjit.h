#ifndef QDP_PRIMCOLORMATJIT_H
#define QDP_PRIMCOLORMATJIT_H


namespace QDP {


template <class T> class PColorMatrixJIT : public PMatrixJIT<T, PColorMatrixJIT>
{
public:

  PColorMatrixJIT() {}

  PColorMatrixJIT( llvm::Value * base , JitDataFormat fmt , NestedIndex args , Shape shape ) : PMatrixJIT<T, PColorMatrixJIT>(base,fmt,args,shape) {}

  
  // for eval Shape
  PColorMatrixJIT( Shape shape ) : PMatrixJIT<T, PColorMatrixJIT>(shape) {}
  

  PColorMatrixJIT( const PColorMatrixJIT& rhs ): PMatrixJIT<T, PColorMatrixJIT>(rhs) {}
  
  Shape::Axis axis() const { return Shape::Axis::Color; }

  int getN() const
  {
    int ret = shape().extent(axis());

    return ret;
  }

  int this_size() const
  {
    int ret = getN() * getN();

    return ret;
  }

  Shape shape() const { return PMatrixJIT<T, PColorMatrixJIT>::shape(); }


  template<class T1>
  PColorMatrixJIT operator=(PScalarJIT<T1> rhs)
  {
    this->assign(rhs);
    return *this;
  }


  template<class T1>
  PColorMatrixJIT operator=(PColorMatrixJIT<T1> rhs) 
  {
    this->assign(rhs);
    return *this;
  }


  PColorMatrixJIT operator=(PColorMatrixJIT rhs) 
  {
    this->assign(rhs);
    return *this;
  }
};




template<class T>
class WordSize< PColorMatrixJIT<T> >
{
public:
  static int value( const Shape& ret )
  {
    return ret.nc() * ret.nc() * WordSize<T>::value( ret );
  }
};
  


  
  template <class T>
  struct JITType< PColorMatrixJIT<T> >
  {
    typedef PColorMatrixJIT< typename JITType<T>::Type_t >  Type_t;
  };




template<class T1>
struct WordType<PColorMatrixJIT<T1> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};


template<class T1>
struct SinglePrecType<PColorMatrixJIT<T1> >
{
  typedef PColorMatrixJIT<typename SinglePrecType<T1>::Type_t> Type_t;
};

template<class T1>
struct DoublePrecType<PColorMatrixJIT<T1> >
{
  typedef PColorMatrixJIT<typename DoublePrecType<T1>::Type_t> Type_t;
};




template<class T>
struct InternalScalar<PColorMatrixJIT<T> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};


template<class T>
struct PrimitiveScalar<PColorMatrixJIT<T> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};


template<class T>
struct LatticeScalar<PColorMatrixJIT<T> > {
  typedef PColorMatrixJIT<typename LatticeScalar<T>::Type_t>  Type_t;
};






template<class T1, class Op>
struct UnaryReturn<PColorMatrixJIT<T1>, Op> {
  typedef PColorMatrixJIT<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2>, Op> {
  typedef PColorMatrixJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<PColorMatrixJIT<T1>, PColorMatrixJIT<T2>, Op> {
  typedef PColorMatrixJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<PColorMatrixJIT<T1>, PScalarJIT<T2>, Op> {
  typedef PColorMatrixJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};



template<class T>
struct UnaryReturn<PColorMatrixJIT<T>, FnTrace > {
  typedef PScalarJIT<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PColorMatrixJIT<T>, FnRealTrace > {
  typedef PScalarJIT<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PColorMatrixJIT<T>, FnImagTrace > {
  typedef PScalarJIT<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PColorMatrixJIT<T>, FnNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PColorMatrixJIT<T>, FnLocalNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrixJIT<T1>, PColorMatrixJIT<T2>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrixJIT<T1>, PScalarJIT<T2>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrixJIT<T1>, PColorMatrixJIT<T2>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrixJIT<T1>, PScalarJIT<T2>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrixJIT<T1>, PColorMatrixJIT<T2>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrixJIT<T1>, PScalarJIT<T2>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrixJIT<T1>, PColorMatrixJIT<T2>, FnInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrixJIT<T1>, PScalarJIT<T2>, FnInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2>, FnInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrixJIT<T1>, PColorMatrixJIT<T2>, FnLocalInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PColorMatrixJIT<T1>, PScalarJIT<T2>, FnLocalInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2>, FnLocalInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};



template<class T1>
struct UnaryReturn<PColorMatrixJIT<T1>, FnIsFinite> {
  typedef PScalarJIT< typename UnaryReturn<T1, FnIsFinite >::Type_t > Type_t;
};




template<class T>
struct UnaryReturn<PColorMatrixJIT<T>, FnTraceColor > {
  typedef PScalarJIT<typename UnaryReturn<T, FnTraceColor>::Type_t>  Type_t;
};

template<class T>
typename UnaryReturn<PColorMatrixJIT<T>, FnTraceColor>::Type_t
traceColor(PColorMatrixJIT<T> s1)
{
  auto d = unary_return<FnTraceColor>( s1 );

  d.elem() = s1.elem(0,0);
  JitForLoop i( 1 , s1.getN() );
  {
    d.elem() += s1.elem(i,i);
  }
  i.end();
  
  return d;
}



template<class T1, class T2>
struct BinaryReturn<PColorMatrixJIT<T1>, PColorMatrixJIT<T2>, FnTraceColorMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
typename BinaryReturn<PColorMatrixJIT<T1>, PColorMatrixJIT<T2>, FnTraceColorMultiply>::Type_t
traceColorMultiply(PColorMatrixJIT<T1> l, PColorMatrixJIT<T2> r)
{
  assert(l.getN() == r.getN());

  auto d = binary_return<FnTraceColorMultiply>( l , r );

  d.elem() = l.elem(0,0) * r.elem(0,0);
  JitForLoop k( 1 , l.getN() );
  {
    d.elem() += l.elem(0,k) * r.elem(k,0);
  }
  k.end();
  
  JitForLoop j( 1 , l.getN() );
  {
    JitForLoop k( 0 , l.getN() );
    {
      d.elem() += l.elem(j,k) * r.elem(k,j);
    }
    k.end();
  }
  j.end();
  
  return d;
}


template<class T1, class T2>
struct BinaryReturn<PColorMatrixJIT<T1>, PScalarJIT<T2>, FnTraceColorMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PColorMatrixJIT<T1>, PScalarJIT<T2>, FnTraceColorMultiply>::Type_t
traceColorMultiply(PColorMatrixJIT<T1> l,  PScalarJIT<T2> r)
{
  auto d = binary_return<FnTraceColorMultiply>( l , r );

  d.elem() = l.elem(0,0) * r.elem();
  JitForLoop k( 1 , l.getN() );
  {
    d.elem() += l.elem(k,k) * r.elem();
  }
  k.end();
  
  return d;
}


template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2>, FnTraceColorMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2>, FnTraceColorMultiply>::Type_t
traceColorMultiply( PScalarJIT<T1> l, PColorMatrixJIT<T2> r)
{
  auto d = binary_return<FnTraceColorMultiply>( l , r );

  d.elem() = l.elem() * r.elem(0,0);
  JitForLoop k( 1 , r.getN() );
  {
    d.elem() += l.elem() * r.elem(k,k);
  }
  k.end();
  
  return d;
}



template <class T>
struct UnaryReturn<PColorMatrixJIT<T>, FnTransposeColor > {
  typedef PColorMatrixJIT<typename UnaryReturn<T, FnTransposeColor>::Type_t> Type_t;
};



template<class T>
typename UnaryReturn<PColorMatrixJIT<T>, FnTransposeColor >::Type_t
transposeColor(PColorMatrixJIT<T> s1)
{
  auto d = unary_return<FnTransposeColor>( s1 );

  JitForLoop i( 0 , s1.getN() );
  {
    JitForLoop j( 0 , s1.getN() );
    {
      d.elem(i,j) = s1.elem(j,i);
    }
    j.end();
  }
  i.end();
  
  return d;
}



template<class T1, class T2>
struct BinaryReturn<PColorVectorJIT<T1>, PColorVectorJIT<T2>, FnOuterProduct> {
  typedef PColorMatrixJIT<typename BinaryReturn<T1, T2, FnOuterProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
typename BinaryReturn<PColorVectorJIT<T1>, PColorVectorJIT<T2>, FnOuterProduct>::Type_t
outerProduct( PColorVectorJIT<T1> l, PColorVectorJIT<T2> r)
{
  assert(l.getN() == r.getN());

  auto d = binary_return<FnOuterProduct>( l,r );

  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = outerProduct(l.elem(i),r.elem(j));
    }
    j.end();
  }
  i.end();

  return d;
}



template<class T>
struct UnaryReturn<PColorMatrixJIT<T>, FnPeekColorMatrixJIT > {
  typedef PScalarJIT<T>  Type_t;
};


template<class T>
struct UnaryReturn<PColorMatrixJIT<T>, FnPeekColorMatrix > {
  typedef PScalarJIT<T>  Type_t;
};



template<class T>
typename UnaryReturn<PColorMatrixJIT<T>, FnPeekColorMatrixJIT>::Type_t
peekColor( PColorMatrixJIT<T> l, llvm::Value* row, llvm::Value* col)
{
  auto d = unary_return<FnPeekColorMatrixJIT>(l);
  
  d.elem() = l.elem(row,col);
  return d;
}



template<class T1, class T2>
void
pokeColor(PColorMatrixJIT<T1> l,  PScalarJIT<T2> r, llvm::Value* row, llvm::Value* col)
{
  l.elem(row,col) = r.elem();
}


template<class T1, class T2, class T3>
struct TrinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2>, PColorMatrixJIT<T3>, FnWhere> {
  typedef PColorMatrixJIT<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};



template<class T1, class T2, class T3>
struct TrinaryReturn<PColorMatrixJIT<T1>, PColorMatrixJIT<T2>, PColorMatrixJIT<T3>, FnColorContract>
{
  typedef PScalarJIT<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};


/*!
 * Performs:
 *  \f$dest = \sum_{i1,i2,i3,j1,j2,j3} \epsilon^{i1,j1,k1}\epsilon^{i2,j2,k2} Q1^{i1,i2} Q2^{j1,j2} Q3^{k1,k2}\f$
 *
 * This routine is completely unrolled for 3 colors
 */
template<class T1, class T2, class T3>
typename TrinaryReturn<PColorMatrixJIT<T1>, PColorMatrixJIT<T2>, PColorMatrixJIT<T3>, FnColorContract>::Type_t
colorContract(PColorMatrixJIT<T1> s1, PColorMatrixJIT<T2> s2, PColorMatrixJIT<T3> s3)
{
  auto d = trinary_return<FnColorContract>( s1,s2,s3 );

  // Permutations: +(0,1,2)+(1,2,0)+(2,0,1)-(1,0,2)-(0,2,1)-(2,1,0)

  // d = \epsilon^{i1,j1,k1}\epsilon^{i2,j2,k2} Q1^{i1,i2} Q2^{j1,j2} Q3^{k1,k2}
  d.elem() = (s1.elem(0,0)*s2.elem(1,1)
           -  s1.elem(1,0)*s2.elem(0,1)
           -  s1.elem(0,1)*s2.elem(1,0)
           +  s1.elem(1,1)*s2.elem(0,0))*s3.elem(2,2)
           + (s1.elem(1,1)*s2.elem(2,2)
           -  s1.elem(2,1)*s2.elem(1,2)
           -  s1.elem(1,2)*s2.elem(2,1)
           +  s1.elem(2,2)*s2.elem(1,1))*s3.elem(0,0)
           + (s1.elem(2,2)*s2.elem(0,0)
           -  s1.elem(0,2)*s2.elem(2,0)
           -  s1.elem(2,0)*s2.elem(0,2)
           +  s1.elem(0,0)*s2.elem(2,2))*s3.elem(1,1)

           + (s1.elem(1,0)*s2.elem(2,1)
           -  s1.elem(2,0)*s2.elem(1,1)
           -  s1.elem(1,1)*s2.elem(2,0)
           +  s1.elem(2,1)*s2.elem(1,0))*s3.elem(0,2)
           + (s1.elem(1,2)*s2.elem(2,0)
           -  s1.elem(2,2)*s2.elem(1,0)
           -  s1.elem(1,0)*s2.elem(2,2)
           +  s1.elem(2,0)*s2.elem(1,2))*s3.elem(0,1)

           + (s1.elem(2,0)*s2.elem(0,1)
           -  s1.elem(0,0)*s2.elem(2,1)
           -  s1.elem(2,1)*s2.elem(0,0)
           +  s1.elem(0,1)*s2.elem(2,0))*s3.elem(1,2)
           + (s1.elem(2,1)*s2.elem(0,2)
           -  s1.elem(0,1)*s2.elem(2,2)
           -  s1.elem(2,2)*s2.elem(0,1)
           +  s1.elem(0,2)*s2.elem(2,1))*s3.elem(1,0)

           + (s1.elem(0,1)*s2.elem(1,2)
           -  s1.elem(1,1)*s2.elem(0,2)
           -  s1.elem(0,2)*s2.elem(1,1)
           +  s1.elem(1,2)*s2.elem(0,1))*s3.elem(2,0)
           + (s1.elem(0,2)*s2.elem(1,0)
           -  s1.elem(1,2)*s2.elem(0,0)
           -  s1.elem(0,0)*s2.elem(1,2)
           +  s1.elem(1,0)*s2.elem(0,2))*s3.elem(2,1);

  return d;
}









/*!
 * Performs:
 *  \f$dest^{k2,k1} = \sum_{i1,i2,j1,j2} \epsilon^{i1,j1,k1}\epsilon^{i2,j2,k2} Q1^{i1,i2} Q2^{j1,j2}\f$
 *
 * This routine is completely unrolled for 3 colors
 */
template<class T1, class T2>
typename BinaryReturn<PColorMatrixJIT<T1>, PColorMatrixJIT<T2>, FnQuarkContractXX>::Type_t
quarkContractXX(PColorMatrixJIT<T1> s1, PColorMatrixJIT<T2> s2)
{
  assert(s1.getN() == s2.getN());
  assert(s1.getN() == 3);
  
  auto d = binary_return<FnQuarkContractXX>( s1,s2 );

#if 1
  // Permutations: +(0,1,2)+(1,2,0)+(2,0,1)-(1,0,2)-(0,2,1)-(2,1,0)

  // k1 = 0, k2 = 0
  // d(0,0) = eps^{i1,j1,0}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(1,2,0),-(2,1,0)    +(1,2,0),-(2,1,0)
  d.elem(0,0) = s1.elem(1,1)*s2.elem(2,2)
              - s1.elem(1,2)*s2.elem(2,1)
              - s1.elem(2,1)*s2.elem(1,2)
              + s1.elem(2,2)*s2.elem(1,1);

  // k1 = 1, k2 = 0
  // d(0,1) = eps^{i1,j1,1}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(2,0,1),-(0,2,1)    +(1,2,0),-(2,1,0)    
  d.elem(0,1) = s1.elem(2,1)*s2.elem(0,2)
              - s1.elem(2,2)*s2.elem(0,1)
              - s1.elem(0,1)*s2.elem(2,2)
              + s1.elem(0,2)*s2.elem(2,1);

  // k1 = 2, k2 = 0
  // d(0,2) = eps^{i1,j1,2}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(0,1,2),-(1,0,2)    +(1,2,0),-(2,1,0)    
  d.elem(0,2) = s1.elem(0,1)*s2.elem(1,2)
              - s1.elem(0,2)*s2.elem(1,1)
              - s1.elem(1,1)*s2.elem(0,2)
              + s1.elem(1,2)*s2.elem(0,1);

  // k1 = 0, k2 = 1
  // d(1,0) = eps^{i1,j1,0}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(1,2,0),-(2,1,0)    +(2,0,1),-(0,2,1)
  d.elem(1,0) = s1.elem(1,2)*s2.elem(2,0)
              - s1.elem(1,0)*s2.elem(2,2)
              - s1.elem(2,2)*s2.elem(1,0)
              + s1.elem(2,0)*s2.elem(1,2);

  // k1 = 1, k2 = 1
  // d(1,1) = eps^{i1,j1,1}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(2,0,1),-(0,2,1)    +(2,0,1),-(0,2,1)
  d.elem(1,1) = s1.elem(2,2)*s2.elem(0,0)
              - s1.elem(2,0)*s2.elem(0,2)
              - s1.elem(0,2)*s2.elem(2,0)
              + s1.elem(0,0)*s2.elem(2,2);

  // k1 = 2, k2 = 1
  // d(1,1) = eps^{i1,j1,2}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(0,1,2),-(1,0,2)    +(2,0,1),-(0,2,1)
  d.elem(1,2) = s1.elem(0,2)*s2.elem(1,0)
              - s1.elem(0,0)*s2.elem(1,2)
              - s1.elem(1,2)*s2.elem(0,0)
              + s1.elem(1,0)*s2.elem(0,2);

  // k1 = 0, k2 = 2
  // d(2,0) = eps^{i1,j1,0}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(1,2,0),-(2,1,0)    +(0,1,2),-(1,0,2)
  d.elem(2,0) = s1.elem(1,0)*s2.elem(2,1)
              - s1.elem(1,1)*s2.elem(2,0)
              - s1.elem(2,0)*s2.elem(1,1)
              + s1.elem(2,1)*s2.elem(1,0);

  // k1 = 1, k2 = 2
  // d(2,1) = eps^{i1,j1,1}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(2,0,1),-(0,2,1)    +(0,1,2),-(1,0,2)
  d.elem(2,1) = s1.elem(2,0)*s2.elem(0,1)
              - s1.elem(2,1)*s2.elem(0,0)
              - s1.elem(0,0)*s2.elem(2,1)
              + s1.elem(0,1)*s2.elem(2,0);

  // k1 = 2, k2 = 2
  // d(2,2) = eps^{i1,j1,2}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(0,1,2),-(1,0,2)    +(0,1,2),-(1,0,2)
  d.elem(2,2) = s1.elem(0,0)*s2.elem(1,1)
              - s1.elem(0,1)*s2.elem(1,0)
              - s1.elem(1,0)*s2.elem(0,1)
              + s1.elem(1,1)*s2.elem(0,0);
  
#else
  
  typedef typename BinaryReturn<PColorMatrixJIT<T1>, PColorMatrixJIT<T2>, FnQuarkContractXX>::Type_t::Sub_t  T_return;

  JitStackMatrix< T1 , 3 > s1_a( s1 );
  JitStackMatrix< T2 , 3 > s2_a( s2 );
  JitStackMatrix< T_return , 3 > d_a;

  JitForLoop loop_j(0);
  JitForLoop loop_i(0);

  d_a.elemJIT( loop_i.index() , loop_j.index() ) =
				   s1_a.elemJIT( llvm_epsilon_1st( 1 , loop_j.index() ) , llvm_epsilon_2nd( 1 , loop_i.index() ) ) *
				   s2_a.elemJIT( llvm_epsilon_1st( 2 , loop_j.index() ) , llvm_epsilon_2nd( 2 , loop_i.index() ) )
				 - s1_a.elemJIT( llvm_epsilon_1st( 1 , loop_j.index() ) , llvm_epsilon_2nd( 2 , loop_i.index() ) ) *
				   s2_a.elemJIT( llvm_epsilon_1st( 2 , loop_j.index() ) , llvm_epsilon_2nd( 1 , loop_i.index() ) )
				 - s1_a.elemJIT( llvm_epsilon_1st( 2 , loop_j.index() ) , llvm_epsilon_2nd( 1 , loop_i.index() ) ) *
				   s2_a.elemJIT( llvm_epsilon_1st( 1 , loop_j.index() ) , llvm_epsilon_2nd( 2 , loop_i.index() ) )
				 + s1_a.elemJIT( llvm_epsilon_1st( 2 , loop_j.index() ) , llvm_epsilon_2nd( 2 , loop_i.index() ) ) *
      				   s2_a.elemJIT( llvm_epsilon_1st( 1 , loop_j.index() ) , llvm_epsilon_2nd( 1 , loop_i.index() ) );
				 
  loop_i.end();
  loop_j.end();

  for(int i=0; i < 3; ++i)
    for(int j=0; j < 3; ++j)
      d.elem(i,j).setup( d_a.elemJIT(i,j) );


#endif
  return d;
}




} // namespace QDP

#endif
