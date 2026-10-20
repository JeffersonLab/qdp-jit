#ifndef QDP_PRIMSPINVECJIT_H
#define QDP_PRIMSPINVECJIT_H


namespace QDP {


template <class T> class PSpinVectorJIT: public BaseJIT< T >
{
public:
  PSpinVectorJIT(){}

  PSpinVectorJIT( llvm::Value * base , JitDataFormat fmt , NestedIndex args , Shape shape ) : BaseJIT< T >(base,fmt,args,shape) {}

  // for eval Shape
  PSpinVectorJIT( Shape shape ) : BaseJIT< T >(shape) {}

  
  Shape::Axis axis() const { return Shape::Axis::Spin; }
  
  Shape shape() const { return BaseJIT< T >::shape(); }

  int getN() const
  {
    return shape().extent(axis());
  }

  int this_size() const
  {
    return getN();
  }

  
  template<class T1> 
  PSpinVectorJIT operator=( PSpinVectorJIT<T1> rhs)
  {
    JitForLoop loop( 0 , getN() );
    {
      this->elem( loop.index() ) = rhs.elem( loop.index() );
    }
    loop.end();
    
    return *this;
  }


  PSpinVectorJIT operator=( PSpinVectorJIT rhs)
  {
    JitForLoop loop( 0 , getN() );
    {
      this->elem( loop.index() ) = rhs.elem( loop.index() );
    }
    loop.end();
    
    return *this;
  }



  template<class T1>
  PSpinVectorJIT operator+=(PSpinVectorJIT<T1> rhs) 
    {
      JitForLoop loop( 0 , getN() );
      {
	this->elem( loop.index() ) += rhs.elem( loop.index() );
      }
      loop.end();

      return *this;
    }


  template<class T1>
  PSpinVectorJIT operator*=(PScalarJIT<T1> rhs) 
    {
      JitForLoop loop( 0 , getN() );
      {
	this->elem( loop.index() ) *= rhs.elem();
      }
      loop.end();

      return *this;
    }

  template<class T1>
  PSpinVectorJIT operator-=(PSpinVectorJIT<T1> rhs) 
    {
      JitForLoop loop( 0 , getN() );
      {
	this->elem( loop.index() ) -= rhs.elem( loop.index() );
      }
      loop.end();

      return *this;
    }


  template<class T1>
  PSpinVectorJIT operator/=(PScalarJIT<T1> rhs) 
    {
      JitForLoop loop( 0 , getN() );
      {
	this->elem( loop.index() ) /= rhs.elem();
      }
      loop.end();

      return *this;
    }

};




template<class T>
class WordSize< PSpinVectorJIT<T> >
{
public:
  static int value( const Shape& ret )
  {
    return ret.ns() * WordSize<T>::value( ret );
  }
};





template <class T>
struct IsWordVec< PSpinVectorJIT<T > >
{
  constexpr static bool value = IsWordVec<T>::value;
};


template<class T1>
struct ScalarType< PSpinVectorJIT<T1> >
{
  typedef PSpinVectorJIT< typename ScalarType<T1>::Type_t  > Type_t;
};




template<class T1>
struct BASEType<PSpinVectorJIT<T1> > 
{
  typedef PSpinVector<typename BASEType<T1>::Type_t>  Type_t;
};



template<class T1>
struct WordType<PSpinVectorJIT<T1> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};


template<class T1>
struct SinglePrecType< PSpinVectorJIT<T1> > 
{
  typedef PSpinVectorJIT< typename SinglePrecType<T1>::Type_t> Type_t;
};

template<class T1>
struct DoublePrecType< PSpinVectorJIT<T1> > 
{
  typedef PSpinVectorJIT< typename DoublePrecType<T1>::Type_t> Type_t;
};


template<class T>
struct InternalScalar<PSpinVectorJIT<T> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};


template<class T>
struct PrimitiveScalar<PSpinVectorJIT<T> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};


template<class T>
struct LatticeScalar<PSpinVectorJIT<T> > {
  typedef PSpinVectorJIT<typename LatticeScalar<T>::Type_t>  Type_t;
};






template<class T1, class Op>
struct UnaryReturn<PSpinVectorJIT<T1>, Op> {
  typedef PSpinVectorJIT<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<PScalarJIT<T1>, PSpinVectorJIT<T2>, Op> {
  typedef PSpinVectorJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn< PSpinMatrixJIT<T1>, PSpinVectorJIT<T2>, Op> {
  typedef PSpinVectorJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};



template<class T1, class T2, template <class> class C1, class Op>
struct BinaryReturn< PMatrixJIT<T1,C1>, PSpinVectorJIT<T2>, Op> {
  typedef PSpinVectorJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};



template<class T1, class T2, class Op>
struct BinaryReturn<PSpinVectorJIT<T1>, PScalarJIT<T2>, Op> {
  typedef PSpinVectorJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<PSpinVectorJIT<T1>, PSpinVectorJIT<T2>, Op> {
  typedef PSpinVectorJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

  




template<class T1>
typename UnaryReturn<PSpinVectorJIT<T1>, OpUnaryPlus>::Type_t
operator+(PSpinVectorJIT<T1> l)
{
  auto d = unary_return<OpUnaryPlus>( l );
 
  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = +l.elem(i);
  }
  i.end();
  
  return d;
}


template<class T1>
typename UnaryReturn<PSpinVectorJIT<T1>, OpUnaryMinus>::Type_t
operator-(PSpinVectorJIT<T1> l)
{
  auto d = unary_return<OpUnaryMinus>( l );
  
  JitForLoop loop( 0 , l.getN() );
  {
    d.elem( loop.index() ) = -l.elem( loop.index() );
  }
  loop.end();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<PSpinVectorJIT<T1>, PSpinVectorJIT<T2>, OpAdd>::Type_t
operator+(PSpinVectorJIT<T1> l, PSpinVectorJIT<T2> r)
{
  assert(l.getN() == r.getN());
  auto d = binary_return<OpAdd>( l , r );
  
  JitForLoop loop( 0 , l.getN() );
  {
    d.elem( loop.index() ) = l.elem( loop.index() ) + r.elem( loop.index() );
  }
  loop.end();

  return d;
}

  

template<class T1, class T2>
typename BinaryReturn<PSpinVectorJIT<T1>, PSpinVectorJIT<T2>, OpSubtract>::Type_t
operator-(PSpinVectorJIT<T1> l, PSpinVectorJIT<T2> r)
{
  assert(l.getN() == r.getN());
  auto d = binary_return<OpSubtract>( l , r );

  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = l.elem(i) - r.elem(i);
  }
  i.end();
  
  return d;
}



template<class T1, class T2>
typename BinaryReturn<PSpinVectorJIT<T1>, PScalarJIT<T2>, OpMultiply>::Type_t
operator*(PSpinVectorJIT<T1> l, const PScalarJIT<T2>& r)
{
  auto d = binary_return<OpMultiply>( l , r );

  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = l.elem(i) * r.elem();
  }
  i.end();
  
  return d;
}


template<class T1, class T2>
typename BinaryReturn<PSpinVectorJIT<T1>, PScalarJIT<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(PSpinVectorJIT<T1> l, const PScalarJIT<T2>& r)
{
  auto d = binary_return<OpMultiplyAdj>( l , r );

  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = multiplyAdj(l.elem(i), r.elem());
  }
  i.end();
  
  return d;
}



template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PSpinVectorJIT<T2>, OpMultiply>::Type_t
operator*(PScalarJIT<T1> l, PSpinVectorJIT<T2> r)
{
  auto d = binary_return<OpMultiply>( l , r );

  JitForLoop loop( 0 , r.getN() );
  {
    d.elem( loop.index() ) = l.elem() * r.elem( loop.index() );
  }
  loop.end();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PSpinVectorJIT<T2>, OpAdjMultiply>::Type_t
adjMultiply(PScalarJIT<T1> l, PSpinVectorJIT<T2> r)
{
  auto d = binary_return<OpAdjMultiply>( l , r );
  
  JitForLoop i( 0 , r.getN() );
  {
    d.elem( i.index() ) = adjMultiply( l.elem() , r.elem( i.index() ) );
  }
  i.end();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<PSpinMatrixJIT<T1>, PSpinVectorJIT<T2>, OpMultiply>::Type_t
operator*(const PSpinMatrixJIT<T1>& l, PSpinVectorJIT<T2> r)
{
  assert(l.getN() == r.getN());
  auto d = binary_return<OpAdjMultiply>( l , r );
  
  JitForLoop i( 0 , l.getN() );
  {
    d.elem( i.index() ) = l.elem( i.index() , 0 ) * r.elem( 0 );
    JitForLoop j( 1 , l.getN() );
    {
      d.elem( i.index() ) += l.elem( i.index() , j.index() ) * r.elem( j.index() );
    }
    j.end();
  }
  i.end();

  return d;
}



template<class T1, class T2,  template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PSpinVectorJIT<T2>, OpMultiply>::Type_t
operator*(const PSpinMatrixJIT<T1>& l, PSpinVectorJIT<T2> r)
{
  assert(l.getN() == r.getN());
  auto d = binary_return<OpMultiply>( l , r );
  
  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = l.elem(i,0) * r.elem(0);
    JitForLoop j( 1 , l.getN() );
    {
      d.elem(i) += l.elem(i,j) * r.elem(j);
    }
    j.end();
  }
  i.end();
  
  return d;
}


template<class T1, class T2>
typename BinaryReturn<PSpinMatrixJIT<T1>, PSpinVectorJIT<T2>, OpAdjMultiply>::Type_t
adjMultiply(const PSpinMatrixJIT<T1>& l, PSpinVectorJIT<T2> r)
{
  assert(l.getN() == r.getN());
  auto d = binary_return<OpAdjMultiply>( l , r );
  
  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = adjMultiply( l.elem(0,i) , r.elem(0) );
    JitForLoop j( 1 , l.getN() );
    {
      d.elem(i) += adjMultiply( l.elem(j,i) , r.elem(j) );
    }
    j.end();
  }
  i.end();

  return d;
}



template<class T1, class T2, template<class> class C1>
typename BinaryReturn<PMatrixJIT<T1,C1>, PSpinVectorJIT<T2>, OpAdjMultiply>::Type_t
adjMultiply(const PMatrixJIT<T1,C1>& l, PSpinVectorJIT<T2> r)
{
  assert(l.getN() == r.getN());
  auto d = binary_return<OpAdjMultiply>( l , r );
  
  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = adjMultiply( l.elem(0,i) , r.elem(0) );
    JitForLoop j( 1 , l.getN() );
    {
      d.elem(i) += adjMultiply( l.elem(j,i) , r.elem(j) );
    }
    j.end();
  }
  i.end();

  return d;
}

template<class T1, class T2>
typename BinaryReturn<PSpinVectorJIT<T1>, PScalarJIT<T2>, OpDivide>::Type_t
operator/(PSpinVectorJIT<T1> l, const PScalarJIT<T2>& r)
{
  auto d = binary_return<OpDivide>( l , r );
  
  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = l.elem(i) / r.elem();
  }
  i.end();
  
  return d;
}




template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnReal>::Type_t
real(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnReal>( s1 );
  
  JitForLoop i( 0 , s1.getN() );
  {
    d.elem(i) = real(s1.elem(i));
  }
  i.end();
  
  return d;
}



template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnImag>::Type_t
imag(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnImag>( s1 );
  
  JitForLoop i( 0 , s1.getN() );
  {
    d.elem(i) = imag(s1.elem(i));
  }
  i.end();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<PSpinVectorJIT<T1>, PSpinVectorJIT<T2>, FnCmplx>::Type_t
cmplx(PSpinVectorJIT<T1> s1, PSpinVectorJIT<T2> s2)
{
  assert(s1.getN() == s2.getN());
  auto d = binary_return<FnCmplx>( s1 , s2 );
  
  JitForLoop i( 0 , s1.getN() );
  {
    d.elem(i) = cmplx(s1.elem(i), s2.elem(i));
  }
  i.end();
  
  return d;
}



template<class T1>
struct UnaryReturn<PSpinVectorJIT<T1>, FnIsFinite> {
  typedef PScalarJIT< typename UnaryReturn<T1, FnIsFinite>::Type_t > Type_t;
};


  
template<class T1>
typename UnaryReturn<PSpinVectorJIT<T1>, FnIsFinite>::Type_t
isfinite(PSpinVectorJIT<T1> l)
{
  auto d = unary_return<FnIsFinite>( l );

  d.elem() = isfinite(l.elem(0));

  JitForLoop i( 0 , l.getN() );
  {
    d.elem() &= isfinite(l.elem(i));
  }
  i.end();
  
  return d;
}

  





template<class T1>
typename UnaryReturn<PSpinVectorJIT<T1>, FnConjugate>::Type_t
conj(PSpinVectorJIT<T1> l)
{
  auto d = unary_return<FnConjugate>( l );
  
  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = conj(l.elem(i));
  }
  i.end();
  
  return d;
}


template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnTimesI>::Type_t
timesI(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnTimesI>( s1 );
  
  JitForLoop i( 0 , s1.getN() );
  {
    d.elem(i) = timesI(s1.elem(i));
  }
  i.end();
  
  return d;
}


template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnTimesMinusI>::Type_t
timesMinusI(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnTimesMinusI>( s1 );
  
  JitForLoop i( 0 , s1.getN() );
  {
    d.elem(i) = timesMinusI(s1.elem(i));
  }
  i.end();
  
  return d;
}




template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnPeekColorVectorJIT >::Type_t
peekColor(PSpinVectorJIT<T> l, llvm::Value* row)
{
  auto d = unary_return<FnPeekColorVectorJIT>(l);

  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = peekColor(l.elem(i),row);
  }
  i.end();
  
  return d;
}



template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnPeekColorMatrixJIT >::Type_t
peekColor(PSpinVectorJIT<T> l, llvm::Value* row, llvm::Value* col)
{
  auto d = unary_return<FnPeekColorMatrixJIT>(l);

  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = peekColor(l.elem(i),row,col);
  }
  i.end();
  
  return d;
}



template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnPeekSpinVectorJIT > {
  typedef PScalarJIT<T>  Type_t;
};


template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnPeekSpinVector > {
  typedef PScalarJIT<T>  Type_t;
};



template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnPeekSpinVectorJIT>::Type_t
peekSpin(PSpinVectorJIT<T> l, llvm::Value* row)
{
  auto d = unary_return<FnPeekSpinVectorJIT>(l);

  d.elem() = l.elem(row);

  return d;
}




template<class T1, class T2>
void
pokeColor(PSpinVectorJIT<T1> l, PSpinVectorJIT<T2> r, llvm::Value* row)
{
  JitForLoop i( 0 , l.getN() );
  {
    pokeColor(l.elem(i),r.elem(i),row);
  }
  i.end();
}



template<class T1, class T2>
void
pokeColor(PSpinVectorJIT<T1> l, PSpinVectorJIT<T2> r, llvm::Value* row, llvm::Value* col)
{
  JitForLoop i( 0 , l.getN() );
  {
    pokeColor(l.elem(i),r.elem(i),row,col);
  }
  i.end();
}



template<class T1, class T2>
void
pokeSpin(PSpinVectorJIT<T1> l, PScalarJIT<T2> r, llvm::Value* row)
{
  l.elem(row) = r.elem();
}



template<class T>
void zero_rep(PSpinVectorJIT<T> dest) 
{
  JitForLoop i( 0 , dest.getN() );
  {
    zero_rep(dest.elem(i));
  }
  i.end();
}



template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnLocalNorm2>::Type_t
localNorm2(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnLocalNorm2>( s1 );
  
  d.elem() = localNorm2(s1.elem(0));
  JitForLoop i( 1 , s1.getN() );
  {
    d.elem() += localNorm2(s1.elem(i));
  }
  i.end();

  return d;
}



template<class T1, class T2, class T3>
struct TrinaryReturn<PScalarJIT<T1>, PSpinVectorJIT<T2>, PSpinVectorJIT<T3>, FnWhere> {
  typedef PSpinVectorJIT<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};



template<class T1, class T2, class T3>
typename TrinaryReturn<PScalarJIT<T1>, PSpinVectorJIT<T2>, PSpinVectorJIT<T3>, FnWhere>::Type_t
where(PScalarJIT<T1> a, PSpinVectorJIT<T2> b, PSpinVectorJIT<T3> c)
{
  auto d = trinary_return<FnWhere>( a,b,c );
  
  JitForLoop i( 0 , b.getN() );
  {
    d.elem(i) = where(a.elem(), b.elem(i), c.elem(i));
  }
  i.end();
  
  return d;
}



template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};


template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnLocalNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<PSpinVectorJIT<T1>, PSpinVectorJIT<T2>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<PSpinVectorJIT<T1>, PSpinVectorJIT<T2>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<PSpinVectorJIT<T1>, PSpinVectorJIT<T2>, FnInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<PSpinVectorJIT<T1>, PSpinVectorJIT<T2>, FnLocalInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};


template<class T1, class T2>
PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>
localInnerProduct(PSpinVectorJIT<T1> s1, PSpinVectorJIT<T2> s2)
{
  auto d = binary_return<FnLocalInnerProduct>( s1 , s2 );
  
  d.elem() = localInnerProduct(s1.elem(0), s2.elem(0));
  JitForLoop i( 1 , s1.getN() );
  {
    d.elem() += localInnerProduct(s1.elem(i), s2.elem(i));
  }
  i.end();
  
  return d;
}


template<class T1, class T2>
PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>
localInnerProductReal(PSpinVectorJIT<T1> s1, PSpinVectorJIT<T2> s2)
{
  auto d = binary_return<FnLocalInnerProductReal>( s1 , s2 );

  d.elem() = localInnerProductReal(s1.elem(0), s2.elem(0));
  JitForLoop i( 1 , s1.getN() );
  {
    d.elem() += localInnerProductReal(s1.elem(i), s2.elem(i));
  }
  i.end();
  
  return d;
}


template<class T2>
struct BinaryReturn<GammaType, PSpinVectorJIT<T2>, OpGammaTypeMultiply> {
  typedef PSpinVectorJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};


template<class T2>
struct BinaryReturn<GammaTypeDP, PSpinVectorJIT<T2>, OpGammaTypeMultiply> {
  typedef PSpinVectorJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};




template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinProject > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinProject>::Type_t >  Type_t;
};


template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir0Plus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinProjectDir0Plus>::Type_t >  Type_t;
};


template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir1Plus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinProjectDir1Plus>::Type_t >  Type_t;
};


template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir2Plus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinProjectDir2Plus>::Type_t >  Type_t;
};

template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir3Plus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinProjectDir3Plus>::Type_t >  Type_t;
};

template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir0Minus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinProjectDir0Minus>::Type_t > Type_t;
};

template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir1Minus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinProjectDir1Minus>::Type_t >  Type_t;
};

template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir2Minus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinProjectDir2Minus>::Type_t >  Type_t;
};

template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir3Minus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinProjectDir3Minus>::Type_t >  Type_t;
};



template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstruct > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinReconstruct>::Type_t >  Type_t;
};


template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir0Plus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinReconstructDir0Plus>::Type_t >  Type_t;
};


template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir1Plus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinReconstructDir1Plus>::Type_t >  Type_t;
};

template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir2Plus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinReconstructDir2Plus>::Type_t >  Type_t;
};

template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir3Plus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinReconstructDir3Plus>::Type_t >  Type_t;
};

template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir0Minus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinReconstructDir0Minus>::Type_t >  Type_t;
};

template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir1Minus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinReconstructDir1Minus>::Type_t >  Type_t;
};

template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir2Minus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinReconstructDir2Minus>::Type_t >  Type_t;
};

template<class T>
struct UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir3Minus > {
  typedef PSpinVectorJIT<typename UnaryReturn<T, FnSpinReconstructDir3Minus>::Type_t >  Type_t;
};



template<class T,  class T1, class T2, class T3>
void
fill_random_jit(PSpinVectorJIT<T> d, T1 seed, T2 skewed_seed, const T3& seed_mult)
{
  JitForLoop i( 0 , d.getN() );
  {
    fill_random_jit(d.elem(i), seed, skewed_seed, seed_mult);
  }
  i.end();
}



template<class T>
void
fill_gaussian_jit(PSpinVectorJIT<T> d, PSpinVectorJIT<T> r1, PSpinVectorJIT<T> r2)
{
  JitForLoop i( 0 , d.getN() );
  {
    fill_gaussian_jit(d.elem(i), r1.elem(i), r2.elem(i));
  }
  i.end();
}


template<class T2>
typename BinaryReturn<GammaType, PSpinVectorJIT<T2>, OpGammaTypeMultiply>::Type_t
operator*(const GammaType& g, PSpinVectorJIT<T2> r)
{
  auto d = binary_return<OpGammaTypeMultiply>(g,r);

  switch ( g.elem() ) {
  case 0:
    d.elem(0) =  r.elem(0);
    d.elem(1) =  r.elem(1);
    d.elem(2) =  r.elem(2);
    d.elem(3) =  r.elem(3);
    break;
  case 1:
    d.elem(0) = timesI(r.elem(3));
    d.elem(1) = timesI(r.elem(2));
    d.elem(2) = timesMinusI(r.elem(1));
    d.elem(3) = timesMinusI(r.elem(0));
    break;
  case 2:
    d.elem(0) = -r.elem(3);
    d.elem(1) =  r.elem(2);
    d.elem(2) =  r.elem(1);
    d.elem(3) = -r.elem(0);
    break;
  case 3:
    d.elem(0) = timesMinusI(r.elem(0));
    d.elem(1) = timesI(r.elem(1));
    d.elem(2) = timesMinusI(r.elem(2));
    d.elem(3) = timesI(r.elem(3));
    break;
  case 4:
    d.elem(0) = timesI(r.elem(2));
    d.elem(1) = timesMinusI(r.elem(3));
    d.elem(2) = timesMinusI(r.elem(0));
    d.elem(3) = timesI(r.elem(1));
    break;
  case 5:
    d.elem(0) = -r.elem(1);
    d.elem(1) =  r.elem(0);
    d.elem(2) = -r.elem(3);
    d.elem(3) =  r.elem(2);
    break;
  case 6:
    d.elem(0) = timesMinusI(r.elem(1));
    d.elem(1) = timesMinusI(r.elem(0));
    d.elem(2) = timesMinusI(r.elem(3));
    d.elem(3) = timesMinusI(r.elem(2));
    break;
  case 7:
    d.elem(0) =  r.elem(2);
    d.elem(1) =  r.elem(3);
    d.elem(2) = -r.elem(0);
    d.elem(3) = -r.elem(1);
    break;
  case 8:
    d.elem(0) =  r.elem(2);
    d.elem(1) =  r.elem(3);
    d.elem(2) =  r.elem(0);
    d.elem(3) =  r.elem(1);
    break;
  case 9:
    d.elem(0) = timesI(r.elem(1));
    d.elem(1) = timesI(r.elem(0));
    d.elem(2) = timesMinusI(r.elem(3));
    d.elem(3) = timesMinusI(r.elem(2));
    break;
  case 10:
    d.elem(0) = -r.elem(1);
    d.elem(1) =  r.elem(0);
    d.elem(2) =  r.elem(3);
    d.elem(3) = -r.elem(2);
    break;
  case 11:
    d.elem(0) = timesMinusI(r.elem(2));
    d.elem(1) = timesI(r.elem(3));
    d.elem(2) = timesMinusI(r.elem(0));
    d.elem(3) = timesI(r.elem(1));
    break;
  case 12:
    d.elem(0) = timesI(r.elem(0));
    d.elem(1) = timesMinusI(r.elem(1));
    d.elem(2) = timesMinusI(r.elem(2));
    d.elem(3) = timesI(r.elem(3));
    break;
  case 13:
    d.elem(0) = -r.elem(3);
    d.elem(1) =  r.elem(2);
    d.elem(2) = -r.elem(1);
    d.elem(3) =  r.elem(0);
    break;
  case 14:
    d.elem(0) = timesMinusI(r.elem(3));
    d.elem(1) = timesMinusI(r.elem(2));
    d.elem(2) = timesMinusI(r.elem(1));
    d.elem(3) = timesMinusI(r.elem(0));
    break;
  case 15:
    d.elem(0) =  r.elem(0);
    d.elem(1) =  r.elem(1);
    d.elem(2) = -r.elem(2);
    d.elem(3) = -r.elem(3);
    break;
  default:
    QDPIO::cout << "Invalid gamma matrix number: " << g.elem() << std::endl;
    QDP_abort(1);
  }
  
  return d;
}




template<class T2>
typename BinaryReturn<GammaTypeDP, PSpinVectorJIT<T2>, OpGammaTypeMultiply>::Type_t
operator*(const GammaTypeDP& g, PSpinVectorJIT<T2> r)
{
  auto d = binary_return<OpGammaTypeMultiply>(g,r);

  switch ( g.elem() ) {
  case 0:
    d.elem(0) =  r.elem(0);
    d.elem(1) =  r.elem(1);
    d.elem(2) =  r.elem(2);
    d.elem(3) =  r.elem(3);
    break;
  case 1:
    d.elem(0) = timesMinusI(r.elem(3));
    d.elem(1) = timesMinusI(r.elem(2));
    d.elem(2) = timesI(r.elem(1));
    d.elem(3) = timesI(r.elem(0));
    break;
  case 2:
    d.elem(0) = -r.elem(3);
    d.elem(1) =  r.elem(2);
    d.elem(2) =  r.elem(1);
    d.elem(3) = -r.elem(0);
    break;
  case 3:
    d.elem(0) = timesI(r.elem(0));
    d.elem(1) = timesMinusI(r.elem(1));
    d.elem(2) = timesI(r.elem(2));
    d.elem(3) = timesMinusI(r.elem(3));
    break;
  case 4:
    d.elem(0) = timesMinusI(r.elem(2));
    d.elem(1) = timesI(r.elem(3));
    d.elem(2) = timesI(r.elem(0));
    d.elem(3) = timesMinusI(r.elem(1));
    break;
  case 5:
    d.elem(0) = -r.elem(1);
    d.elem(1) =  r.elem(0);
    d.elem(2) = -r.elem(3);
    d.elem(3) =  r.elem(2);
    break;
  case 6:
    d.elem(0) = timesI(r.elem(1));
    d.elem(1) = timesI(r.elem(0));
    d.elem(2) = timesI(r.elem(3));
    d.elem(3) = timesI(r.elem(2));
    break;
  case 7:
    d.elem(0) =  r.elem(2);
    d.elem(1) =  r.elem(3);
    d.elem(2) = -r.elem(0);
    d.elem(3) = -r.elem(1);
    break;
  case 8:
    d.elem(0) =  r.elem(0);
    d.elem(1) =  r.elem(1);
    d.elem(2) = -r.elem(2);
    d.elem(3) = -r.elem(3);
    break;
  case 9:
    d.elem(0) = timesI(r.elem(3));
    d.elem(1) = timesI(r.elem(2));
    d.elem(2) = timesI(r.elem(1));
    d.elem(3) = timesI(r.elem(0));
    break;
  case 10:
    d.elem(0) =  r.elem(3);
    d.elem(1) = -r.elem(2);
    d.elem(2) =  r.elem(1);
    d.elem(3) = -r.elem(0);
    break;
  case 11:
    d.elem(0) = timesI(r.elem(0));
    d.elem(1) = timesMinusI(r.elem(1));
    d.elem(2) = timesMinusI(r.elem(2));
    d.elem(3) = timesI(r.elem(3));
    break;
  case 12:
    d.elem(0) = timesI(r.elem(2));
    d.elem(1) = timesMinusI(r.elem(3));
    d.elem(2) = timesI(r.elem(0));
    d.elem(3) = timesMinusI(r.elem(1));
    break;
  case 13:
    d.elem(0) = -r.elem(1);
    d.elem(1) =  r.elem(0);
    d.elem(2) =  r.elem(3);
    d.elem(3) = -r.elem(2);
    break;
  case 14:
    d.elem(0) = timesI(r.elem(1));
    d.elem(1) = timesI(r.elem(0));
    d.elem(2) = timesMinusI(r.elem(3));
    d.elem(3) = timesMinusI(r.elem(2));
    break;
  case 15:
    d.elem(0) = -r.elem(2);
    d.elem(1) = -r.elem(3);
    d.elem(2) = -r.elem(0);
    d.elem(3) = -r.elem(1);
    break;
  default:
    QDPIO::cout << "Invalid gamma matrix number: " << g.elem() << std::endl;
    QDP_abort(1);
  }
  
  return d;
}

  


template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir0Minus>::Type_t
spinProjectDir0Minus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir0Minus>( s1 );

  /*                              ( 1  0  0 -i)  ( a0 )    ( a0 - i a3 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -i  0)  ( a1 )  = ( a1 - i a2 )
   *                    0         ( 0  i  1  0)  ( a2 )    ( a2 + i a1 )
   *                              ( i  0  0  1)  ( a3 )    ( a3 + i a0 )

   * Therefore the top components are

   *      ( b0r + i b0i )  =  ( {a0r + a3i} + i{a0i - a3r} )
   *      ( b1r + i b1i )     ( {a1r + a2i} + i{a1i - a2r} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r - a1i} + i{a2i + a1r} )  =  ( - b1i + i b1r )
   *      ( b3r + i b3i )     ( {a3r - a0i} + i{a3i + a0r} )     ( - b0i + i b0r ) 
   */
  d.elem(0) = s1.elem(0) - timesI(s1.elem(3));
  d.elem(1) = s1.elem(1) - timesI(s1.elem(2));

  return d;
}

template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir1Minus>::Type_t
spinProjectDir1Minus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir1Minus>( s1 );

  /*                              ( 1  0  0  1)  ( a0 )    ( a0 + a3 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -1  0)  ( a1 )  = ( a1 - a2 )
   *                    1         ( 0 -1  1  0)  ( a2 )    ( a2 - a1 )
   *                              ( 1  0  0  1)  ( a3 )    ( a3 + a0 )
	 
   * Therefore the top components are
      
   *      ( b0r + i b0i )  =  ( {a0r + a3r} + i{a0i + a3i} )
   *      ( b1r + i b1i )     ( {a1r - a2r} + i{a1i - a2i} )
      
   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r - a1r} + i{a2i - a1i} )  =  ( - b1r - i b1i )
   *      ( b3r + i b3i )     ( {a3r + a0r} + i{a3i + a0i} )     (   b0r + i b0i ) 
   */
  d.elem(0) = s1.elem(0) + s1.elem(3);
  d.elem(1) = s1.elem(1) - s1.elem(2);

  return d;
}
    
template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir2Minus>::Type_t
spinProjectDir2Minus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir2Minus>( s1 );

  /*                              ( 1  0 -i  0)  ( a0 )    ( a0 - i a2 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1  0  i)  ( a1 )  = ( a1 + i a3 )
   *                    2         ( i  0  1  0)  ( a2 )    ( a2 + i a0 )
   *                              ( 0 -i  0  1)  ( a3 )    ( a3 - i a1 )

   * Therefore the top components are
      
   *      ( b0r + i b0i )  =  ( {a0r + a2i} + i{a0i - a2r} )
   *      ( b1r + i b1i )     ( {a1r - a3i} + i{a1i + a3r} )
      
   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r - a0i} + i{a2i + a0r} )  =  ( - b0i + i b0r )
   *      ( b3r + i b3i )     ( {a3r + a1i} + i{a3i - a1r} )     (   b1i - i b1r )
   */
  d.elem(0) = s1.elem(0) - timesI(s1.elem(2));
  d.elem(1) = s1.elem(1) + timesI(s1.elem(3));

  return d;
}
    
template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir3Minus>::Type_t
spinProjectDir3Minus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir3Minus>( s1 );

  /*                              ( 1  0 -1  0)  ( a0 )    ( a0 - a2 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1  0 -1)  ( a1 )  = ( a1 - a3 )
   *                    3         (-1  0  1  0)  ( a2 )    ( a2 - a0 )
   *                              ( 0 -1  0  1)  ( a3 )    ( a3 - a1 )
      
   * Therefore the top components are
      
   *      ( b0r + i b0i )  =  ( {a0r - a2r} + i{a0i - a2i} )
   *      ( b1r + i b1i )     ( {a1r - a3r} + i{a1i - a3i} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r - a0r} + i{a2i - a0i} )  =  ( - b0r - i b0i )
   *      ( b3r + i b3i )     ( {a3r - a1r} + i{a3i - a1i} )     ( - b1r - i b1i ) 
   */
  d.elem(0) = s1.elem(0) - s1.elem(2);
  d.elem(1) = s1.elem(1) - s1.elem(3);

  return d;
}

template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir0Plus>::Type_t
spinProjectDir0Plus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir0Plus>( s1 );

  /*                              ( 1  0  0 +i)  ( a0 )    ( a0 + i a3 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1 +i  0)  ( a1 )  = ( a1 + i a2 )
   *                    0         ( 0 -i  1  0)  ( a2 )    ( a2 - i a1 )
   *                              (-i  0  0  1)  ( a3 )    ( a3 - i a0 )

   * Therefore the top components are

   *      ( b0r + i b0i )  =  ( {a0r - a3i} + i{a0i + a3r} )
   *      ( b1r + i b1i )     ( {a1r - a2i} + i{a1i + a2r} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r + a1i} + i{a2i - a1r} )  =  ( b1i - i b1r )
   *      ( b3r + i b3i )     ( {a3r + a0i} + i{a3i - a0r} )     ( b0i - i b0r ) 
   */
  d.elem(0) = s1.elem(0) + timesI(s1.elem(3));
  d.elem(1) = s1.elem(1) + timesI(s1.elem(2));

  return d;
}

template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir1Plus>::Type_t
spinProjectDir1Plus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir1Plus>( s1 );

  /*                              ( 1  0  0 -1)  ( a0 )    ( a0 - a3 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  1  0)  ( a1 )  = ( a1 + a2 )
   *                    1         ( 0  1  1  0)  ( a2 )    ( a2 + a1 )
   *                              (-1  0  0  1)  ( a3 )    ( a3 - a0 )

   * Therefore the top components are

   *      ( b0r + i b0i )  =  ( {a0r - a3r} + i{a0i - a3i} )
   *      ( b1r + i b1i )     ( {a1r + a2r} + i{a1i + a2i} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r + a1r} + i{a2i + a1i} )  =  (   b1r + i b1i )
   *      ( b3r + i b3i )     ( {a3r - a0r} + i{a3i - a0i} )     ( - b0r - i b0i ) 
   */
  d.elem(0) = s1.elem(0) - s1.elem(3);
  d.elem(1) = s1.elem(1) + s1.elem(2);

  return d;
}

template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir2Plus>::Type_t
spinProjectDir2Plus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir2Plus>( s1 );

  /*                              ( 1  0  i  0)  ( a0 )    ( a0 + i a2 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0 -i)  ( a1 )  = ( a1 - i a3 )
   *                    2         (-i  0  1  0)  ( a2 )    ( a2 - i a0 )
   *                              ( 0  i  0  1)  ( a3 )    ( a3 + i a1 )

   * Therefore the top components are

   *      ( b0r + i b0i )  =  ( {a0r - a2i} + i{a0i + a2r} )
   *      ( b1r + i b1i )     ( {a1r + a3i} + i{a1i - a3r} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r + a0i} + i{a2i - a0r} )  =  (   b0i - i b0r )
   *      ( b3r + i b3i )     ( {a3r - a1i} + i{a3i + a1r} )     ( - b1i + i b1r ) 
   */
  d.elem(0) = s1.elem(0) + timesI(s1.elem(2));
  d.elem(1) = s1.elem(1) - timesI(s1.elem(3));

  return d;
}

template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinProjectDir3Plus>::Type_t
spinProjectDir3Plus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir3Plus>( s1 );

  /*                              ( 1  0  1  0)  ( a0 )    ( a0 + a2 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0  1)  ( a1 )  = ( a1 + a3 )
   *                    3         ( 1  0  1  0)  ( a2 )    ( a2 + a0 )
   *                              ( 0  1  0  1)  ( a3 )    ( a3 + a1 )

   * Therefore the top components are

   *      ( b0r + i b0i )  =  ( {a0r + a2r} + i{a0i + a2i} )
   *      ( b1r + i b1i )     ( {a1r + a3r} + i{a1i + a3i} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r + a0r} + i{a2i + a0i} )  =  ( b0r + i b0i )
   *      ( b3r + i b3i )     ( {a3r + a1r} + i{a3i + a1i} )     ( b1r + i b1i ) 
   */
  d.elem(0) = s1.elem(0) + s1.elem(2);
  d.elem(1) = s1.elem(1) + s1.elem(3);

  return d;
}



template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir0Minus>::Type_t
spinReconstructDir0Minus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir0Minus>( s1 );

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = timesI(s1.elem(1));
  d.elem(3) = timesI(s1.elem(0));

  return d;
}

template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir1Minus>::Type_t
spinReconstructDir1Minus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir1Minus>( s1 );

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = -s1.elem(1);
  d.elem(3) = s1.elem(0);

  return d;
}


template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir2Minus>::Type_t
spinReconstructDir2Minus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir2Minus>( s1 );

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = timesI(s1.elem(0));
  d.elem(3) = timesMinusI(s1.elem(1));

  return d;
}

template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir3Minus>::Type_t
spinReconstructDir3Minus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir3Minus>( s1 );

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = -s1.elem(0);
  d.elem(3) = -s1.elem(1);

  return d;
}

template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir0Plus>::Type_t
spinReconstructDir0Plus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir0Plus>( s1 );

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = timesMinusI(s1.elem(1));
  d.elem(3) = timesMinusI(s1.elem(0));

  return d;
}

template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir1Plus>::Type_t
spinReconstructDir1Plus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir1Plus>( s1 );

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = s1.elem(1);
  d.elem(3) = -s1.elem(0);

  return d;
}

template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir2Plus>::Type_t
spinReconstructDir2Plus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir2Plus>( s1 );

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = timesMinusI(s1.elem(0));
  d.elem(3) = timesI(s1.elem(1));

  return d;
}

template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnSpinReconstructDir3Plus>::Type_t
spinReconstructDir3Plus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir3Plus>( s1 );

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = s1.elem(0);
  d.elem(3) = s1.elem(1);

  return d;
}





template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnChiralProjectPlus>::Type_t
chiralProjectPlus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnChiralProjectPlus>( s1 );

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  zero_rep(d.elem(2));
  zero_rep(d.elem(3));

  return d;
}


template<class T>
typename UnaryReturn<PSpinVectorJIT<T>, FnChiralProjectMinus>::Type_t
chiralProjectMinus(PSpinVectorJIT<T> s1)
{
  auto d = unary_return<FnChiralProjectMinus>( s1 );

  zero_rep(d.elem(0));
  zero_rep(d.elem(1));
  d.elem(2) = s1.elem(2);
  d.elem(3) = s1.elem(3);

  return d;
}




} // namespace QDP

#endif
