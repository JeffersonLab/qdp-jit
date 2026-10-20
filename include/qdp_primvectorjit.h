#ifndef QDP_PRIMVECTORJIT_H
#define QDP_PRIMVECTORJIT_H



namespace QDP {


template <class T, template<class> class C> class PVectorJIT : public BaseJIT< T >
{
public:
  typedef C<T>  CC;

  // for eval Shape
  PVectorJIT( Shape shape ) : BaseJIT< T >(shape) {}
  
  PVectorJIT( llvm::Value * base , JitDataFormat fmt , NestedIndex args , Shape shape ) : BaseJIT< T >(base,fmt,args,shape) {}

  
  int getN() const
  {
    return this->shape().extent( axis() );
  }

  Shape::Axis axis() const
  {
    return static_cast<const CC*>(this)->axis();
  }


  int this_size() const { return getN(); }

  PVectorJIT operator=(PVectorJIT rhs) 
  {
    JitForLoop i( 0 , getN() );
    {
      this->elem( i.index() ) = rhs.elem( i.index() );
    }
    i.end();

    return *this;
  }


  template<class T1>
  CC assign(C<T1> rhs) 
  {
    JitForLoop i( 0 , getN() );
    {
      this->elem( i.index() ) = rhs.elem( i.index() );
    }
    i.end();

    return *static_cast<CC*>(this);
  }



  CC assign(C<T> rhs) 
  {
    JitForLoop i( 0 , getN() );
    {
      this->elem( i.index() ) = rhs.elem( i.index() );
    }
    i.end();

    return *static_cast<CC*>(this);
  }

  


  template<class T1>
  CC operator=(C<T1> rhs) 
  {
    JitForLoop i( 0 , getN() );
    {
      this->elem( i.index() ) = rhs.elem( i.index() );
    }
    i.end();

    return *static_cast<CC*>(this);
  }


  CC operator=(C<T> rhs) 
  {

    JitForLoop i( 0 , getN() );
    {
      this->elem( i.index() ) = rhs.elem( i.index() );
    }
    i.end();

    return *static_cast<CC*>(this);
  }



  template<class T1>
  CC operator+=(C<T1> rhs) 
  {
    JitForLoop i( 0 , getN() );
    {
      this->elem(i) += rhs.elem(i);
    }
    i.end();
      
    return *static_cast<CC*>(this);
  }



  template<class T1>
  CC operator-=(C<T1> rhs) 
  {
    JitForLoop i( 0 , getN() );
    {
      this->elem(i) -= rhs.elem(i);
    }
    i.end();
      
    return *static_cast<CC*>(this);
  }


  
  template<class T1>
  CC operator*=(PScalarJIT<T1> rhs) 
  {
    JitForLoop i( 0 , getN() );
    {
      this->elem(i) *= rhs.elem();
    }
    i.end();
    
    return *static_cast<CC*>(this);
  }


  
  template<class T1>
  CC operator/=(PScalarJIT<T1> rhs) 
  {
    JitForLoop i( 0 , getN() );
    {
      this->elem(i) /= rhs.elem();
    }
    i.end();
      
    return *static_cast<CC*>(this);
  }

};



  template <class T, template<class> class C >
  struct IsWordVec< PVectorJIT<T,C > >
  {
    constexpr static bool value = IsWordVec<T>::value;
  };

  

template<class T1, template<class> class C>
struct WordType<PVectorJIT<T1,C> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

template<class T1, template<class> class C> 
struct SinglePrecType< PVectorJIT<T1,C> >
{
  typedef PVectorJIT< typename SinglePrecType<T1>::Type_t, C> Type_t;
};

template<class T1, template<class> class C> 
struct DoublePrecType< PVectorJIT<T1,C> >
{
  typedef PVectorJIT< typename DoublePrecType<T1>::Type_t, C> Type_t;
};


template<class T, template<class> class C>
struct InternalScalar<PVectorJIT<T,C> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};


template<class T, template<class> class C>
struct PrimitiveScalar<PVectorJIT<T,C> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};


template<class T, template<class> class C>
struct LatticeScalar<PVectorJIT<T,C> > {
  typedef C<typename LatticeScalar<T>::Type_t>  Type_t;
};






template<class T1, template<class> class C, class Op>
struct UnaryReturn<PVectorJIT<T1,C>, Op> {
  typedef C<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C, class Op>
struct BinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,C>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};



template<class T1, class T2, template<class> class C1, template<class> class C2, class Op>
struct BinaryReturn<PMatrixJIT<T1,C1>, PVectorJIT<T2,C2>, Op> {
  typedef C2<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


  

template<class T1, class T2, template<class> class C, class Op>
struct BinaryReturn<PVectorJIT<T1,C>, PScalarJIT<T2>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};



template<class T1, class T2, template<class> class C, class Op>
struct BinaryReturn<PVectorJIT<T1,C>, PVectorJIT<T2,C>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};




template<class T1, template<class> class C>
typename UnaryReturn<PVectorJIT<T1,C>, OpUnaryPlus>::Type_t
operator+(PVectorJIT<T1,C> l)
{
  auto d = unary_return<OpUnaryPlus>(l);
  
  JitForLoop i( 0 , l.getN() );
  {
    d.elem( i ) = +l.elem( i );
  }
  i.end();

  return d;
}


template<class T1, template<class> class C>
typename UnaryReturn<PVectorJIT<T1,C>, OpUnaryMinus>::Type_t
operator-(PVectorJIT<T1,C> l)
{
  auto d = unary_return<OpUnaryMinus>( l );
  
  JitForLoop i( 0 , l.getN() );
  {
    d.elem( i ) = -l.elem( i );
  }
  i.end();

  return d;
}


template<class T1, class T2, template<class> class C>
typename BinaryReturn<PVectorJIT<T1,C>, PVectorJIT<T2,C>, OpAdd>::Type_t
operator+(PVectorJIT<T1,C> l, PVectorJIT<T2,C> r)
{
  assert( l.getN() == r.getN() );
  auto d = binary_return<OpAdd>( l , r );

  JitForLoop i( 0 , l.getN() );
  {
    d.elem( i ) = l.elem( i ) + r.elem( i );
  }
  i.end();

  return d;
}


template<class T1, class T2, template<class> class C>
typename BinaryReturn<PVectorJIT<T1,C>, PVectorJIT<T2,C>, OpSubtract>::Type_t
operator-(PVectorJIT<T1,C> l, PVectorJIT<T2,C> r)
{
  assert( l.getN() == r.getN() );
  auto d = binary_return<OpSubtract>(l,r);
  
  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = l.elem(i) - r.elem(i);
  }
  i.end();
  
  return d;
}



template<class T1, class T2, template<class> class C>
typename BinaryReturn<PVectorJIT<T1,C>, PScalarJIT<T2>, OpMultiply>::Type_t
operator*(PVectorJIT<T1,C> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpMultiply>(l,r);
  
  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = l.elem(i) * r.elem();
  }
  i.end();

  return d;
}


template<class T1, class T2, template<class> class C>
typename BinaryReturn<PVectorJIT<T1,C>, PScalarJIT<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(PVectorJIT<T1,C> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpMultiplyAdj>(l,r);

  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = multiplyAdj(l.elem(i), r.elem());
  }
  i.end();

  return d;
}



template<class T1, class T2, template<class> class C>
typename BinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,C>, OpMultiply>::Type_t
operator*(PScalarJIT<T1> l, PVectorJIT<T2,C> r)
{
  auto d = binary_return<OpMultiply>(l,r);

  JitForLoop i( 0 , r.getN() );
  {
    d.elem(i) = l.elem() * r.elem(i);
  }
  i.end();
  
  return d;
}


template<class T1, class T2, template<class> class C>
typename BinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,C>, OpAdjMultiply>::Type_t
adjMultiply(PScalarJIT<T1> l, PVectorJIT<T2,C> r)
{
  auto d = binary_return<OpAdjMultiply>(l,r);

  JitForLoop i( 0 , r.getN() );
  {
    d.elem(i) = adjMultiply(l.elem(), r.elem(i));
  }
  i.end();
    
  return d;
}




template<class T1, class T2, template<class> class C1, template<class> class C2>
typename BinaryReturn<PMatrixJIT<T1,C1>, PVectorJIT<T2,C2>, OpMultiply>::Type_t
operator*(PMatrixJIT<T1,C1> l, PVectorJIT<T2,C2> r)
{
  assert( l.getN() == r.getN() );
  auto d = binary_return<OpMultiply>( l , r );

  JitForLoop i0( 0 , l.getN() );
  i0.unroll_full();
  {
    d.elem( i0 ) = l.elem( i0 , 0 ) * r.elem( 0 );
  }
  i0.end();

  JitForLoop j( 1 , l.getN() );
  {
    JitForLoop i( 0 , l.getN() );
    i.unroll_full();
    {
      d.elem( i ) += l.elem( i , j ) * r.elem( j );
    }
    i.end();
  }
  j.end();

  return d;
}




template<class T1, class T2, template<class> class C1, template<class> class C2>
typename BinaryReturn<PMatrixJIT<T1,C1>, PVectorJIT<T2,C2>, OpAdjMultiply>::Type_t
adjMultiply(PMatrixJIT<T1,C1> l, PVectorJIT<T2,C2> r)
{
  assert( l.getN() == r.getN() );
  auto d = binary_return<OpAdjMultiply>( l , r );

  JitForLoop i0( 0 , l.getN() );
  i0.unroll_full();
  {
    d.elem( i0 ) = adjMultiply( l.elem( 0 , i0 ) , r.elem( 0 ));
  }
  i0.end();

  JitForLoop j( 1 , l.getN() );
  {
    JitForLoop i( 0 , l.getN() );
    i.unroll_full();
    {
      d.elem( i ) += adjMultiply( l.elem( j , i ) , r.elem( j ) );
    }
    i.end();
  }
  j.end();

  return d;
}


template<class T1, class T2, template<class> class C>
typename BinaryReturn<PVectorJIT<T1,C>, PScalarJIT<T2>, OpDivide>::Type_t
operator/(PVectorJIT<T1,C> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpDivide>( l , r );

  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = l.elem(i) / r.elem();
  }
  i.end();
  
  return d;
}




template<class T, template<class> class C>
typename UnaryReturn<PVectorJIT<T,C>, FnReal>::Type_t
real(PVectorJIT<T,C> s1)
{
  auto d = unary_return<FnReal>( s1 );

  JitForLoop i( 0 , s1.getN() );
  {
    d.elem(i) = real(s1.elem(i));
  }
  i.end();
  
  return d;
}



template<class T, template<class> class C>
typename UnaryReturn<PVectorJIT<T,C>, FnImag>::Type_t
imag(PVectorJIT<T,C> s1)
{
  auto d = unary_return<FnImag>( s1 );

  JitForLoop i( 0 , s1.getN() );
  {
    d.elem(i) = imag(s1.elem(i));
  }
  i.end();

  return d;
}



template<class T1, class T2, template<class> class C>
typename BinaryReturn<PVectorJIT<T1,C>, PVectorJIT<T2,C>, FnCmplx>::Type_t
cmplx(PVectorJIT<T1,C> s1, PVectorJIT<T2,C> s2)
{
  assert( s1.getN() == s2.getN() );
  auto d = binary_return<FnCmplx>( s1 , s2 );

  JitForLoop i( 0 , s1.getN() );
  {
    d.elem(i) = cmplx(s1.elem(i), s2.elem(i));
  }
  i.end();

  return d;
}




template<class T1, template<class> class C>
struct UnaryReturn<PVectorJIT<T1,C>, FnIsFinite> {
  typedef PScalarJIT< typename UnaryReturn<T1, FnIsFinite >::Type_t > Type_t;
};

  
template<class T1, template<class> class C>
typename UnaryReturn<PVectorJIT<T1,C>, FnIsFinite>::Type_t
isfinite(PVectorJIT<T1,C> l)
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




template<class T1, template<class> class C>
typename UnaryReturn<PVectorJIT<T1,C>, FnConjugate>::Type_t
conj(PVectorJIT<T1,C> l)
{
  auto d = unary_return<FnConjugate>( l );

  JitForLoop i( 0 , l.getN() );
  {
    d.elem(i) = conj(l.elem(i));
  }
  i.end();
  
  return d;
}


template<class T, template<class> class C>
typename UnaryReturn<PVectorJIT<T,C>, FnTimesI>::Type_t
timesI(PVectorJIT<T,C> s1)
{
  auto d = unary_return<FnTimesI>(s1);
  
  JitForLoop i( 0 , s1.getN() );
  {
    d.elem(i) = timesI(s1.elem(i));
  }
  i.end();

  return d;
}


template<class T, template<class> class C>
typename UnaryReturn<PVectorJIT<T,C>, FnTimesMinusI>::Type_t
timesMinusI(PVectorJIT<T,C> s1)
{
  auto d = unary_return<FnTimesMinusI>(s1);

  JitForLoop i( 0 , s1.getN() );
  {
    d.elem(i) = timesMinusI(s1.elem(i));
  }
  i.end();
  
  return d;
}



template<class T, template<class> class C>
void zero_rep(PVectorJIT<T,C> dest) 
{
  JitForLoop i( 0 , dest.getN() );
  {
    zero_rep(dest.elem(i));
  }
  i.end();
}




template<class T, template<class> class C, class T1, class T2, class T3>
void
fill_random_jit(PVectorJIT<T,C> d, T1 seed, T2 skewed_seed, const T3& seed_mult)
{
  JitForLoop i( 0 , d.getN() );
  {
    fill_random_jit(d.elem(i), seed, skewed_seed, seed_mult);
  }
  i.end();
}



template<class T, template<class> class C>
void
fill_gaussian_jit(PVectorJIT<T,C> d, PVectorJIT<T,C> r1, PVectorJIT<T,C> r2)
{
  JitForLoop i( 0 , d.getN() );
  {
    fill_gaussian_jit(d.elem(i), r1.elem(i), r2.elem(i));
  }
  i.end();
}





template<class T, template<class> class C>
struct UnaryReturn<PVectorJIT<T,C>, FnNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, template<class> class C>
struct UnaryReturn<PVectorJIT<T,C>, FnLocalNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T, template<class> class C>
typename UnaryReturn<PVectorJIT<T,C>, FnLocalNorm2>::Type_t
localNorm2(PVectorJIT<T,C> s1)
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



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PVectorJIT<T1,C>, PVectorJIT<T2,C>, FnInnerProduct > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
struct BinaryReturn<PVectorJIT<T1,C>, PVectorJIT<T2,C>, FnLocalInnerProduct > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>
localInnerProduct(PVectorJIT<T1,C> s1, PVectorJIT<T2,C> s2)
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




template<class T1, class T2, template<class> class C>
struct BinaryReturn<PVectorJIT<T1,C>, PVectorJIT<T2,C>, FnInnerProductReal > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PVectorJIT<T1,C>, PVectorJIT<T2,C>, FnLocalInnerProductReal > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};



template<class T1, class T2, template<class> class C>
PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>
localInnerProductReal(PVectorJIT<T1,C> s1, PVectorJIT<T2,C> s2)
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



template<class T1, class T2, class T3, template<class> class C>
struct TrinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,C>, PVectorJIT<T3,C>, FnWhere> {
  typedef C<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};



template<class T1, class T2, class T3, template<class> class C>
typename TrinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,C>, PVectorJIT<T3,C>, FnWhere>::Type_t
where(PScalarJIT<T1> a, PVectorJIT<T2,C> b, PVectorJIT<T3,C> c)
{
  assert(b.getN() == c.getN());

  auto d = trinary_return<FnWhere>(a,b,c);

  JitForLoop i( 0 , b.getN() );
  {
    d.elem(i) = where(a.elem(), b.elem(i), c.elem(i));
  }
  i.end();
  
  return d;
}



} // namespace QDP

#endif
