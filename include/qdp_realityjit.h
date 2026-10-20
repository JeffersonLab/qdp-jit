#ifndef QDP_REALITYJIT_H
#define QDP_REALITYJIT_H


namespace QDP {


template<class T>
class RScalarJIT: public BaseJIT< T >
{
public:
  int getN() const override { return 1; }
  int this_size() const { return getN(); }

  RScalarJIT() {}

  ~RScalarJIT() {}

  RScalarJIT( llvm::Value * base , JitDataFormat fmt , NestedIndex args , const Shape& shape ) : BaseJIT< T >(base,fmt,args) {}

  template<class T1>
  RScalarJIT(RScalarJIT<T1> rhs):BaseJIT< T >(rhs)  {}

  RScalarJIT(const RScalarJIT& rhs):BaseJIT< T >(rhs) {}

  RScalarJIT operator=( RScalarJIT rhs) {
    this->elem() = rhs.elem();
    return *this;
  }
  

  template<class T1>
  RScalarJIT operator=( RScalarJIT<T1> rhs) {
    this->elem() = rhs.elem();
    return *this;
  }


  RScalarJIT operator=( typename WordType<T>::Type_t rhs) {
    this->elem() = rhs;
    return *this;
  }

  
  template<class T1>
  RScalarJIT operator+=(RScalarJIT<T1> rhs) 
  {
    this->elem() += rhs.elem();
    return *this;
  }


  template<class T1>
  RScalarJIT operator-=(RScalarJIT<T1> rhs) 
  {
    this->elem() -= rhs.elem();
    return *this;
  }

  
  template<class T1>
  RScalarJIT operator*=(RScalarJIT<T1> rhs) 
  {
    this->elem() *= rhs.elem();
    return *this;
  }


  template<class T1>
  RScalarJIT operator/=(RScalarJIT<T1> rhs) 
  {
    this->elem() /= rhs.elem();
    return *this;
  }


  template<class T1>
  RScalarJIT operator%=(RScalarJIT<T1> rhs) 
  {
    this->elem() %= rhs.elem();
    return *this;
  }


  template<class T1>
  RScalarJIT operator|=(RScalarJIT<T1> rhs) 
  {
    this->elem() |= rhs.elem();
    return *this;
  }


  template<class T1>
  RScalarJIT operator&=(RScalarJIT<T1> rhs) 
  {
    this->elem() &= rhs.elem();
    return *this;
  }


  template<class T1>
  RScalarJIT operator^=(RScalarJIT<T1> rhs) 
  {
    this->elem() ^= rhs.elem();
    return *this;
  }


  template<class T1>
  RScalarJIT operator<<=(RScalarJIT<T1> rhs) 
  {
    this->elem() <<= rhs.elem();
    return *this;
  }


  template<class T1>
  RScalarJIT operator>>=(RScalarJIT<T1> rhs) 
  {
    this->elem() >>= rhs.elem();
    return *this;
  }


  T elem() const { return BaseJIT< T >::elem(0); }
};

 




  
template<class T>
class RComplexJIT : public BaseJIT< T >
{
public:
  int getN() const override { return 2; }
  int this_size() const { return getN(); }

  RComplexJIT() {}

  ~RComplexJIT() {}

  RComplexJIT( llvm::Value * base , JitDataFormat fmt , NestedIndex args , const Shape& shape ) : BaseJIT< T >(base,fmt,args) {}

   
  RComplexJIT operator=( RComplexJIT rhs) 
    {
      this->real() = rhs.real();
      this->imag() = rhs.imag();
      return *this;
    }

  
  template<class T1>
  RComplexJIT operator=(RScalarJIT<T1> rhs) 
    {
      this->real() = rhs.elem();
      zero_rep(this->imag());
      return *this;
    }

  
  template<class T1>
  RComplexJIT operator+=(RScalarJIT<T1> rhs) 
    {
      this->real() += rhs.elem();
      return *this;
    }


  template<class T1>
  RComplexJIT operator-=(RScalarJIT<T1> rhs) 
    {
      this->real() -= rhs.elem();
      return *this;
    }


  template<class T1>
  RComplexJIT operator*=(RScalarJIT<T1> rhs) 
    {
      this->real() *= rhs.elem();
      this->imag() *= rhs.elem();
      return *this;
    }


  template<class T1>
  RComplexJIT operator/=(RScalarJIT<T1> rhs) 
    {
      this->real() /= rhs.elem();
      this->imag() /= rhs.elem();
      return *this;
    }


  template<class T1>
  RComplexJIT operator+=( RComplexJIT<T1> rhs) 
    {
      this->real() += rhs.real();
      this->imag() += rhs.imag();
      return *this;
    }


  template<class T1>
  RComplexJIT operator-=( RComplexJIT<T1> rhs) 
    {
      this->real() -= rhs.real();
      this->imag() -= rhs.imag();
      return *this;
    }


  template<class T1>
  RComplexJIT operator*=( RComplexJIT<T1> rhs) 
    {
      auto d = StackAllocJit<RComplexJIT<T1>>::alloc();
      d = *this * rhs;

      this->real() = d.real();
      this->imag() = d.imag();
      return *this;
    }


  template<class T1>
  RComplexJIT operator/=( RComplexJIT<T1> rhs) 
    {
      auto d = StackAllocJit<RComplexJIT<T1>>::alloc();
      d = *this / rhs;

      this->real() = d.real();
      this->imag() = d.imag();
      return *this;
    }

  
  template<class T1>
  RComplexJIT operator=( RComplexJIT<T1> rhs) 
    {
      this->real() = rhs.real();
      this->imag() = rhs.imag();
      return *this;
    }


  T real() const { return BaseJIT< T >::elem(0); }
  T imag() const { return BaseJIT< T >::elem(1); }
};



template<class T>
class WordSize< RScalarJIT<T> >
{
public:
  static int value( const Shape& ret )
  {
    return 1 * WordSize<T>::value( ret );
  }
};


template<class T>
class WordSize< RComplexJIT<T> >
{
public:
  static int value( const Shape& ret )
  {
    return 2 * WordSize<T>::value( ret );
  }
};










  template <class T>
  struct IsWordVec< RScalarJIT<T> >
  {
    constexpr static bool value = IsWordVec<T>::value;
  };


  template <class T>
  struct IsWordVec< RComplexJIT<T> >
  {
    constexpr static bool value = IsWordVec<T>::value;
  };


template<class T>
struct ScalarType<RScalarJIT<T> >
{
  typedef RScalarJIT< typename ScalarType<T>::Type_t > Type_t;
};


template<class T>
struct ScalarType<RComplexJIT<T> >
{
  typedef RComplexJIT< typename ScalarType<T>::Type_t > Type_t;
};



  


template<class T>
struct WordType<RScalarJIT<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

template<class T>
struct WordType<RComplexJIT<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};


template<class T> 
struct SinglePrecType<RScalarJIT<T> >
{
  typedef RScalarJIT<typename SinglePrecType<T>::Type_t>  Type_t;
};

template<class T> 
struct SinglePrecType<RComplexJIT<T> >
{
  typedef RComplexJIT<typename SinglePrecType<T>::Type_t>  Type_t;
};

template<class T> 
struct DoublePrecType<RScalarJIT<T> >
{
  typedef RScalarJIT<typename DoublePrecType<T>::Type_t>  Type_t;
};

template<class T> 
struct DoublePrecType<RComplexJIT<T> >
{
  typedef RComplexJIT<typename DoublePrecType<T>::Type_t>  Type_t;
};



template<class T>
struct InternalScalar<RScalarJIT<T> > {
  typedef RScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};

template<class T>
struct InternalScalar<RComplexJIT<T> > {
  typedef RScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};



template<class T>
struct PrimitiveScalar<RScalarJIT<T> > {
  typedef RScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

template<class T>
struct PrimitiveScalar<RComplexJIT<T> > {
  typedef RScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};


template<class T>
struct LatticeScalar<RScalarJIT<T> > {
  typedef RScalarJIT<typename LatticeScalar<T>::Type_t>  Type_t;
};

template<class T>
struct LatticeScalar<RComplexJIT<T> > {
  typedef RComplexJIT<typename LatticeScalar<T>::Type_t>  Type_t;
};



template<class T>
struct RealScalar<RScalarJIT<T> > {
  typedef RScalarJIT<typename RealScalar<T>::Type_t>  Type_t;
};

template<class T>
struct RealScalar<RComplexJIT<T> > {
  typedef RScalarJIT<typename RealScalar<T>::Type_t>  Type_t;
};







template<class T1, class Op>
struct UnaryReturn<RScalarJIT<T1>, Op> {
  typedef RScalarJIT<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};


template<class T1, class Op>
struct UnaryReturn<RComplexJIT<T1>, Op> {
  typedef RComplexJIT<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, Op> {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, Op> {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, Op> {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, Op> {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};






template<class T1, class T2, class T3>
struct TrinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, RScalarJIT<T3>, FnColorContract> {
  typedef RScalarJIT<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};




template<class T1, class T2, class T3>
struct TrinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, RComplexJIT<T3>, FnColorContract> {
  typedef RComplexJIT<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};



template<class T>
struct UnaryReturn<RScalarJIT<T>, OpNot > {
  typedef RScalarJIT<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
};


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, OpNot>::Type_t
operator!(RScalarJIT<T1> l)
{
  auto d = unary_return<OpNot>( l );

  d.elem() = !l.elem();

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, OpUnaryPlus>::Type_t
operator+(RScalarJIT<T1> l)
{
  auto d = unary_return<OpUnaryPlus>( l );

  d.elem() = +l.elem();

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, OpUnaryMinus>::Type_t
operator-(RScalarJIT<T1> l)
{
  auto d = unary_return<OpUnaryMinus>( l );

  d.elem() = -l.elem();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpAdd>::Type_t
operator+(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpAdd>( l , r );

  d.elem() = l.elem() + r.elem();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpSubtract>::Type_t
operator-(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpSubtract>( l , r );

  d.elem() = l.elem() - r.elem();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpMultiply>::Type_t
operator*(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpMultiply>( l , r );
  
  d.elem() = l.elem() * r.elem();

  return d;
}




template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpAdjMultiply>::Type_t
adjMultiply(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpAdjMultiply>( l , r );

  d.elem() = l.elem() * r.elem();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpMultiplyAdj>( l , r );

  d.elem() = l.elem() * r.elem();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpAdjMultiplyAdj>( l , r );

  d.elem() = l.elem() * r.elem();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpDivide>::Type_t
operator/(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpDivide>( l , r );

  d.elem() = l.elem() / r.elem();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpLeftShift>::Type_t
operator<<(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpLeftShift>( l , r );

  d.elem() = l.elem() << r.elem();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpRightShift>::Type_t
operator>>(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpRightShift>( l , r );

  d.elem() = l.elem() >> r.elem();

  return d;
}


template<class T1, class T2 >
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpMod>::Type_t
operator%(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpMod>( l , r );

  d.elem() = l.elem() % r.elem();

  return d;
}


template<class T1, class T2 >
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpBitwiseXor>::Type_t
operator^(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpBitwiseXor>( l , r );

  d.elem() = l.elem() ^ r.elem();

  return d;
}


template<class T1, class T2 >
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpBitwiseAnd>::Type_t
operator&(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpBitwiseAnd>( l , r );

  d.elem() = l.elem() & r.elem();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpBitwiseOr>::Type_t
operator|(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpBitwiseOr>( l , r );

  d.elem() = l.elem() | r.elem();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpLT>::Type_t
operator<(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpLT>( l , r );

  d.elem() = l.elem() < r.elem();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpLE>::Type_t
operator<=(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpLE>( l , r );

  d.elem() = l.elem() <= r.elem();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpGT>::Type_t
operator>(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpGT>( l , r );

  d.elem() = l.elem() > r.elem();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpGE>::Type_t
operator>=(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpGE>( l , r );

  d.elem() = l.elem() >= r.elem();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpEQ>::Type_t
operator==(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpEQ>( l , r );

  d.elem() = l.elem() == r.elem();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpNE>::Type_t
operator!=(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpNE>( l , r );

  d.elem() = l.elem() != r.elem();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpAnd>::Type_t
operator&&(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpAnd>( l , r );

  d.elem() = l.elem() && r.elem();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpOr>::Type_t
operator||(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpOr>( l , r );

  d.elem() = l.elem() || r.elem();

  return d;
}




template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnAdjoint>::Type_t
adj(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnAdjoint>( s1 );

  d.elem() = s1.elem();

  return d;
}



template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnConjugate>::Type_t
conj(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnConjugate>( s1 );

  d.elem() = s1.elem();

  return d;
}



template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnTranspose>::Type_t
transpose(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnTranspose>( s1 );

  d.elem() = s1.elem();

  return d;
}



template<class T>
struct UnaryReturn<RScalarJIT<T>, FnTrace > {
  typedef RScalarJIT<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnTrace>::Type_t
trace(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnTrace>( s1 );

  d.elem() = s1.elem();

  return d;
}



template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnRealTrace>::Type_t
realTrace(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnRealTrace>( s1 );

  d.elem() = s1.elem();

  return d;
}



template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnImagTrace>::Type_t
imagTrace(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnImagTrace>( s1 );

  d.elem() = s1.elem();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnTraceMultiply>::Type_t
traceMultiply(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<FnTraceMultiply>( l,r );

  d.elem() = l.elem() * r.elem();

  return d;
}



template<class T>
typename UnaryReturn<RScalarJIT<T>, FnReal>::Type_t
real(RScalarJIT<T> s1)
{
  auto d = unary_return<FnReal>( s1 );

  d.elem() = s1.elem();

  return d;
}



template<class T>
typename UnaryReturn<RScalarJIT<T>, FnImag>::Type_t
imag(RScalarJIT<T> s1)
{
  auto d = unary_return<FnImag>( s1 );

  zero_rep( d.elem() );

  return d;
}



template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnArcCos>::Type_t
acos(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnArcCos>( s1 );

  d.elem() = acos(s1.elem());

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnArcSin>::Type_t
asin(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnArcSin>( s1 );

  d.elem() = asin(s1.elem());

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnArcTan>::Type_t
atan(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnArcTan>( s1 );

  d.elem() = atan(s1.elem());

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnCeil>::Type_t
ceil(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnCeil>( s1 );

  d.elem() = ceil(s1.elem());

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnCos>::Type_t
cos(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnCos>( s1 );

  d.elem() = cos(s1.elem());

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnHypCos>::Type_t
cosh(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnHypCos>( s1 );

  d.elem() = cosh(s1.elem());

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnExp>::Type_t
exp(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnExp>( s1 );

  d.elem() = exp(s1.elem());

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnFabs>::Type_t
fabs(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnFabs>( s1 );

  d.elem() = fabs(s1.elem());

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnFloor>::Type_t
floor(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnFloor>( s1 );

  d.elem() = floor(s1.elem());

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnLog>::Type_t
log(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnLog>( s1 );

  d.elem() = log(s1.elem());

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnLog10>::Type_t
log10(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnLog10>( s1 );

  d.elem() = log10(s1.elem());

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnSin>::Type_t
sin(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnSin>( s1 );

  d.elem() = sin(s1.elem());

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnHypSin>::Type_t
sinh(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnHypSin>( s1 );

  d.elem() = sinh(s1.elem());

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnSqrt>::Type_t
sqrt(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnSqrt>( s1 );

  d.elem() = sqrt(s1.elem());

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnIsFinite>::Type_t
isfinite(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnIsFinite>( s1 );

  d.elem() = isfinite(s1.elem());

  return d;
}



template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnTan>::Type_t
tan(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnTan>( s1 );

  d.elem() = tan(s1.elem());

  return d;
}


template<class T1>
typename UnaryReturn<RScalarJIT<T1>, FnHypTan>::Type_t
tanh(RScalarJIT<T1> s1)
{
  auto d = unary_return<FnHypTan>( s1 );

  d.elem() = tanh(s1.elem());

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnPow>::Type_t
pow(RScalarJIT<T1> s1, RScalarJIT<T2> s2)
{
  auto d = binary_return<FnPow>( s1 , s2 );

  d.elem() = pow(s1.elem(), s2.elem());

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnArcTan2>::Type_t
atan2(RScalarJIT<T1> s1, RScalarJIT<T2> s2)
{
  auto d = binary_return<FnArcTan2>( s1 , s2 );

  d.elem() = atan2(s1.elem(), s2.elem());

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnOuterProduct>::Type_t
outerProduct(RScalarJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<FnOuterProduct>( l , r );

  d.elem() = l.elem() * r.elem();

  return d;
}


template<class T>
typename UnaryReturn<RScalarJIT<T>, FnLocalNorm2>::Type_t
localNorm2(RScalarJIT<T> s1)
{
  auto d = unary_return<FnLocalNorm2>( s1 );

  d.elem() = localNorm2(s1.elem());

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(RScalarJIT<T1> s1, RScalarJIT<T2> s2)
{
  auto d = binary_return<FnLocalInnerProduct>( s1 , s2 );

  d.elem() = localInnerProduct(s1.elem(), s2.elem());

  return d;
}

template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnLocalColorInnerProduct>::Type_t
localColorInnerProduct(RScalarJIT<T1> s1, RScalarJIT<T2> s2)
{
  auto d = binary_return<FnLocalColorInnerProduct>( s1 , s2 );

  d.elem() = localColorInnerProduct(s1.elem(), s2.elem());

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(RScalarJIT<T1> s1, RScalarJIT<T2> s2)
{
  auto d = binary_return<FnLocalInnerProductReal>( s1 , s2 );

  d.elem() = localInnerProduct(s1.elem(), s2.elem());

  return d;
}



template<class T1, class T2, class T3>
struct TrinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, RScalarJIT<T3>, FnWhere> {
  typedef RScalarJIT<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};


template<class T1, class T2, class T3>
typename TrinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, RScalarJIT<T3>, FnWhere>::Type_t
where(RScalarJIT<T1> a, RScalarJIT<T2> b, RScalarJIT<T3> c)
{
  auto d = trinary_return<FnWhere>( a,b,c );

  d.elem() = where(a.elem(), b.elem(), c.elem());

  return d;
}


template<class T> 
inline
void zero_rep(RScalarJIT<T> dest) 
{
  zero_rep(dest.elem());
}



template<class T, class T1, class T2, class T3>
void
fill_random_jit(RScalarJIT<T> d, T1 seed, T2 skewed_seed, const T3& seed_mult)
{
  fill_random_jit(d.elem(), seed, skewed_seed, seed_mult);
}


template<class T, class T1, class T2, class T3>
void
fill_random_jit(RComplexJIT<T> d, T1 seed, T2 skewed_seed, const T3& seed_mult)
{
  fill_random_jit(d.real(), seed, skewed_seed, seed_mult);
  fill_random_jit(d.imag(), seed, skewed_seed, seed_mult);
}



template<class T>
void
fill_gaussian_jit( RScalarJIT<T> d, RScalarJIT<T> r1, RScalarJIT<T> r2)
{
  auto two_pi = stack_alloc_jit<T>();
  auto minus2 = stack_alloc_jit<T>();
  two_pi = 6.283185307;
  minus2 = -2.0;
  
  auto w_g_r = cos( r2.elem() * two_pi );

  auto w_r1 = sqrt( minus2 * log(r1.elem()) );

  d.elem() = w_r1 * w_g_r;
}



template<class T1>
typename UnaryReturn<RComplexJIT<T1>, OpUnaryPlus>::Type_t
operator+(RComplexJIT<T1> l)
{
  auto d = unary_return<OpUnaryPlus>( l );
  
  d.real() = +l.real();
  d.imag() = +l.imag();

  return d;
}



template<class T1>
typename UnaryReturn<RComplexJIT<T1>, OpUnaryMinus>::Type_t
operator-(RComplexJIT<T1> l)
{
  auto d = unary_return<OpUnaryMinus>( l );
  
  d.real() = -l.real();
  d.imag() = -l.imag();

  return d;
}




template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpAdd>::Type_t
operator+(RComplexJIT<T1> l, RComplexJIT<T2> r)
{
  auto d = binary_return<OpAdd>( l , r );

  d.real() = l.real() + r.real();
  d.imag() = l.imag() + r.imag();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, OpAdd>::Type_t
operator+(RComplexJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpAdd>( l , r );

  d.real() = l.real()+r.elem();
  d.imag() = l.imag();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, OpAdd>::Type_t
operator+(RScalarJIT<T1> l, RComplexJIT<T2> r)
{
  auto d = binary_return<OpAdd>( l , r );

  d.real() = l.elem()+r.real();
  d.imag() = r.imag();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpSubtract>::Type_t
operator-(RComplexJIT<T1> l, RComplexJIT<T2> r)
{
  auto d = binary_return<OpSubtract>( l , r );

  d.real() = l.real() - r.real();
  d.imag() = l.imag() - r.imag();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, OpSubtract>::Type_t
operator-(RComplexJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpSubtract>( l , r );

  d.real() = l.real() - r.elem();
  d.imag() = l.imag();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, OpSubtract>::Type_t
operator-(RScalarJIT<T1> l, RComplexJIT<T2> r)
{
  auto d = binary_return<OpSubtract>( l , r );

  d.real() = l.elem() - r.real();
  d.imag() = -r.imag();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpMultiply>::Type_t
operator*(RComplexJIT<T1> l, RComplexJIT<T2> r)
{
  auto d = binary_return<OpMultiply>( l , r );
  
  d.real() = l.real()*r.real() - l.imag()*r.imag();
  d.imag() = l.real()*r.imag() + l.imag()*r.real();
  
  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, OpMultiply>::Type_t
operator*(RScalarJIT<T1> l, RComplexJIT<T2> r)
{
  auto d = binary_return<OpMultiply>( l , r );

  d.real() = l.elem()*r.real();
  d.imag() = l.elem()*r.imag();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, OpMultiply>::Type_t
operator*(RComplexJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<OpMultiply>( l , r );

  d.real() = l.real()*r.elem();
  d.imag() = l.imag()*r.elem();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpAdjMultiply>::Type_t
adjMultiply(RComplexJIT<T1> l, RComplexJIT<T2> r)
{
  auto d = binary_return<OpMultiply>( l , r );

  d.real() = l.real()*r.real() + l.imag()*r.imag();
  d.imag() = l.real()*r.imag() - l.imag()*r.real();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(RComplexJIT<T1> l, RComplexJIT<T2> r)
{
  auto d = binary_return<OpMultiplyAdj>( l , r );

  d.real() = l.real()*r.real() + l.imag()*r.imag();
  d.imag() = l.imag()*r.real() - l.real()*r.imag();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(RComplexJIT<T1> l, RComplexJIT<T2> r)
{
  auto d = binary_return<OpAdjMultiplyAdj>( l , r );

  d.real() = l.real()*r.real() - l.imag()*r.imag();
  d.imag() = -(l.real()*r.imag() + l.imag()*r.real());

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpDivide>::Type_t
operator/(RComplexJIT<T1> l, RComplexJIT<T2> r)
{
  typedef typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpDivide>::Type_t  Ret_t;

  auto d = binary_return<OpDivide>( l , r );

  auto tmp = create_word< typename WordType<Ret_t>::Type_t >( 1.0 ) / (r.real()*r.real() + r.imag()*r.imag());;
  
  d.real() = (l.real()*r.real() + l.imag()*r.imag()) * tmp;
  d.imag() = (l.imag()*r.real() - l.real()*r.imag()) * tmp;

  return d;

}


template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, OpDivide>::Type_t
operator/(RComplexJIT<T1> l, RScalarJIT<T2> r)
{
  typedef typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, OpDivide>::Type_t  Ret_t;

  auto d = binary_return<OpDivide>( l , r );
  auto tmp = create_word< typename WordType<Ret_t>::Type_t >( 1.0 ) / r.elem();
  
  d.real() = l.real() * tmp;
  d.imag() = l.imag() * tmp;

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, OpDivide>::Type_t
operator/(RScalarJIT<T1> l, RComplexJIT<T2> r)
{
  typedef typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, OpDivide>::Type_t  Ret_t;

  auto d = binary_return<OpDivide>( l , r );
  auto tmp = create_word< typename WordType<Ret_t>::Type_t >( 1.0 ) / (r.real()*r.real() + r.imag()*r.imag());
  
  d.real() = l.elem() * r.real() * tmp;
  d.imag() = -l.elem() * r.imag() * tmp;

  return d;
}



template<class T1>
typename UnaryReturn<RComplexJIT<T1>, FnAdjoint>::Type_t
adj(RComplexJIT<T1> l)
{
  auto d = unary_return<FnAdjoint>( l );
  
  d.real() = l.real();
  d.imag() = -l.imag();

  return d;
}


template<class T1>
typename UnaryReturn<RComplexJIT<T1>, FnConjugate>::Type_t
conj(RComplexJIT<T1> l)
{
  auto d = unary_return<FnConjugate>( l );
  
  d.real() = l.real();
  d.imag() = -l.imag();

  return d;
}


template<class T1>
typename UnaryReturn<RComplexJIT<T1>, FnTranspose>::Type_t
transpose(RComplexJIT<T1> l)
{
  auto d = unary_return<FnTranspose>( l );
  
  d.real() = l.real();
  d.imag() = l.imag();

  return d;
}



template<class T1>
typename UnaryReturn<RComplexJIT<T1>, FnTrace>::Type_t
trace(RComplexJIT<T1> s1)
{
  auto d = unary_return<FnTrace>( s1 );
  
  d.real() = s1.real();
  d.imag() = s1.imag();

  return d;
}



template<class T>
struct UnaryReturn<RComplexJIT<T>, FnRealTrace > {
  typedef RScalarJIT<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};




template<class T1>
typename UnaryReturn<RComplexJIT<T1>, FnRealTrace>::Type_t
realTrace(RComplexJIT<T1> s1)
{
  auto d = unary_return<FnRealTrace>( s1 );
  
  d.elem() = s1.real();

  return d;
}



template<class T>
struct UnaryReturn<RComplexJIT<T>, FnImagTrace > {
  typedef RScalarJIT<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};


template<class T1>
typename UnaryReturn<RComplexJIT<T1>, FnImagTrace>::Type_t
imagTrace(RComplexJIT<T1> s1)
{
  auto d = unary_return<FnImagTrace>( s1 );
  
  d.elem() = s1.imag();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpMultiply>::Type_t
traceMultiply(RComplexJIT<T1> l, RComplexJIT<T2> r)
{
  auto d = binary_return<OpMultiply>( l , r );
  
  d.real() = l.real()*r.real() - l.imag()*r.imag();
  d.imag() = l.real()*r.imag() + l.imag()*r.real();

  return d;
}



template<class T>
struct UnaryReturn<RComplexJIT<T>, FnReal > {
  typedef RScalarJIT<typename UnaryReturn<T, FnReal>::Type_t>  Type_t;
};


template<class T1>
typename UnaryReturn<RComplexJIT<T1>, FnReal>::Type_t
real(RComplexJIT<T1> s1)
{
  auto d = unary_return<FnReal>( s1 );
  
  d.elem() = s1.real();

  return d;
}


template<class T>
struct UnaryReturn<RComplexJIT<T>, FnImag > {
  typedef RScalarJIT<typename UnaryReturn<T, FnImag>::Type_t>  Type_t;
};


template<class T1>
typename UnaryReturn<RComplexJIT<T1>, FnImag>::Type_t
imag(RComplexJIT<T1> s1)
{
  auto d = unary_return<FnReal>( s1 );
  
  d.elem() = s1.imag();

  return d;
}



template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnCmplx > {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, FnCmplx>::Type_t>  Type_t;
};


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnCmplx>::Type_t
cmplx(RScalarJIT<T1> s1, RScalarJIT<T2> s2)
{
  auto d = binary_return<FnCmplx>( s1 , s2 );
  
  d.real() = s1.elem();
  d.imag() = s2.elem();

  return d;
}




template<class T>
struct UnaryReturn<RScalarJIT<T>, FnTimesI > {
  typedef RComplexJIT<typename UnaryReturn<T, FnTimesI>::Type_t>  Type_t;
};


template<class T>
typename UnaryReturn<RScalarJIT<T>, FnTimesI>::Type_t
timesI(RScalarJIT<T> s1)
{
  auto d = unary_return<FnTimesI>( s1 );
  
  zero_rep( d.real() );
  d.imag() = s1.elem();

  return d;
}


template<class T>
typename UnaryReturn<RComplexJIT<T>, FnTimesI>::Type_t
timesI(RComplexJIT<T> s1)
{
  auto d = unary_return<FnTimesI>( s1 );

  d.real() = -s1.imag();
  d.imag() = s1.real();

  return d;
}



template<class T>
struct UnaryReturn<RScalarJIT<T>, FnTimesMinusI > {
  typedef RComplexJIT<typename UnaryReturn<T, FnTimesMinusI>::Type_t>  Type_t;
};


template<class T>
typename UnaryReturn<RScalarJIT<T>, FnTimesMinusI>::Type_t
timesMinusI(RScalarJIT<T> s1)
{
  auto d = unary_return<FnTimesMinusI>( s1 );
  
  zero_rep( d.real() );
  d.imag() = -s1.elem();

  return d;
}



template<class T>
typename UnaryReturn<RComplexJIT<T>, FnTimesMinusI>::Type_t
timesMinusI(RComplexJIT<T> s1)
{
  auto d = unary_return<FnTimesMinusI>( s1 );

  d.real() = s1.imag();
  d.imag() = -s1.real();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, FnOuterProduct>::Type_t
outerProduct(RComplexJIT<T1> l, RComplexJIT<T2> r)
{
  auto d = binary_return<FnOuterProduct>( l,r );
  
  d.real() = l.real()*r.real() + l.imag()*r.imag();
  d.imag() = l.imag()*r.real() - l.real()*r.imag();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, FnOuterProduct>::Type_t
outerProduct(RComplexJIT<T1> l, RScalarJIT<T2> r)
{
  auto d = binary_return<FnOuterProduct>( l,r );
  
  d.real() = l.real()*r.elem();
  d.imag() = l.imag()*r.elem();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, FnOuterProduct>::Type_t
outerProduct(RScalarJIT<T1> l, RComplexJIT<T2> r)
{
  auto d = binary_return<FnOuterProduct>( l,r );
  
  d.real() = l.elem()*r.real();
  d.imag() = -l.elem()*r.imag();

  return d;
}



template<class T>
struct UnaryReturn<RComplexJIT<T>, FnNorm2 > {
  typedef RScalarJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<RComplexJIT<T>, FnLocalNorm2 > {
  typedef RScalarJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};


template<class T>
typename UnaryReturn<RComplexJIT<T>, FnLocalNorm2>::Type_t
localNorm2(RComplexJIT<T> s1)
{
  auto d = unary_return<FnLocalNorm2>( s1 );
  
  d.elem() = localNorm2(s1.real()) + localNorm2(s1.imag());

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(RComplexJIT<T1> l, RComplexJIT<T2> r)
{
  auto d = binary_return<FnLocalInnerProduct>( l,r );
  
  d.real() = localInnerProduct(l.real() , r.real()) + localInnerProduct(l.imag() , r.imag());
  d.imag() = localInnerProduct(l.real() , r.imag()) - localInnerProduct(l.imag() , r.real());

  return d;
}


template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, FnLocalColorInnerProduct>::Type_t
localColorInnerProduct(RComplexJIT<T1> l, RComplexJIT<T2> r)
{
  auto d = binary_return<FnLocalColorInnerProduct>( l,r );
  
  d.real() = localColorInnerProduct(l.real(),r.real()) + localColorInnerProduct(l.imag(),r.imag());
  d.imag() = localColorInnerProduct(l.real(),r.imag()) - localColorInnerProduct(l.imag(),r.real());

  return d;
}

  

template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, FnInnerProductReal > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};



template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, FnLocalInnerProductReal > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};


template<class T1, class T2>
typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(RComplexJIT<T1> l, RComplexJIT<T2> r)
{
  auto d = binary_return<FnLocalInnerProductReal>( l,r );
  
  d.elem() = localInnerProduct(l.real(),r.real()) + localInnerProduct(l.imag(),r.imag());

  return d;
}



template<class T1, class T2, class T3>
struct TrinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, RComplexJIT<T3>, FnWhere> {
  typedef RComplexJIT<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};


template<class T1, class T2, class T3>
typename TrinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, RComplexJIT<T3>, FnWhere>::Type_t
where(RScalarJIT<T1> a, RComplexJIT<T2> b, RComplexJIT<T3> c)
{
  auto d = trinary_return<FnWhere>( a,b,c );

  d.real() = where(a.elem(), b.real(), c.real());
  d.imag() = where(a.elem(), b.imag(), c.imag());

  return d;
}


template<class T1, class T2, class T3>
struct TrinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, RComplexJIT<T3>, FnWhere> {
  typedef RComplexJIT<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
typename TrinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, RComplexJIT<T3>, FnWhere>::Type_t
where(RScalarJIT<T1> a, RScalarJIT<T2> b, RComplexJIT<T3> c)
{
  auto d = trinary_return<FnWhere>( a,b,c );

  Shape shape;
  auto zero = StackAllocJit< T2 >::alloc( shape );
  zero_rep(zero);
  
  d.real() = where(a.elem(), b.elem(), c.real());
  d.imag() = where(a.elem(), zero    , c.imag());

  return d;
}



template<class T1, class T2, class T3>
struct TrinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, RScalarJIT<T3>, FnWhere> {
  typedef RComplexJIT<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
typename TrinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, RScalarJIT<T3>, FnWhere>::Type_t
where(RScalarJIT<T1> a, RComplexJIT<T2> b, RScalarJIT<T3> c)
{
  auto d = trinary_return<FnWhere>( a,b,c );

  Shape shape;
  auto zero = StackAllocJit< T3 >::alloc( shape );
  zero_rep(zero);
  
  d.real() = where(a.elem(), b.real(), c.elem());
  d.imag() = where(a.elem(), b.imag(), zero );

  return d;
}


template<class T> 
inline
void zero_rep(RComplexJIT<T> dest) 
{
  zero_rep(dest.real());
  zero_rep(dest.imag());
}

template<class T>
void
fill_gaussian_jit(RComplexJIT<T> d, RComplexJIT<T> r1, RComplexJIT<T> r2)
{
  auto two_pi = stack_alloc_jit<T>();
  auto minus2 = stack_alloc_jit<T>();
  two_pi = 6.283185307;
  minus2 = -2.0;
  
  auto w_r2_r = r2.real() * two_pi;
  auto w_g_r = cos( w_r2_r );
  auto w_g_i = sin( w_r2_r );

  auto w_r1_r = sqrt( minus2 * log(r1.real()) );

  d.real() = w_r1_r * w_g_r;
  d.imag() = w_r1_r * w_g_i;
}




template<class T1>
struct UnaryReturn<RComplexJIT<T1>, FnIsFinite> {
  typedef RScalarJIT<typename UnaryReturn<T1, FnIsFinite>::Type_t>  Type_t;
};


template<class T1>
typename UnaryReturn<RComplexJIT<T1>, FnIsFinite>::Type_t
isfinite(RComplexJIT<T1> s1)
{
  auto d = unary_return<FnIsFinite>( s1 );
  
  d.elem() = isfinite(s1.real()) && isfinite(s1.imag());

  return d;
}




} // namespace QDP

#endif
