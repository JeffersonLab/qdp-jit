#ifndef QDP_PRIMSCALARJIT_H
#define QDP_PRIMSCALARJIT_H


namespace QDP {


  template<class T> class PScalarJIT: public BaseJIT< T >
  {
  public:
    typedef T SubType_t;
    
    int getN() const 
    {
      return 1;
    }

    int this_size() const
    {
      return 1;
    }

    PScalarJIT(): BaseJIT< T >()
    {
    }


    PScalarJIT( llvm::Value * base , JitDataFormat fmt , NestedIndex args , Shape shape ) : BaseJIT< T >(base,fmt,args,shape)
    {
    }

    //for eval Shape
    PScalarJIT( Shape shape ) : BaseJIT< T >(shape) {}

    
    PScalarJIT( const T& rhs )
    {
      BaseJIT<T>::setup( rhs );
    }



    PScalarJIT( llvm::Value * base , JitDataFormat fmt , NestedIndex args = NestedIndex() )
    {
      BaseJIT<T>::setup( base , fmt , args );
    }

    

    
    template<class T1>
    PScalarJIT operator=( PScalarJIT<T1> rhs) {
      elem() = rhs.elem();
      return *this;
    }


    PScalarJIT operator=( const PScalarJIT rhs) {
      elem() = rhs.elem();
      return *this;
    }


    PScalarJIT operator=( typename WordType<T>::Type_t rhs) {
      elem() = rhs;
      return *this;
    }



    template<class T1>
    PScalarJIT operator+=(PScalarJIT<T1> rhs) 
    {
      elem() += rhs.elem();
      return *this;
    }


    template<class T1>
    PScalarJIT operator-=(PScalarJIT<T1> rhs) 
    {
      elem() -= rhs.elem();
      return *this;
    }


    template<class T1>
    PScalarJIT operator*=(PScalarJIT<T1> rhs) 
    {
      elem() *= rhs.elem();
      return *this;
    }


    template<class T1>
    PScalarJIT operator/=(PScalarJIT<T1> rhs) 
    {
      elem() /= rhs.elem();
      return *this;
    }


    template<class T1>
    PScalarJIT operator%=(PScalarJIT<T1> rhs) 
    {
      elem() %= rhs.elem();
      return *this;
    }


    template<class T1>
    PScalarJIT operator|=(PScalarJIT<T1> rhs) 
    {
      elem() |= rhs.elem();
      return *this;
    }


    template<class T1>
    PScalarJIT operator&=(PScalarJIT<T1> rhs) 
    {
      elem() &= rhs.elem();
      return *this;
    }


    template<class T1>
    PScalarJIT operator^=(PScalarJIT<T1> rhs) 
    {
      elem() ^= rhs.elem();
      return *this;
    }


    template<class T1>
    PScalarJIT operator<<=(PScalarJIT<T1> rhs) 
    {
      elem() <<= rhs.elem();
      return *this;
    }


    template<class T1>
    PScalarJIT operator>>=(PScalarJIT<T1> rhs) 
    {
      elem() >>= rhs.elem();
      return *this;
    }

  
  public:
    T elem() const      { return BaseJIT< T >::elem(); }
  };


template<class T>
class WordSize< PScalarJIT<T> >
{
public:
  static int value( const Shape& ret )
  {
    return 1 * WordSize<T>::value( ret );
  }
};

  




template<class T>
istream& operator>>(istream& s, PScalarJIT<T> d)
{
  return s >> d.elem();
}

template<class T>
StandardInputStream& operator>>(StandardInputStream& s, PScalarJIT<T> d)
{
  return s >> d.elem();
}

template<class T>
ostream& operator<<(ostream& s, PScalarJIT<T> d)
{
  return s << d.elem();
}

template<class T>
StandardOutputStream& operator<<(StandardOutputStream& s, PScalarJIT<T> d)
{
  return s << d.elem();
}

template<class T>
TextReader& operator>>(TextReader& txt, PScalarJIT<T> d)
{
  return txt >> d.elem();
}

template<class T>
TextWriter& operator<<(TextWriter& txt, PScalarJIT<T> d)
{
  return txt << d.elem();
}

#ifndef QDP_NO_LIBXML2
//! XML output
template<class T>
XMLWriter& operator<<(XMLWriter& xml, PScalarJIT<T> d)
{
  return xml << d.elem();
}

//! XML input
template<class T>
void read(XMLReader& xml, const string& path, PScalarJIT<T> d)
{
  read(xml, path, d.elem());
}
#endif

/*! @} */  // end of group primscalar






template<class T>
struct ScalarType<PScalarJIT<T> >
{
  typedef PScalarJIT< typename ScalarType<T>::Type_t > Type_t;
};


template<class T> 
struct JITType< PScalarJIT<T> >
{
  typedef PScalarJIT<typename JITType<T>::Type_t>  Type_t;
};

template<class T> 
struct BASEType< PScalarJIT<T> >
{
  typedef PScalar<typename BASEType<T>::Type_t>  Type_t;
};



template<class T>
struct WordType<PScalarJIT<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};


template<class T>
struct SinglePrecType<PScalarJIT<T> >
{
  typedef PScalarJIT< typename SinglePrecType<T>::Type_t > Type_t;
};

template<class T>
struct DoublePrecType<PScalarJIT<T> >
{
  typedef PScalarJIT< typename DoublePrecType<T>::Type_t > Type_t;
};


template<class T>
struct InternalScalar<PScalarJIT<T> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};


template<class T>
struct RealScalar<PScalarJIT<T> > {
  typedef PScalarJIT<typename RealScalar<T>::Type_t>  Type_t;
};


template<class T>
struct PrimitiveScalar<PScalarJIT<T> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};


template<class T>
struct LatticeScalar<PScalarJIT<T> > {
  typedef PScalarJIT<typename LatticeScalar<T>::Type_t>  Type_t;
};



template<class T> 
void 
zero_rep(PScalarJIT<T> dest) 
{
  zero_rep(dest.elem());
}


template<class T, class T1, class T2,class T3>
void
fill_random_jit( PScalarJIT<T> d, T1 seed, T2 skewed_seed, const T3& seed_mult)
{
  fill_random_jit(d.elem(), seed, skewed_seed, seed_mult);
}


template<class T,class T2>
void
fill_gaussian_jit(PScalarJIT<T> d, PScalarJIT<T2> r1, PScalarJIT<T2> r2)
{
  fill_gaussian_jit(d.elem(), r1.elem(), r2.elem());
}



template<class T1, class Op>
struct UnaryReturn<PScalarJIT<T1>, Op> {
  typedef PScalarJIT<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, Op> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};



template<class T1>
typename UnaryReturn<PScalarJIT<T1>, OpNot>::Type_t
operator!(PScalarJIT<T1> l)
{
  auto d = unary_return<OpNot>( l );

  d = !l.elem();

  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, OpUnaryPlus>::Type_t
operator+(PScalarJIT<T1> l)
{
  auto d = unary_return<OpUnaryPlus>( l );

  d = +l.elem();

  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, OpUnaryMinus>::Type_t
operator-(PScalarJIT<T1> l)
{
  auto d = unary_return<OpUnaryMinus>( l );

  d = -l.elem();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpAdd>::Type_t
operator+(PScalarJIT<T1> l, const PScalarJIT<T2>& r)
{
  auto d =binary_return<OpAdd>( l , r );

  d = l.elem() + r.elem();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpSubtract>::Type_t
operator-(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpSubtract>( l , r );

  d = l.elem() - r.elem();

  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpMultiply>::Type_t
operator*(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpMultiply>( l , r );
  
  d.elem() = l.elem() * r.elem();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpAdjMultiply>::Type_t
adjMultiply(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpAdjMultiply>( l , r );

  d.elem() = adjMultiply(l.elem() , r.elem());

  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  return multiplyAdj(l.elem() , r.elem());
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpAdjMultiplyAdj>( l , r );

  d.elem() = adjMultiplyAdj(l.elem() , r.elem());

  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpDivide>::Type_t
operator/(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpDivide>( l , r );

  d.elem() = l.elem() / r.elem();

  return d;
}


 

template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpLeftShift>::Type_t
operator<<(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpLeftShift>( l , r );

  d.elem() = l.elem() << r.elem();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpRightShift>::Type_t
operator>>(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpRightShift>( l , r );

  d.elem() = l.elem() >> r.elem();

  return d;
}


template<class T1, class T2 >
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpMod>::Type_t
operator%(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpMod>( l , r );

  d.elem() = l.elem() % r.elem();

  return d;
}

template<class T1, class T2 >
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpBitwiseXor>::Type_t
operator^(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpBitwiseXor>( l , r );

  d.elem() = l.elem() ^ r.elem();

  return d;
}

template<class T1, class T2 >
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpBitwiseAnd>::Type_t
operator&(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpBitwiseAnd>( l , r );

  d.elem() = l.elem() & r.elem();

  return d;
}

template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpBitwiseOr>::Type_t
operator|(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpBitwiseOr>( l , r );

  d.elem() = l.elem() | r.elem();

  return d;
}




template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpLT>::Type_t
operator<(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpLT>( l , r );

  d.elem() = l.elem() < r.elem();

  return d;
}




template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpLE>::Type_t
operator<=(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpLE>( l , r );

  d.elem() = l.elem() <= r.elem();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpGT>::Type_t
operator>(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpGT>( l , r );

  d.elem() = l.elem() > r.elem();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpGE>::Type_t
operator>=(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpGE>( l , r );

  d.elem() = l.elem() >= r.elem();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpEQ>::Type_t
operator==(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpEQ>( l , r );

  d.elem() = l.elem() == r.elem();

  return d;
}




template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpNE>::Type_t
operator!=(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpNE>( l , r );

  d.elem() = l.elem() != r.elem();

  return d;
}




template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpAnd>::Type_t
operator&&(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpAnd>( l , r );

  d.elem() = l.elem() && r.elem();

  return d;
}




template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpOr>::Type_t
operator||(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpOr>( l , r );

  d.elem() = l.elem() || r.elem();

  return d;
}





template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnAdjoint>::Type_t
adj(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnAdjoint>(s1);
  d.elem() = adj(s1.elem());
  return d;
}



template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnConjugate>::Type_t
conj(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnConjugate>(s1);
  d.elem() = conj(s1.elem());
  return d;
}



template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnTranspose>::Type_t
transpose(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnTranspose>(s1);
  d.elem() = transpose(s1.elem());
  return d;
}




template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnTrace>::Type_t
trace(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnTrace>(s1);
  d.elem() = trace(s1.elem());
  return d;
}



template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnRealTrace>::Type_t
realTrace(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnRealTrace>(s1);
  d.elem() = realTrace(s1.elem());
  return d;
}



template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnImagTrace>::Type_t
imagTrace(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnImagTrace>(s1);
  d.elem() = imagTrace(s1.elem());
  return d;
}



template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnTraceColor>::Type_t
traceColor(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnTraceColor>(s1);
  d.elem() = traceColor(s1.elem());
  return d;
}



template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnTraceSpin>::Type_t
traceSpin(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnTraceSpin>(s1);
  d.elem() = traceSpin(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnTransposeSpin>::Type_t
transposeSpin(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnTransposeSpin>(s1);
  d.elem() = transposeSpin(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnTransposeColor>::Type_t
transposeColor(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnTransposeColor>(s1);
  d.elem() = transposeColor(s1.elem());
  return d;
}



template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnTraceMultiply>::Type_t
traceMultiply(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<FnTraceMultiply>(l,r);
  d.elem() = traceMultiply(l.elem(), r.elem());
  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnTraceColorMultiply>::Type_t
traceColorMultiply(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<FnTraceColorMultiply>(l,r);
  d.elem() = traceMultiply(l.elem(), r.elem());
  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<FnTraceSpinMultiply>(l,r);
  d.elem() = traceMultiply(l.elem(), r.elem());
  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnTraceSpinOuterProduct>::Type_t
traceSpinOuterProduct(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<FnTraceSpinOuterProduct>(l,r);
  d.elem() = traceSpinOuterProduct(l.elem(), r.elem());
  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnOuterProduct>::Type_t
outerProduct(PScalarJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<FnOuterProduct>(l,r);
  d.elem() = outerProduct(l.elem(),r.elem());
  return d;
}



template<class T>
typename UnaryReturn<PScalarJIT<T>, FnReal>::Type_t
real(const PScalarJIT<T>& s1)
{
  auto d = unary_return<FnReal>(s1);
  d.elem() = real(s1.elem());
  return d;
}



template<class T>
typename UnaryReturn<PScalarJIT<T>, FnImag>::Type_t
imag(const PScalarJIT<T>& s1)
{
  auto d = unary_return<FnImag>(s1);
  d.elem() = imag(s1.elem());
  return d;
}



template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnArcCos>::Type_t
acos(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnArcCos>(s1);
  d.elem() = acos(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnArcSin>::Type_t
asin(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnArcSin>(s1);
  d.elem() = asin(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnArcTan>::Type_t
atan(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnArcTan>(s1);
  d.elem() = atan(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnCeil>::Type_t
ceil(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnCeil>(s1);
  d.elem() = ceil(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnCos>::Type_t
cos(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnCos>(s1);
  d.elem() = cos(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnHypCos>::Type_t
cosh(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnHypCos>(s1);
  d.elem() = cosh(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnExp>::Type_t
exp(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnExp>(s1);
  d.elem() = exp(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnFabs>::Type_t
fabs(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnFabs>(s1);
  d.elem() = fabs(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnFloor>::Type_t
floor(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnFloor>(s1);
  d.elem() = floor(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnLog>::Type_t
log(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnLog>(s1);
  d.elem() = log(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnLog10>::Type_t
log10(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnLog10>(s1);
  d.elem() = log10(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnSin>::Type_t
sin(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnSin>(s1);
  d.elem() = sin(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnHypSin>::Type_t
sinh(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnHypSin>(s1);
  d.elem() = sinh(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnSqrt>::Type_t
sqrt(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnSqrt>(s1);
  d.elem() = sqrt(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnIsFinite>::Type_t
isfinite( PScalarJIT<T1> s1)
{
  auto d = unary_return<FnIsFinite>(s1);
  d.elem() = isfinite(s1.elem());
  return d;
}



template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnTan>::Type_t
tan(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnTan>(s1);
  d.elem() = tan(s1.elem());
  return d;
}


template<class T1>
typename UnaryReturn<PScalarJIT<T1>, FnHypTan>::Type_t
tanh(PScalarJIT<T1> s1)
{
  auto d = unary_return<FnHypTan>(s1);
  d.elem() = tanh(s1.elem());
  return d;
}



template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnPow>::Type_t
pow(PScalarJIT<T1> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnPow>(s1,s2);
  d.elem() = pow(s1.elem(), s2.elem());
  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnArcTan2>::Type_t
atan2(PScalarJIT<T1> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnArcTan2>(s1,s2);
  d.elem() = atan2(s1.elem(), s2.elem());
  return d;
}



template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnCmplx>::Type_t
cmplx(PScalarJIT<T1> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnCmplx>(s1,s2);
  d.elem() = cmplx(s1.elem(), s2.elem());
  return d;
}


template<class T>
typename UnaryReturn<PScalarJIT<T>, FnTimesI>::Type_t
timesI(const PScalarJIT<T>& s1)
{
  auto d = unary_return<FnTimesI>(s1);
  d.elem() = timesI(s1.elem());
  return d;
}


template<class T>
typename UnaryReturn<PScalarJIT<T>, FnTimesMinusI>::Type_t
timesMinusI(const PScalarJIT<T>& s1)
{
  auto d = unary_return<FnTimesMinusI>(s1);
  d.elem() = timesMinusI(s1.elem());
  return d;
}




template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSeedToFloat>::Type_t
seedToFloat( PScalarJIT<T> s1)
{
  auto d = unary_return<FnSeedToFloat>(s1);
  d.elem() = seedToFloat(s1.elem());
  return d;
}




template<class T>
typename UnaryReturn<PScalarJIT<T>, FnPeekColorVectorJIT>::Type_t
peekColor( PScalarJIT<T> l, llvm::Value * row)
{
  auto d = unary_return<FnPeekColorVectorJIT>(l);
  d.elem() = peekColor(l.elem(),row);
  return d;
}


template<class T>
typename UnaryReturn<PScalarJIT<T>, FnPeekColorMatrixJIT>::Type_t
peekColor( PScalarJIT<T> l, llvm::Value * row, llvm::Value * col)
{
  auto d = unary_return<FnPeekColorMatrixJIT>(l);
  d.elem() = peekColor(l.elem(),row,col);
  return d;
}



template<class T>
typename UnaryReturn<PScalarJIT<T>, FnPeekSpinVectorJIT>::Type_t
peekSpin( PScalarJIT<T> l, llvm::Value * row)
{
  auto d = unary_return<FnPeekSpinVectorJIT>(l);
  d.elem() = peekSpin(l.elem(),row);
  return d;
}


template<class T>
typename UnaryReturn<PScalarJIT<T>, FnPeekSpinMatrixJIT>::Type_t
peekSpin( PScalarJIT<T> l, llvm::Value * row, llvm::Value * col)
{
  auto d = unary_return<FnPeekSpinMatrixJIT>(l);
  d.elem() = peekSpin(l.elem(),row,col);
  return d;
}


template<class T1, class T2>
void
pokeColor(PScalarJIT<T1> l, PScalarJIT<T2> r, llvm::Value * row)
{
  pokeColor(l.elem(),r.elem(),row);
}


template<class T1, class T2>
void
pokeColor(PScalarJIT<T1> l, PScalarJIT<T2> r, llvm::Value * row, llvm::Value * col)
{
  pokeColor(l.elem(),r.elem(),row,col);
}


template<class T1, class T2>
void
pokeSpin(PScalarJIT<T1> l, PScalarJIT<T2> r, llvm::Value * row)
{
  pokeSpin(l.elem(),r.elem(),row);
}


template<class T1, class T2>
void
pokeSpin(PScalarJIT<T1> l, PScalarJIT<T2> r, llvm::Value * row, llvm::Value * col)
{
  pokeSpin(l.elem(),r.elem(),row,col);
}



template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir0Minus>::Type_t
spinProjectDir0Minus( PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir0Minus>(s1);
  d.elem() =  spinProjectDir0Minus(s1.elem());
  return d;
}


template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir0Minus>::Type_t
spinReconstructDir0Minus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir0Minus>(s1);
  d.elem() =  spinReconstructDir0Minus(s1.elem());
  return d;
}



template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir1Minus>::Type_t
spinProjectDir1Minus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir1Minus>(s1);
  d.elem() =  spinProjectDir1Minus(s1.elem());
  return d;
}


template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir1Minus>::Type_t
spinReconstructDir1Minus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir1Minus>(s1);
  d.elem() =  spinReconstructDir1Minus(s1.elem());
  return d;
}



template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir2Minus>::Type_t
spinProjectDir2Minus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir2Minus>(s1);
  d.elem() =  spinProjectDir2Minus(s1.elem());
  return d;
}


template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir2Minus>::Type_t
spinReconstructDir2Minus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir2Minus>(s1);
  d.elem() =  spinReconstructDir2Minus(s1.elem());
  return d;
}



template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir3Minus>::Type_t
spinProjectDir3Minus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir3Minus>(s1);
  d.elem() =  spinProjectDir3Minus(s1.elem());
  return d;
}


template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir3Minus>::Type_t
spinReconstructDir3Minus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir3Minus>(s1);
  d.elem() =  spinReconstructDir3Minus(s1.elem());
  return d;
}



template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir0Plus>::Type_t
spinProjectDir0Plus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir0Plus>(s1);
  d.elem() =  spinProjectDir0Plus(s1.elem());
  return d;
}


template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir0Plus>::Type_t
spinReconstructDir0Plus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir0Plus>(s1);
  d.elem() =  spinReconstructDir0Plus(s1.elem());
  return d;
}



template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir1Plus>::Type_t
spinProjectDir1Plus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir1Plus>(s1);
  d.elem() =  spinProjectDir1Plus(s1.elem());
  return d;
}


template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir1Plus>::Type_t
spinReconstructDir1Plus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir1Plus>(s1);
  d.elem() =  spinReconstructDir1Plus(s1.elem());
  return d;
}



template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir2Plus>::Type_t
spinProjectDir2Plus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir2Plus>(s1);
  d.elem() =  spinProjectDir2Plus(s1.elem());
  return d;
}


template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir2Plus>::Type_t
spinReconstructDir2Plus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir2Plus>(s1);
  d.elem() =  spinReconstructDir2Plus(s1.elem());
  return d;
}



template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir3Plus>::Type_t
spinProjectDir3Plus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinProjectDir3Plus>(s1);
  d.elem() =  spinProjectDir3Plus(s1.elem());
  return d;
}


template<class T>
typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir3Plus>::Type_t
spinReconstructDir3Plus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnSpinReconstructDir3Plus>(s1);
  d.elem() =  spinReconstructDir3Plus(s1.elem());
  return d;
}


template<class T>
typename UnaryReturn<PScalarJIT<T>, FnChiralProjectPlus>::Type_t
chiralProjectPlus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnChiralProjectPlus>(s1);
  d.elem() =  chiralProjectPlus(s1.elem());
  return d;
}


template<class T>
typename UnaryReturn<PScalarJIT<T>, FnChiralProjectMinus>::Type_t
chiralProjectMinus(PScalarJIT<T> s1)
{
  auto d = unary_return<FnChiralProjectMinus>(s1);
  d.elem() =  chiralProjectMinus(s1.elem());
  return d;
}



template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnQuarkContract13>::Type_t
quarkContract13(PScalarJIT<T1> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnQuarkContract13>(s1,s2);
  d.elem() =  quarkContract13(s1.elem(), s2.elem());
  return d;
}

template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnQuarkContract14>::Type_t
quarkContract14(PScalarJIT<T1> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnQuarkContract14>(s1,s2);
  d.elem() =  quarkContract14(s1.elem(), s2.elem());
  return d;
}

template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnQuarkContract23>::Type_t
quarkContract23(PScalarJIT<T1> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnQuarkContract23>(s1,s2);
  d.elem() =  quarkContract23(s1.elem(), s2.elem());
  return d;
}

template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnQuarkContract24>::Type_t
quarkContract24(PScalarJIT<T1> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnQuarkContract24>(s1,s2);
  d.elem() =  quarkContract24(s1.elem(), s2.elem());
  return d;
}

template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnQuarkContract12>::Type_t
quarkContract12(PScalarJIT<T1> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnQuarkContract12>(s1,s2);
  d.elem() =  quarkContract12(s1.elem(), s2.elem());
  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnQuarkContract34>::Type_t
quarkContract34(PScalarJIT<T1> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnQuarkContract34>(s1,s2);
  d.elem() =  quarkContract34(s1.elem(), s2.elem());
  return d;
}



template<class T1, class T2, class T3>
struct TrinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, PScalarJIT<T3>, FnColorContract> {
  typedef PScalarJIT<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};



template<class T1, class T2, class T3>
typename TrinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, PScalarJIT<T3>, FnColorContract>::Type_t
colorContract(PScalarJIT<T1> s1, PScalarJIT<T2> s2, const PScalarJIT<T3>& s3)
{
  auto d = trinary_return<FnColorContract>(s1,s2,s3);

  d.elem() = colorContract(s1.elem(), s2.elem(), s3.elem());

  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnColorVectorContract>::Type_t
colorVectorContract(PScalarJIT<T1> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnColorVectorContract>(s1,s2);

  d.elem() =  colorVectorContract(s1.elem(), s2.elem());

  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnColorCrossProduct>::Type_t
colorCrossProduct(PScalarJIT<T1> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnColorCrossProduct>(s1,s2);

  d.elem() =  colorCrossProduct(s1.elem(), s2.elem());

  return d;
}



template<class T>
typename UnaryReturn<PScalarJIT<T>, FnLocalNorm2>::Type_t
localNorm2(PScalarJIT<T> s1)
{
  auto d = unary_return<FnLocalNorm2>(s1);
  
  d.elem() = localNorm2(s1.elem());

  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(PScalarJIT<T1> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnLocalInnerProduct>(s1,s2);
  
  d.elem() = localInnerProduct(s1.elem(), s2.elem());

  return d;
}


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnLocalColorInnerProduct>::Type_t
localColorInnerProduct(PScalarJIT<T1> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnLocalColorInnerProduct>(s1,s2);

  d.elem() = localColorInnerProduct(s1.elem(), s2.elem());

  return d;
}

  
template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(PScalarJIT<T1> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnLocalInnerProductReal>(s1,s2);

  d.elem() = localInnerProductReal(s1.elem(), s2.elem());

  return d;
}



template<class T1, class T2, class T3>
struct TrinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, PScalarJIT<T3>, FnWhere> {
  typedef PScalarJIT<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};


template<class T1, class T2, class T3>
typename TrinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, PScalarJIT<T3>, FnWhere>::Type_t
where(PScalarJIT<T1> a, PScalarJIT<T2> b, PScalarJIT<T3> c)
{
  return where(a.elem(), b.elem(), c.elem());
}


} // namespace QDP

#endif
