#ifndef QDP_WORDJIT_H
#define QDP_WORDJIT_H


namespace QDP {

  template<class T>
  class WordJIT 
  {
    llvm::Value *     m_base;
    NestedIndex index;
    JitDataFormat data_format_;

    bool setup_m = false;

  public:
    typedef T Type_t;
    
    WordJIT( llvm::Value * base , JitDataFormat fmt , NestedIndex args = {} ): m_base(base), index(args), data_format_(fmt), setup_m(true) {}

    WordJIT(const WordJIT& rhs): m_base(rhs.m_base), index(rhs.index), data_format_(rhs.data_format_), setup_m(rhs.setup_m)
    {}


    Shape shape() const {
      Shape dummy;
      return dummy;
    }


    llvm::Value *     get_base() const { return m_base; }
    NestedIndex get_index() const { return index; }
    JitDataFormat data_format() const { return data_format_; }

    llvm::Value *     get_offset() const 
    {
      assert(setup_m);
      return datalayout( data_format_.order , index );
    }

    void setup( llvm::Value * base , JitDataFormat fmt , NestedIndex args )
    {
      m_base = base;
      data_format_ = fmt;
      index = args;
      setup_m = true;
    }

    void setup( llvm::Value * base , JitDataFormat fmt , NestedIndex args , Shape shape )
    {
      m_base = base;
      data_format_ = fmt;
      index = args;
      setup_m = true;
    }
    
    WordJIT() {}


    void store(llvm::Value* val) const
    {
      assert(setup_m);
      llvm::Value* v = llvm_cast( llvm_get_type<T>() , val );
      if ( data_format_.half() )
	llvm_store_ptr_idx<jit_half_t>( llvm_cast( llvm_get_type<jit_half_t>() , v ) , get_base() , get_offset() );
      else
	llvm_store_ptr_idx<T>( v , get_base() , get_offset() );
    }


    llvm::Value* elem() const
    {
      if ( data_format_.half() )
	return llvm_cast( llvm_get_type<T>() , llvm_load_ptr_idx<jit_half_t>( get_base() , get_offset() ) );
      return llvm_load_ptr_idx<T>( get_base() , get_offset() );
    }


    llvm::Value* get_val() const
    {
      return elem();
    }

    
    WordJIT operator=(T val)
    {
      store(llvm_create_value(val));
      return *this;
    }
    
    
    WordJIT operator=(llvm::Value* val)
    {
      store(val);
      return *this;
    }

    
    template<class T1>
    WordJIT operator=(WordJIT<T1> s1)
    {
      store(s1.elem());
      return *this;
    }

    WordJIT operator=( WordJIT s1)
    {
      store(s1.elem());
      return *this;
    }

    
    template<class T1>
    WordJIT operator+=(WordJIT<T1> rhs) 
    {
      store( llvm_add( elem() , rhs.elem() ) );
      return *this;
    }


    template<class T1>
    WordJIT operator-=(WordJIT<T1> rhs) 
    {
      store( llvm_sub( elem() , rhs.elem() ) );
      return *this;
    }


    template<class T1>
    WordJIT operator*=(WordJIT<T1> rhs) 
    {
      store( llvm_mul( elem() , rhs.elem() ) );
      return *this;
    }


    template<class T1>
    WordJIT operator/=(WordJIT<T1> rhs) 
    {
      store( llvm_div( elem() , rhs.elem() ) );
      return *this;
    }


    template<class T1>
    WordJIT operator%=(WordJIT<T1> rhs) 
    {
      store( llvm_rem( elem() , rhs.elem() ) );
      return *this;
    }

    
    template<class T1>
    WordJIT operator|=(WordJIT<T1> rhs) 
    {
      store( llvm_or( elem() , rhs.elem() ) );
      return *this;
    }

    
    template<class T1>
    WordJIT operator&=(WordJIT<T1> rhs) 
    {
      store( llvm_and( elem() , rhs.elem() ) );
      return *this;
    }

    
    template<class T1>
    WordJIT operator^=(WordJIT<T1> rhs) 
    {
      store( llvm_xor( elem() , rhs.elem() ) );
      return *this;
    }
    

    template<class T1>
    WordJIT operator<<=(WordJIT<T1> rhs) 
    {
      store( llvm_shl( elem() , rhs.elem() ) );
      return *this;
    }


    template<class T1>
    WordJIT operator>>=(WordJIT<T1> rhs) 
    {
      store( llvm_shr( elem() , rhs.elem() ) );
      return *this;
    }
  };





  template<class T>
  class WordSize< WordJIT<T> >
  {
  public:
    static int value( const Shape& ret )
    {
      return 1;
    }
  };


  template <class T>
  struct IsWordVec< WordJIT<T> >
  {
    constexpr static bool value = false;
  };

  template <class T>
  struct IsWordVec< WordVecJIT<T> >
  {
    constexpr static bool value = true;
  };

  template<class T>
  struct ScalarType<WordJIT<T> >
  {
    typedef WordJIT< T > Type_t;
  };
  
  template<class T>
  struct InternalScalar<WordJIT<T> > {
    typedef WordJIT<T>  Type_t;
  };

  
  template<class T>
  struct RealScalar<WordJIT<T> > {
    typedef WordJIT<typename RealScalar<T>::Type_t>  Type_t;
  };


  template<class T> 
  struct JITType< WordJIT<T> >
  {
    typedef WordJIT<T>  Type_t;
  };
  
  

  template<class T> 
  struct WordType<WordJIT<T> >
  {
    typedef T  Type_t;
  };


  template<class T>
  struct UnaryReturn<WordJIT<T>, FnSeedToFloat> {
    typedef WordJIT<typename UnaryReturn<T, FnSeedToFloat>::Type_t>  Type_t;
  };

  
  template<class T>
  struct UnaryReturn<WordJIT<T>, FnIsFinite> {
    typedef WordJIT<typename UnaryReturn<T, FnIsFinite>::Type_t>  Type_t;
  };


  template<class T, class Op>
  struct UnaryReturn<WordJIT<T>, Op> {
    typedef WordJIT<typename UnaryReturn<T, Op>::Type_t>  Type_t;
  };



  template<class T1, class T2, class Op>
  struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, Op> {
    typedef WordJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };



  template<class T1>
  typename UnaryReturn<WordJIT<T1>, OpNot>::Type_t
  operator!(WordJIT<T1> l)
  {
    auto d = unary_return<OpNot>(l);

    d.store( llvm_not( l.elem() ) );
    
    return d;
  }


  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpAdd>::Type_t
  operator+(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpAdd>(l,r);

    d.store( llvm_add( l.elem() , r.elem() ) );
    
    return d;
  }


  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpSubtract>::Type_t
  operator-(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpSubtract>(l,r);

    d.store( llvm_sub( l.elem() , r.elem() ) );
    
    return d;
  }


  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpMultiply>::Type_t
  operator*(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpMultiply>(l,r);

    d.store( llvm_mul( l.elem() , r.elem() ) );
    
    return d;
  }


  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpDivide>::Type_t
  operator/(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpDivide>(l,r);

    d.store( llvm_div( l.elem() , r.elem() ) );
    
    return d;
  }


  template<class T1>
  typename UnaryReturn<WordJIT<T1>, OpUnaryMinus>::Type_t
  operator-(WordJIT<T1> l)
  {
    auto d = unary_return<OpUnaryMinus>(l);

    d.store( llvm_neg( l.elem() ) );
    
    return d;
  }


  template<class T1>
  typename UnaryReturn<WordJIT<T1>, OpUnaryPlus>::Type_t
  operator+(WordJIT<T1> l)
  {
    auto d = unary_return<OpUnaryPlus>(l);

    d.store( l.elem() );
    
    return d;
  }



  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpLeftShift>::Type_t
  operator<<(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpLeftShift>(l,r);

    d.store( llvm_shl( l.elem() , r.elem() ) );
    
    return d;
  }



  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpRightShift>::Type_t
  operator>>(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpRightShift>(l,r);

    d.store( llvm_shr( l.elem() , r.elem() ) );
    
    return d;
  }



  template<class T1, class T2 >
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpMod>::Type_t
  operator%(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpMod>(l,r);

    d.store( llvm_rem( l.elem() , r.elem() ) );
    
    return d;
  }




  template<class T1, class T2 >
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpBitwiseXor>::Type_t
  operator^(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpBitwiseXor>(l,r);

    d.store( llvm_xor( l.elem() , r.elem() ) );
    
    return d;
  }


  template<class T1, class T2 >
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpBitwiseAnd>::Type_t
  operator&(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpBitwiseAnd>(l,r);

    d.store( llvm_and( l.elem() , r.elem() ) );
    
    return d;
  }



  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpBitwiseOr>::Type_t
  operator|(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpBitwiseOr>(l,r);

    d.store( llvm_or( l.elem() , r.elem() ) );
    
    return d;
  }



  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpLT>::Type_t
  operator<(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpLT>(l,r);

    d.store( llvm_lt( l.elem() , r.elem() ) );
    
    return d;
  }


  template<class T1, class T2 >
  struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpLE > {
    typedef WordJIT<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
  };

  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpLE>::Type_t
  operator<=(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpLE>(l,r);

    d.store( llvm_le( l.elem() , r.elem() ) );
    
    return d;
  }


  template<class T1, class T2 >
  struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpGT > {
    typedef WordJIT<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
  };

  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpGT>::Type_t
  operator>(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpGT>(l,r);

    d.store( llvm_gt( l.elem() , r.elem() ) );
    
    return d;
  }



  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpGE>::Type_t
  operator>=(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpGE>(l,r);

    d.store( llvm_ge( l.elem() , r.elem() ) );
    
    return d;
  }



  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpEQ>::Type_t
  operator==(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpEQ>(l,r);

    d.store( llvm_eq( l.elem() , r.elem() ) );
    
    return d;
  }



  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpNE>::Type_t
  operator!=(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpNE>(l,r);

    d.store( llvm_ne( l.elem() , r.elem() ) );
    
    return d;
  }




  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpAnd>::Type_t
  operator&&(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpAnd>(l,r);

    d.store( llvm_and( l.elem() , r.elem() ) );
    
    return d;
  }



  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpOr>::Type_t
  operator||(WordJIT<T1> l, WordJIT<T2> r)
  {
    auto d = binary_return<OpOr>(l,r);

    d.store( llvm_or( l.elem() , r.elem() ) );
    
    return d;
  }


  template<class T1, class T2, class T3>
  struct TrinaryReturn<WordJIT<T1>, WordJIT<T2>, WordJIT<T3>, FnWhere > {
    typedef WordJIT<typename BinaryReturn<T2, T3, FnWhere>::Type_t>  Type_t;
  };


  template<class T1, class T2, class T3>
  typename TrinaryReturn<WordJIT<T1>, WordJIT<T2>, WordJIT<T3>, FnWhere >::Type_t
  where( WordJIT<T1> a , WordJIT<T2> b , WordJIT<T3> c )
  {
    auto d = trinary_return<FnWhere>(a,b,c);

    d.store( jit_ternary( a.elem() , b.elem() , c.elem() ) );
    
    return d;
  }


  template<class T>
  struct UnaryReturn<WordJIT<T>, FnLocalNorm2 > {
    typedef WordJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
  };

  
  template<class T>
  typename UnaryReturn<WordJIT<T>, FnLocalNorm2>::Type_t
  localNorm2(WordJIT<T> s1)
  {
    auto d = unary_return<FnLocalNorm2>(s1);
    
    d.store( llvm_mul( s1.elem() , s1.elem() ) );
    
    return d;
  }



  template<class T, class T1, class T2, class T3>
  void
  fill_random_jit(WordJIT<T> d, T1 seed, T2 skewed_seed, const T3& seed_mult)
  {
    d.store( seedToFloat(skewed_seed).elem().elem().elem().get_val() );

    seed        = seed        * seed_mult;
    skewed_seed = skewed_seed * seed_mult;
  }

  

  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnLocalInnerProduct>::Type_t
  localInnerProduct(WordJIT<T1> s1, WordJIT<T2> s2)
  {
    auto d = binary_return<FnLocalInnerProduct>(s1,s2);
    
    d.store( llvm_mul( s1.elem() , s2.elem() ) );
    
    return d;
  }


  

  template<class T1, class T2>
  typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnLocalInnerProductReal>::Type_t
  localInnerProductReal(WordJIT<T1> s1, WordJIT<T2> s2)
  {
    auto d = binary_return<FnLocalInnerProductReal>(s1,s2);
    
    d.store( llvm_mul( s1.elem() , s2.elem() ) );
    
    return d;
  }



  template<class T>
  void zero_rep(WordJIT<T> dest)
  {
    dest.store( llvm_create_value( (T)0 ) );
  }



  typename UnaryReturn<WordJIT<float>, FnCeil>::Type_t ceil( WordJIT<float> s1);
  typename UnaryReturn<WordJIT<float>, FnFloor>::Type_t floor( WordJIT<float> s1);
  typename UnaryReturn<WordJIT<float>, FnFabs>::Type_t fabs( WordJIT<float> s1);
  typename UnaryReturn<WordJIT<float>, FnSqrt>::Type_t sqrt( WordJIT<float> s1);
  typename UnaryReturn<WordJIT<float>, FnArcCos>::Type_t acos( WordJIT<float> s1);
  typename UnaryReturn<WordJIT<float>, FnArcSin>::Type_t asin( WordJIT<float> s1);
  typename UnaryReturn<WordJIT<float>, FnArcTan>::Type_t atan( WordJIT<float> s1);
  typename UnaryReturn<WordJIT<float>, FnCos>::Type_t cos( WordJIT<float> s1);
  typename UnaryReturn<WordJIT<float>, FnHypCos>::Type_t cosh( WordJIT<float> s1);
  typename UnaryReturn<WordJIT<float>, FnExp>::Type_t exp( WordJIT<float> s1);
  typename UnaryReturn<WordJIT<float>, FnLog>::Type_t log(WordJIT<float> s1);
  typename UnaryReturn<WordJIT<float>, FnLog10>::Type_t log10( WordJIT<float> s1);
  typename UnaryReturn<WordJIT<float>, FnSin>::Type_t sin( WordJIT<float> s1);
  typename UnaryReturn<WordJIT<float>, FnHypSin>::Type_t sinh( WordJIT<float> s1);
  typename UnaryReturn<WordJIT<float>, FnTan>::Type_t tan( WordJIT<float> s1);
  typename UnaryReturn<WordJIT<float>, FnHypTan>::Type_t tanh( WordJIT<float> s1);

  typename UnaryReturn<WordJIT<float>, FnIsFinite>::Type_t isfinite( WordJIT<float> s1);

  typename UnaryReturn<WordJIT<double>, FnCeil>::Type_t ceil( WordJIT<double> s1);
  typename UnaryReturn<WordJIT<double>, FnFloor>::Type_t floor( WordJIT<double> s1);
  typename UnaryReturn<WordJIT<double>, FnFabs>::Type_t fabs( WordJIT<double> s1);
  typename UnaryReturn<WordJIT<double>, FnSqrt>::Type_t sqrt( WordJIT<double> s1);
  typename UnaryReturn<WordJIT<double>, FnArcCos>::Type_t acos( WordJIT<double> s1);
  typename UnaryReturn<WordJIT<double>, FnArcSin>::Type_t asin( WordJIT<double> s1);
  typename UnaryReturn<WordJIT<double>, FnArcTan>::Type_t atan( WordJIT<double> s1);
  typename UnaryReturn<WordJIT<double>, FnCos>::Type_t cos( WordJIT<double> s1);
  typename UnaryReturn<WordJIT<double>, FnHypCos>::Type_t cosh( WordJIT<double> s1);
  typename UnaryReturn<WordJIT<double>, FnExp>::Type_t exp( WordJIT<double> s1);
  typename UnaryReturn<WordJIT<double>, FnLog>::Type_t log( WordJIT<double> s1);
  typename UnaryReturn<WordJIT<double>, FnLog10>::Type_t log10( WordJIT<double> s1);
  typename UnaryReturn<WordJIT<double>, FnSin>::Type_t sin( WordJIT<double> s1);
  typename UnaryReturn<WordJIT<double>, FnHypSin>::Type_t sinh( WordJIT<double> s1);
  typename UnaryReturn<WordJIT<double>, FnTan>::Type_t tan( WordJIT<double> s1);
  typename UnaryReturn<WordJIT<double>, FnHypTan>::Type_t tanh( WordJIT<double> s1);

  typename UnaryReturn<WordJIT<double>, FnIsFinite>::Type_t isfinite( WordJIT<double> s1);

  typename BinaryReturn<WordJIT<float>, WordJIT<float>, FnPow>::Type_t pow( WordJIT<float> s1,  WordJIT<float> s2);
  typename BinaryReturn<WordJIT<double>, WordJIT<double>, FnPow>::Type_t pow( WordJIT<double> s1,  WordJIT<double> s2);
  typename BinaryReturn<WordJIT<double>, WordJIT<float>, FnPow>::Type_t pow( WordJIT<double> s1,  WordJIT<float> s2);
  typename BinaryReturn<WordJIT<float>, WordJIT<double>, FnPow>::Type_t pow( WordJIT<float> s1,  WordJIT<double> s2);

  typename BinaryReturn<WordJIT<float>, WordJIT<float>, FnArcTan2>::Type_t   atan2( WordJIT<float> s1 ,  WordJIT<float> s2);
  typename BinaryReturn<WordJIT<double>, WordJIT<double>, FnArcTan2>::Type_t atan2( WordJIT<double> s1,  WordJIT<double> s2);
  typename BinaryReturn<WordJIT<double>, WordJIT<float>, FnArcTan2>::Type_t  atan2( WordJIT<double> s1,  WordJIT<float> s2);
  typename BinaryReturn<WordJIT<float>, WordJIT<double>, FnArcTan2>::Type_t  atan2( WordJIT<float> s1 ,  WordJIT<double> s2);



template<class T>
WordJIT<T> operator*( REAL f , const WordJIT<T>& rhs )
{
  auto fj = create_word<T>(f);
  return fj * rhs;
}

template<class T>
WordJIT<T> operator*( const WordJIT<T>& rhs , REAL f )
{
  auto fj = create_word<T>(f);
  return rhs * fj;
}

template<class T>
WordJIT<T> operator+( REAL f , const WordJIT<T>& rhs )
{
  auto fj = create_word<T>(f);
  return fj + rhs;
}

template<class T>
WordJIT<T> operator+( const WordJIT<T>& rhs , REAL f )
{
  auto fj = create_word<T>(f);
  return rhs + fj;
}

template<class T>
WordJIT<T> operator-( REAL f , const WordJIT<T>& rhs )
{
  auto fj = create_word<T>(f);
  return fj - rhs;
}

template<class T>
WordJIT<T> operator-( const WordJIT<T>& rhs , REAL f )
{
  auto fj = create_word<T>(f);
  return rhs - fj;
}

template<class T>
WordJIT<T> operator/( REAL f , const WordJIT<T>& rhs )
{
  auto fj = create_word<T>(f);
  return fj / rhs;
}

template<class T>
WordJIT<T> operator/( const WordJIT<T>& rhs , REAL f )
{
  auto fj = create_word<T>(f);
  return rhs / fj;
}

  

template<class T>
decltype(auto) operator<( REAL f , const WordJIT<T>& rhs )
{
  auto fj = create_word<T>(f);
  return fj < rhs;
}

template<class T>
decltype(auto) operator<( const WordJIT<T>& rhs , REAL f )
{
  auto fj = create_word<T>(f);
  return rhs < fj;
}


  
} // namespace QDP

#endif
