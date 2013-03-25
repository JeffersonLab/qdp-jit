// -*- C++ -*-
//
// QDP data parallel interface
//


#ifndef QDP_WORDREG_H
#define QDP_WORDREG_H



namespace QDP {


  template<class T>
  class WordREG 
  {
  public:

    // Default constructing should be possible
    // then there is no need for MPL index when
    // construction a PMatrix<T,N>
    WordREG(): setup_m(false) {}

    WordREG(int i) {
      val = jit_val_create_const_int( i );
      setup_m=true;
    }

    WordREG(float f) {
      val = jit_val_create_const_float( f );
      setup_m=true;
    }

    WordREG(double f) {
      val = jit_val_create_const_float( f );
      setup_m=true;
    }
    
    WordREG(const WordREG& rhs) {
      assert(rhs.get_val());
      val = jit_val_create_copy( rhs.get_val() );
      setup_m=true;
    }

    void setup(jit_value_t v) {
      assert(v);
      val=v;
      setup_m=true;
    }

    void setup(const WordJIT<T>& wj) {
      val = jit_ins_load( wj.getAddress() , 0 , jit_type<T>::value );
      setup_m=true;
    }

    void replace(const WordREG& rhs ) {
      assert(val);
      jit_ins_mov_no_create( val , rhs.get_val() );
    }


    jit_value_t    get_val() const  { 
      assert(setup_m);
      return val; 
    }

    friend void swap(WordREG& lhs,WordREG& rhs) {
      std::swap( lhs.val , rhs.val );
      std::swap( lhs.setup_m , rhs.setup_m );
    }

    WordREG& operator=(WordREG rhs) {
      swap(*this,rhs);
      return *this;
    }

    template<class T1>
    WordREG(const WordREG<T1>& rhs) 
    {
      val = jit_val_create_convert( getFunc(rhs) , jit_type<T>::value , rhs.get_val() );
      setup_m = true;
    }

    // template<class T1>
    // inline
    // WordREG& operator=(const WordREG<T1>& rhs) 
    // {
    //   val = jit_val_create_convert( getFunc(*this) , jit_type<T>::value , rhs.get_val() );
    //   return *this;
    // }


    //! WordREG += WordREG
    template<class T1>
    inline
    WordREG& operator+=(const WordREG<T1>& rhs) 
    {
      val = jit_ins_add( val , rhs.get_val() );
      return *this;
    }

    //! WordREG -= WordREG
    template<class T1>
    inline
    WordREG& operator-=(const WordREG<T1>& rhs) 
    {
      val = jit_ins_sub( val , rhs.get_val() );
      return *this;
    }

    //! WordREG *= WordREG
    template<class T1>
    inline
    WordREG& operator*=(const WordREG<T1>& rhs) 
    {
      val = jit_ins_mul( val , rhs.get_val() );
      return *this;
    }

    //! WordREG /= WordREG
    template<class T1>
    inline
    WordREG& operator/=(const WordREG<T1>& rhs) 
    {
      val = jit_ins_div( val , rhs.get_val() );
      return *this;
    }

    //! WordREG %= WordREG
    template<class T1>
    inline
    WordREG& operator%=(const WordREG<T1>& rhs) 
    {
      val = jit_ins_mod( val , rhs.get_val() );
      return *this;
    }

    //! WordREG |= WordREG
    template<class T1>
    inline
    WordREG& operator|=(const WordREG<T1>& rhs) 
    {
      val = jit_ins_or( val , rhs.get_val() );
      return *this;
    }

    //! WordREG &= WordREG
    template<class T1>
    inline
    WordREG& operator&=(const WordREG<T1>& rhs) 
    {
      val = jit_ins_and( val , rhs.get_val() );
      return *this;
    }

    //! WordREG ^= WordREG
    template<class T1>
    inline
    WordREG& operator^=(const WordREG<T1>& rhs) 
    {
      val = jit_ins_xor( val , rhs.get_val() );
      return *this;
    }

    //! WordREG <<= WordREG
    template<class T1>
    inline
    WordREG& operator<<=(const WordREG<T1>& rhs) 
    {
      val = jit_ins_shl( val , rhs.get_val() );
      return *this;
    }

    //! WordREG >>= WordREG
    template<class T1>
    inline
    WordREG& operator>>=(const WordREG<T1>& rhs) 
    {
      val = jit_ins_shr( val , rhs.get_val() );
      return *this;
    }

  private:
    bool setup_m;
    jit_value_t    val;
  };



  template <class T>
  jit_function_t getFunc(const WordREG<T>& l) {
    return getFunc( l.get_val() );
  }


//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

  template<class T>
  struct InternalScalar<WordREG<T> > {
    typedef WordREG<T>  Type_t;
  };

  template<class T>
  struct RealScalar<WordREG<T> > {
    typedef WordREG<typename RealScalar<T>::Type_t>  Type_t;
  };


  template<class T> 
  struct JITType< WordREG<T> >
  {
    typedef WordJIT<T>  Type_t;
  };
  


  template<class T> 
  struct WordType<WordREG<T> >
  {
    typedef T  Type_t;
  };


  template<class T>
  struct UnaryReturn<WordREG<T>, FnSeedToFloat> {
    typedef WordREG<typename UnaryReturn<T, FnSeedToFloat>::Type_t>  Type_t;
  };


  template<class T>
  inline typename UnaryReturn<WordREG<T>, FnSeedToFloat>::Type_t
  seedToFloat(const WordREG<T>& s1)
  {
    printme<typename UnaryReturn<WordREG<T>, FnSeedToFloat>::Type_t>();
    typename UnaryReturn<WordREG<T>, FnSeedToFloat>::Type_t d;
    assert(!"ni");
    //val = jit_val_create_convert( getFunc(*this) , jit_type<T>::value , rhs.get_val() );

    return d;
  }

  

  // Default binary(WordREG,WordREG) -> WordREG
#if 0
  template<class T1, class T2, class Op>
  struct BinaryReturn<WordREG<T1>, WordREG<T2>, Op> {
    typedef WordREG<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };
#endif


  template<class T>
  struct UnaryReturn<WordREG<T>, OpNot > {
    typedef WordREG<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
  };
  
  template<class T1>
  inline typename UnaryReturn<WordREG<T1>, OpNot>::Type_t
  operator!(const WordREG<T1>& l)
  {
    typename UnaryReturn<WordREG<T1>, OpNot>::Type_t ret;
    ret.setup( jit_ins_not( l.get_val() ) );
    return ret;
  }


  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpAdd>::Type_t
  operator+(const WordREG<T1>& l, const WordREG<T2>& r)
  {
    typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpAdd>::Type_t ret;
    ret.setup( jit_ins_add( l.get_val() , r.get_val() ) );
    return ret;
  }


  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpSubtract>::Type_t
  operator-(const WordREG<T1>& l, const WordREG<T2>& r)
  {
    typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpSubtract>::Type_t ret;
    ret.setup( jit_ins_sub( l.get_val() , r.get_val() ) );
    return ret;
  }


  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpMultiply>::Type_t
  operator*(const WordREG<T1>& l, const WordREG<T2>& r)
  {
    typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpMultiply>::Type_t ret;
    ret.setup( jit_ins_mul( l.get_val() , r.get_val() ) );
    return ret;
  }


  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpDivide>::Type_t
  operator/(const WordREG<T1>& l, const WordREG<T2>& r)
  {
    typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpDivide>::Type_t ret;
    ret.setup( jit_ins_div( l.get_val() , r.get_val() ) );
    return ret;
  }


  template<class T1>
  inline typename UnaryReturn<WordREG<T1>, OpUnaryMinus>::Type_t
  operator-(const WordREG<T1>& l)
  {
    typename UnaryReturn<WordREG<T1>, OpUnaryMinus>::Type_t ret;
    ret.setup( jit_ins_neg( l.get_val() ) );
    return ret;
  }





template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpLeftShift > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpLeftShift>::Type_t
operator<<(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpLeftShift>::Type_t ret;
  ret.setup( jit_ins_shl( l.get_val() , r.get_val() ) );
  return ret;
}



template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpRightShift > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpRightShift>::Type_t
operator>>(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpRightShift>::Type_t ret;
  ret.setup( jit_ins_shr( l.get_val() , r.get_val() ) );
  return ret;
}



template<class T1, class T2 >
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpMod>::Type_t
operator%(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpMod>::Type_t ret;
  ret.setup( jit_ins_rem( l.get_val() , r.get_val() ) );
  return ret;
}




template<class T1, class T2 >
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseXor>::Type_t
operator^(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseXor>::Type_t ret;
  ret.setup( jit_ins_xor( l.get_val() , r.get_val() ) );
  return ret;
}


template<class T1, class T2 >
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseAnd>::Type_t
operator&(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseAnd>::Type_t ret;
  ret.setup( jit_ins_and( l.get_val() , r.get_val() ) );
  return ret;
}





template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseOr>::Type_t
operator|(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseOr>::Type_t ret;
  ret.setup( jit_ins_or( l.get_val() , r.get_val() ) );
  return ret;
}



// Comparisons
template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpLT > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpLT>::Type_t
operator<(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpLT>::Type_t ret;
  ret.setup( jit_ins_lt( l.get_val() , r.get_val() ) );
  return ret;
}


template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpLE > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpLE>::Type_t
operator<=(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpLE>::Type_t ret;
  ret.setup( jit_ins_le( l.get_val() , r.get_val() ) );
  return ret;
}


template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpGT > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpGT>::Type_t
operator>(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpGT>::Type_t ret;
  ret.setup( jit_ins_gt( l.get_val() , r.get_val() ) );
  return ret;
}


template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpGE > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpGE>::Type_t
operator>=(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpGE>::Type_t ret;
  ret.setup( jit_ins_ge( l.get_val() , r.get_val() ) );
  return ret;
}


template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpEQ > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpEQ>::Type_t
operator==(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpEQ>::Type_t ret;
  ret.setup( jit_ins_eq( l.get_val() , r.get_val() ) );
  return ret;
}


template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpNE > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpNE>::Type_t
operator!=(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpNE>::Type_t ret;
  ret.setup( jit_ins_ne( l.get_val() , r.get_val() ) );
  return ret;
}



template<class T1, class T2>
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpAnd > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpAnd>::Type_t
operator&&(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpAnd>::Type_t ret;
  ret.setup( jit_ins_and( l.get_val() , r.get_val() ) );
  return ret;
}


template<class T1, class T2>
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpOr > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpOr>::Type_t
operator||(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpOr>::Type_t ret;
  ret.setup( jit_ins_or( l.get_val() , r.get_val() ) );
  return ret;
}


  template<class T1, class T2, class T3>
  inline typename TrinaryReturn<WordREG<T1>, WordREG<T2>, WordREG<T3>, FnWhere >::Type_t
  where(const WordREG<T1> &a, const WordREG<T2> &b, const WordREG<T3> &c)
  {
    typename TrinaryReturn<WordREG<T1>, WordREG<T2>, WordREG<T3>, FnWhere >::Type_t ret;
    ret.setup( jit_ins_selp( getFunc(a) , b.get_val() , c.get_val() , a.get_val() ) );
    return ret;
  }


// ArcCos
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnArcCos>::Type_t
acos(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnArcCos>::Type_t ret;
  ret.setup( jit_ins_acos( s1.get_val() ) );
  return ret;
}

// ArcSin
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnArcSin>::Type_t
asin(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnArcSin>::Type_t ret;
  ret.setup( jit_ins_asin( s1.get_val() ) );
  return ret;
}

// ArcTan
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnArcTan>::Type_t
atan(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnArcTan>::Type_t ret;
  ret.setup( jit_ins_atan( s1.get_val() ) );
  return ret;
}

// Ceil(ing)
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnCeil>::Type_t
ceil(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnCeil>::Type_t ret;
  ret.setup( jit_ins_ceil( s1.get_val() ) );
  return ret;
}

// Cos
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnCos>::Type_t
cos(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnCos>::Type_t ret;
  ret.setup( jit_ins_cos( s1.get_val() ) );
  return ret;
}

// Cosh
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnHypCos>::Type_t
cosh(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnHypCos>::Type_t ret;
  ret.setup( jit_ins_cosh( s1.get_val() ) );
  return ret;
}

// Exp
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnExp>::Type_t
exp(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnExp>::Type_t ret;
  ret.setup( jit_ins_exp( s1.get_val() ) );
  return ret;
}

// Fabs
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnFabs>::Type_t
fabs(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnFabs>::Type_t ret;
  ret.setup( jit_ins_fabs( s1.get_val() ) );
  return ret;
}

// Floor
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnFloor>::Type_t
floor(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnFloor>::Type_t ret;
  ret.setup( jit_ins_floor( s1.get_val() ) );
  return ret;
}

// Log
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnLog>::Type_t
log(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnLog>::Type_t ret;
  ret.setup( jit_ins_log( s1.get_val() ) );
  return ret;
}

// Log10
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnLog10>::Type_t
log10(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnLog10>::Type_t ret;
  ret.setup( jit_ins_log10( s1.get_val() ) );
  return ret;
}

// Sin
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnSin>::Type_t
sin(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnSin>::Type_t ret;
  ret.setup( jit_ins_sin( s1.get_val() ) );
  return ret;
}

// Sinh
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnHypSin>::Type_t
sinh(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnHypSin>::Type_t ret;
  ret.setup( jit_ins_sinh( s1.get_val() ) );
  return ret;
}

// Sqrt
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnSqrt>::Type_t
sqrt(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnSqrt>::Type_t ret;
  ret.setup( jit_ins_sqrt( s1.get_val() ) );
  return ret;
}

// Tan
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnTan>::Type_t
tan(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnTan>::Type_t ret;
  ret.setup( jit_ins_tan( s1.get_val() ) );
  return ret;
}

// Tanh
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnHypTan>::Type_t
tanh(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnHypTan>::Type_t ret;
  ret.setup( jit_ins_tanh( s1.get_val() ) );
  return ret;
}


//! WordREG<T> = pow(WordREG<T> , WordREG<T>)
template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, FnPow>::Type_t
pow(const WordREG<T1>& s1, const WordREG<T2>& s2)
{
  typename UnaryReturn<WordREG<T1>, FnHypTan>::Type_t ret;
  ret.setup( jit_ins_pow( s1.get_val() , s2.get_val() ) );
  return ret;
}

//! WordREG<T> = atan2(WordREG<T> , WordREG<T>)
template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, FnArcTan2>::Type_t
atan2(const WordREG<T1>& s1, const WordREG<T2>& s2)
{
  typename UnaryReturn<WordREG<T1>, FnArcTan2>::Type_t ret;
  ret.setup( jit_ins_atan2( s1.get_val() , s2.get_val() ) );
  return ret;
}


template<class T>
struct UnaryReturn<WordREG<T>, FnLocalNorm2 > {
  typedef WordREG<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<WordREG<T>, FnLocalNorm2>::Type_t
localNorm2(const WordREG<T>& s1)
{
  typename UnaryReturn<WordREG<T>, FnLocalNorm2>::Type_t ret;
  ret.setup( jit_ins_mul( s1.get_val() , s1.get_val() ) );
  return ret;
}




//! WordREG<T> = InnerProduct(adj(WordREG<T1>)*WordREG<T2>)
template<class T1, class T2>
struct BinaryReturn<WordREG<T1>, WordREG<T2>, FnInnerProduct > {
  typedef WordREG<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<WordREG<T1>, WordREG<T2>, FnLocalInnerProduct > {
  typedef WordREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(const WordREG<T1>& s1, const WordREG<T2>& s2)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, FnLocalInnerProduct>::Type_t ret;
  ret.setup( jit_ins_mul( s1.get_val() , s1.get_val() ) );
  return ret;
}


//! WordREG<T> = InnerProductReal(adj(PMatrix<T1>)*PMatrix<T1>)
// Real-ness is eaten at this level
template<class T1, class T2>
struct BinaryReturn<WordREG<T1>, WordREG<T2>, FnInnerProductReal > {
  typedef WordREG<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<WordREG<T1>, WordREG<T2>, FnLocalInnerProductReal > {
  typedef WordREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const WordREG<T1>& s1, const WordREG<T2>& s2)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, FnLocalInnerProductReal>::Type_t ret;
  ret.setup( jit_ins_mul( s1.get_val() , s1.get_val() ) );
  return ret;
}




} // namespace QDP

#endif
