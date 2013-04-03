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

    WordREG(int i);
    WordREG(double f);

    WordREG(const WordREG& rhs) {
      assert(rhs.get_val());
      val = jit_val_create_copy( rhs.get_val() );
      setup_m=true;
    }

    void setup(jit_value_t v);

    void setup(const WordJIT<T>& wj) {
      val = jit_ins_load( wj.getAddress() , 0 , jit_type<T>::value );
      setup_m=true;
    }

    void replace(const WordREG& rhs ) {
      //assert( rhs.get_val()->get_type() == jit_type<T>::value );
      if (val) {
	jit_ins_mov_no_create( val , rhs.get_val() );	
      } else {
	val = rhs.get_val();
      }
      setup_m = true;
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
      setup(rhs.get_val());
      // val = jit_val_create_convert( getFunc(rhs) , jit_type<T>::value , rhs.get_val() );
      // setup_m = true;
    }

    template<class T1>
    inline
    WordREG& operator=(const WordREG<T1>& rhs) 
    {
      setup(rhs.get_val());
      //val = jit_val_create_convert( getFunc(*this) , jit_type<T>::value , rhs.get_val() );
      return *this;
    }


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





  template<>
  WordREG<double>::WordREG(int i);

  template<>
  WordREG<double>::WordREG(double f);

  template<>
  WordREG<float>::WordREG(int i);

  template<>
  WordREG<float>::WordREG(double f);

  template<>
  WordREG<int>::WordREG(int i);

  template<>
  WordREG<int>::WordREG(double f);



  template<>
  void WordREG<int>::setup(jit_value_t v);

  template<>
  void WordREG<float>::setup(jit_value_t v);

  template<>
  void WordREG<double>::setup(jit_value_t v);

  template<>
  void WordREG<bool>::setup(jit_value_t v);





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


// Ceil(ing)
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnCeil>::Type_t
ceil(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnCeil>::Type_t ret;
  ret.setup( jit_ins_ceil( s1.get_val() ) );
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

// Sqrt
template<class T1>
inline typename UnaryReturn<WordREG<T1>, FnSqrt>::Type_t
sqrt(const WordREG<T1>& s1)
{
  typename UnaryReturn<WordREG<T1>, FnSqrt>::Type_t ret;
  ret.setup( jit_ins_sqrt( s1.get_val() ) );
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
  ret.setup( jit_ins_mul( s1.get_val() , s2.get_val() ) );
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
  ret.setup( jit_ins_mul( s1.get_val() , s2.get_val() ) );
  return ret;
}


  inline void 
  zero_rep(WordREG<double>& dest) 
  {
    dest.replace( WordREG<double>(0.) );
  }

  inline void 
  zero_rep(WordREG<float>& dest) 
  {
    dest.replace( WordREG<float>(0.) );
  }

  inline void 
  zero_rep(WordREG<int>& dest) 
  {
    dest.replace( WordREG<int>(0) );
  }




typename UnaryReturn<WordREG<float>, FnArcCos>::Type_t acos(const WordREG<float>& s1);
typename UnaryReturn<WordREG<float>, FnArcSin>::Type_t asin(const WordREG<float>& s1);
typename UnaryReturn<WordREG<float>, FnArcTan>::Type_t atan(const WordREG<float>& s1);
typename UnaryReturn<WordREG<float>, FnCos>::Type_t cos(const WordREG<float>& s1);
typename UnaryReturn<WordREG<float>, FnHypCos>::Type_t cosh(const WordREG<float>& s1);
typename UnaryReturn<WordREG<float>, FnExp>::Type_t exp(const WordREG<float>& s1);
typename UnaryReturn<WordREG<float>, FnLog>::Type_t log(const WordREG<float>& s1);
typename UnaryReturn<WordREG<float>, FnLog10>::Type_t log10(const WordREG<float>& s1);
typename UnaryReturn<WordREG<float>, FnSin>::Type_t sin(const WordREG<float>& s1);
typename UnaryReturn<WordREG<float>, FnHypSin>::Type_t sinh(const WordREG<float>& s1);
typename UnaryReturn<WordREG<float>, FnTan>::Type_t tan(const WordREG<float>& s1);
typename UnaryReturn<WordREG<float>, FnHypTan>::Type_t tanh(const WordREG<float>& s1);
typename BinaryReturn<WordREG<float>, WordREG<float>, FnPow>::Type_t pow(const WordREG<float>& s1, const WordREG<float>& s2);
typename BinaryReturn<WordREG<float>, WordREG<float>, FnArcTan2>::Type_t atan2(const WordREG<float>& s1, const WordREG<float>& s2);


typename UnaryReturn<WordREG<double>, FnSin>::Type_t sin(const WordREG<double>& s1);
typename UnaryReturn<WordREG<double>, FnArcCos>::Type_t acos(const WordREG<double>& s1);
typename UnaryReturn<WordREG<double>, FnArcSin>::Type_t asin(const WordREG<double>& s1);
typename UnaryReturn<WordREG<double>, FnArcTan>::Type_t atan(const WordREG<double>& s1);
typename UnaryReturn<WordREG<double>, FnCos>::Type_t cos(const WordREG<double>& s1);
typename UnaryReturn<WordREG<double>, FnHypCos>::Type_t cosh(const WordREG<double>& s1);
typename UnaryReturn<WordREG<double>, FnExp>::Type_t exp(const WordREG<double>& s1);
typename UnaryReturn<WordREG<double>, FnLog>::Type_t log(const WordREG<double>& s1);
typename UnaryReturn<WordREG<double>, FnLog10>::Type_t log10(const WordREG<double>& s1);
typename UnaryReturn<WordREG<double>, FnSin>::Type_t sin(const WordREG<double>& s1);
typename UnaryReturn<WordREG<double>, FnHypSin>::Type_t sinh(const WordREG<double>& s1);
typename UnaryReturn<WordREG<double>, FnTan>::Type_t tan(const WordREG<double>& s1);
typename UnaryReturn<WordREG<double>, FnHypTan>::Type_t tanh(const WordREG<double>& s1);
typename BinaryReturn<WordREG<double>, WordREG<double>, FnPow>::Type_t pow(const WordREG<double>& s1, const WordREG<double>& s2);
typename BinaryReturn<WordREG<double>, WordREG<double>, FnArcTan2>::Type_t atan2(const WordREG<double>& s1, const WordREG<double>& s2);








} // namespace QDP

#endif
