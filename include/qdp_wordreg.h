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


  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpAdd>::Type_t
  operator+(const WordREG<T1>& l, const WordREG<T2>& r)
  {
    typedef typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpAdd>::Type_t Ret_t;
    jit_value_t new_tmp = jit_ins_add( l.get_val() , r.get_val() );
    Ret_t tmp;
    tmp.setup(new_tmp);
    return tmp;
  }


  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpSubtract>::Type_t
  operator-(const WordREG<T1>& l, const WordREG<T2>& r)
  {
    typedef typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpSubtract>::Type_t Ret_t;
    jit_value_t new_tmp = jit_ins_sub( l.get_val() , r.get_val() );
    Ret_t tmp;
    tmp.setup(new_tmp);
    return tmp;
  }


  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpMultiply>::Type_t
  operator*(const WordREG<T1>& l, const WordREG<T2>& r)
  {
    typedef typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpMultiply>::Type_t Ret_t;
    jit_value_t new_tmp = jit_ins_mul( l.get_val() , r.get_val() );
    Ret_t tmp;
    tmp.setup( new_tmp );
    return tmp;
  }


  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpDivide>::Type_t
  operator/(const WordREG<T1>& l, const WordREG<T2>& r)
  {
    typedef typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpDivide>::Type_t Ret_t;
    jit_value_t new_tmp = jit_ins_div( l.get_val() , r.get_val() );
    Ret_t tmp;
    tmp.setup( new_tmp );
    return tmp;
  }

#if 0
  template<class T1>
  inline typename UnaryReturn<WordREG<T1>, OpUnaryMinus>::Type_t
  operator-(const WordREG<T1>& l)
  {
    typedef typename UnaryReturn<WordREG<T1>, OpUnaryMinus>::Type_t  Ret_t;
    jit_value_t new_tmp = jit_ins_neg( l.get_val() );
    Ret_t tmp;
    tmp.setup( l.get_func() , new_tmp );
    return tmp;
  }
#endif  



#if 0
template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpLeftShift > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpLeftShift>::Type_t
operator<<(const WordREG<T1>& l, const WordREG<T2>& r)
{
  return l.elem() << r.elem();
}
#endif


template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpRightShift > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpRightShift>::Type_t
operator>>(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typedef typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpRightShift>::Type_t Ret_t;
  jit_value_t new_tmp = jit_ins_shr( l.get_val() , r.get_val() );
  Ret_t tmp;
  tmp.setup( new_tmp );
  return tmp;
}


#if 0
template<class T1, class T2 >
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpMod>::Type_t
operator%(const WordREG<T1>& l, const WordREG<T2>& r)
{
  return l.elem() % r.elem();
}

template<class T1, class T2 >
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseXor>::Type_t
operator^(const WordREG<T1>& l, const WordREG<T2>& r)
{
  return l.elem() ^ r.elem();
}
#endif 



template<class T1, class T2 >
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseAnd>::Type_t
operator&(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typedef typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseAnd>::Type_t Ret_t;
  jit_value_t new_tmp = jit_ins_bit_and( l.get_val() , r.get_val() );
  Ret_t tmp;
  tmp.setup( new_tmp );
  return tmp;
}




#if 0
template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseOr>::Type_t
operator|(const WordREG<T1>& l, const WordREG<T2>& r)
{
  return l.elem() | r.elem();
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
  return l.elem() < r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpLE > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpLE>::Type_t
operator<=(const WordREG<T1>& l, const WordREG<T2>& r)
{
  return l.elem() <= r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpGT > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpGT>::Type_t
operator>(const WordREG<T1>& l, const WordREG<T2>& r)
{
  return l.elem() > r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpGE > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpGE>::Type_t
operator>=(const WordREG<T1>& l, const WordREG<T2>& r)
{
  return l.elem() >= r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpEQ > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpEQ>::Type_t
operator==(const WordREG<T1>& l, const WordREG<T2>& r)
{
  return l.elem() == r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpNE > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpNE>::Type_t
operator!=(const WordREG<T1>& l, const WordREG<T2>& r)
{
  return l.elem() != r.elem();
}


template<class T1, class T2>
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpAnd > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpAnd>::Type_t
operator&&(const WordREG<T1>& l, const WordREG<T2>& r)
{
  return l.elem() && r.elem();
}


template<class T1, class T2>
struct BinaryReturn<WordREG<T1>, WordREG<T2>, OpOr > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpOr>::Type_t
operator||(const WordREG<T1>& l, const WordREG<T2>& r)
{
  return l.elem() || r.elem();
}
#endif


} // namespace QDP

#endif
