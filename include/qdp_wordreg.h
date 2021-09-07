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
    WordREG() {}


    WordREG(const WordREG& rhs)
    {
      val = rhs.get_val();
    }


    explicit WordREG(llvm::Value * rhs)
    {
      setup(rhs);
    }


    void setup(llvm::Value * v) {
      val = llvm_cast( llvm_get_type<T>() , v );
    }

    void setup(const WordJIT<T>& wj) {
      llvm::Value *val_j = llvm_load_ptr_idx( wj.getBaseReg() , wj.getOffset() );
      setup( val_j );
    }

    void setup_value(const WordJIT<T>& wj) {
      llvm::Value *val_j = wj.getBaseReg();
      setup( val_j );
    }

    
    llvm::Value *get_val() const { return val; }


    WordREG& operator=(const WordREG& rhs) {
      val = rhs.get_val();      
      return *this;
    }

    template<class T1>
    WordREG(const WordREG<T1>& rhs)
    {
      setup(rhs.get_val());
    }

    template<class T1>
    inline
    WordREG& operator=(const WordREG<T1>& rhs) 
    {
      setup(rhs.get_val());
      return *this;
    }


    //! WordREG += WordREG
    template<class T1>
    inline
    WordREG& operator+=(const WordREG<T1>& rhs) 
    {
      val = llvm_add( val , rhs.get_val() );
      return *this;
    }

    //! WordREG -= WordREG
    template<class T1>
    inline
    WordREG& operator-=(const WordREG<T1>& rhs) 
    {
      val = llvm_sub( val , rhs.get_val() );
      return *this;
    }

    //! WordREG *= WordREG
    template<class T1>
    inline
    WordREG& operator*=(const WordREG<T1>& rhs) 
    {
      val = llvm_mul( val , rhs.get_val() );
      return *this;
    }

    //! WordREG /= WordREG
    template<class T1>
    inline
    WordREG& operator/=(const WordREG<T1>& rhs) 
    {
      val = llvm_div( val , rhs.get_val() );
      return *this;
    }

    //! WordREG %= WordREG
    template<class T1>
    inline
    WordREG& operator%=(const WordREG<T1>& rhs) 
    {
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
      val = llvm_mod( val , rhs.get_val() );
      return *this;
#endif
    }

    //! WordREG |= WordREG
    template<class T1>
    inline
    WordREG& operator|=(const WordREG<T1>& rhs) 
    {
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
      val = llvm_or( val , rhs.get_val() );
      return *this;
#endif
    }

    //! WordREG &= WordREG
    template<class T1>
    inline
    WordREG& operator&=(const WordREG<T1>& rhs) 
    {
      val = llvm_and( val , rhs.get_val() );
      return *this;
    }

    //! WordREG ^= WordREG
    template<class T1>
    inline
    WordREG& operator^=(const WordREG<T1>& rhs) 
    {
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
      val = llvm_xor( val , rhs.get_val() );
      return *this;
#endif
    }

    //! WordREG <<= WordREG
    template<class T1>
    inline
    WordREG& operator<<=(const WordREG<T1>& rhs) 
    {
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
      val = llvm_shl( val , rhs.get_val() );
      return *this;
#endif
    }

    //! WordREG >>= WordREG
    template<class T1>
    inline
    WordREG& operator>>=(const WordREG<T1>& rhs) 
    {
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
      val = llvm_shr( val , rhs.get_val() );
      return *this;
#endif
    }

  private:
    llvm::Value *    val;
  };




  template<class T>
  class WordVecREG 
  {
  public:

    // Default constructing should be possible
    // then there is no need for MPL index when
    // construction a PMatrix<T,N>
    WordVecREG() {}


    WordVecREG(const WordVecREG& rhs)
    {
      val = rhs.get_val();
    }


    explicit WordVecREG(llvm::Value * rhs)
    {
      setup(rhs);
    }


    void setup(llvm::Value * v) {
      //val = llvm_cast( llvm_get_type<T>() , v );
      QDPIO::cout << "not doing cast in WordVecREG::setup\n";
      val = v;
    }

    void setup(const WordVecJIT<T>& wj) {
      llvm::Value *val_j = llvm_load_ptr_idx( wj.getBaseReg() , wj.getOffset() );
      setup( val_j );
    }

    
    llvm::Value *get_val() const { return val; }


    WordVecREG& operator=(const WordVecREG& rhs) {
      val = rhs.get_val();      
      return *this;
    }

    template<class T1>
    WordVecREG(const WordVecREG<T1>& rhs)
    {
      setup(rhs.get_val());
    }

    template<class T1>
    inline
    WordVecREG& operator=(const WordVecREG<T1>& rhs) 
    {
      setup(rhs.get_val());
      return *this;
    }


    //! WordVecREG += WordVecREG
    template<class T1>
    inline
    WordVecREG& operator+=(const WordVecREG<T1>& rhs) 
    {
      val = llvm_add( val , rhs.get_val() );
      return *this;
    }

    //! WordVecREG -= WordVecREG
    template<class T1>
    inline
    WordVecREG& operator-=(const WordVecREG<T1>& rhs) 
    {
      val = llvm_sub( val , rhs.get_val() );
      return *this;
    }

    //! WordVecREG *= WordVecREG
    template<class T1>
    inline
    WordVecREG& operator*=(const WordVecREG<T1>& rhs) 
    {
      val = llvm_mul( val , rhs.get_val() );
      return *this;
    }

    //! WordVecREG /= WordVecREG
    template<class T1>
    inline
    WordVecREG& operator/=(const WordVecREG<T1>& rhs) 
    {
      val = llvm_div( val , rhs.get_val() );
      return *this;
    }

    //! WordVecREG %= WordVecREG
    template<class T1>
    inline
    WordVecREG& operator%=(const WordVecREG<T1>& rhs) 
    {
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
      val = llvm_mod( val , rhs.get_val() );
      return *this;
#endif
    }

    //! WordVecREG |= WordVecREG
    template<class T1>
    inline
    WordVecREG& operator|=(const WordVecREG<T1>& rhs) 
    {
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
      val = llvm_or( val , rhs.get_val() );
      return *this;
#endif
    }

    //! WordVecREG &= WordVecREG
    template<class T1>
    inline
    WordVecREG& operator&=(const WordVecREG<T1>& rhs) 
    {
      val = llvm_and( val , rhs.get_val() );
      return *this;
    }

    //! WordVecREG ^= WordVecREG
    template<class T1>
    inline
    WordVecREG& operator^=(const WordVecREG<T1>& rhs) 
    {
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
      val = llvm_xor( val , rhs.get_val() );
      return *this;
#endif
    }

    //! WordVecREG <<= WordVecREG
    template<class T1>
    inline
    WordVecREG& operator<<=(const WordVecREG<T1>& rhs) 
    {
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
      val = llvm_shl( val , rhs.get_val() );
      return *this;
#endif
    }

    //! WordVecREG >>= WordVecREG
    template<class T1>
    inline
    WordVecREG& operator>>=(const WordVecREG<T1>& rhs) 
    {
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
      val = llvm_shr( val , rhs.get_val() );
      return *this;
#endif
    }

  private:
    llvm::Value *    val;
  };

  



//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

  template<class T>
  struct InternalScalar<WordREG<T> > {
    typedef WordREG<T>  Type_t;
  };

  template<class T>
  struct InternalScalar<WordVecREG<T> > {
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
  struct JITType< WordVecREG<T> >
  {
    typedef WordVecJIT<T>  Type_t;
  };

  

  template<class T> 
  struct WordType<WordREG<T> >
  {
    typedef T  Type_t;
  };

  template<class T> 
  struct WordType<WordVecREG<T> >
  {
    typedef T  Type_t;
  };


  template<class T>
  struct UnaryReturn<WordREG<T>, FnSeedToFloat> {
    typedef WordREG<typename UnaryReturn<T, FnSeedToFloat>::Type_t>  Type_t;
  };

  
  template<class T>
  struct UnaryReturn<WordREG<T>, FnIsFinite> {
    typedef WordREG<typename UnaryReturn<T, FnIsFinite>::Type_t>  Type_t;
  };



  // Default binary(WordREG,WordREG) -> WordREG
#if 1
  template<class T1, class T2, class Op>
  struct BinaryReturn<WordREG<T1>, WordREG<T2>, Op> {
    typedef WordREG<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };
#endif

  template<class T1, class T2, class Op>
  struct BinaryReturn<WordVecREG<T1>, WordREG<T2>, Op> {
    typedef WordVecREG<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };



  template<class T>
  struct UnaryReturn<WordREG<T>, OpNot > {
    typedef WordREG<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
  };
  
  template<class T1>
  inline typename UnaryReturn<WordREG<T1>, OpNot>::Type_t
  operator!(const WordREG<T1>& l)
  {
    typename UnaryReturn<WordREG<T1>, OpNot>::Type_t ret;
    ret.setup( llvm_not( l.get_val() ) );
    return ret;
  }


  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpAdd>::Type_t
  operator+(const WordREG<T1>& l, const WordREG<T2>& r)
  {
    typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpAdd>::Type_t ret;
    ret.setup( llvm_add( l.get_val() , r.get_val() ) );
    return ret;
  }


  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpSubtract>::Type_t
  operator-(const WordREG<T1>& l, const WordREG<T2>& r)
  {
    typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpSubtract>::Type_t ret;
    ret.setup( llvm_sub( l.get_val() , r.get_val() ) );
    return ret;
  }


  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpMultiply>::Type_t
  operator*(const WordREG<T1>& l, const WordREG<T2>& r)
  {
    typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpMultiply>::Type_t ret;
    ret.setup( llvm_mul( l.get_val() , r.get_val() ) );
    return ret;
  }


  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpDivide>::Type_t
  operator/(const WordREG<T1>& l, const WordREG<T2>& r)
  {
    typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpDivide>::Type_t ret;
    ret.setup( llvm_div( l.get_val() , r.get_val() ) );
    return ret;
  }


  template<class T1>
  inline typename UnaryReturn<WordREG<T1>, OpUnaryMinus>::Type_t
  operator-(const WordREG<T1>& l)
  {
    typename UnaryReturn<WordREG<T1>, OpUnaryMinus>::Type_t ret;
    ret.setup( llvm_neg( l.get_val() ) );
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
  ret.setup( llvm_shl( l.get_val() , r.get_val() ) );
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
  ret.setup( llvm_shr( l.get_val() , r.get_val() ) );
  return ret;
}



template<class T1, class T2 >
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpMod>::Type_t
operator%(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpMod>::Type_t ret;
  ret.setup( llvm_rem( l.get_val() , r.get_val() ) );
  return ret;
}




template<class T1, class T2 >
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseXor>::Type_t
operator^(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseXor>::Type_t ret;
  ret.setup( llvm_xor( l.get_val() , r.get_val() ) );
  return ret;
}


template<class T1, class T2 >
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseAnd>::Type_t
operator&(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseAnd>::Type_t ret;
  ret.setup( llvm_and( l.get_val() , r.get_val() ) );
  return ret;
}



template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseOr>::Type_t
operator|(const WordREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordREG<T2>, OpBitwiseOr>::Type_t ret;
  ret.setup( llvm_or( l.get_val() , r.get_val() ) );
  return ret;
}



// *************************************************

  template<class T1>
  inline typename UnaryReturn<WordVecREG<T1>, OpUnaryMinus>::Type_t
  operator-(const WordVecREG<T1>& l)
  {
    typename UnaryReturn<WordVecREG<T1>, OpUnaryMinus>::Type_t ret;
    ret.setup( llvm_neg( l.get_val() ) );
    return ret;
  }



// *************************************************
// Binary operators: vec, vec


  template<class T1, class T2>
  inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpAdd>::Type_t
  operator+(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
  {
    typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpAdd>::Type_t ret;
    ret.setup( llvm_add( l.get_val() , r.get_val() ) );
    return ret;
  }

  template<class T1, class T2>
  inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpSubtract>::Type_t
  operator-(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
  {
    typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpSubtract>::Type_t ret;
    ret.setup( llvm_sub( l.get_val() , r.get_val() ) );
    return ret;
  }

  template<class T1, class T2>
  inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpMultiply>::Type_t
  operator*(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
  {
    typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpMultiply>::Type_t ret;
    ret.setup( llvm_mul( l.get_val() , r.get_val() ) );
    return ret;
  }

  template<class T1, class T2>
  inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpDivide>::Type_t
  operator/(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
  {
    typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpDivide>::Type_t ret;
    ret.setup( llvm_div( l.get_val() , r.get_val() ) );
    return ret;
  }


template<class T1, class T2 >
struct BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpLeftShift > {
  typedef WordVecREG<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpLeftShift>::Type_t
operator<<(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpLeftShift>::Type_t ret;
  ret.setup( llvm_shl( l.get_val() , r.get_val() ) );
  return ret;
}

template<class T1, class T2 >
struct BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpRightShift > {
  typedef WordVecREG<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpRightShift>::Type_t
operator>>(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpRightShift>::Type_t ret;
  ret.setup( llvm_shr( l.get_val() , r.get_val() ) );
  return ret;
}

template<class T1, class T2 >
inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpMod>::Type_t
operator%(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpMod>::Type_t ret;
  ret.setup( llvm_rem( l.get_val() , r.get_val() ) );
  return ret;
}

template<class T1, class T2 >
inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpBitwiseXor>::Type_t
operator^(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpBitwiseXor>::Type_t ret;
  ret.setup( llvm_xor( l.get_val() , r.get_val() ) );
  return ret;
}

template<class T1, class T2 >
inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpBitwiseAnd>::Type_t
operator&(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpBitwiseAnd>::Type_t ret;
  ret.setup( llvm_and( l.get_val() , r.get_val() ) );
  return ret;
}

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpBitwiseOr>::Type_t
operator|(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpBitwiseOr>::Type_t ret;
  ret.setup( llvm_or( l.get_val() , r.get_val() ) );
  return ret;
}



// *************************************************
// Binary operators: scalar, vec


  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpAdd>::Type_t
  operator+(const WordREG<T1>& l, const WordVecREG<T2>& r)
  {
    typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpAdd>::Type_t ret;
    ret.setup( llvm_add( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
    return ret;
  }

  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpSubtract>::Type_t
  operator-(const WordREG<T1>& l, const WordVecREG<T2>& r)
  {
    typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpSubtract>::Type_t ret;
    ret.setup( llvm_sub( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
    return ret;
  }

  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpMultiply>::Type_t
  operator*(const WordREG<T1>& l, const WordVecREG<T2>& r)
  {
    typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpMultiply>::Type_t ret;
    ret.setup( llvm_mul( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
    return ret;
  }

  template<class T1, class T2>
  inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpDivide>::Type_t
  operator/(const WordREG<T1>& l, const WordVecREG<T2>& r)
  {
    typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpDivide>::Type_t ret;
    ret.setup( llvm_div( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
    return ret;
  }


template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpLeftShift > {
  typedef WordVecREG<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpLeftShift>::Type_t
operator<<(const WordREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpLeftShift>::Type_t ret;
  ret.setup( llvm_shl( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
  return ret;
}

template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpRightShift > {
  typedef WordVecREG<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpRightShift>::Type_t
operator>>(const WordREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpRightShift>::Type_t ret;
  ret.setup( llvm_shr( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
  return ret;
}

template<class T1, class T2 >
inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpMod>::Type_t
operator%(const WordREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpMod>::Type_t ret;
  ret.setup( llvm_rem( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
  return ret;
}

template<class T1, class T2 >
inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpBitwiseXor>::Type_t
operator^(const WordREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpBitwiseXor>::Type_t ret;
  ret.setup( llvm_xor( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
  return ret;
}

template<class T1, class T2 >
inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpBitwiseAnd>::Type_t
operator&(const WordREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpBitwiseAnd>::Type_t ret;
  ret.setup( llvm_and( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
  return ret;
}

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpBitwiseOr>::Type_t
operator|(const WordREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpBitwiseOr>::Type_t ret;
  ret.setup( llvm_or( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
  return ret;
}





// *************************************************
// Binary operators: vec, scalar


  template<class T1, class T2>
  inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpAdd>::Type_t
  operator+(const WordVecREG<T1>& l, const WordREG<T2>& r)
  {
    typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpAdd>::Type_t ret;
    ret.setup( llvm_add( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
    return ret;
  }

  template<class T1, class T2>
  inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpSubtract>::Type_t
  operator-(const WordVecREG<T1>& l, const WordREG<T2>& r)
  {
    typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpSubtract>::Type_t ret;
    ret.setup( llvm_sub( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
    return ret;
  }

  template<class T1, class T2>
  inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpMultiply>::Type_t
  operator*(const WordVecREG<T1>& l, const WordREG<T2>& r)
  {
    typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpMultiply>::Type_t ret;
    ret.setup( llvm_mul( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
    return ret;
  }

  template<class T1, class T2>
  inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpDivide>::Type_t
  operator/(const WordVecREG<T1>& l, const WordREG<T2>& r)
  {
    typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpDivide>::Type_t ret;
    ret.setup( llvm_div( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
    return ret;
  }


template<class T1, class T2 >
struct BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpLeftShift > {
  typedef WordVecREG<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpLeftShift>::Type_t
operator<<(const WordVecREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpLeftShift>::Type_t ret;
  ret.setup( llvm_shl( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
  return ret;
}

template<class T1, class T2 >
struct BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpRightShift > {
  typedef WordVecREG<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpRightShift>::Type_t
operator>>(const WordVecREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpRightShift>::Type_t ret;
  ret.setup( llvm_shr( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
  return ret;
}

template<class T1, class T2 >
inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpMod>::Type_t
operator%(const WordVecREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpMod>::Type_t ret;
  ret.setup( llvm_rem( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
  return ret;
}

template<class T1, class T2 >
inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpBitwiseXor>::Type_t
operator^(const WordVecREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpBitwiseXor>::Type_t ret;
  ret.setup( llvm_xor( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
  return ret;
}

template<class T1, class T2 >
inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpBitwiseAnd>::Type_t
operator&(const WordVecREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpBitwiseAnd>::Type_t ret;
  ret.setup( llvm_and( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
  return ret;
}

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpBitwiseOr>::Type_t
operator|(const WordVecREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpBitwiseOr>::Type_t ret;
  ret.setup( llvm_or( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
  return ret;
}




// *************************************************





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
  ret.setup( llvm_lt( l.get_val() , r.get_val() ) );
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
  ret.setup( llvm_le( l.get_val() , r.get_val() ) );
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
  ret.setup( llvm_gt( l.get_val() , r.get_val() ) );
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
  ret.setup( llvm_ge( l.get_val() , r.get_val() ) );
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
  ret.setup( llvm_eq( l.get_val() , r.get_val() ) );
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
  ret.setup( llvm_ne( l.get_val() , r.get_val() ) );
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
  ret.setup( llvm_and( l.get_val() , r.get_val() ) );
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
  ret.setup( llvm_or( l.get_val() , r.get_val() ) );
  return ret;
}



// **************************************************
// Comparison, mixed; vec, scalar

template<class T1, class T2 >
struct BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpLT > {
  typedef WordVecREG<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpLT>::Type_t
operator<(const WordVecREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpLT>::Type_t ret;
  ret.setup( llvm_lt( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
  return ret;
}


 template<class T1, class T2 >
 struct BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpLE > {
   typedef WordVecREG<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpLE>::Type_t
operator<=(const WordVecREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpLE>::Type_t ret;
  ret.setup( llvm_le( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
  return ret;
}


 template<class T1, class T2 >
 struct BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpGT > {
   typedef WordVecREG<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpGT>::Type_t
operator>(const WordVecREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpGT>::Type_t ret;
  ret.setup( llvm_gt( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
  return ret;
}


 template<class T1, class T2 >
 struct BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpGE > {
   typedef WordVecREG<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpGE>::Type_t
operator>=(const WordVecREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpGE>::Type_t ret;
  ret.setup( llvm_ge( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
  return ret;
}


 template<class T1, class T2 >
 struct BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpEQ > {
   typedef WordVecREG<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpEQ>::Type_t
operator==(const WordVecREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpEQ>::Type_t ret;
  ret.setup( llvm_eq( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
  return ret;
}


 template<class T1, class T2 >
 struct BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpNE > {
   typedef WordVecREG<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpNE>::Type_t
operator!=(const WordVecREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpNE>::Type_t ret;
  ret.setup( llvm_ne( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
  return ret;
}



 template<class T1, class T2>
 struct BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpAnd > {
   typedef WordVecREG<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpAnd>::Type_t
operator&&(const WordVecREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpAnd>::Type_t ret;
  ret.setup( llvm_and( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
  return ret;
}


 template<class T1, class T2>
 struct BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpOr > {
   typedef WordVecREG<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpOr>::Type_t
operator||(const WordVecREG<T1>& l, const WordREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordREG<T2>, OpOr>::Type_t ret;
  ret.setup( llvm_or( l.get_val() , llvm_fill_vector( r.get_val() ) ) );
  return ret;
}

// *************************************************
// Comparisons: vec, vec

template<class T1, class T2 >
struct BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpLT > {
  typedef WordVecREG<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpLT>::Type_t
operator<(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpLT>::Type_t ret;
  ret.setup( llvm_lt( l.get_val() , r.get_val() ) );
  return ret;
}


 template<class T1, class T2 >
 struct BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpLE > {
   typedef WordVecREG<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpLE>::Type_t
operator<=(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpLE>::Type_t ret;
  ret.setup( llvm_le( l.get_val() , r.get_val() ) );
  return ret;
}


 template<class T1, class T2 >
 struct BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpGT > {
   typedef WordVecREG<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpGT>::Type_t
operator>(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpGT>::Type_t ret;
  ret.setup( llvm_gt( l.get_val() , r.get_val() ) );
  return ret;
}


 template<class T1, class T2 >
 struct BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpGE > {
   typedef WordVecREG<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpGE>::Type_t
operator>=(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpGE>::Type_t ret;
  ret.setup( llvm_ge( l.get_val() , r.get_val() ) );
  return ret;
}


 template<class T1, class T2 >
 struct BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpEQ > {
   typedef WordVecREG<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpEQ>::Type_t
operator==(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpEQ>::Type_t ret;
  ret.setup( llvm_eq( l.get_val() , r.get_val() ) );
  return ret;
}


 template<class T1, class T2 >
 struct BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpNE > {
   typedef WordVecREG<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpNE>::Type_t
operator!=(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpNE>::Type_t ret;
  ret.setup( llvm_ne( l.get_val() , r.get_val() ) );
  return ret;
}



 template<class T1, class T2>
 struct BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpAnd > {
   typedef WordVecREG<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpAnd>::Type_t
operator&&(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpAnd>::Type_t ret;
  ret.setup( llvm_and( l.get_val() , r.get_val() ) );
  return ret;
}


 template<class T1, class T2>
 struct BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpOr > {
   typedef WordVecREG<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpOr>::Type_t
operator||(const WordVecREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordVecREG<T1>, WordVecREG<T2>, OpOr>::Type_t ret;
  ret.setup( llvm_or( l.get_val() , r.get_val() ) );
  return ret;
}


// *************************************************
// Comparisons: scalar, vec

template<class T1, class T2 >
struct BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpLT > {
  typedef WordREG<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpLT>::Type_t
operator<(const WordREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpLT>::Type_t ret;
  ret.setup( llvm_lt( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
  return ret;
}


 template<class T1, class T2 >
 struct BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpLE > {
   typedef WordREG<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpLE>::Type_t
operator<=(const WordREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpLE>::Type_t ret;
  ret.setup( llvm_le( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
  return ret;
}


 template<class T1, class T2 >
 struct BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpGT > {
   typedef WordREG<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpGT>::Type_t
operator>(const WordREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpGT>::Type_t ret;
  ret.setup( llvm_gt( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
  return ret;
}


 template<class T1, class T2 >
 struct BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpGE > {
   typedef WordREG<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpGE>::Type_t
operator>=(const WordREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpGE>::Type_t ret;
  ret.setup( llvm_ge( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
  return ret;
}


 template<class T1, class T2 >
 struct BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpEQ > {
   typedef WordREG<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpEQ>::Type_t
operator==(const WordREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpEQ>::Type_t ret;
  ret.setup( llvm_eq( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
  return ret;
}


 template<class T1, class T2 >
 struct BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpNE > {
   typedef WordREG<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpNE>::Type_t
operator!=(const WordREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpNE>::Type_t ret;
  ret.setup( llvm_ne( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
  return ret;
}



 template<class T1, class T2>
 struct BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpAnd > {
   typedef WordREG<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpAnd>::Type_t
operator&&(const WordREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpAnd>::Type_t ret;
  ret.setup( llvm_and( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
  return ret;
}


 template<class T1, class T2>
 struct BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpOr > {
   typedef WordREG<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
 };

template<class T1, class T2>
inline typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpOr>::Type_t
operator||(const WordREG<T1>& l, const WordVecREG<T2>& r)
{
  typename BinaryReturn<WordREG<T1>, WordVecREG<T2>, OpOr>::Type_t ret;
  ret.setup( llvm_or( llvm_fill_vector( l.get_val() ) , r.get_val() ) );
  return ret;
}




// ************************************************


  template<class T1, class T2, class T3>
  inline typename TrinaryReturn<WordREG<T1>, WordREG<T2>, WordREG<T3>, FnWhere >::Type_t
  where(const WordREG<T1> &a, const WordREG<T2> &b, const WordREG<T3> &c)
  {
    typename TrinaryReturn<WordREG<T1>, WordREG<T2>, WordREG<T3>, FnWhere >::Type_t ret;

    ret.setup( jit_ternary( a.get_val() , b.get_val() , c.get_val() ) );
    
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
  ret.setup( llvm_mul( s1.get_val() , s1.get_val() ) );
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
  ret.setup( llvm_mul( s1.get_val() , s2.get_val() ) );
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
  ret.setup( llvm_mul( s1.get_val() , s2.get_val() ) );
  return ret;
}


  inline void 
  zero_rep(WordREG<double>& dest) 
  {
    //llvm_mov( dest.get_val() , 
    dest.setup(llvm_create_value(0.0));
  }

  inline void 
  zero_rep(WordREG<jit_half_t>& dest) 
  {
    //llvm_mov( dest.get_val() , llvm_create_value(0.0) );
    dest.setup(llvm_create_value(0.0));
  }

inline void 
  zero_rep(WordREG<float>& dest) 
  {
    //llvm_mov( dest.get_val() , llvm_create_value(0.0) );
    dest.setup(llvm_create_value(0.0));
  }

  inline void 
  zero_rep(WordREG<int>& dest)
  {
    //llvm_mov( dest.get_val() , llvm_create_value(0) );
    dest.setup(llvm_create_value(0));
  }





typename UnaryReturn<WordREG<float>, FnCeil>::Type_t ceil(const WordREG<float>& s1);
typename UnaryReturn<WordREG<float>, FnFloor>::Type_t floor(const WordREG<float>& s1);
typename UnaryReturn<WordREG<float>, FnFabs>::Type_t fabs(const WordREG<float>& s1);
typename UnaryReturn<WordREG<float>, FnSqrt>::Type_t sqrt(const WordREG<float>& s1);
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

typename UnaryReturn<WordREG<float>, FnIsFinite>::Type_t isfinite(const WordREG<float>& s1);

typename UnaryReturn<WordREG<double>, FnCeil>::Type_t ceil(const WordREG<double>& s1);
typename UnaryReturn<WordREG<double>, FnFloor>::Type_t floor(const WordREG<double>& s1);
typename UnaryReturn<WordREG<double>, FnFabs>::Type_t fabs(const WordREG<double>& s1);
typename UnaryReturn<WordREG<double>, FnSqrt>::Type_t sqrt(const WordREG<double>& s1);
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

typename UnaryReturn<WordREG<double>, FnIsFinite>::Type_t isfinite(const WordREG<double>& s1);

typename BinaryReturn<WordREG<float>, WordREG<float>, FnPow>::Type_t pow(const WordREG<float>& s1, const WordREG<float>& s2);
typename BinaryReturn<WordREG<double>, WordREG<double>, FnPow>::Type_t pow(const WordREG<double>& s1, const WordREG<double>& s2);
typename BinaryReturn<WordREG<float>, WordREG<float>, FnPow>::Type_t pow(const WordREG<double>& s1, const WordREG<float>& s2);
typename BinaryReturn<WordREG<float>, WordREG<float>, FnPow>::Type_t pow(const WordREG<float>& s1, const WordREG<double>& s2);


typename BinaryReturn<WordREG<float>, WordREG<float>, FnArcTan2>::Type_t atan2(const WordREG<float>& s1, const WordREG<float>& s2);
typename BinaryReturn<WordREG<double>, WordREG<double>, FnArcTan2>::Type_t atan2(const WordREG<double>& s1, const WordREG<double>& s2);
typename BinaryReturn<WordREG<float>, WordREG<float>, FnArcTan2>::Type_t atan2(const WordREG<double>& s1, const WordREG<float>& s2);
typename BinaryReturn<WordREG<float>, WordREG<float>, FnArcTan2>::Type_t atan2(const WordREG<float>& s1, const WordREG<double>& s2);








} // namespace QDP

#endif
