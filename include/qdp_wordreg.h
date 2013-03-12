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

    explicit WordREG(int i) {
      val = jit_val_create_const_int( i );
      setup_m=true;
    }

    explicit WordREG(float f) {
      val = jit_val_create_const_float( f );
      setup_m=true;
    }

    explicit WordREG(double f) {
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
      val = jit_ins_load( wj.getBaseReg() , wj.getLevel() * sizeof(typename WordType<T>::Type_t) , jit_type<T>::value );
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
    bool setup_m=false;
    jit_value_t    val;
  };



  template <class T>
  jit_function_t getFunc(const WordREG<T>& l) {
    return getFunc( l.get_val() );
  }




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
  

  // Default binary(WordREG,WordREG) -> WordREG
  template<class T1, class T2, class Op>
  struct BinaryReturn<WordREG<T1>, WordREG<T2>, Op> {
    typedef WordREG<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };



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



} // namespace QDP

#endif
