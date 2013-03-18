// -*- C++ -*-
//
// QDP data parallel interface
//


#ifndef QDP_WORDJIT_H
#define QDP_WORDJIT_H



namespace QDP {


  template<class T>
  class WordJIT 
  {
  public:
    enum {Size_t = 1};

    // Default constructing should be possible
    // then there is no need for MPL index when
    // construction a PMatrix<T,N>
    WordJIT(): setup_m(false) {}


    void setup(jit_function_t func_, jit_value_t r_base_, jit_value_t full_, jit_value_t level_ ) {
      func          = func_;
      r_base        = r_base_;
      offset_full   = full_;
      offset_level  = level_;
      setup_m = true;
    }


    jit_value_t getAddress() const {
      jit_value_t ws         = jit_val_create_const_int( sizeof(typename WordType<T>::Type_t) );
      jit_value_t lev_mul_ws = jit_ins_mul ( offset_level , ws );
      jit_value_t address    = jit_ins_add( r_base , lev_mul_ws );
      return address;
    }


    template<class T1>
    void operator=(const WordREG<T1>& s1) {
      assert(setup_m);
      jit_ins_store( getAddress() , 0 , jit_type<T>::value , s1.get_val() );
    }


    //! WordJIT += WordJIT
    template<class T1>
    inline
    WordJIT& operator+=(const WordREG<T1>& rhs) 
    {
      jit_value_t tmp = jit_ins_load( getAddress() , 0 , jit_type<T>::value );
      jit_value_t tmp2 = jit_ins_add( tmp , rhs.get_val() );
      jit_ins_store( getAddress() , 0 , jit_type<T>::value , tmp2 );
      return *this;
    }

    //! WordJIT -= WordJIT
    template<class T1>
    inline
    WordJIT& operator-=(const WordREG<T1>& rhs) 
    {
      jit_value_t tmp = jit_ins_load( getAddress() , 0 , jit_type<T>::value );
      jit_value_t tmp2 = jit_ins_sub( tmp , rhs.get_val() );
      jit_ins_store( getAddress() , 0 , jit_type<T>::value , tmp2 );
      return *this;
    }

    //! WordJIT *= WordJIT
    template<class T1>
    inline
    WordJIT& operator*=(const WordREG<T1>& rhs) 
    {
      jit_value_t tmp = jit_ins_load( getAddress() , 0 , jit_type<T>::value );
      jit_value_t tmp2 = jit_ins_mul( tmp , rhs.get_val() );
      jit_ins_store( getAddress() , 0 , jit_type<T>::value , tmp2 );
      return *this;
    }

    //! WordJIT /= WordJIT
    template<class T1>
    inline
    WordJIT& operator/=(const WordREG<T1>& rhs) 
    {
      jit_value_t tmp = jit_ins_load( getAddress() , 0 , jit_type<T>::value );
      jit_value_t tmp2 = jit_ins_div( tmp , rhs.get_val() );
      jit_ins_store( getAddress() , 0 , jit_type<T>::value , tmp2 );
      return *this;
    }

    //! WordJIT %= WordJIT
    template<class T1>
    inline
    WordJIT& operator%=(const WordREG<T1>& rhs) 
    {
      jit_value_t tmp = jit_ins_load( getAddress() , 0 , jit_type<T>::value );
      jit_value_t tmp2 = jit_ins_mod( tmp , rhs.get_val() );
      jit_ins_store( getAddress() , 0 , jit_type<T>::value , tmp2 );
      return *this;
    }

    //! WordJIT |= WordJIT
    template<class T1>
    inline
    WordJIT& operator|=(const WordREG<T1>& rhs) 
    {
      jit_value_t tmp = jit_ins_load( getAddress() , 0 , jit_type<T>::value );
      jit_value_t tmp2 = jit_ins_or( tmp , rhs.get_val() );
      jit_ins_store( getAddress() , 0 , jit_type<T>::value , tmp2 );
      return *this;
    }

    //! WordJIT &= WordJIT
    template<class T1>
    inline
    WordJIT& operator&=(const WordREG<T1>& rhs) 
    {
      jit_value_t tmp = jit_ins_load( getAddress() , 0 , jit_type<T>::value );
      jit_value_t tmp2 = jit_ins_and( tmp , rhs.get_val() );
      jit_ins_store( getAddress() , 0 , jit_type<T>::value , tmp2 );
      return *this;
    }

    //! WordJIT ^= WordJIT
    template<class T1>
    inline
    WordJIT& operator^=(const WordREG<T1>& rhs) 
    {
      jit_value_t tmp = jit_ins_load( getAddress() , 0 , jit_type<T>::value );
      jit_value_t tmp2 = jit_ins_xor( tmp , rhs.get_val() );
      jit_ins_store( getAddress() , 0 , jit_type<T>::value , tmp2 );
      return *this;
    }

    //! WordJIT <<= WordJIT
    template<class T1>
    inline
    WordJIT& operator<<=(const WordREG<T1>& rhs) 
    {
      jit_value_t tmp = jit_ins_load( getAddress() , 0 , jit_type<T>::value );
      jit_value_t tmp2 = jit_ins_shl( tmp , rhs.get_val() );
      jit_ins_store( getAddress() , 0 , jit_type<T>::value , tmp2 );
      return *this;
    }

    //! WordJIT >>= WordJIT
    template<class T1>
    inline
    WordJIT& operator>>=(const WordREG<T1>& rhs) 
    {
      jit_value_t tmp = jit_ins_load( getAddress() , 0 , jit_type<T>::value );
      jit_value_t tmp2 = jit_ins_shr( tmp , rhs.get_val() );
      jit_ins_store( getAddress() , 0 , jit_type<T>::value , tmp2 );
      return *this;
    }


    jit_function_t get_func() const { assert(setup_m); return func;}
    jit_value_t getBaseReg() const { assert(setup_m); return r_base; }
    jit_value_t getFull() const { assert(setup_m); return offset_full; }
    jit_value_t getLevel() const { assert(setup_m); return offset_level; }

  private:
    template<class T1>
    void operator=(const WordJIT<T1>& s1);
    void operator=(const WordJIT& s1);

    jit_function_t func;
    jit_value_t    r_base;
    jit_value_t    offset_full;
    jit_value_t    offset_level;
    bool setup_m;
  };


  template<class T>
  struct REGType< WordJIT<T> >
  {
    typedef WordREG<typename REGType<T>::Type_t>  Type_t;
  };



  template<class T> 
  struct WordType<WordJIT<T> >
  {
    typedef T  Type_t;
  };


  // Default binary(WordJIT,WordJIT) -> WordJIT
  template<class T1, class T2, class Op>
  struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, Op> {
    typedef WordJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };


} // namespace QDP

#endif
