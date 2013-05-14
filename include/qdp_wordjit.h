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
    WordJIT(): 
      setup_m(false)
      // r_base(jit_ptx_type::u64),
      // offset_full(jit_ptx_type::s32),
      // offset_level(jit_ptx_type::s32)
    {
    }

#if 0
    void stack_setup( const llvm::Value *& base_m , IndexDomainVector args ) {
      r_base = base_m;
      offset = datalayout_stack( args );
      setup_m = true;
    }
#endif

    void setup( const llvm::Value *& base_m , JitDeviceLayout lay , IndexDomainVector args ) {
      r_base = base_m;
      offset = datalayout( lay , args );
      setup_m = true;
    }

    // void setup( llvm::Value * r_base_, llvm::Value * full_, llvm::Value * level_ ) {
    //   r_base        = r_base_;
    //   offset_full   = full_;
    //   offset_level  = level_;
    //   setup_m = true;
    // }


    // llvm::Value * getAddress() const {
    //   llvm::Value * ws         = llvm::Value *( sizeof(typename WordType<T>::Type_t) );
    //   llvm::Value * lev_mul_ws = jit_ins_mul ( offset_level , ws );
    //   llvm::Value * address    = jit_ins_add( r_base , lev_mul_ws );
    //   return address;
    // }

    // llvm::Value * getOffset() const {
    //   return offset;
    // }

    // llvm::Value * getOffset() const {
    //   llvm::Value * ws         = llvm_create_value( sizeof(typename WordType<T>::Type_t) );
    //   llvm::Value * lev_mul_ws = jit_ins_mul ( offset_level , ws );
    //   return lev_mul_ws;
    // }


    template<class T1>
    void operator=(const WordREG<T1>& s1) {
      assert(setup_m);
      jit_ins_store( r_base , getOffset() , jit_type<T>::value , s1.get_val() );
    }


    //! WordJIT += WordJIT
    template<class T1>
    inline
    WordJIT& operator+=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = jit_ins_load( r_base , getOffset() , jit_type<T>::value );
      llvm::Value * tmp2 = jit_ins_add( tmp , rhs.get_val() );
      jit_ins_store( r_base , getOffset() , jit_type<T>::value , tmp2 );
      return *this;
    }

    //! WordJIT -= WordJIT
    template<class T1>
    inline
    WordJIT& operator-=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = jit_ins_load( r_base , getOffset() , jit_type<T>::value );
      llvm::Value * tmp2 = jit_ins_sub( tmp , rhs.get_val() );
      jit_ins_store( r_base , getOffset() , jit_type<T>::value , tmp2 );
      return *this;
    }

    //! WordJIT *= WordJIT
    template<class T1>
    inline
    WordJIT& operator*=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = jit_ins_load( r_base , getOffset() , jit_type<T>::value );
      llvm::Value * tmp2 = jit_ins_mul( tmp , rhs.get_val() );
      jit_ins_store( r_base , getOffset() , jit_type<T>::value , tmp2 );
      return *this;
    }

    //! WordJIT /= WordJIT
    template<class T1>
    inline
    WordJIT& operator/=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = jit_ins_load( r_base , getOffset() , jit_type<T>::value );
      llvm::Value * tmp2 = jit_ins_div( tmp , rhs.get_val() );
      jit_ins_store( r_base , getOffset() , jit_type<T>::value , tmp2 );
      return *this;
    }

    //! WordJIT %= WordJIT
    template<class T1>
    inline
    WordJIT& operator%=(const WordREG<T1>& rhs) 
    {
      std::cout << __PRETTY_FUNCTION__ << "\n"; QDP_error_exit("ni");
      return *this;
    }

    //! WordJIT |= WordJIT
    template<class T1>
    inline
    WordJIT& operator|=(const WordREG<T1>& rhs) 
    {
      std::cout << __PRETTY_FUNCTION__ << "\n"; QDP_error_exit("ni");
      return *this;
    }

    //! WordJIT &= WordJIT
    template<class T1>
    inline
    WordJIT& operator&=(const WordREG<T1>& rhs) 
    {
      std::cout << __PRETTY_FUNCTION__ << "\n"; QDP_error_exit("ni");
      return *this;
    }

    //! WordJIT ^= WordJIT
    template<class T1>
    inline
    WordJIT& operator^=(const WordREG<T1>& rhs) 
    {
      std::cout << __PRETTY_FUNCTION__ << "\n"; QDP_error_exit("ni");
      return *this;
    }

    //! WordJIT <<= WordJIT
    template<class T1>
    inline
    WordJIT& operator<<=(const WordREG<T1>& rhs) 
    {
      std::cout << __PRETTY_FUNCTION__ << "\n"; QDP_error_exit("ni");
      return *this;
    }

    //! WordJIT >>= WordJIT
    template<class T1>
    inline
    WordJIT& operator>>=(const WordREG<T1>& rhs) 
    {
      std::cout << __PRETTY_FUNCTION__ << "\n"; QDP_error_exit("ni");
      return *this;
    }


    llvm::Value * getBaseReg() const { assert(setup_m); return r_base; }
    llvm::Value * getOffset() const { assert(setup_m); return offset; }
    // llvm::Value * getFull() const { assert(setup_m); return offset_full; }
    // llvm::Value * getLevel() const { assert(setup_m); return offset_level; }

  private:
    template<class T1>
    void operator=(const WordJIT<T1>& s1);
    void operator=(const WordJIT& s1);

    llvm::Value *     r_base;
    llvm::Value *     offset;
    // llvm::Value *    offset_full;
    // llvm::Value *    offset_level;
    bool setup_m;
  };


  template<class T1>
  inline typename UnaryReturn<WordREG<T1>, OpUnaryMinus>::Type_t
  operator-(const WordJIT<T1>& l)
  {
    typename UnaryReturn<WordREG<T1>, OpUnaryMinus>::Type_t ret;
    ret.setup(l);
    return -ret.elem();
  }


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


  template<class T, class T1, class T2> 
  inline
  void copymask(WordJIT<T>& d, const WordREG<T1>& mask, const WordREG<T2>& s1)
  {
    std::cout << __PRETTY_FUNCTION__ << "\n"; QDP_error_exit("ni");

    //jit_ins_store( d.getAddress() , 0 , jit_type<T>::value , s1.get_val() , mask.get_val() );
  }



  inline void 
  zero_rep(WordJIT<double>& dest)
  {
    jit_ins_store( dest.getBaseReg() , dest.getOffset() , jit_type<double>::value , llvm_create_value( 0.0 ) );
  }

  inline void 
  zero_rep(WordJIT<float>& dest)
  {
    jit_ins_store( dest.getBaseReg() , dest.getOffset() , jit_type<float>::value , llvm_create_value( 0.0 ) );
  }

  inline void 
  zero_rep(WordJIT<int>& dest)
  {
    jit_ins_store( dest.getBaseReg() , dest.getOffset() , jit_type<int>::value , llvm_create_value( 0 ) );
  }


  //! dest  = random  
  template<class T, class T1, class T2, class T3>
  inline void
  fill_random(WordJIT<T>& d, T1& seed, T2& skewed_seed, const T3& seed_mult)
  {
    d = seedToFloat( skewed_seed ).elem().elem().elem();
    seed        = seed        * seed_mult;
    skewed_seed = skewed_seed * seed_mult;
  }



} // namespace QDP

#endif
