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
    enum {ScalarSize_t = 1};

    WordJIT(): setup_m(false)
    {
    }

    void setup( llvm::Value * base_m , JitDeviceLayout lay , IndexDomainVector args ) {
      r_base = base_m;
      offset = datalayout( lay , args );
      setup_m = true;
    }

    

    template<class T1>
    void operator=(const WordREG<T1>& s1) {
      assert(setup_m);
      llvm_store_ptr_idx( s1.get_val() , r_base , offset );
    }


    //! WordJIT += WordJIT
    template<class T1>
    inline
    WordJIT& operator+=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_load_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_add( tmp , rhs.get_val() );
      llvm_store_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    //! WordJIT -= WordJIT
    template<class T1>
    inline
    WordJIT& operator-=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_load_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_sub( tmp , rhs.get_val() );
      llvm_store_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    //! WordJIT *= WordJIT
    template<class T1>
    inline
    WordJIT& operator*=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_load_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_mul( tmp , rhs.get_val() );
      llvm_store_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    //! WordJIT /= WordJIT
    template<class T1>
    inline
    WordJIT& operator/=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_load_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_div( tmp , rhs.get_val() );
      llvm_store_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    //! WordJIT %= WordJIT
    template<class T1>
    inline
    WordJIT& operator%=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_load_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_rem( tmp , rhs.get_val() );
      llvm_store_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    //! WordJIT |= WordJIT
    template<class T1>
    inline
    WordJIT& operator|=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_load_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_or( tmp , rhs.get_val() );
      llvm_store_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    //! WordJIT &= WordJIT
    template<class T1>
    inline
    WordJIT& operator&=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_load_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_and( tmp , rhs.get_val() );
      llvm_store_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    //! WordJIT ^= WordJIT
    template<class T1>
    inline
    WordJIT& operator^=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_load_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_xor( tmp , rhs.get_val() );
      llvm_store_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    //! WordJIT <<= WordJIT
    template<class T1>
    inline
    WordJIT& operator<<=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_load_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_shl( tmp , rhs.get_val() );
      llvm_store_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    //! WordJIT >>= WordJIT
    template<class T1>
    inline
    WordJIT& operator>>=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_load_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_shr( tmp , rhs.get_val() );
      llvm_store_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }


    llvm::Value * getBaseReg() const { assert(setup_m); return r_base; }
    llvm::Value * getOffset() const { assert(setup_m); return offset; }

  private:
    llvm::Value *     r_base;
    llvm::Value *     offset;
    bool setup_m;
  };



#if defined (QDP_CODEGEN_VECTOR)
  template<class T>
  class WordVecJIT 
  {
  public:
    enum {ScalarSize_t = 1};

    WordVecJIT(): setup_m(false)
    {
    }

    void setup( llvm::Value * base_m , JitDeviceLayout lay , IndexDomainVector args ) {
      r_base = base_m;
      offset = datalayout( lay , args );
      setup_m = true;
    }

    

    template<class T1>
    void operator=(const WordREG<T1>& s1) {
      assert(setup_m);
      llvm_vecstore_ptr_idx( llvm_fill_vector( s1.get_val() ) , r_base , offset );
    }


    template<class T1>
    void operator=(const WordVecREG<T1>& s1) {
      assert(setup_m);
      llvm_vecstore_ptr_idx( s1.get_val() , r_base , offset );
    }


    // In-place operators: vec

    template<class T1>
    inline
    WordVecJIT& operator+=(const WordVecREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_add( tmp , rhs.get_val() );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator-=(const WordVecREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_sub( tmp , rhs.get_val() );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator*=(const WordVecREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_mul( tmp , rhs.get_val() );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator/=(const WordVecREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_div( tmp , rhs.get_val() );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator%=(const WordVecREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_rem( tmp , rhs.get_val() );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator|=(const WordVecREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_or( tmp , rhs.get_val() );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator&=(const WordVecREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_and( tmp , rhs.get_val() );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator^=(const WordVecREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_xor( tmp , rhs.get_val() );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator<<=(const WordVecREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_shl( tmp , rhs.get_val() );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator>>=(const WordVecREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_shr( tmp , rhs.get_val() );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }


    // **************************************
    // In-place operators: scalar

    template<class T1>
    inline
    WordVecJIT& operator+=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_add( tmp , llvm_fill_vector( rhs.get_val() ) );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator-=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_sub( tmp , llvm_fill_vector( rhs.get_val() ) );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator*=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_mul( tmp , llvm_fill_vector( rhs.get_val() ) );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator/=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_div( tmp , llvm_fill_vector( rhs.get_val() ) );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator%=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_rem( tmp , llvm_fill_vector( rhs.get_val() ) );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator|=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_or( tmp , llvm_fill_vector( rhs.get_val() ) );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator&=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_and( tmp , llvm_fill_vector( rhs.get_val() ) );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator^=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_xor( tmp , llvm_fill_vector( rhs.get_val() ) );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator<<=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_shl( tmp , llvm_fill_vector( rhs.get_val() ) );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    template<class T1>
    inline
    WordVecJIT& operator>>=(const WordREG<T1>& rhs) 
    {
      llvm::Value * tmp = llvm_vecload_ptr_idx( r_base , offset );
      llvm::Value * tmp2 = llvm_shr( tmp , llvm_fill_vector( rhs.get_val() ) );
      llvm_vecstore_ptr_idx( tmp2 , r_base , offset );
      return *this;
    }

    

    llvm::Value * getBaseReg() const { assert(setup_m); return r_base; }
    llvm::Value * getOffset() const { assert(setup_m); return offset; }

  private:
    llvm::Value *     r_base;
    llvm::Value *     offset;
    bool setup_m;
  };
#endif



  

  template<class T, class Op>
  struct UnaryReturn<WordJIT<T>, Op> {
    typedef WordJIT<typename UnaryReturn<T, Op>::Type_t>  Type_t;
  };

#if defined (QDP_CODEGEN_VECTOR)  
  template<class T, class Op>
  struct UnaryReturn<WordVecJIT<T>, Op> {
    typedef WordVecJIT<typename UnaryReturn<T, Op>::Type_t>  Type_t;
  };
#endif

  template<>
  struct UnaryReturn<float, FnIsFinite> {
    typedef bool  Type_t;
  };

  template<>
  struct UnaryReturn<double, FnIsFinite> {
    typedef bool  Type_t;
  };

  

  template<class T1>
  inline typename UnaryReturn<WordREG<T1>, OpUnaryMinus>::Type_t
  operator-(const WordJIT<T1>& l)
  {
    typename UnaryReturn<WordREG<T1>, OpUnaryMinus>::Type_t ret;
    ret.setup(l);
    return -ret.elem();
  }


#if defined (QDP_CODEGEN_VECTOR)  
  template<class T1>
  inline typename UnaryReturn<WordVecREG<T1>, OpUnaryMinus>::Type_t
  operator-(const WordVecJIT<T1>& l)
  {
    typename UnaryReturn<WordVecREG<T1>, OpUnaryMinus>::Type_t ret;
    ret.setup(l);
    return -ret.elem();
  }
#endif
  
  // ***********


#if defined (QDP_CODEGEN_VECTOR)  
  template<class T>
  struct ScalarType<WordVecJIT<T> >
  {
    typedef WordJIT< T > Type_t;
  };
#endif
  
  template<class T>
  struct ScalarType<WordJIT<T> >
  {
    typedef WordJIT< T > Type_t;
  };
  
  template<class T>
  struct REGType< WordJIT<T> >
  {
    typedef WordREG<typename REGType<T>::Type_t>  Type_t;
  };

#if defined (QDP_CODEGEN_VECTOR)  
  template<class T>
  struct REGType< WordVecJIT<T> >
  {
    typedef WordVecREG<typename REGType<T>::Type_t>  Type_t;
  };
#endif
  
  // ***********
  
  template<class T>
  struct BASEType< WordJIT<T> >
  {
    typedef Word<typename BASEType<T>::Type_t>  Type_t;
  };

#if defined (QDP_CODEGEN_VECTOR)  
  template<class T>
  struct BASEType< WordVecJIT<T> >
  {
    typedef WordVec<typename BASEType<T>::Type_t>  Type_t;
  };
#endif  

  // **********  
  
  template<class T> 
  struct WordType<WordJIT<T> >
  {
    typedef T  Type_t;
  };

#if defined (QDP_CODEGEN_VECTOR)  
  template<class T> 
  struct WordType<WordVecJIT<T> >
  {
    typedef T  Type_t;
  };
#endif

  

  // Default binary(WordJIT,WordJIT) -> WordJIT
  template<class T1, class T2, class Op>
  struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, Op> {
    typedef WordJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };

#if defined (QDP_CODEGEN_VECTOR)
  template<class T1, class T2, class Op>
  struct BinaryReturn<WordVecJIT<T1>, WordVecJIT<T2>, Op> {
    typedef WordVecJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };

  template<class T1, class T2, class Op>
  struct BinaryReturn<WordVecJIT<T1>, WordJIT<T2>, Op> {
    typedef WordVecJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };

  template<class T1, class T2, class Op>
  struct BinaryReturn<WordJIT<T1>, WordVecJIT<T2>, Op> {
    typedef WordVecJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };
#endif

  

  inline void 
  zero_rep(WordJIT<double> dest)
  {
    llvm_store_ptr_idx( llvm_create_value( 0.0 ) , dest.getBaseReg() , dest.getOffset() );
  }

  inline void 
  zero_rep(WordJIT<jit_half_t> dest)
  {
    llvm_store_ptr_idx( llvm_create_value( 0.0 ) , dest.getBaseReg() , dest.getOffset() );
  }

  inline void 
  zero_rep(WordJIT<float> dest)
  {
    llvm_store_ptr_idx( llvm_create_value( 0.0 ) , dest.getBaseReg() , dest.getOffset() );
  }

  inline void 
  zero_rep(WordJIT<int> dest)
  {
    llvm_store_ptr_idx( llvm_create_value( 0 ) , dest.getBaseReg() , dest.getOffset() );
  }


  // *****************

#if defined (QDP_CODEGEN_VECTOR)  
  inline void 
  zero_rep(WordVecJIT<double> dest)
  {
    llvm_vecstore_ptr_idx( llvm_fill_vector( llvm_create_value( 0.0 ) ) , dest.getBaseReg() , dest.getOffset() );
  }

  inline void 
  zero_rep(WordVecJIT<jit_half_t> dest)
  {
    llvm_vecstore_ptr_idx( llvm_fill_vector( llvm_create_value( 0.0 ) ) , dest.getBaseReg() , dest.getOffset() );
  }

  inline void 
  zero_rep(WordVecJIT<float> dest)
  {
    llvm_vecstore_ptr_idx( llvm_fill_vector( llvm_create_value( 0.0 ) ) , dest.getBaseReg() , dest.getOffset() );
  }

  inline void 
  zero_rep(WordVecJIT<int> dest)
  {
    llvm_vecstore_ptr_idx( llvm_fill_vector( llvm_create_value( 0.0 ) ) , dest.getBaseReg() , dest.getOffset() );
  }
#endif

  
  
  template<class T, class T1, class T2, class T3>
  inline void
  fill_random_jit(WordJIT<T> d, T1 seed, T2 skewed_seed, const T3& seed_mult)
  {
    typedef typename REGType<typename BASEType< T2 >::Type_t >::Type_t T2REG;
    typedef typename REGType<typename BASEType< T1 >::Type_t >::Type_t T1REG;
    T2REG sk;
    T1REG se;
    sk.setup(skewed_seed);
    se.setup(seed);

    d = seedToFloat( sk ).elem().elem().elem();

    seed        = se * seed_mult;
    skewed_seed = sk * seed_mult;
  }


#if defined (QDP_CODEGEN_VECTOR)
  template<class T, class T1, class T2, class T3>
  inline void
  fill_random_jit(WordVecJIT<T> d, T1 seed, T2 skewed_seed, const T3& seed_mult)
  {
    QDPIO::cout << "fill_random_jit not yet implemented\n";
  }
#endif
  


} // namespace QDP

#endif
