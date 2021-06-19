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
    WordJIT(): setup_m(false)
    {
    }

    void setup( llvm::Value * base_m , JitDeviceLayout lay , IndexDomainVector args ) {
      r_base = base_m;
      offset = datalayout( lay , args );
      setup_m = true;
    }

    
    llvm::Value* get_base() const
    {
      if (!setup_m)
	{
	  QDPIO::cerr << "internal error: WordJIT not setup but requesting base" << std::endl;
	  QDP_abort(1);
	}
      return r_base;
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
    // llvm::Value * getFull() const { assert(setup_m); return offset_full; }
    // llvm::Value * getLevel() const { assert(setup_m); return offset_level; }

  private:
    //template<class T1>
    //void operator=(const WordJIT<T1>& s1);
    //void operator=(const WordJIT& s1);

    llvm::Value *     r_base;
    llvm::Value *     offset;
    // llvm::Value *    offset_full;
    // llvm::Value *    offset_level;
    bool setup_m;
  };


  template<class T, class Op>
  struct UnaryReturn<WordJIT<T>, Op> {
    typedef WordJIT<typename UnaryReturn<T, Op>::Type_t>  Type_t;
  };


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


  template<class T>
  struct REGType< WordJIT<T> >
  {
    typedef WordREG<typename REGType<T>::Type_t>  Type_t;
  };

  
  template<class T>
  struct BASEType< WordJIT<T> >
  {
    typedef Word<typename BASEType<T>::Type_t>  Type_t;
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
    JitIf ifCopy( mask.get_val() );
    {
      d = s1;
    }
    ifCopy.end();
  }



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

  
  template<class T, class T1, class T2, class T3>
  inline void
  fill_random(WordJIT<T> d, T1 seed, T2 skewed_seed, const T3& seed_mult)
  {
    typedef typename REGType<typename BASEType< T2 >::Type_t >::Type_t T2REG;
    typedef typename REGType<typename BASEType< T1 >::Type_t >::Type_t T1REG;
    T2REG sk;
    T1REG se;
    sk.setup(skewed_seed);
    se.setup(seed);
#if 1
    d = seedToFloat( sk ).elem().elem().elem();

    seed        = se * seed_mult;
    skewed_seed = sk * seed_mult;
#else
    d = seedToFloat( skewed_seed ).elem().elem().elem();

    seed        = se * seed_mult;
    skewed_seed = sk * seed_mult;
#endif
  }



} // namespace QDP

#endif
