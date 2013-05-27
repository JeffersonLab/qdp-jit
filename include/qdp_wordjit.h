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

    void setup( llvm::Value * base_m , JitDeviceLayout lay , IndexDomainVector args ) {
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
    //   llvm::Value * lev_mul_ws = llvm_mul ( offset_level , ws );
    //   llvm::Value * address    = llvm_add( r_base , lev_mul_ws );
    //   return address;
    // }

    // llvm::Value * getOffset() const {
    //   llvm::Value * ws         = llvm_create_value( sizeof(typename WordType<T>::Type_t) );
    //   llvm::Value * lev_mul_ws = llvm_mul ( offset_level , ws );
    //   return lev_mul_ws;
    // }


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
    llvm::BasicBlock * block_copy      = llvm_new_basic_block();
    llvm::BasicBlock * block_not_copy  = llvm_new_basic_block();
    llvm::BasicBlock * block_copy_exit = llvm_new_basic_block();
    llvm_cond_branch( mask.get_val() , block_copy , block_not_copy );
    {
      llvm_set_insert_point(block_not_copy);
      llvm_branch( block_copy_exit );
    }
    {
      llvm_set_insert_point(block_copy);
      d = s1;
      llvm_branch( block_copy_exit );
    }
    llvm_set_insert_point(block_copy_exit);
  }



  inline void 
  zero_rep(WordJIT<double>& dest)
  {
    llvm_store_ptr_idx( llvm_create_value( 0.0 ) , dest.getBaseReg() , dest.getOffset() );
  }

  inline void 
  zero_rep(WordJIT<float>& dest)
  {
    llvm::outs() << "dest.getBaseReg()->dump(); = "; dest.getBaseReg()->dump();
    llvm_store_ptr_idx( llvm_create_value( 0.0 ) , dest.getBaseReg() , dest.getOffset() );
  }

  inline void 
  zero_rep(WordJIT<int>& dest)
  {
    llvm_store_ptr_idx( llvm_create_value( 0 ) , dest.getBaseReg() , dest.getOffset() );
  }


  template<class T1, class T3>
  void random_seed_mul(T1& seed, const T3& seed_mult)
  {
    llvm::Value* s0 = seed.elem().elem(0).elem().get_val();
    llvm::Value* s1 = seed.elem().elem(1).elem().get_val();
    llvm::Value* s2 = seed.elem().elem(2).elem().get_val();
    llvm::Value* s3 = seed.elem().elem(3).elem().get_val();

    llvm::Value* m0 = seed_mult.elem().elem(0).elem().get_val();
    llvm::Value* m1 = seed_mult.elem().elem(1).elem().get_val();
    llvm::Value* m2 = seed_mult.elem().elem(2).elem().get_val();
    llvm::Value* m3 = seed_mult.elem().elem(3).elem().get_val();

    std::vector<llvm::Value*> ret = llvm_seedMultiply(s0,s1,s2,s3, m0,m1,m2,m3);

    seed.elem().elem(0).elem().setup( ret.at(0) );
    seed.elem().elem(1).elem().setup( ret.at(1) );
    seed.elem().elem(2).elem().setup( ret.at(2) );
    seed.elem().elem(3).elem().setup( ret.at(3) );
  }

  //! dest  = random  
  template<class T, class T1, class T2, class T3>
  inline void
  fill_random(WordJIT<T>& d, T1& seed, T2& skewed_seed, const T3& seed_mult)
  {
    d = seedToFloat( skewed_seed ).elem().elem().elem();

    random_seed_mul( seed        , seed_mult );
    random_seed_mul( skewed_seed , seed_mult );

    // seed        = seed        * seed_mult;
    // skewed_seed = skewed_seed * seed_mult;
  }



} // namespace QDP

#endif
