#ifndef QDP_PRIMJITBASE
#define QDP_PRIMJITBASE

namespace QDP {

  template<class T, int N >
  class BaseJIT {

    T F[N];
    bool setup_m;

    // llvm::Value * full;
    // llvm::Value * level;

    llvm::Value *       m_base;
    IndexDomainVector partial_offset;
    JitDeviceLayout::LayoutEnum   layout;

  public:
    BaseJIT(): 
      setup_m(false)
      // full(jit_ptx_type::s32),
      // level(jit_ptx_type::s32),
      // r_base(jit_ptx_type::u64)
    {}

    ~BaseJIT() {}

    enum { ThisSize = N };                 // Size in T's
    enum { Size_t = ThisSize * T::Size_t}; // Size in registers


    T& arrayF(int i) { assert(setup_m); return F[i]; }
    const T& arrayF(int i) const { assert(setup_m); return F[i]; }


    // void stack_setup( const llvm::Value *& stack_base , IndexDomainVector args = IndexDomainVector() ) {
    //   m_base = stack_base;
    //   partial_offset = args;
    //   for (int i = 0 ; i < N ; i++ ) {
    // 	IndexDomainVector args_curry = args;
    // 	args_curry.push_back( make_pair( N , llvm_create_value(i) ) );
    // 	F[i].stack_setup( m_base , args_curry );
    //   }
    //   setup_m = true;
    // }


    void setup( llvm::Value * base , JitDeviceLayout::LayoutEnum lay , IndexDomainVector args = IndexDomainVector() ) {
      m_base = base;
      layout = lay;
      partial_offset = args;
      for (int i = 0 ; i < N ; i++ ) {
	IndexDomainVector args_curry = args;
	args_curry.push_back( make_pair( N , llvm_create_value(i) ) );
	F[i].setup( m_base , lay , args_curry );
      }
      setup_m = true;
    }

    // void setup( const llvm::Value *& r_base_, const llvm::Value *& full_, const llvm::Value *& level_ ) {
    //   full = full_;
    //   level = level_;
    //   r_base = r_base_;
    //   for (int i = 0 ; i < N ; i++ ) 
    // 	F[i].setup( r_base , 
    // 		    llvm_mul( full  , llvm_create_value(N) ) ,
    // 		    llvm_add( level , llvm_mul( full , llvm_create_value(i) ) ) );
    //   setup_m = true;
    // }


    T getJitElem( llvm::Value * index ) {
      assert(setup_m);
      T ret;
      IndexDomainVector args = partial_offset;
      args.push_back( make_pair( N , index ) );
      ret.setup( m_base , layout , args );
      return ret;
#if 0
      T ret;
      ret.setup( r_base , 
		 llvm_mul( full  , llvm_create_value(N) ) ,
		 llvm_add( level , llvm_mul( full , index ) ) );
      return ret;
#endif
    }


    typename REGType<T>::Type_t getRegElem( llvm::Value * index ) {
      T jit;
      IndexDomainVector args = partial_offset;
      args.push_back( make_pair( N , index ) );
      jit.setup( m_base , JitDeviceLayout::LayoutScalar, args );
      typename REGType<T>::Type_t ret_reg;
      ret_reg.setup( jit );
      return ret_reg;
#if 0
      QDP_error_exit("getRegElem ni");
      llvm::Value * ws = llvm_create_value( sizeof(typename WordType<T>::Type_t) );
      llvm::Value * idx_mul_length = llvm_mul( index  , llvm_mul( ws , full ) );
      llvm::Value * base           = llvm_add( r_base , idx_mul_length );
      T ret_jit;
      ret_jit.setup( base ,
		     llvm_mul( full  , llvm_create_value(N) ) ,
		     level );
      typename REGType<T>::Type_t ret_reg;
      ret_reg.setup( ret_jit );
      return ret_reg;
#endif
    }



  };

}

#endif
