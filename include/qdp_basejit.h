#ifndef QDP_PRIMJITBASE
#define QDP_PRIMJITBASE

namespace QDP {

  template<class T, int N >
  class BaseJIT {

    T F[N];
    bool setup_m;

    // jit_value_t full;
    // jit_value_t level;

    jit_value_t       m_base;
    IndexDomainVector partial_offset;
    JitDeviceLayout   layout;

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


    // void stack_setup( const jit_value_t& stack_base , IndexDomainVector args = IndexDomainVector() ) {
    //   m_base = stack_base;
    //   partial_offset = args;
    //   for (int i = 0 ; i < N ; i++ ) {
    // 	IndexDomainVector args_curry = args;
    // 	args_curry.push_back( make_pair( N , create_jit_value(i) ) );
    // 	F[i].stack_setup( m_base , args_curry );
    //   }
    //   setup_m = true;
    // }


    void setup( const jit_value_t& base , JitDeviceLayout lay , IndexDomainVector args = IndexDomainVector() ) {
      m_base = base;
      layout = lay;
      partial_offset = args;
      for (int i = 0 ; i < N ; i++ ) {
	IndexDomainVector args_curry = args;
	args_curry.push_back( make_pair( N , create_jit_value(i) ) );
	F[i].setup( m_base , lay , args_curry );
      }
      setup_m = true;
    }

    // void setup( const jit_value_t& r_base_, const jit_value_t& full_, const jit_value_t& level_ ) {
    //   full = full_;
    //   level = level_;
    //   r_base = r_base_;
    //   for (int i = 0 ; i < N ; i++ ) 
    // 	F[i].setup( r_base , 
    // 		    jit_ins_mul( full  , create_jit_value(N) ) ,
    // 		    jit_ins_add( level , jit_ins_mul( full , create_jit_value(i) ) ) );
    //   setup_m = true;
    // }


    T getJitElem( jit_value_t index ) {
      assert(setup_m);
      T ret;
      IndexDomainVector args = partial_offset;
      args.push_back( make_pair( N , index ) );
      ret.setup( m_base , layout , args );
      return ret;
#if 0
      T ret;
      ret.setup( r_base , 
		 jit_ins_mul( full  , create_jit_value(N) ) ,
		 jit_ins_add( level , jit_ins_mul( full , index ) ) );
      return ret;
#endif
    }


    typename REGType<T>::Type_t getRegElem( jit_value_t index ) {
      T jit;
      IndexDomainVector args = partial_offset;
      args.push_back( make_pair( N , index ) );
      jit.setup( m_base , JitDeviceLayout::Scalar, args );
      typename REGType<T>::Type_t ret_reg;
      ret_reg.setup( jit );
      return ret_reg;
#if 0
      QDP_error_exit("getRegElem ni");
      jit_value_t ws = create_jit_value( sizeof(typename WordType<T>::Type_t) );
      jit_value_t idx_mul_length = jit_ins_mul( index  , jit_ins_mul( ws , full ) );
      jit_value_t base           = jit_ins_add( r_base , idx_mul_length );
      T ret_jit;
      ret_jit.setup( base ,
		     jit_ins_mul( full  , create_jit_value(N) ) ,
		     level );
      typename REGType<T>::Type_t ret_reg;
      ret_reg.setup( ret_jit );
      return ret_reg;
#endif
    }



  };

}

#endif
