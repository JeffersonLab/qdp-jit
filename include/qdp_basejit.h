#ifndef QDP_PRIMJITBASE
#define QDP_PRIMJITBASE

namespace QDP {

  template<class T, int N >
  class BaseJIT {

    T F[N];
    bool setup_m;
    jit_function_t func;
    jit_value_t full;
    jit_value_t level;
    jit_value_t r_base;

  public:
    BaseJIT(): setup_m(false) {}
    ~BaseJIT() {}

    enum { ThisSize = N };                 // Size in T's
    enum { Size_t = ThisSize * T::Size_t}; // Size in registers


    T& arrayF(int i) { assert(setup_m); return F[i]; }
    const T& arrayF(int i) const { assert(setup_m); return F[i]; }

    void setup(jit_function_t func_, jit_value_t r_base_, jit_value_t full_, jit_value_t level_ ) {
      func = func_;
      full = full_;
      level = level_;
      r_base = r_base_;
      for (int i = 0 ; i < N ; i++ ) 
	F[i].setup( func , 
		    r_base , 
		    jit_ins_mul( full  , jit_val_create_const_int(N) ) ,
		    jit_ins_add( level , jit_ins_mul( full , jit_val_create_const_int(i) ) ) );
      setup_m = true;
    }


    T getJitElem( jit_value_t index ) {
      T ret;
      ret.setup( func, r_base , 
		 jit_ins_mul( full  , jit_val_create_const_int(N) ) ,
		 jit_ins_add( level , jit_ins_mul( full , index ) ) );
      return ret;
    }


    typename REGType<T>::Type_t getRegElem( jit_value_t index ) {
#if 0
      std::cout << "getRegElem " << full << " " << level << " " << sizeof(typename WordType<T>::Type_t) << "\n";
      jit_value_t full_r  = jit_val_create_const_int( full );
      jit_value_t level_r = jit_val_create_const_int( level );
      jit_value_t ws_r    = jit_val_create_const_int( sizeof(typename WordType<T>::Type_t) );
      jit_value_t tmp0    = jit_ins_mul(ws_r,full_r);
      jit_value_t tmp1    = jit_ins_mul( index , tmp0 );
#endif

      jit_value_t ws = jit_val_create_const_int( sizeof(typename WordType<T>::Type_t) );
      jit_value_t idx_mul_length = jit_ins_mul( index , jit_ins_mul( ws,full ) );

      jit_value_t tmp1_u32= jit_val_create_convert(func,jit_ptx_type::u32,idx_mul_length);
      jit_value_t base    = jit_ins_add(r_base,tmp1_u32);
      T ret_jit;
      ret_jit.setup( func , 
		     base , 
		     jit_ins_mul( full  , jit_val_create_const_int(N) ) ,
		     level );
      typename REGType<T>::Type_t ret_reg;
      ret_reg.setup( ret_jit );
      return ret_reg;
    }



  };

}

#endif
