#ifndef QDP_PRIMJITBASE
#define QDP_PRIMJITBASE

namespace QDP {

  template<class T, int N >
  class BaseJIT {

    T F[N];
    bool setup_m = false;
    jit_function_t func;
    int full,level;
    jit_value_t r_base;

  public:

    enum { ThisSize = N };                 // Size in T's
    enum { Size_t = ThisSize * T::Size_t}; // Size in registers


    T& arrayF(int i) { assert(setup_m); return F[i]; }
    const T& arrayF(int i) const { assert(setup_m); return F[i]; }

    void setup(jit_function_t func_, jit_value_t r_base_, int full_, int level_ ) {
      func = func_;
      full = full_;
      level = level_;
      r_base = r_base_;
      for (int i = 0 ; i < N ; i++ ) 
	F[i].setup( func, r_base , full * N , level + full * i );
      setup_m = true;
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

      jit_value_t idx_mul_length = jit_ins_mul( index , 
						jit_val_create_const_int( sizeof(typename WordType<T>::Type_t) * 
									  full ) );

      jit_value_t tmp1_u32= jit_val_create_convert(func,jit_ptx_type::u32,idx_mul_length);
      jit_value_t base    = jit_ins_add(r_base,tmp1_u32);
      T ret_jit;
      ret_jit.setup(func,base,full*N,level);
      typename REGType<T>::Type_t ret_reg;
      ret_reg.setup( ret_jit );
      return ret_reg;
    }



  };

}

#endif
