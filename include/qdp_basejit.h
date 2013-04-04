#ifndef QDP_PRIMJITBASE
#define QDP_PRIMJITBASE

namespace QDP {

  template<class T, int N >
  class BaseJIT {

    T F[N];
    bool setup_m;
    jit_value full;
    jit_value level;
    jit_value r_base;

  public:
    BaseJIT(): setup_m(false) {}
    ~BaseJIT() {}

    enum { ThisSize = N };                 // Size in T's
    enum { Size_t = ThisSize * T::Size_t}; // Size in registers


    T& arrayF(int i) { assert(setup_m); return F[i]; }
    const T& arrayF(int i) const { assert(setup_m); return F[i]; }

    void setup( const jit_value& r_base_, const jit_value& full_, const jit_value& level_ ) {
      full = full_;
      level = level_;
      r_base = r_base_;
      for (int i = 0 ; i < N ; i++ ) 
	F[i].setup( r_base , 
		    jit_ins_mul( full  , jit_value(N) ) ,
		    jit_ins_add( level , jit_ins_mul( full , jit_value(i) ) ) );
      setup_m = true;
    }


    T getJitElem( jit_value index ) {
      T ret;
      ret.setup( r_base , 
		 jit_ins_mul( full  , jit_value(N) ) ,
		 jit_ins_add( level , jit_ins_mul( full , index ) ) );
      return ret;
    }


    typename REGType<T>::Type_t getRegElem( jit_value index ) {
      jit_value ws( sizeof(typename WordType<T>::Type_t) );
      jit_value idx_mul_length = jit_ins_mul( index  , jit_ins_mul( ws , full ) );
      jit_value base           = jit_ins_add( r_base , idx_mul_length );
      T ret_jit;
      ret_jit.setup( base ,
		     jit_ins_mul( full  , jit_value(N) ) ,
		     level );
      typename REGType<T>::Type_t ret_reg;
      ret_reg.setup( ret_jit );
      return ret_reg;
    }



  };

}

#endif
