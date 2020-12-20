#ifndef QDP_PRIMJITBASE
#define QDP_PRIMJITBASE

namespace QDP {

  template<class T, int N >
  class BaseJIT {

    std::array<T,N> F;
    bool setup_m;

    llvm::Value *       m_base;
    IndexDomainVector partial_offset;
    JitDeviceLayout   layout;

  public:
    enum { ThisSize = N };                 // Size in T's
    enum { Size_t = ThisSize * T::Size_t}; // Size in registers

    
    //Default constructor
    BaseJIT(): setup_m(false) {}

    
    //Copy constructor
    BaseJIT(const BaseJIT& rhs): partial_offset(rhs.partial_offset),
				 setup_m(rhs.setup_m),      
				 m_base(rhs.m_base),
				 layout(rhs.layout),
				 F(rhs.F)
    {
    }

    
    //Copy assignment
    BaseJIT& operator=(const BaseJIT& rhs)
    {
      if(&rhs == this)
	return *this;
      
      for ( int i = 0 ; i < N ; ++i )
	F[i] = rhs.F[i];
      partial_offset = rhs.partial_offset;
      setup_m = rhs.setup_m;
      m_base  = rhs.m_base;
      layout  = rhs.layout;
      return *this;
    }


    
    


    T& arrayF(int i) { assert(setup_m); return F[i]; }
    const T& arrayF(int i) const { assert(setup_m); return F[i]; }



    void setup( llvm::Value * base , JitDeviceLayout lay , IndexDomainVector args = IndexDomainVector() ) {
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


    T getJitElem( llvm::Value * index ) {
      if (!setup_m)
	{
	  QDPIO::cerr << "qdp-jit internal error: BaseJIT::getJitElem elem not set up.\n";
	  QDP_abort(1);
	}
      T ret;
      IndexDomainVector args = partial_offset;
      args.push_back( make_pair( N , index ) );
      ret.setup( m_base , layout , args );
      return ret;
    }


    typename REGType<T>::Type_t getRegElem( llvm::Value * index ) {
      if (!setup_m)
	{
	  QDPIO::cerr << "qdp-jit internal error: BaseJIT::getJitElem elem not set up.\n";
	  QDP_abort(1);
	}
      T jit;
      IndexDomainVector args = partial_offset;
      args.push_back( make_pair( N , index ) );
      jit.setup( m_base , JitDeviceLayout::Scalar, args );
      typename REGType<T>::Type_t ret_reg;
      ret_reg.setup( jit );
      return ret_reg;
    }

  };


  
  
  
}

#endif
