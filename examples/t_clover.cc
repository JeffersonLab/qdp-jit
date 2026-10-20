
#include "qdp.h"

using namespace QDP;

namespace QDP
{
  class PackForQUDATimer {
    double acc_time;
    PackForQUDATimer(): acc_time(0.0) {}
  public:
    static PackForQUDATimer& Instance() {
      static PackForQUDATimer singleton;
      return singleton;
    }

    double&        get()       { return acc_time; }
    const double&  get() const { return acc_time; }
  };


  template<typename T>
  class PComp: public BaseView<T>
  {
  public:
    typedef T Sub_t;
    enum { ThisSize = 2 };
    Shape shape() const { return BaseView<T>::shape(); }
    PComp( Shape shape , typename WordType<T>::Type_t *base_ ): BaseView<T>(shape,base_) {}

    PComp operator=(const PComp& rhs)
    {
      for (int i = 0 ; i < 2 ; ++i )
	elem(i) = rhs.elem(i);
      return *this;
    }

    T elem(int i) const { return BaseView<T>::elem(i); }

  private:
    //T comp[2];
  };

  template<class T>
  class WordSize< PComp<T> >
  {
  public:
    static int value( const Shape& ret )
    {
      return 2 * WordSize<T>::value( ret );
    }
  };

  template<class T>
  struct GetLimit<PComp<T>,0>
  {
    static int limit(const Shape& shape)
    {
      return 2;
    }
  };


  template<typename T>
  struct PCompJIT: public BaseJIT<T>
  {
    using BaseJIT<T>::BaseJIT;
    int getN() const { return 2; }

    int this_size() const { return 2; }
    template<class T1>
    PCompJIT operator=( const PCompJIT<T1>& rhs) {
      this->elem(0) = rhs.elem(0);
      this->elem(1) = rhs.elem(1);
      return *this;
    }

    PCompJIT operator=( const PCompJIT& rhs) {
      this->elem(0) = rhs.elem(0);
      this->elem(1) = rhs.elem(1);
      return *this;
    }
  };


  template<class T> 
  struct ScalarType<PComp<T> >
  {
    typedef PComp<typename ScalarType<T>::Type_t>  Type_t;
  };

  template<class T> 
  struct ScalarType<PCompJIT<T> >
  {
    typedef PCompJIT<typename ScalarType<T>::Type_t>  Type_t;
  };
  
  template<class T> 
  struct JITType<PComp<T> >
  {
    typedef PCompJIT<typename JITType<T>::Type_t>  Type_t;
  };

  template<class T>
  struct WordType<PComp<T> > 
  {
    typedef typename WordType<T>::Type_t  Type_t;
  };

  template<class T>
  struct WordType<PCompJIT<T> > 
  {
    typedef typename WordType<T>::Type_t  Type_t;
  };






  template<typename T>
  class PTriDia: public BaseView<T>
  {
  public:
    typedef T Sub_t;
    enum { ThisSize = 2*Nc };
    Shape shape() const { return BaseView<T>::shape(); }
    PTriDia( Shape shape , typename WordType<T>::Type_t *base_ ): BaseView<T>(shape,base_) {}

    PTriDia operator=(const PTriDia& rhs)
    {
      for (int i = 0 ; i < 2*Nc ; ++i )
	elem(i) = rhs.elem(i);
      return *this;
    }

    T elem(int i) const { return BaseView<T>::elem(i); }
  private:
    //T diag[2*Nc];
  };

  template<class T>
  class WordSize< PTriDia<T> >
  {
  public:
    static int value( const Shape& ret )
    {
      return 2 * Nc * WordSize<T>::value( ret );
    }
  };

  template<class T>
  struct GetLimit<PTriDia<T>,0>
  {
    static int limit(const Shape& shape)
    {
      return 2*Nc;
    }
  };


  

  template<typename T>
  struct PTriDiaJIT: public BaseJIT<T>
  {
    using BaseJIT<T>::BaseJIT;

    int getN() const { return 2 * Nc; }
    int this_size() const { return 2 * Nc; }
    
    template<class T1>
    PTriDiaJIT operator=( const PTriDiaJIT<T1>& rhs)
    {
      JitForLoop i( 0 , 2 * Nc );
      {
	this->elem( i.index() ) = rhs.elem( i.index() );
      }
      i.end();
      return *this;
    }

    PTriDiaJIT operator=( const PTriDiaJIT& rhs)
    {
      JitForLoop i( 0 , 2 * Nc );
      {
	this->elem( i.index() ) = rhs.elem( i.index() );
      }
      i.end();
      return *this;
    }
  };

  


  template<class T> 
  struct ScalarType<PTriDia<T> >
  {
    typedef PTriDia<typename ScalarType<T>::Type_t>  Type_t;
  };
  
  template<class T> 
  struct ScalarType<PTriDiaJIT<T> >
  {
    typedef PTriDiaJIT<typename ScalarType<T>::Type_t>  Type_t;
  };

  
  template<class T> 
  struct JITType<PTriDia<T> >
  {
    typedef PTriDiaJIT<typename JITType<T>::Type_t>  Type_t;
  };


  template<class T>
  struct WordType<PTriDia<T> > 
  {
    typedef typename WordType<T>::Type_t  Type_t;
  };

  template<class T>
  struct WordType<PTriDiaJIT<T> > 
  {
    typedef typename WordType<T>::Type_t  Type_t;
  };






  template<typename T>
  class PTriOff: public BaseView<T>
  {
  public:
    typedef T Sub_t;
    enum { ThisSize = 2*Nc*Nc-Nc };
    Shape shape() const { return BaseView<T>::shape(); }
    PTriOff( Shape shape , typename WordType<T>::Type_t *base_ ): BaseView<T>(shape,base_) {}

    PTriOff operator=(const PTriOff& rhs)
    {
      for (int i = 0 ; i < 2*Nc*Nc-Nc ; ++i )
	elem(i) = rhs.elem(i);
      return *this;
    }

    T elem(int i) const { return BaseView<T>::elem(i); }
    
  private:
    //T offd[2*Nc*Nc-Nc];
  };


  template<class T>
  class WordSize< PTriOff<T> >
  {
  public:
    static int value( const Shape& ret )
    {
      return ( 2 * Nc * Nc - Nc ) * WordSize<T>::value( ret );
    }
  };

  template<class T>
  struct GetLimit<PTriOff<T>,0>
  {
    static int limit(const Shape& shape)
    {
      return 2*Nc*Nc-Nc;
    }
  };

  
  
  template<typename T>
  struct PTriOffJIT: public BaseJIT<T>
  {
    using BaseJIT<T>::BaseJIT;

    int getN() const { return this_size(); }
    int this_size() const { return 2 * Nc * Nc - Nc; }

    template<class T1>
    PTriOffJIT operator=( const PTriOffJIT<T1>& rhs) {
      JitForLoop i( 0 , 2*Nc*Nc-Nc );
      {
	this->elem( i.index() ) = rhs.elem( i.index() );
      }
      i.end();
      return *this;
    }
    
    PTriOffJIT operator=( const PTriOffJIT& rhs) {
      JitForLoop i( 0 , 2*Nc*Nc-Nc );
      {
	this->elem( i.index() ) = rhs.elem( i.index() );
      }
      i.end();
      return *this;
    }
  };



  template<class T> 
  struct ScalarType<PTriOff<T> >
  {
    typedef PTriOff<typename ScalarType<T>::Type_t>  Type_t;
  };
  
  template<class T> 
  struct ScalarType<PTriOffJIT<T> >
  {
    typedef PTriOffJIT<typename ScalarType<T>::Type_t>  Type_t;
  };


  template<class T> 
  struct JITType<PTriOff<T> >
  {
    typedef PTriOffJIT<typename JITType<T>::Type_t>  Type_t;
  };


  template<class T>
  struct WordType<PTriOff<T> > 
  {
    typedef typename WordType<T>::Type_t  Type_t;
  };

  template<class T>
  struct WordType<PTriOffJIT<T> > 
  {
    typedef typename WordType<T>::Type_t  Type_t;
  };



  template<class T>
  struct LeafFunctor<PComp<T>, PrintTag>
  {
    typedef int Type_t;
    static int apply(const PrintTag &f)
    { 
      f.os_m << "PComp<";
      LeafFunctor<T,PrintTag>::apply(f);
      f.os_m << ">"; 
      return 0;
    }
  };

  template<class T>
  struct LeafFunctor<PTriDia<T>, PrintTag>
  {
    typedef int Type_t;
    static int apply(const PrintTag &f)
    { 
      f.os_m << "PTriDia<";
      LeafFunctor<T,PrintTag>::apply(f);
      f.os_m << ">"; 
      return 0;
    }
  };

  template<class T>
  struct LeafFunctor<PTriOff<T>, PrintTag>
  {
    typedef int Type_t;
    static int apply(const PrintTag &f)
    { 
      f.os_m << "PTriOff<";
      LeafFunctor<T,PrintTag>::apply(f);
      f.os_m << ">"; 
      return 0;
    }
  };
}


namespace QDP {

template<typename RealT,typename U,typename X,typename Y>
  void function_make_clov_exec(JitFunction& function, const ExprPlan& plan, 
			       const RealT& diag_mass, 
			       const U& f0,
			       const U& f1,
			       const U& f2,
			       const U& f3,
			       const U& f4,
			       const U& f5,
			       X& tri_dia,
			       Y& tri_off)
  {
#ifdef QDP_DEEP_LOG
    function.type_W = typeid(REAL).name();
    //function.set_dest_id( tri_dia.getId() );
    function.set_dest_id( tri_off.getId() );
    function.set_is_lat(true);
#endif

    const LatticeLayout layout = f0.layout();

    int th_count = layout.sitesOnNode();

    LaunchArgs args( plan );
    args.add_iteration( th_count , layout.sitesOnNode() , all.getIdSiteTable(layout) );
    args.add_plan( plan , all );
    args.launch( function , th_count );
  }


template<typename RealT,typename U,typename X,typename Y>
  void function_make_clov_build(JitFunction& function, const ExprPlan& plan,
				const RealT& diag_mass, 
				const U& f0,
				const U& f1,
				const U& f2,
				const U& f3,
				const U& f4,
				const U& f5,
				const X& tri_dia,
				const Y& tri_off)
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

    typedef typename WordType<RealT>::Type_t REALT;

    KernelBuilder k( "make_clov" , __PRETTY_FUNCTION__ , plan );   // th_count, latsize, site table

    typedef typename LeafFunctor<RealT, MGParamLeaf>::Type_t  RealTJIT;
    RealTJIT diag_mass_jit( k.view( diag_mass ) );

    typedef typename LeafFunctor<U, MGParamLeaf>::Type_t  UJIT;
    UJIT f0_jit( k.view( f0 ) );
    UJIT f1_jit( k.view( f1 ) );
    UJIT f2_jit( k.view( f2 ) );
    UJIT f3_jit( k.view( f3 ) );
    UJIT f4_jit( k.view( f4 ) );
    UJIT f5_jit( k.view( f5 ) );

    typedef typename LeafFunctor<X, MGParamLeaf>::Type_t  XJIT;
    XJIT tri_dia_jit( k.view( tri_dia ) );

    typedef typename LeafFunctor<Y, MGParamLeaf>::Type_t  YJIT;
    YJIT tri_off_jit( k.view( tri_off ) );

    llvm::Value* r_idx = k.site();

    auto f0_j = f0_jit.elem( r_idx );
    auto f1_j = f1_jit.elem( r_idx );
    auto f2_j = f2_jit.elem( r_idx );
    auto f3_j = f3_jit.elem( r_idx );
    auto f4_j = f4_jit.elem( r_idx );
    auto f5_j = f5_jit.elem( r_idx );

    auto tri_dia_j = tri_dia_jit.elem( r_idx );
    auto tri_off_j = tri_off_jit.elem( r_idx );

    for(int jj = 0; jj < 2; jj++) {
      for(int ii = 0; ii < 2*Nc; ii++) {
	tri_dia_j.elem(jj).elem(ii) = diag_mass_jit.elem().elem().elem();
	//tri[site].diag[jj][ii] = diag_mass.elem().elem().elem();
      }
    }


    typedef RComplexJIT<WordJIT<REALT> > C_t;
    typedef RScalarJIT<WordJIT<REALT> >  R_t;
    
    C_t E_minus = StackAllocJit< C_t >::alloc();
    C_t B_minus = StackAllocJit< C_t >::alloc();
    C_t ctmp_0 = StackAllocJit< C_t >::alloc();
    C_t ctmp_1 = StackAllocJit< C_t >::alloc();
    R_t rtmp_0 = StackAllocJit< R_t >::alloc();
    R_t rtmp_1 = StackAllocJit< R_t >::alloc();

    JitForLoop i( 0 , Nc );
    {
      ctmp_0 = f5_j.elem().elem(i,i);
      ctmp_0 -= f0_j.elem().elem(i,i);
      rtmp_0 = imag(ctmp_0);
      tri_dia_j.elem(0).elem(i) += rtmp_0;
	  
      tri_dia_j.elem(0).elem(i+Nc) -= rtmp_0;
	  
      ctmp_1 = f5_j.elem().elem(i,i);
      ctmp_1 += f0_j.elem().elem(i,i);
      rtmp_1 = imag(ctmp_1);
      tri_dia_j.elem(1).elem(i) -= rtmp_1;
	  
      tri_dia_j.elem(1).elem(i+Nc) += rtmp_1;
    }
    i.end();

    for(int i = 1; i < Nc; ++i) {
      for(int j = 0; j < i; ++j) {
	    
	int elem_ij  = i*(i-1)/2 + j;
	int elem_tmp = (i+Nc)*(i+Nc-1)/2 + j+Nc;
	    
	ctmp_0 = f0_j.elem().elem(i,j);
	ctmp_0 -= f5_j.elem().elem(i,j);
	tri_off_j.elem(0).elem(elem_ij) = timesI(ctmp_0);
	    
	zero_rep( tri_off_j.elem(0).elem(elem_tmp) );
	tri_off_j.elem(0).elem(elem_tmp) -= tri_off_j.elem(0).elem(elem_ij);// * -1.0;
	    
	ctmp_1 = f5_j.elem().elem(i,j);
	ctmp_1 += f0_j.elem().elem(i,j);
	tri_off_j.elem(1).elem(elem_ij) = timesI(ctmp_1);
	    
	zero_rep( tri_off_j.elem(1).elem(elem_tmp) );
	tri_off_j.elem(1).elem(elem_tmp) -= tri_off_j.elem(1).elem(elem_ij);
      }
    }

    for(int i = 0; i < Nc; ++i) {
      for(int j = 0; j < Nc; ++j) {
	    
	int elem_ij  = (i+Nc)*(i+Nc-1)/2 + j;
	    
	//E_minus = timesI(f2_j.elem().elem(i,j));
	E_minus = f2_j.elem().elem(i,j);
	E_minus = timesI( E_minus );

	E_minus += f4_j.elem().elem(i,j);
	    
	//B_minus = timesI(f3_j.elem().elem(i,j));
	B_minus = f3_j.elem().elem(i,j);
	B_minus = timesI( B_minus );

	B_minus -= f1_j.elem().elem(i,j);
	    
	tri_off_j.elem(0).elem(elem_ij) = B_minus - E_minus;
	    
	tri_off_j.elem(1).elem(elem_ij) = E_minus + B_minus;
      }
    }

    //    std::cout << __PRETTY_FUNCTION__ << ": leaving\n";

    k.finish( function );
  }


template<typename T,typename X,typename Y>
  void function_ldagdlinv_exec( JitFunction& function, const ExprPlan& plan,
				T& tr_log_diag,
				X& tri_dia,
				Y& tri_off,
				const Subset& s)
  {
#ifdef QDP_DEEP_LOG
    function.type_W = typeid(REAL).name();
    function.set_dest_id( tr_log_diag.getId() );
    function.set_is_lat(true);
#endif
    const LatticeLayout layout = tri_dia.layout();

    int th_count = s.numSiteTable(layout);

    LaunchArgs args( plan );
    args.add_iteration( th_count , tr_log_diag.layout().sitesOnNode() , s.getIdSiteTable(layout) );
    args.add_plan( plan , s );
    args.launch( function , th_count );

  }


template<typename U,typename T,typename X,typename Y>
  void function_ldagdlinv_build(JitFunction& function, const ExprPlan& plan,
				const T& tr_log_diag,
				const X& tri_dia,
				const Y& tri_off,
				const Subset& s)
  {
    typedef typename WordType<U>::Type_t REALT;

    //std::cout << __PRETTY_FUNCTION__ << " entering\n";

    KernelBuilder k( "ldagdlinv" , __PRETTY_FUNCTION__ , plan );   // th_count, latsize, site table

    typedef typename LeafFunctor<T, MGParamLeaf>::Type_t  TJIT;
    TJIT tr_log_diag_jit( k.view( tr_log_diag ) );

    typedef typename LeafFunctor<X, MGParamLeaf>::Type_t  XJIT;
    XJIT tri_dia_jit( k.view( tri_dia ) );

    typedef typename LeafFunctor<Y, MGParamLeaf>::Type_t  YJIT;
    YJIT tri_off_jit( k.view( tri_off ) );

    llvm::Value* r_idx = k.site();

    auto tr_log_diag_j = tr_log_diag_jit.elem(r_idx);
    auto tri_dia_j     = tri_dia_jit.elem(r_idx);
    auto tri_off_j     = tri_off_jit.elem(r_idx);



    typedef RScalarJIT <WordJIT<REALT> >  R_t;
    typedef RComplexJIT<WordJIT<REALT> >  C_t;

    R_t zip = StackAllocJit< R_t >::alloc();
    zero_rep(zip);
    int N = 2*Nc;
      
    //int site_neg_logdet=0;
  
    auto inv_d    = stack_alloc_array_jit<R_t>(6);
    auto inv_offd = stack_alloc_array_jit<C_t>(15);
    auto v        = stack_alloc_array_jit<C_t>(6);
    auto diag_g   = stack_alloc_array_jit<R_t>(6);

    C_t A_ii = StackAllocJit< C_t >::alloc();
    R_t one  = StackAllocJit< R_t >::alloc();
    one = 1.0;
    
    for(int block=0; block < 2; block++) {


      for(int i=0; i < N; i++) {
	inv_d[i] = tri_dia_j.elem(block).elem(i);
      }

      for(int i=0; i < 15; i++) { 
	inv_offd[i] = tri_off_j.elem(block).elem(i);
      }

      for(int j=0; j < N; ++j) { 
	    
	for(int i=0; i < j; i++) { 
	  int elem_ji = j*(j-1)/2 + i;
	      
	  A_ii = cmplx( inv_d[i], zip );
	  v[i] = A_ii*adj(inv_offd[elem_ji]);
	}

	v[j] = cmplx(inv_d[j],zip);
	    
	for(int k=0; k < j; k++) { 
	  int elem_jk = j*(j-1)/2 + k;
	  v[j] -= inv_offd[elem_jk]*v[k];
	}
	    
	inv_d[j] = real( v[j] );
	    
	for(int k=j+1; k < N; k++) { 
	  int elem_kj = k*(k-1)/2 + j;
	  for(int l=0; l < j; l++) { 
	    int elem_kl = k*(k-1)/2 + l;
	    inv_offd[elem_kj] -= inv_offd[elem_kl] * v[l];
	  }
	  inv_offd[elem_kj] /= v[j];
	}
      }

      // Now fix up the inverse
      //one.elem() = (REALT)1;
	  
      for(int i=0; i < N; i++) { 
	diag_g[i] = one/inv_d[i];
      
	// Compute the trace log
	// NB we are always doing trace log | A | 
	// (because we are always working with actually A^\dagger A
	//  even in one flavour case where we square root)
	tr_log_diag_j.elem().elem() += log(fabs(inv_d[i]));
	// However, it is worth counting just the no of negative logdets
#if 0
	if( inv_d[i].elem() < 0 ) { 
	  site_neg_logdet++;
	}
#endif
      }

      // Now we need to invert the L D L^\dagger 
      // We can do this by solving:
      //  L D L^\dagger M^{-1} = 1   
      // This can be done by solving L D X = 1  (X = L^\dagger M^{-1})
      // Then solving L^\dagger M^{-1} = X
      // LD is lower diagonal and so X will also be lower diagonal.
      // LD X = 1 can be solved by forward substitution.
      // Likewise L^\dagger is strictly upper triagonal and so
      // L^\dagger M^{-1} = X can be solved by forward substitution.
      
      for(int k = 0; k < N; ++k) {

	for(int i = 0; i < k; ++i) {
	  zero_rep(v[i]);
	}
	    
	/*# Forward substitution */
	    
	// The first element is the inverse of the diagonal
	v[k] = cmplx(diag_g[k],zip);
	    
	for(int i = k+1; i < N; ++i) {
	  zero_rep(v[i]);
	      
	  for(int j = k; j < i; ++j) {
	    int elem_ij = i*(i-1)/2+j;	
		
	    // subtract l_ij*d_j*x_{kj}
	    v[i] -= inv_offd[elem_ij] *inv_d[j]*v[j];
		
	  }
	      
	  // scale out by 1/d_i
	  v[i] *= diag_g[i];
	}
	    
	/*# Backward substitution */
	// V[N-1] remains unchanged
	// Start from V[N-2]
	    
	for(int i = N-2; (int)i >= (int)k; --i) {
	  for(int j = i+1; j < N; ++j) {
	    int elem_ji = j*(j-1)/2 + i;
	    // Subtract terms of typ (l_ji)*x_kj
	    v[i] -= adj(inv_offd[elem_ji]) * v[j];
	  }
	}
	    
	/*# Overwrite column k of invcl.offd */
	inv_d[k] = real(v[k]);
	for(int i = k+1; i < N; ++i) {

	  int elem_ik = i*(i-1)/2+k;
	  inv_offd[elem_ik] = v[i];
	}
      }

      // Overwrite original data
      for(int i=0; i < N; i++) { 
	tri_dia_j.elem(block).elem(i) = inv_d[i];
      }
      for(int i=0; i < 15; i++) { 
	tri_off_j.elem(block).elem(i) = inv_offd[i];
      }
    }

    //    std::cout << __PRETTY_FUNCTION__ << " leaving\n";

    k.finish( function );
  }


template<typename U,typename X,typename Y>
  void function_triacntr_exec( JitFunction& function, const ExprPlan& plan,
			       U& B,
			       const X& tri_dia,
			       const Y& tri_off,
			       int mat,
			       const Subset& s)
  {
#ifdef QDP_DEEP_LOG
    function.type_W = typeid(REAL).name();
    function.set_dest_id( B.getId() );
    function.set_is_lat(true);
#endif
    const LatticeLayout layout = tri_dia.layout();

    int th_count = s.numSiteTable(layout);

    LaunchArgs args( plan );
    args.add_iteration( th_count , tri_dia.layout().sitesOnNode() , s.getIdSiteTable(layout) );
    args.add_int( mat );
    args.add_plan( plan , s );
    args.launch( function , th_count );

  }


template<typename U,typename X,typename Y>
  void function_triacntr_build( JitFunction& function, const ExprPlan& plan,
				const U& B,
				const X& tri_dia,
				const Y& tri_off,
				int mat,
				const Subset& s)
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

    typedef typename WordType<U>::Type_t REALT;

    KernelBuilder k( "triacntr" , __PRETTY_FUNCTION__ , plan );   // th_count, latsize, site table

    ParamRef p_mat    = k.add_param<int>();

    typedef typename LeafFunctor<U, MGParamLeaf>::Type_t  UJIT;
    UJIT B_jit( k.view( B ) );

    typedef typename LeafFunctor<X, MGParamLeaf>::Type_t  XJIT;
    XJIT tri_dia_jit( k.view( tri_dia ) );

    typedef typename LeafFunctor<Y, MGParamLeaf>::Type_t  YJIT;
    YJIT tri_off_jit( k.view( tri_off ) );

    llvm::Value* r_idx = k.site();

    llvm::Value * r_mat    = llvm_derefParam( p_mat );

    auto B_j = B_jit.elem(r_idx);
    auto tri_dia_j = tri_dia_jit.elem(r_idx);
    auto tri_off_j = tri_off_jit.elem(r_idx);

    typedef RScalarJIT <WordJIT<REALT> >  R_t;
    typedef RComplexJIT<WordJIT<REALT> >  C_t;

    JitSwitch sw(r_mat);
    {
      /*# gamma(   0)   1  0  0  0            # ( 0000 )  --> 0 */
      /*#               0  1  0  0 */
      /*#               0  0  1  0 */
      /*#               0  0  0  1 */
      /*# From diagonal part */
      sw.case_begin(0);
      {

	C_t lctmp0   = StackAllocJit< C_t >::alloc();
	R_t lr_zero0 = StackAllocJit< R_t >::alloc();
	R_t lrtmp0   = StackAllocJit< R_t >::alloc();

	zero_rep(lr_zero0);
	    
	for(int i0 = 0; i0 < Nc; ++i0)
	  {
	    lrtmp0 = tri_dia_j.elem(0).elem(i0);
	    lrtmp0 += tri_dia_j.elem(0).elem(i0+Nc);
	    lrtmp0 += tri_dia_j.elem(1).elem(i0);
	    lrtmp0 += tri_dia_j.elem(1).elem(i0+Nc);
	    B_j.elem().elem(i0,i0) = cmplx(lrtmp0,lr_zero0);
	  }
	    
	/*# From lower triangular portion */
	int elem_ij0 = 0;
	for(int i0 = 1; i0 < Nc; ++i0) {
	      
	  int elem_ijb0 = (i0+Nc)*(i0+Nc-1)/2 + Nc;
	      
	  for(int j0 = 0; j0 < i0; ++j0) {
		
	    lctmp0 = tri_off_j.elem(0).elem(elem_ij0);
	    lctmp0 += tri_off_j.elem(0).elem(elem_ijb0);
	    lctmp0 += tri_off_j.elem(1).elem(elem_ij0);
	    lctmp0 += tri_off_j.elem(1).elem(elem_ijb0);
		
	    B_j.elem().elem(j0,i0) = lctmp0;
	    B_j.elem().elem(i0,j0) = adj(lctmp0);
		
	    elem_ij0++;
	    elem_ijb0++;
	  }
	}
      }
      sw.case_end();

      /*# gamma(  12)  -i  0  0  0            # ( 0011 )  --> 3 */
      /*#               0  i  0  0 */
      /*#               0  0 -i  0 */
      /*#               0  0  0  i */
      /*# From diagonal part */
      sw.case_begin( 3 );
      {

	C_t lctmp3   = StackAllocJit< C_t >::alloc();
	R_t lr_zero3 = StackAllocJit< R_t >::alloc();
	R_t lrtmp3   = StackAllocJit< R_t >::alloc();
	
	lr_zero3 = 0;
	    
	for(int i3 = 0; i3 < Nc; ++i3) {
	      
	  lrtmp3 = tri_dia_j.elem(0).elem(i3+Nc);
	  lrtmp3 -= tri_dia_j.elem(0).elem(i3);
	  lrtmp3 -= tri_dia_j.elem(1).elem(i3);
	  lrtmp3 += tri_dia_j.elem(1).elem(i3+Nc);
	  B_j.elem().elem(i3,i3) = cmplx(lr_zero3,lrtmp3);
	}
	    
	/*# From lower triangular portion */
	int elem_ij3 = 0;
	for(int i3 = 1; i3 < Nc; ++i3) {
	      
	  int elem_ijb3 = (i3+Nc)*(i3+Nc-1)/2 + Nc;
	      
	  for(int j3 = 0; j3 < i3; ++j3) {
		
	    lctmp3 = tri_off_j.elem(0).elem(elem_ijb3);
	    lctmp3 -= tri_off_j.elem(0).elem(elem_ij3);
	    lctmp3 -= tri_off_j.elem(1).elem(elem_ij3);
	    lctmp3 += tri_off_j.elem(1).elem(elem_ijb3);
		
	    B_j.elem().elem(j3,i3) = timesI(adj(lctmp3));
	    B_j.elem().elem(i3,j3) = timesI(lctmp3);
		
	    elem_ij3++;
	    elem_ijb3++;
	  }
	}
      }
      sw.case_end();

      /*# gamma(  13)   0 -1  0  0            # ( 0101 )  --> 5 */
      /*#               1  0  0  0 */
      /*#               0  0  0 -1 */
      /*#               0  0  1  0 */
      sw.case_begin( 5 );
      {

	C_t lctmp5 = StackAllocJit< C_t >::alloc();
	R_t lrtmp5 = StackAllocJit< R_t >::alloc();
	
	for(int i5 = 0; i5 < Nc; ++i5) {
	      
	  int elem_ij5 = (i5+Nc)*(i5+Nc-1)/2;
	      
	  for(int j5 = 0; j5 < Nc; ++j5) {
		
	    int elem_ji5 = (j5+Nc)*(j5+Nc-1)/2 + i5;
		
	    lctmp5 = adj(tri_off_j.elem(0).elem(elem_ji5));
	    lctmp5 -= tri_off_j.elem(0).elem(elem_ij5);
	    lctmp5 += adj(tri_off_j.elem(1).elem(elem_ji5));
	    lctmp5 -= tri_off_j.elem(1).elem(elem_ij5);
		
	    B_j.elem().elem(i5,j5) = lctmp5;
		
	    elem_ij5++;
	  }
	}
      }
      sw.case_end();

      /*# gamma(  23)   0 -i  0  0            # ( 0110 )  --> 6 */
      /*#              -i  0  0  0 */
      /*#               0  0  0 -i */
      /*#               0  0 -i  0 */
      sw.case_begin( 6 );
      {

	C_t lctmp6 = StackAllocJit< C_t >::alloc();
	R_t lrtmp6 = StackAllocJit< R_t >::alloc();

	for(int i6 = 0; i6 < Nc; ++i6) {
	      
	  int elem_ij6 = (i6+Nc)*(i6+Nc-1)/2;
	      
	  for(int j6 = 0; j6 < Nc; ++j6) {
		
	    int elem_ji6 = (j6+Nc)*(j6+Nc-1)/2 + i6;
	
	    lctmp6 = adj(tri_off_j.elem(0).elem(elem_ji6));
	    lctmp6 += tri_off_j.elem(0).elem(elem_ij6);
	    lctmp6 += adj(tri_off_j.elem(1).elem(elem_ji6));
	    lctmp6 += tri_off_j.elem(1).elem(elem_ij6);
		
	    B_j.elem().elem(i6,j6) = timesMinusI(lctmp6);
		
	    elem_ij6++;
	  }
	}
      }
      sw.case_end();

      /*# gamma(  14)   0  i  0  0            # ( 1001 )  --> 9 */
      /*#               i  0  0  0 */
      /*#               0  0  0 -i */
      /*#               0  0 -i  0 */
      sw.case_begin( 9 );
      {

	C_t lctmp9 = StackAllocJit< C_t >::alloc();
	R_t lrtmp9 = StackAllocJit< R_t >::alloc();

	for(int i9 = 0; i9 < Nc; ++i9) {
	      
	  int elem_ij9 = (i9+Nc)*(i9+Nc-1)/2;
	      
	  for(int j9 = 0; j9 < Nc; ++j9) {
		
	    int elem_ji9 = (j9+Nc)*(j9+Nc-1)/2 + i9;
	
	    lctmp9 = adj(tri_off_j.elem(0).elem(elem_ji9));
	    lctmp9 += tri_off_j.elem(0).elem(elem_ij9);
	    lctmp9 -= adj(tri_off_j.elem(1).elem(elem_ji9));
	    lctmp9 -= tri_off_j.elem(1).elem(elem_ij9);
		
	    B_j.elem().elem(i9,j9) = timesI(lctmp9);
		
	    elem_ij9++;
	  }
	}
      }
      sw.case_end();

      /*# gamma(  24)   0 -1  0  0            # ( 1010 )  --> 10 */
      /*#               1  0  0  0 */
      /*#               0  0  0  1 */
      /*#               0  0 -1  0 */
      sw.case_begin( 10 );
      {

	C_t lctmp10 = StackAllocJit< C_t >::alloc();
	R_t lrtmp10 = StackAllocJit< R_t >::alloc();

	for(int i10 = 0; i10 < Nc; ++i10) {
	      
	  int elem_ij10 = (i10+Nc)*(i10+Nc-1)/2;
	      
	  for(int j10 = 0; j10 < Nc; ++j10) {
		
	    int elem_ji10 = (j10+Nc)*(j10+Nc-1)/2 + i10;
	
	    lctmp10 = adj(tri_off_j.elem(0).elem(elem_ji10));
	    lctmp10 -= tri_off_j.elem(0).elem(elem_ij10);
	    lctmp10 -= adj(tri_off_j.elem(1).elem(elem_ji10));
	    lctmp10 += tri_off_j.elem(1).elem(elem_ij10);
		
	    B_j.elem().elem(i10,j10) = lctmp10;
		
	    elem_ij10++;
	  }
	}
      }
      sw.case_end();

      /*# gamma(  34)   i  0  0  0            # ( 1100 )  --> 12 */
      /*#               0 -i  0  0 */
      /*#               0  0 -i  0 */
      /*#               0  0  0  i */
      /*# From diagonal part */
      sw.case_begin( 12 );
      {

	C_t lctmp12   = StackAllocJit< C_t >::alloc();
	R_t lr_zero12 = StackAllocJit< R_t >::alloc();
	R_t lrtmp12   = StackAllocJit< R_t >::alloc();

	lr_zero12 = 0;
	    
	for(int i12 = 0; i12 < Nc; ++i12) {
      
	  lrtmp12 = tri_dia_j.elem(0).elem(i12);
	  lrtmp12 -= tri_dia_j.elem(0).elem(i12+Nc);
	  lrtmp12 -= tri_dia_j.elem(1).elem(i12);
	  lrtmp12 += tri_dia_j.elem(1).elem(i12+Nc);
	  B_j.elem().elem(i12,i12) = cmplx(lr_zero12,lrtmp12);
	}
	    
	/*# From lower triangular portion */
	int elem_ij12 = 0;
	for(int i12 = 1; i12 < Nc; ++i12) {
	      
	  int elem_ijb12 = (i12+Nc)*(i12+Nc-1)/2 + Nc;
	      
	  for(int j12 = 0; j12 < i12; ++j12) {
	
	    lctmp12 = tri_off_j.elem(0).elem(elem_ij12);
	    lctmp12 -= tri_off_j.elem(0).elem(elem_ijb12);
	    lctmp12 -= tri_off_j.elem(1).elem(elem_ij12);
	    lctmp12 += tri_off_j.elem(1).elem(elem_ijb12);
		
	    B_j.elem().elem(i12,j12) = timesI(lctmp12);
	    B_j.elem().elem(j12,i12) = timesI(adj(lctmp12));
		
	    elem_ij12++;
	    elem_ijb12++;
	  }
	}
      }
      sw.case_end();

      sw.case_default();
      {
      }
      sw.case_end();
    }

    k.finish( function );
  }


template<typename T,typename X,typename Y>
  void function_apply_clov_exec(JitFunction& function, const ExprPlan& plan,
				T& chi,
				const T& psi,
				const X& tri_dia,
				const Y& tri_off,
				const Subset& s)
  {
#ifdef QDP_DEEP_LOG
    function.type_W = typeid(REAL).name();
    function.set_dest_id( chi.getId() );
    function.set_is_lat(true);
#endif
    const LatticeLayout layout = psi.layout();

    int th_count = s.numSiteTable(layout);

    LaunchArgs args( plan );
    args.add_iteration( th_count , chi.layout().sitesOnNode() , s.getIdSiteTable(layout) );
    args.add_plan( plan , s );
    args.launch( function , th_count );

  }


template<typename T,typename X,typename Y>
  void function_apply_clov_build( JitFunction& function, const ExprPlan& plan,
				  const T& chi,
				  const T& psi,
				  const X& tri_dia,
				  const Y& tri_off,
				  const Subset& s)
  {
    KernelBuilder k( "apply_clov" , __PRETTY_FUNCTION__ , plan );   // th_count, latsize, site table

    typedef typename LeafFunctor<T, MGParamLeaf>::Type_t  TJIT;
    TJIT chi_jit( k.view( chi ) );
    TJIT psi_jit( k.view( psi ) );

    typedef typename LeafFunctor<X, MGParamLeaf>::Type_t  XJIT;
    XJIT tri_dia_jit( k.view( tri_dia ) );

    typedef typename LeafFunctor<Y, MGParamLeaf>::Type_t  YJIT;
    YJIT tri_off_jit( k.view( tri_off ) );

    llvm::Value* r_idx = k.site();

    auto chi_j = chi_jit.elem(r_idx);
    auto psi_j = psi_jit.elem(r_idx);
    auto tri_dia_j = tri_dia_jit.elem(r_idx);
    auto tri_off_j = tri_off_jit.elem(r_idx);

    auto chi_s = stack_alloc_jit< decltype(chi_j) >( chi_jit.shape() );

    int n = 2*Nc;

    for(int i = 0; i < n; ++i)
      {
	chi_s.elem((0*n+i)/3).elem((0*n+i)%3) = tri_dia_j.elem(0).elem(i) * psi_j.elem((0*n+i)/3).elem((0*n+i)%3);
	// cchi[0*n+i] = tri[site].diag[0][i] * ppsi[0*n+i];

	chi_s.elem((1*n+i)/3).elem((1*n+i)%3) = tri_dia_j.elem(1).elem(i) * psi_j.elem((1*n+i)/3).elem((1*n+i)%3);
	// cchi[1*n+i] = tri[site].diag[1][i] * ppsi[1*n+i];
      }

    int kij = 0;  
    for(int i = 0; i < n; ++i)
      {
	for(int j = 0; j < i; j++)
	  {
	    chi_s.elem((0*n+i)/3).elem((0*n+i)%3) += tri_off_j.elem(0).elem(kij) * psi_j.elem((0*n+j)/3).elem((0*n+j)%3);
	    // cchi[0*n+i] += tri[site].offd[0][kij] * ppsi[0*n+j];

	    chi_s.elem((0*n+j)/3).elem((0*n+j)%3) += conj(tri_off_j.elem(0).elem(kij)) * psi_j.elem((0*n+i)/3).elem((0*n+i)%3);
	    // cchi[0*n+j] += conj(tri[site].offd[0][kij]) * ppsi[0*n+i];

	    chi_s.elem((1*n+i)/3).elem((1*n+i)%3) += tri_off_j.elem(1).elem(kij) * psi_j.elem((1*n+j)/3).elem((1*n+j)%3);
	    // cchi[1*n+i] += tri[site].offd[1][kij] * ppsi[1*n+j];

	    chi_s.elem((1*n+j)/3).elem((1*n+j)%3) += conj(tri_off_j.elem(1).elem(kij)) * psi_j.elem((1*n+i)/3).elem((1*n+i)%3);
	    // cchi[1*n+j] += conj(tri[site].offd[1][kij]) * ppsi[1*n+i];

	    kij++;
	  }
      }

    chi_j = chi_s;

    k.finish( function );
  }

}


// ---------------------------------------------------------------------
// ---------------------------------------------------------------------

typedef LatticeFermion      T;
typedef LatticeColorMatrix  U;
typedef WordType<T>::Type_t REALT;
typedef OLattice< PScalar< PScalar< RScalar< Word< REALT> > > > > LatticeREAL;
typedef OScalar<  PScalar< PScalar< RScalar< Word< REALT> > > > > RealT;
typedef OLattice<PComp<PTriDia<RScalar <Word<REALT> > > > >  X;
typedef OLattice<PComp<PTriOff<RComplex<Word<REALT> > > > >  Y;

void makeClov( const multi1d<U>& f , const RealT& diag_mass , X& tri_dia , Y& tri_off )
{
  static JitFunctionCache cache;
  U f0 = f[0]; U f1 = f[1]; U f2 = f[2]; U f3 = f[3]; U f4 = f[4]; U f5 = f[5];
  ExprPlan plan = describe_leaves( diag_mass , f0 , f1 , f2 , f3 , f4 , f5 , tri_dia , tri_off );
  JitFunction& function = cache.get( plan , [&]( JitFunction& fn ) {
      function_make_clov_build( fn , plan, diag_mass, f0,f1,f2,f3,f4,f5, tri_dia , tri_off );
    } );
  function_make_clov_exec(function, plan, diag_mass, f0,f1,f2,f3,f4,f5,tri_dia, tri_off);
}

void apply( T& chi , const T& psi , const X& tri_dia , const Y& tri_off , int cb )
{
  static JitFunctionCache cache;
  ExprPlan plan = describe_leaves( chi , psi , tri_dia , tri_off );
  JitFunction& function = cache.get( plan , [&]( JitFunction& fn ) {
      function_apply_clov_build( fn , plan, chi, psi, tri_dia, tri_off, rb[cb] );
    } );
  function_apply_clov_exec(function, plan, chi, psi, tri_dia, tri_off, rb[cb] );
}

void triacntr( U& B , const X& tri_dia , const Y& tri_off , int mat , int cb )
{
  static JitFunctionCache cache;
  ExprPlan plan = describe_leaves( B , tri_dia , tri_off );
  JitFunction& function = cache.get( plan , [&]( JitFunction& fn ) {
      function_triacntr_build<U>( fn , plan, B, tri_dia, tri_off, mat, rb[cb] );
    } );
  function_triacntr_exec(function, plan, B, tri_dia, tri_off, mat, rb[cb] );
}

void ldagdlinv( LatticeREAL& tr_log_diag , X& tri_dia , Y& tri_off , int cb )
{
  tr_log_diag[rb[cb]] = zero;
  static JitFunctionCache cache;
  ExprPlan plan = describe_leaves( tr_log_diag , tri_dia , tri_off );
  JitFunction& function = cache.get( plan , [&]( JitFunction& fn ) {
      function_ldagdlinv_build<U>( fn , plan, tr_log_diag, tri_dia, tri_off, rb[cb] );
    } );
  function_ldagdlinv_exec(function, plan, tr_log_diag, tri_dia, tri_off, rb[cb] );
}


int main(int argc, char **argv)
{
  QDP_initialize(&argc, &argv);

  int lat = 8;
  for (int i = 1; i < argc; i++)
    if (strcmp(argv[i], "-lat") == 0)
      sscanf(argv[++i], "%d", &lat);

  multi1d<int> nrow(Nd);
  nrow = lat;
  Layout::setLattSize(nrow);
  Layout::create();

  multi1d<U> f(6);
  for (int i = 0; i < 6; ++i)
    gaussian(f[i]);

  RealT diag_mass = 4.25;

  X tri_dia;
  Y tri_off;

  makeClov( f , diag_mass , tri_dia , tri_off );

  T psi, chi;
  gaussian(psi);

  char buf[256];
  auto print = [&]( const char* what , double v ) {
    char line[320];
    snprintf( line , sizeof(line) , "CLOVER %-24s %.12e" , what , v );
    QDPIO::cout << line << std::endl;
  };

  for (int cb = 0; cb < 2; ++cb)
    {
      chi = zero;
      apply( chi , psi , tri_dia , tri_off , cb );
      print( cb == 0 ? "apply cb=0" : "apply cb=1" , toDouble(norm2(chi, rb[cb])) );
    }

  int mats[] = { 0 , 3 , 5 , 6 , 9 , 10 , 12 , 15 };
  for (int mat : mats)
    for (int cb = 0; cb < 2; ++cb)
      {
	U B;
	B = zero;
	triacntr( B , tri_dia , tri_off , mat , cb );
	snprintf( buf , sizeof(buf) , "triacntr mat=%d cb=%d" , mat , cb );
	print( buf , toDouble(norm2(B, rb[cb])) );
      }

  LatticeREAL tr_log_diag;
  for (int cb = 0; cb < 2; ++cb)
    {
      ldagdlinv( tr_log_diag , tri_dia , tri_off , cb );
      print( cb == 0 ? "trlog cb=0" : "trlog cb=1" , toDouble(sum(tr_log_diag, rb[cb])) );
    }

  for (int cb = 0; cb < 2; ++cb)
    {
      chi = zero;
      apply( chi , psi , tri_dia , tri_off , cb );
      print( cb == 0 ? "apply inv cb=0" : "apply inv cb=1" , toDouble(norm2(chi, rb[cb])) );
    }

  QDP_finalize();
}
