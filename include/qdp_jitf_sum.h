#ifndef QDP_JITF_SUM_H
#define QDP_JITF_SUM_H

//#include "qmp.h"

namespace QDP {

  void function_sum_convert_ind_exec( JitFunction& function, 
				      int size, int threads, int blocks, int shared_mem_usage,
				      int in_id, int out_id, int siteTableId );

  void function_sum_convert_exec( JitFunction& function, 
				  int size, int threads, int blocks, int shared_mem_usage,
				  int in_id, int out_id);

  void function_sum_exec( JitFunction& function, 
			  int size, int threads, int blocks, int shared_mem_usage,
			  int in_id, int out_id);


  void function_bool_reduction_exec( JitFunction& function, 
				     int size, int threads, int blocks, int shared_mem_usage,
				     int in_id, int out_id);


  
  template<class T1 , class RHS >
  void function_sum_convert_ind_expr_exec( JitFunction& function, 
					   int size, int threads, int blocks, int shared_mem_usage,
					   const QDPExpr<RHS,OLattice<T1> >& rhs, int out_id, int siteTableId )
  {
    // Make sure 'threads' is a power of two (the jit kernel make this assumption)
    if ( (threads & (threads - 1)) != 0 )
      {
	QDPIO::cerr << "internal error: function_sum_convert_ind_exec not power of 2\n";
	QDP_abort(1);
      }

#ifdef QDP_DEEP_LOG
    function.start = 0;
    function.count = blocks;
    function.set_dest_id( out_id );
#endif


    
    int lo = 0;
    int hi = size;
    
    JitParam jit_lo( QDP_get_global_cache().addJitParamInt( lo ) );
    JitParam jit_hi( QDP_get_global_cache().addJitParamInt( hi ) );

    AddressLeaf addr_leaf(all);
    forEach(rhs, addr_leaf, NullCombine());

    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_lo.get_id() );
    ids.push_back( jit_hi.get_id() );
    ids.push_back( siteTableId );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
      {
	ids.push_back( addr_leaf.ids[i] );
      }
    ids.push_back( out_id );

    jit_launch_explicit_geom( function , ids , getGeom( hi-lo , threads ) , shared_mem_usage );
  }
  


  // T1 input
  // T2 output
  template< class T1 , class T2 , JitDeviceLayout input_layout >
  void
  function_sum_convert_ind_build(JitFunction& function)
  {
#ifdef QDP_DEEP_LOG
    function.size_T = sizeof(T2);
    function.type_W = typeid(typename WordType<T2>::Type_t).name();
#endif
    
    llvm_start_new_function("sum_convert_ind",__PRETTY_FUNCTION__ );

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    typedef typename WordType<T1>::Type_t T1WT;
    typedef typename WordType<T2>::Type_t T2WT;

    ParamRef p_site_perm  = llvm_add_param< int* >(); // Siteperm  array
    ParamRef p_idata      = llvm_add_param< T1WT* >();  // Input  array
    ParamRef p_odata      = llvm_add_param< T2WT* >();  // output array

    OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want coal   access later
    OLatticeJIT<typename JITType<T2>::Type_t> odata(  p_odata );   // want scalar access later

    llvm_derefParam( p_lo ); // r_lo
    llvm::Value* r_hi     = llvm_derefParam( p_hi );

    //llvm_derefParam( p_idata );  // Input  array
    //llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_get_type<T2WT>() );

    typedef typename JITType<T2>::Type_t T2JIT;

    llvm::Value* r_idx = llvm_thread_idx();

    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // this is a power of 2

    IndexDomainVector args;
    args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later
    T2JIT sdata_jit;
    sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );
    zero_rep( sdata_jit );

    llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

    llvm::Value* r_idx_perm = llvm_array_type_indirection( p_site_perm , r_idx );

    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;   
    reg_idata_elem.setup( idata.elem( input_layout , r_idx_perm ) );

    sdata_jit = reg_idata_elem; // This should do the precision conversion (SP->DP)

    llvm_bar_sync();

    
    llvm::Value* r_pow_shr1 = llvm_shr( r_ntidx , llvm_create_value(1) );

    //
    // Reduction loop
    //
    JitForLoopPower loop( r_pow_shr1 );
    {
      JitIf ifInRange( llvm_lt( r_tidx , loop.index() ) );
      {
	llvm::Value * v = llvm_add( loop.index() , r_tidx );

	JitIf ifInRange2( llvm_lt( v , r_ntidx ) );
	{
	  IndexDomainVector args_new;
	  args_new.push_back( make_pair( Layout::sitesOnNode() , 
					 llvm_add( r_tidx , loop.index() ) ) );  // sitesOnNode irrelevant since Scalar access later

	  typename JITType<T2>::Type_t sdata_jit_plus;
	  sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

	  typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg_plus;
	  sdata_reg_plus.setup( sdata_jit_plus );

	  sdata_jit += sdata_reg_plus;
	}
	ifInRange2.end();
      }
      ifInRange.end();

      llvm_bar_sync();
    }
    loop.end();
    //
    // -------------------
    //

    JitIf ifStore( llvm_eq( r_tidx , llvm_create_value(0) ) );
    {
      typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg;   
      sdata_reg.setup( sdata_jit );
      odata.elem( JitDeviceLayout::Scalar , r_block_idx ) = sdata_reg;
    }
    ifStore.end();

    jit_get_function(function);
  }




  // T1 input
  template< JitDeviceLayout input_layout, class T1 , class RHS >
  void
  function_sum_convert_ind_expr_build( JitFunction& function , const QDPExpr<RHS,OLattice<T1> >& rhs )
  {
    typedef typename UnaryReturn< OLattice<T1> , FnSum>::Type_t::SubType_t T2;

#ifdef QDP_DEEP_LOG
    function.size_T = sizeof(T2);
    function.type_W = typeid(typename WordType<T2>::Type_t).name();
#endif
    
    llvm_start_new_function("sum_convert_ind_expr",__PRETTY_FUNCTION__ );

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    //typedef typename WordType<T1>::Type_t T1WT;
    typedef typename WordType<T2>::Type_t T2WT;

    ParamRef p_site_perm  = llvm_add_param< int* >(); // Siteperm  array
    //ParamRef p_idata      = llvm_add_param< T1WT* >();  // Input  array

    ParamLeaf param_leaf;
    typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
    View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

    ParamRef p_odata      = llvm_add_param< T2WT* >();  // output array

    
    //OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want coal   access later
    OLatticeJIT<typename JITType<T2>::Type_t> odata(  p_odata );   // want scalar access later

    llvm_derefParam( p_lo ); // r_lo
    llvm::Value* r_hi     = llvm_derefParam( p_hi );

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_get_type<T2WT>() );

    typedef typename JITType<T2>::Type_t T2JIT;

    llvm::Value* r_idx = llvm_thread_idx();

    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // this is a power of 2

    IndexDomainVector args;
    args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later
    T2JIT sdata_jit;
    sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );
    zero_rep( sdata_jit );

    llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

    llvm::Value* r_idx_perm = llvm_array_type_indirection( p_site_perm , r_idx );

    //typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;   
    //reg_idata_elem.setup( idata.elem( input_layout , r_idx_perm ) );

    //sdata_jit = reg_idata_elem; // This should do the precision conversion (SP->DP)

    OpAssign()( sdata_jit , 
		forEach(rhs_view, ViewLeaf( input_layout , r_idx_perm ), OpCombine()));

    
    llvm_bar_sync();

    
    llvm::Value* r_pow_shr1 = llvm_shr( r_ntidx , llvm_create_value(1) );

    //
    // Reduction loop
    //
    JitForLoopPower loop( r_pow_shr1 );
    {
      JitIf ifInRange( llvm_lt( r_tidx , loop.index() ) );
      {
	llvm::Value * v = llvm_add( loop.index() , r_tidx );

	JitIf ifInRange2( llvm_lt( v , r_ntidx ) );
	{
	  IndexDomainVector args_new;
	  args_new.push_back( make_pair( Layout::sitesOnNode() , 
					 llvm_add( r_tidx , loop.index() ) ) );  // sitesOnNode irrelevant since Scalar access later

	  typename JITType<T2>::Type_t sdata_jit_plus;
	  sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

	  typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg_plus;
	  sdata_reg_plus.setup( sdata_jit_plus );

	  sdata_jit += sdata_reg_plus;
	}
	ifInRange2.end();
      }
      ifInRange.end();

      llvm_bar_sync();
    }
    loop.end();
    //
    // -------------------
    //

    JitIf ifStore( llvm_eq( r_tidx , llvm_create_value(0) ) );
    {
      typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg;   
      sdata_reg.setup( sdata_jit );
      odata.elem( JitDeviceLayout::Scalar , r_block_idx ) = sdata_reg;
    }
    ifStore.end();

    jit_get_function(function);
  }




  


  // T1 input
  // T2 output
  template< class T1 , class T2 , JitDeviceLayout input_layout >
  void
  function_sum_convert_build(JitFunction& function)
  {
#ifdef QDP_DEEP_LOG
    function.size_T = sizeof(T2);
    function.type_W = typeid(typename WordType<T2>::Type_t).name();
#endif
    
    llvm_start_new_function("sum_convert",__PRETTY_FUNCTION__ );

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    typedef typename WordType<T1>::Type_t T1WT;
    typedef typename WordType<T2>::Type_t T2WT;

    ParamRef p_idata      = llvm_add_param< T1WT* >();  // Input  array
    ParamRef p_odata      = llvm_add_param< T2WT* >();  // output array

    OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want coal   access later
    OLatticeJIT<typename JITType<T2>::Type_t> odata(  p_odata );   // want scalar access later

    llvm_derefParam( p_lo ); // r_lo
    llvm::Value* r_hi     = llvm_derefParam( p_hi );

    llvm_derefParam( p_idata );  // Input  array
    llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_get_type<T2WT>() );

    typedef typename JITType<T2>::Type_t T2JIT;

    llvm::Value* r_idx = llvm_thread_idx();

    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // this is a power of 2

    IndexDomainVector args;
    args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later
    T2JIT sdata_jit;
    sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );
    zero_rep( sdata_jit );

    llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;
    //reg_idata_elem.setup( idata.elem( input_layout , r_idx_perm ) );
    reg_idata_elem.setup( idata.elem( input_layout , r_idx ) );

    sdata_jit = reg_idata_elem; // This should do the precision conversion (SP->DP)

    llvm_bar_sync();

    
    llvm::Value* r_pow_shr1 = llvm_shr( r_ntidx , llvm_create_value(1) );

    //
    // Reduction loop
    //
    JitForLoopPower loop( r_pow_shr1 );
    {
      JitIf ifInRange( llvm_lt( r_tidx , loop.index() ) );
      {
	llvm::Value * v = llvm_add( loop.index() , r_tidx );

	JitIf ifInRange2( llvm_lt( v , r_ntidx ) );
	{
	  IndexDomainVector args_new;
	  args_new.push_back( make_pair( Layout::sitesOnNode() , 
					 llvm_add( r_tidx , loop.index() ) ) );  // sitesOnNode irrelevant since Scalar access later

	  typename JITType<T2>::Type_t sdata_jit_plus;
	  sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

	  typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg_plus;
	  sdata_reg_plus.setup( sdata_jit_plus );

	  sdata_jit += sdata_reg_plus;
	}
	ifInRange2.end();
      }
      ifInRange.end();

      llvm_bar_sync();
    }
    loop.end();
    //
    // -------------------
    //

    JitIf ifStore( llvm_eq( r_tidx , llvm_create_value(0) ) );
    {
      typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg;   
      sdata_reg.setup( sdata_jit );
      odata.elem( JitDeviceLayout::Scalar , r_block_idx ) = sdata_reg;
    }
    ifStore.end();

    jit_get_function(function);
  }




  template<class T1>
  void
  function_sum_build(JitFunction& function)
  {
#ifdef QDP_DEEP_LOG
    function.size_T = sizeof(T1);
    function.type_W = typeid(typename WordType<T1>::Type_t).name();
#endif
    
    llvm_start_new_function("sum",__PRETTY_FUNCTION__ );

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    typedef typename WordType<T1>::Type_t WT;

    ParamRef p_idata      = llvm_add_param< WT* >();  // Input  array
    ParamRef p_odata      = llvm_add_param< WT* >();  // output array

    OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want coal   access later
    OLatticeJIT<typename JITType<T1>::Type_t> odata(  p_odata );   // want scalar access later

    llvm_derefParam( p_lo );
    llvm::Value* r_hi     = llvm_derefParam( p_hi );

    llvm_derefParam( p_idata );  // Input  array
    llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_get_type<WT>() );


    typedef typename JITType<T1>::Type_t T1JIT;

    llvm::Value* r_idx = llvm_thread_idx();   

    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // needed later

    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;   
    reg_idata_elem.setup( idata.elem( JitDeviceLayout::Scalar , r_idx ) ); 

    IndexDomainVector args;
    args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later

    T1JIT sdata_jit;
    sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );

    zero_rep( sdata_jit );

    llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

    sdata_jit = reg_idata_elem; // 

    llvm_bar_sync();

    llvm::Value* r_pow_shr1 = llvm_shr( r_ntidx , llvm_create_value(1) );

    //
    // Reduction loop
    //
    JitForLoopPower loop( r_pow_shr1 );
    {
      JitIf ifInRange( llvm_lt( r_tidx , loop.index() ) );
      {
	llvm::Value * v = llvm_add( loop.index() , r_tidx );

	JitIf ifInRange2( llvm_lt( v , r_ntidx ) );
	{
	  IndexDomainVector args_new;
	  args_new.push_back( make_pair( Layout::sitesOnNode() , 
					 llvm_add( r_tidx , loop.index() ) ) );  // sitesOnNode irrelevant since Scalar access later

	  typename JITType<T1>::Type_t sdata_jit_plus;
	  sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

	  typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg_plus;
	  sdata_reg_plus.setup( sdata_jit_plus );

	  sdata_jit += sdata_reg_plus;
	}
	ifInRange2.end();
      }
      ifInRange.end();

      llvm_bar_sync();
    }
    loop.end();
    //
    // -------------------
    //

    JitIf ifStore( llvm_eq( r_tidx , llvm_create_value(0) ) );
    {
      typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg;   
      sdata_reg.setup( sdata_jit );
      odata.elem( JitDeviceLayout::Scalar , r_block_idx ) = sdata_reg;
    }
    ifStore.end();

    jit_get_function(function);
  }




  // T1 input
  // T2 output
  template< class T1 , class T2 , JitDeviceLayout input_layout , class ConvertOp, class ReductionOp >
  void
  function_bool_reduction_convert_build(JitFunction& function)
  {
#ifdef QDP_DEEP_LOG
    function.size_T = sizeof(T2);
    function.type_W = typeid(typename WordType<T2>::Type_t).name();
#endif
    
    llvm_start_new_function("bool_reduction_convert",__PRETTY_FUNCTION__ );

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    typedef typename WordType<T1>::Type_t T1WT;
    typedef typename WordType<T2>::Type_t T2WT;

    ParamRef p_idata      = llvm_add_param< T1WT* >();  // Input  array
    ParamRef p_odata      = llvm_add_param< T2WT* >();  // output array

    OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want coal   access later
    OLatticeJIT<typename JITType<T2>::Type_t> odata(  p_odata );   // want scalar access later

    llvm_derefParam( p_lo ); // r_lo
    llvm::Value* r_hi     = llvm_derefParam( p_hi );

    llvm_derefParam( p_idata );  // Input  array
    llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_get_type<T2WT>() );

    typedef typename JITType<T2>::Type_t T2JIT;

    llvm::Value* r_idx = llvm_thread_idx();

    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // this is a power of 2

    IndexDomainVector args;
    args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later
    T2JIT sdata_jit;
    sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );

    ReductionOp::initNeutral( sdata_jit );
    
    llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;
    //reg_idata_elem.setup( idata.elem( input_layout , r_idx_perm ) );
    reg_idata_elem.setup( idata.elem( input_layout , r_idx ) );

    ConvertOp::apply( sdata_jit , reg_idata_elem );

    llvm_bar_sync();


    llvm::Value* r_pow_shr1 = llvm_shr( r_ntidx , llvm_create_value(1) );

    //
    // Reduction loop
    //
    JitForLoopPower loop( r_pow_shr1 );
    {
      JitIf ifInRange( llvm_lt( r_tidx , loop.index() ) );
      {
	llvm::Value * v = llvm_add( loop.index() , r_tidx );

	JitIf ifInRange2( llvm_lt( v , r_ntidx ) );
	{
	  IndexDomainVector args_new;
	  args_new.push_back( make_pair( Layout::sitesOnNode() , 
					 llvm_add( r_tidx , loop.index() ) ) );  // sitesOnNode irrelevant since Scalar access later

	  typename JITType<T2>::Type_t sdata_jit_plus;
	  sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

	  typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg_plus;
	  sdata_reg_plus.setup( sdata_jit_plus );

	  ReductionOp::apply( sdata_jit , sdata_reg_plus );
	}
	ifInRange2.end();
      }
      ifInRange.end();

      llvm_bar_sync();
    }
    loop.end();
    //
    // -------------------
    //

    JitIf ifStore( llvm_eq( r_tidx , llvm_create_value(0) ) );
    {
      typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg;   
      sdata_reg.setup( sdata_jit );
      odata.elem( JitDeviceLayout::Scalar , r_block_idx ) = sdata_reg;
    }
    ifStore.end();

    jit_get_function(function);
  }

  


  template<class T1, class ReductionOp>
  void 
  function_bool_reduction_build(JitFunction& function)
  {
#ifdef QDP_DEEP_LOG
    function.size_T = sizeof(T1);
    function.type_W = typeid(typename WordType<T1>::Type_t).name();
#endif

    llvm_start_new_function("bool_reduction",__PRETTY_FUNCTION__ );

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    typedef typename WordType<T1>::Type_t WT;

    ParamRef p_idata      = llvm_add_param< WT* >();  // Input  array
    ParamRef p_odata      = llvm_add_param< WT* >();  // output array

    OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want coal   access later
    OLatticeJIT<typename JITType<T1>::Type_t> odata(  p_odata );   // want scalar access later

    llvm_derefParam( p_lo );
    llvm::Value* r_hi     = llvm_derefParam( p_hi );

    llvm_derefParam( p_idata );  // Input  array
    llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_get_type<WT>() );


    typedef typename JITType<T1>::Type_t T1JIT;

    llvm::Value* r_idx = llvm_thread_idx();   

    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // needed later

    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;   
    reg_idata_elem.setup( idata.elem( JitDeviceLayout::Scalar , r_idx ) ); 

    IndexDomainVector args;
    args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later

    T1JIT sdata_jit;
    sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );
    
    ReductionOp::initNeutral( sdata_jit );
    
    llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

    sdata_jit = reg_idata_elem; // This is just copying booleans

    llvm_bar_sync();

    llvm::Value* r_pow_shr1 = llvm_shr( r_ntidx , llvm_create_value(1) );

    //
    // Reduction loop
    //
    JitForLoopPower loop( r_pow_shr1 );
    {
      JitIf ifInRange( llvm_lt( r_tidx , loop.index() ) );
      {
	llvm::Value * v = llvm_add( loop.index() , r_tidx );

	JitIf ifInRange2( llvm_lt( v , r_ntidx ) );
	{
	  IndexDomainVector args_new;
	  args_new.push_back( make_pair( Layout::sitesOnNode() , 
					 llvm_add( r_tidx , loop.index() ) ) );  // sitesOnNode irrelevant since Scalar access later

	  typename JITType<T1>::Type_t sdata_jit_plus;
	  sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

	  typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg_plus;
	  sdata_reg_plus.setup( sdata_jit_plus );

	  ReductionOp::apply( sdata_jit , sdata_reg_plus );
	}
	ifInRange2.end();
      }
      ifInRange.end();

      llvm_bar_sync();
    }
    loop.end();
    //
    // -------------------
    //

    JitIf ifStore( llvm_eq( r_tidx , llvm_create_value(0) ) );
    {
      typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg;   
      sdata_reg.setup( sdata_jit );
      odata.elem( JitDeviceLayout::Scalar , r_block_idx ) = sdata_reg;
    }
    ifStore.end();

    jit_get_function(function);
  }


  

} // namespace

#endif
