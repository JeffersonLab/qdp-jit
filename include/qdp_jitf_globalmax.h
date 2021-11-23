#ifndef QDP_JITF_GLOBMAX_H
#define QDP_JITF_GLOBMAX_H


//#include "qmp.h"

namespace QDP {


  void function_global_max_exec( JitFunction& function, 
				 int size, int threads, int blocks, int shared_mem_usage,
				 int in_id , int out_id);


  template<class T1>
  void 
  function_global_max_build( JitFunction& function )
  {
    llvm_start_new_function("global_max",__PRETTY_FUNCTION__ );

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    typedef typename WordType<T1>::Type_t WT;

    ParamRef p_idata      = llvm_add_param< WT* >();  // Input  array
    ParamRef p_odata      = llvm_add_param< WT* >();  // output array

    OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want coal   access later
    OLatticeJIT<typename JITType<T1>::Type_t> odata(  p_odata );   // want scalar access later

    llvm_derefParam( p_lo );
    llvm::Value* r_hi     = llvm_derefParam( p_hi );

    llvm::Value* r_idx = llvm_thread_idx();   

    llvm_derefParam( p_idata );  // Input  array
    llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_get_type<WT>() );


    typedef typename JITType<T1>::Type_t T1JIT;


    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx();

    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;
    reg_idata_elem.setup( idata.elem( JitDeviceLayout::Scalar , r_idx ) );

    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_1st_elem;
    reg_idata_1st_elem.setup( idata.elem( JitDeviceLayout::Scalar , llvm_create_value(0) ) ); 

    IndexDomainVector args;
    args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );

    T1JIT sdata_jit;
    sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );

    

    JitIf ifInitFirst( llvm_ge( r_idx , r_hi ) );
    {
      sdata_jit = reg_idata_1st_elem;
    }
    ifInitFirst.els();
    {
      sdata_jit = reg_idata_elem;
    }
    ifInitFirst.end();

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

	  typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg;   
	  sdata_reg.setup( sdata_jit );

	  sdata_jit = where( sdata_reg > sdata_reg_plus , sdata_reg , sdata_reg_plus );
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




}
#endif
