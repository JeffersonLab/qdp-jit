#ifndef QDP_KERNEL_SCALAR_EVAL_H
#define QDP_KERNEL_SCALAR_EVAL_H


namespace QDP {

  namespace ScalarEval {

    template<class T, class T1, class Op, class RHS>
    void build_kernel( JitFunction& function , const ExprPlan& plan , OScalar<T>& dest , const Op& op , const QDPExpr<RHS,OScalar<T1> >& rhs )
    {
      llvm_start_new_function( "eval_sca_sca" , __PRETTY_FUNCTION__ , qdp_jit_config_get_hostjit() );

      MGParamLeaf param_leaf( plan , -1 );

      const PlanEntry& e_dest = param_leaf.next( PlanEntry::Kind::Scalar );
      typedef typename JITType< OScalar<T> >::Type_t  FuncRet_t;
      FuncRet_t dest_jit( llvm_add_param< typename WordType<T>::Type_t * >() , e_dest.shape );

      auto op_jit = AddOpParam<Op,MGParamLeaf>::apply(op,param_leaf);

      auto rhs_view = forEach( rhs , param_leaf , TreeCombine() );
      assert( param_leaf.all_entries_used() );

      llvm_create_function();

      MGViewLeaf vl( llvm_create_value(0) , param_leaf );
      op_jit( dest_jit.elem() , forEach( rhs_view , vl , OpCombine() ) );

      jit_get_function( function );

      assert( function.get_num_params() < 0 ||
	      function.get_num_params() == plan.num_leaf_params() );
    }


    inline void exec( JitFunction& function , const ExprPlan& plan )
    {
      // dest (ring buffer, first entry), op params, rhs
      LaunchArgs args( plan );
      args.add_plan( plan , all );

      const int dest_id = args.ids().front();

      args.launch( function , 1 );

      QDP_get_global_cache().assureOnHost( dest_id );
    }


    template<class T, class T1, class Op, class RHS>
    void run( OScalar<T>& dest , const Op& op , const QDPExpr<RHS,OScalar<T1> >& rhs )
    {
      static JitFunctionCache cache;

      ExprPlan plan = describe_sca_sca( dest , op , rhs );

      JitFunction& function = cache.get( plan , [&]( JitFunction& f ) {
	  build_kernel( f , plan , dest , op , rhs );
	} );

      exec( function , plan );
    }

  }

}
#endif
