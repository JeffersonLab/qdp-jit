#ifndef QDP_KERNEL_EVAL_H
#define QDP_KERNEL_EVAL_H

#include<unistd.h>

namespace QDP {

  namespace Eval {

    template<class T, class C1, class Op, class RHS>
    void build_kernel( JitFunction& function , const ExprPlan& plan , OLattice<T>& dest , const Op& op , const QDPExpr<RHS,C1>& rhs ,
		       const char* ftype , bool pointer_attributes )
    {
      std::ostringstream expr;
      expr << std::string(__PRETTY_FUNCTION__) << "_key=" << plan.key();

      KernelBuilder k( ftype , expr.str().c_str() , plan );

      auto dest_jit = k.view( dest );
      auto op_jit = AddOpParam<Op,MGParamLeaf>::apply( op , k.param_leaf() );
      auto rhs_view = k.view( rhs );

      MGViewLeaf vl( k.site() , k.param_leaf() );

      op_jit( forEach( dest_jit , vl , OpCombine() ) ,
	      forEach( rhs_view , vl , OpCombine() ) );

      if ( pointer_attributes )
	llvm_set_pointer_param_attributes( k.param_leaf().dest_base() , !plan.dest_overlap );

      k.finish( function );
    }


    template<class T, class T1, class RHS>
    bool self_assign( const OLattice<T>& dest , const QDPExpr<RHS,OLattice<T1> >& rhs )
    {
      SelfAssignTag tag( dest.getId() );
      return forEach( rhs , tag , BitOrCombine() ) != 0;
    }


    template<class T, class C1, class Op, class RHS>
    void exec( JitFunction& function , const ExprPlan& plan , OLattice<T>& dest , const Op& op , const QDPExpr<RHS,C1>& rhs , const Subset& s )
    {
      const LatticeLayout layout = dest.layout();

#ifdef QDP_DEEP_LOG
      function.type_W = typeid(typename WordType<T>::Type_t).name();
      function.set_is_lat(true);
#endif

      if constexpr ( HasShift<RHS>::value )
	{
	  if ( self_assign( dest , rhs ) )
	    {
	      QDPIO::cout << "The destination appears inside a shift on the right-hand side, e.g. a = shift(a, ...). This is a read/write race and not allowed." << std::endl;
	      QDPIO::cout << "The expression's type information:" << std::endl;

	      std::ostringstream expr;
	      printExprTree( expr , dest, op, rhs );
	      QDPIO::cout << expr.str() << std::endl;

	      QDPIO::cout << __PRETTY_FUNCTION__ << std::endl;

	      sleep(2);
	      QDP_abort(1);
	    }
	}

      if (s.numSiteTable(layout) < 1)
	return;

      int offnode_maps = 0;

      if constexpr ( HasShift<RHS>::value )
	{
	  check_shift_layout( get_layout_from_rhs( rhs ).index() , layout.index() );

	  ShiftPhase1 phase1( s , layout.index() );
	  offnode_maps = forEach(rhs, phase1 , BitOrCombine());
	}

      function.set_dest_id( dest.getId() );

      MasterMap::Tables tables = MasterMap::Instance().get(s,layout,offnode_maps);

      LaunchArgs args( plan );
      args.add_iteration( tables.num_inner , layout.sitesOnNode() , tables.id_inner );
      args.add_plan( plan , s );

      args.launch( function , tables.num_inner );

      if constexpr ( HasShift<RHS>::value )
	{
	  ShiftPhase2 phase2;
	  forEach(rhs, phase2 , NullCombine());

	  if ( tables.num_face > 0 )
	    {
	      args.reset_iteration( tables.num_face , tables.id_face );
	      args.launch( function , tables.num_face );
	    }
	}

#ifdef QDP_DEEP_LOG
      QDP_get_global_logger().log(dest,s);
#endif
    }


    template<class T, class C1, class Op, class RHS>
    void run( OLattice<T>& dest , const Op& op , const QDPExpr<RHS,C1>& rhs , const Subset& s ,
	      const char* ftype = "eval" , bool pointer_attributes = true )
    {
      static JitFunctionCache cache;

      ExprPlan plan = describe_expr( dest , op , rhs );

      JitFunction& function = cache.get( plan , [&]( JitFunction& f ) {
	  build_kernel( f , plan , dest , op , rhs , ftype , pointer_attributes );
	} );

      exec( function , plan , dest , op , rhs , s );
    }

  }

}
#endif
