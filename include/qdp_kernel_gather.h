#ifndef QDP_KERNEL_GATHER_H
#define QDP_KERNEL_GATHER_H

namespace QDP {

  namespace Gather {

    template<class T1, class RHS>
    void build_kernel( JitFunction& function , const ExprPlan& plan , const QDPExpr<RHS,OLattice<T1> >& rhs )
    {
      Shape shape = shape_of(rhs);

      KernelBuilder k( "gather" , __PRETTY_FUNCTION__ , plan , false );

      ParamRef p_soffset = k.add_param<int*>();
      ParamRef p_dest    = k.add_param< typename WordType<T1>::Type_t *>();

      auto rhs_view = k.view( rhs );

      llvm::Value * r_idx      = k.thread();
      llvm::Value * r_idx_site = llvm_array_type_indirection<int>( p_soffset , r_idx );

      NestedIndex idv;
      idv.push_back( llvm_create_value(1) , r_idx );
      typename JITType<T1>::Type_t dest_jit( llvm_derefParam(p_dest) , JitDataOrder::SiteSlow , idv , shape );

      MGViewLeaf vl( r_idx_site , k.param_leaf() );
      dest_jit = forEach( rhs_view , vl , OpCombine() );

      k.finish( function );
    }


    inline void exec( JitFunction& function , const ExprPlan& plan , int send_buf_id , const Map& map , const Subset& subset , int layout_ref , bool set_l0_event )
    {
      if (subset.numSiteTable(Layout::from_index(layout_ref)) < 1)
	return;

      const int th_count = map.soffset(subset,layout_ref).size();

      LaunchArgs args( plan );
      args.add_iteration( th_count , Layout::from_index( layout_ref ).sitesOnNode() );
      args.add_id( map.getSoffsetsId(subset,layout_ref) );
      args.add_id( send_buf_id );
      args.add_plan( plan , subset );
      args.launch( function , th_count , set_l0_event );
    }


    template<class T1, class RHS>
    void run( int send_buf_id , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs , const Subset& subset , int layout_ref , bool set_l0_event )
    {
      static JitFunctionCache cache;

      ExprPlan plan = describe_rhs( rhs , layout_ref );

      JitFunction& function = cache.get( plan , [&]( JitFunction& f ) {
	  build_kernel( f , plan , rhs );
	} );

      exec( function , plan , send_buf_id , map , subset , layout_ref , set_l0_event );
    }

  }

}
#endif
