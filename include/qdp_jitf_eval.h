#ifndef QDP_JITFUNC_EVAL_H
#define QDP_JITFUNC_EVAL_H

#include<type_traits>
#include<unistd.h>

namespace QDP {

  template<class T, class T1, class Op, class RHS>
#if defined (QDP_PROP_OPT)
  typename std::enable_if_t< ! HasProp<RHS>::value >
#else
  void
#endif
  function_build(
#if defined (QDP_CODEGEN_VECTOR)
		 JitFunction& function,
		 JitFunction& function_scalar,
#else
		 JitFunction& function,
#endif
		 const DynKey& key, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
  {
    std::ostringstream expr;
#if 0
    printExprTreeSubset( expr , dest, op, rhs , s , key );
#else
    expr << std::string(__PRETTY_FUNCTION__) << "_key=" << key;
#endif
  
    //
    // SIMD version for L0
    // Scalar version for CUDA/ROCM
    //
    if (1)
    {
      llvm_start_new_function("eval",expr.str().c_str() );

      WorkgroupGuard workgroupGuard;
      ParamRef p_site_table = llvm_add_param<int*>();

      ParamLeaf param_leaf;

      typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
      FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
	  
      auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

      typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
      View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

      llvm::Value* r_idx_thread = llvm_thread_idx();
       
      workgroupGuard.check(r_idx_thread);

      llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );

      ViewLeaf vl( JitDeviceLayout::Coalesced , r_idx );
#if defined (QDP_CODEGEN_VECTOR)
      // The data layout VNode guarantees all receive sites are local
      vl.handle_multi_index = false;
#endif

      op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ), 
	      forEach(rhs_view , vl , OpCombine()));

      jit_get_function( function );
    }
#if defined (QDP_CODEGEN_VECTOR)
    //
    // Scalar version for L0
    {
      llvm_start_new_function("eval_scalar",expr.str().c_str() );

      WorkgroupGuard workgroupGuard;
      ParamRef p_site_table = llvm_add_param<int*>();

      ParamLeafScalar param_leaf;
      
      typedef typename LeafFunctor<OLattice<T>, ParamLeafScalar>::Type_t  FuncRet_t;
      FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
	  
      auto op_jit = AddOpParam<Op,ParamLeafScalar>::apply(op,param_leaf);

      typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeafScalar, TreeCombine>::Type_t View_t;
      View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

      llvm::Value* r_idx_thread = llvm_thread_idx();
       
      workgroupGuard.check(r_idx_thread);

      llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );

      // op_jit( dest_jit.elemScalar( JitDeviceLayout::Coalesced , r_idx ),
      // 	      forEach(rhs_view, ViewLeafScalar( JitDeviceLayout::Coalesced , r_idx ), OpCombine()));
      op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ),
	      forEach(rhs_view , ViewLeaf( JitDeviceLayout::Coalesced , r_idx ) , OpCombine()));

      jit_get_function( function_scalar );
    }
#endif
  }


  namespace {
        template <class T>
    void print_object(const T& a)
    {
      QDPIO::cout << __PRETTY_FUNCTION__ << std::endl;
    }

  }

#if defined (QDP_PROP_OPT)
  template<class T, class T1, class Op, class RHS>
  typename std::enable_if_t< HasProp<RHS>::value >
  function_build(
#if defined (QDP_CODEGEN_VECTOR)
		 JitFunction& function,
		 JitFunction& function_scalar,
#else
		 JitFunction& function,
#endif
		 const DynKey& key, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
  {
    std::ostringstream expr;

    expr << std::string(__PRETTY_FUNCTION__) << "_key=" << key;

    //
    // SIMD version in AVX
    // Scalar version for CUDA/ROCM
    //
    {
      llvm_start_new_function("evalprop",expr.str().c_str() );

      WorkgroupGuard workgroupGuard;

      ParamRef p_site_table = llvm_add_param<int*>();
      ParamLeaf param_leaf;

      typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
      FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

      typedef typename AddOpParam<Op,ParamLeaf>::Type_t OpJit_t;
      OpJit_t op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

      typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
      View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

      llvm::Value* r_idx_thread = llvm_thread_idx();
       
      workgroupGuard.check(r_idx_thread);

      llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );

      std::vector< JitForLoop > loops;
      CreateLoops<T,OpJit_t>::apply( loops , op_jit );

      ViewSpinLeaf viewSpin( JitDeviceLayout::Coalesced , r_idx );
#if defined (QDP_CODEGEN_VECTOR)
      // The data layout VNode guarantees all receive sites are local
      viewSpin.handle_multi_index = false;
#endif

      for( int i = 0 ; i < loops.size() ; ++i )
	viewSpin.indices.push_back( loops.at(i).index() );

      op_jit( viewSpinJit( dest_jit , viewSpin ) , forEach( rhs_view , viewSpin , OpCombine() ) );
 
      for( int i = loops.size() - 1 ; 0 <= i ; --i )
	{
	  loops[i].end();
	}

      jit_get_function( function );
    }

#if defined (QDP_CODEGEN_VECTOR)
    //
    // Scalar version for AVX
    {
      llvm_start_new_function("evalprop_scalar",expr.str().c_str() );

      WorkgroupGuard workgroupGuard;

      ParamRef p_site_table = llvm_add_param<int*>();
      ParamLeafScalar param_leaf;

      typedef typename LeafFunctor<OLattice<T>, ParamLeafScalar>::Type_t  FuncRet_t;
      FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

      typedef typename AddOpParam<Op,ParamLeafScalar>::Type_t OpJit_t;
      OpJit_t op_jit = AddOpParam<Op,ParamLeafScalar>::apply(op,param_leaf);

      typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeafScalar, TreeCombine>::Type_t View_t;
      View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

      llvm::Value* r_idx_thread = llvm_thread_idx();
       
      workgroupGuard.check(r_idx_thread);

      llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );

      std::vector< JitForLoop > loops;
      CreateLoops<T,OpJit_t>::apply( loops , op_jit );

      ViewSpinLeaf viewSpin( JitDeviceLayout::Coalesced , r_idx );

      for( int i = 0 ; i < loops.size() ; ++i )
	viewSpin.indices.push_back( loops.at(i).index() );

      op_jit( viewSpinJit( dest_jit , viewSpin ) , forEach( rhs_view , viewSpin , OpCombine() ) );
      
      for( int i = loops.size() - 1 ; 0 <= i ; --i )
	{
	  loops[i].end();
	}

      jit_get_function( function_scalar );
    }
#endif
  }
#endif  


  template<class T, class T1, class Op, class RHS>
#if defined (QDP_PROP_OPT)
  typename std::enable_if_t< ! HasProp<RHS>::value >
#else
  void
#endif  
  function_check_selfassign(bool& result, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs)
  {
    result = false;
  }


#if defined (QDP_PROP_OPT)
  template<class T, class T1, class Op, class RHS>
  typename std::enable_if_t< HasProp<RHS>::value >
  function_check_selfassign(bool& result, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs)
  {
    result = forEach(rhs, SelfAssignTag(dest.getId()) , BitOrCombine()) > 0;
  }
#endif


  

  template<class T, class T1, class Op, class RHS>
  void 
  function_exec(
#if defined (QDP_CODEGEN_VECTOR)
		JitFunction& function,
		JitFunction& function_scalar,
#else
		JitFunction& function,
#endif
		OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
  {
#ifdef QDP_DEEP_LOG
    function.type_W = typeid(typename WordType<T>::Type_t).name();
    function.set_dest_id( dest.getId() );
    function.set_is_lat(true);
#if defined (QDP_CODEGEN_VECTOR)
    function_scalar.type_W = typeid(typename WordType<T>::Type_t).name();
    function_scalar.set_dest_id( dest.getId() );
    function_scalar.set_is_lat(true);
#endif
#endif
    
    bool has_self_assign;
    function_check_selfassign(has_self_assign , dest, op , rhs );
    if (has_self_assign)
      {
	QDPIO::cout << "Self assignment in the form a=f(a) detected, where 'a' is a propagator. This is not allowed." << std::endl;
	QDPIO::cout << "The expression's type information:" << std::endl;

	std::ostringstream expr;
	printExprTree( expr , dest, op, rhs );
	QDPIO::cout << expr.str() << std::endl;

	QDPIO::cout << __PRETTY_FUNCTION__ << std::endl;

	sleep(2);
	QDP_abort(1);
      }

    if (s.numSiteTable() < 1)
      return;

    ShiftPhase1 phase1(s);
    int offnode_maps = forEach(rhs, phase1 , BitOrCombine());

    AddressLeaf addr_leaf(s);
    forEach(dest, addr_leaf, NullCombine());
    AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
    forEach(rhs, addr_leaf, NullCombine());


    // For tuning
    function.set_dest_id( dest.getId() );
    function.set_enable_tuning();

    //
    // 1st part to inner call: scalar for CUDA/ROCM, vector in case of AVX/VNODE
    if (1)
    {
#if defined (QDP_CODEGEN_VECTOR)
      int th_count = MasterMap::Instance().getCountVNodeInnerSIMD(s,offnode_maps);
#else
      int th_count = MasterMap::Instance().getCountInnerScalar(s,offnode_maps);
#endif      
      WorkgroupGuardExec workgroupGuardExec(th_count);
      std::vector<QDPCache::ArgKey> ids;
      workgroupGuardExec.check(ids);
#if defined (QDP_CODEGEN_VECTOR)
      ids.push_back( MasterMap::Instance().getIdVNodeInnerSIMD(s,offnode_maps) );
#else
      ids.push_back( MasterMap::Instance().getIdInnerScalar(s,offnode_maps) );
#endif
      for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
	ids.push_back( addr_leaf.ids[i] );

      jit_launch(function,th_count,ids);
    }
#if defined (QDP_CODEGEN_VECTOR)
    //
    // 2nd part to inner call: scalar for AVX/VNODE
    if (1)
    {
      int th_count = MasterMap::Instance().getCountVNodeInnerScalar(s,offnode_maps);
      WorkgroupGuardExec workgroupGuardExec(th_count);
      std::vector<QDPCache::ArgKey> ids;
      workgroupGuardExec.check(ids);
      ids.push_back( MasterMap::Instance().getIdVNodeInnerScalar(s,offnode_maps) );
      for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
	ids.push_back( addr_leaf.ids[i] );

      jit_launch(function_scalar,th_count,ids);
    }
#endif

    //
    // in case of offnode comms: Do QMP wait
    //
    ShiftPhase2 phase2;
    forEach(rhs, phase2 , NullCombine());

    // face call: scalar for all targets
    int th_count_face = MasterMap::Instance().getCountFace(s,offnode_maps);
    if ( th_count_face > 0 )
      {
	WorkgroupGuardExec workgroupGuardExec(th_count_face);
	std::vector<QDPCache::ArgKey> ids;
	workgroupGuardExec.check(ids);
	ids.push_back( MasterMap::Instance().getIdFace(s,offnode_maps) );
	for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
	  ids.push_back( addr_leaf.ids[i] );
#if defined (QDP_CODEGEN_VECTOR)
	jit_launch(function_scalar,th_count_face,ids);
#else
	jit_launch(function,th_count_face,ids);
#endif
      }

#ifdef QDP_DEEP_LOG
    jit_deep_log(function);
#endif
    
  }

} // QDP
#endif
