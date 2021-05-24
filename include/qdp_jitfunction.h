#ifndef QDP_JITFUNC_H
#define QDP_JITFUNC_H

#include<type_traits>

namespace QDP {



  template<class OP,class T, class QT1, class QT2>
  void 
  function_OP_exec(JitFunction& function, OSubLattice<T>& ret,
		   const QT1& l,const QT2& r,
		   const Subset& s)
  {
    int th_count = s.numSiteTable();

    if (th_count < 1) {
      //QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI\n";
      return;
    }

    AddressLeaf addr_leaf(s);
    OP op;
    forEach(ret, addr_leaf, NullCombine());
    AddOpAddress<OP,AddressLeaf>::apply(op,addr_leaf);
    forEach(l, addr_leaf, NullCombine());
    forEach(r, addr_leaf, NullCombine());

    JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );

    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_th_count.get_id() );
    ids.push_back( s.getIdSiteTable() );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
      ids.push_back( addr_leaf.ids[i] );
 
    jit_launch(function,th_count,ids);
  }



  
  template<class OP,class T, class T1, class T2, class C1, class C2>
  void
  function_OP_type_subtype_build(JitFunction& function, OSubLattice<T>& ret,
				 const QDPType<T1,C1> & l,const QDPSubType<T2,C2> & r)
  {
    typedef typename QDPType<T1,C1>::Subtype_t    LT;
    typedef typename QDPSubType<T2,C2>::Subtype_t RT;
    
    if (ptx_db::db_enabled)
      {
	llvm_ptx_db( function , __PRETTY_FUNCTION__ );
	if (!function.empty())
	  return;
      }
    
    llvm_start_new_function("localInnerProduct_type_subtype",__PRETTY_FUNCTION__ );

    ParamRef p_th_count     = llvm_add_param<int>();
    ParamRef p_site_table   = llvm_add_param<int*>();      // subset sitetable

    ParamLeaf param_leaf;

    typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   ret_jit(forEach(ret, param_leaf, TreeCombine()));

    OP op;
    auto op_jit = AddOpParam<OP,ParamLeaf>::apply(op,param_leaf);

    typename LeafFunctor<QDPType<T1,C1>   , ParamLeaf>::Type_t   l_jit(forEach(l, param_leaf, TreeCombine()));
    typename LeafFunctor<QDPSubType<T2,C2>, ParamLeaf>::Type_t   r_jit(forEach(r, param_leaf, TreeCombine()));
	
    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
    llvm::Value* r_idx_thread = llvm_thread_idx();

    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

    llvm::Value* r_idx_perm = llvm_array_type_indirection( p_site_table , r_idx_thread );

    typename REGType< typename JITType< LT >::Type_t >::Type_t l_reg;
    l_reg.setup( l_jit.elem( JitDeviceLayout::Coalesced , r_idx_perm ) );

    typename REGType< typename JITType< RT >::Type_t >::Type_t r_reg;
    r_reg.setup( r_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ) );

    ret_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ) = op_jit( l_reg , r_reg );
    
    jit_get_function(function);
  }

  
  template<class OP, class T, class T1, class T2, class C1, class C2>
  void
  function_OP_subtype_type_build(JitFunction& function, OSubLattice<T>& ret,
				 const QDPSubType<T1,C1> & l,const QDPType<T2,C2> & r)
  {
    typedef typename QDPSubType<T1,C1>::Subtype_t LT;
    typedef typename QDPType<T2,C2>::Subtype_t    RT;
    
    if (ptx_db::db_enabled)
      {
	llvm_ptx_db( function , __PRETTY_FUNCTION__ );
	if (!function.empty())
	  return;
      }


    llvm_start_new_function("localInnerProduct_subtype_type",__PRETTY_FUNCTION__ );

    
    ParamRef p_th_count     = llvm_add_param<int>();
    ParamRef p_site_table   = llvm_add_param<int*>();      // subset sitetable

    ParamLeaf param_leaf;

    typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   ret_jit(forEach(ret, param_leaf, TreeCombine()));

    OP op;
    auto op_jit = AddOpParam<OP,ParamLeaf>::apply(op,param_leaf);

    typename LeafFunctor<QDPSubType<T1,C1> , ParamLeaf>::Type_t   l_jit(forEach(l, param_leaf, TreeCombine()));
    typename LeafFunctor<QDPType<T2,C2>    , ParamLeaf>::Type_t   r_jit(forEach(r, param_leaf, TreeCombine()));
	
    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
    llvm::Value* r_idx_thread = llvm_thread_idx();

    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

    llvm::Value* r_idx_perm = llvm_array_type_indirection( p_site_table , r_idx_thread );

    typename REGType< typename JITType< LT >::Type_t >::Type_t l_reg;
    l_reg.setup( l_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ) );   // ok: Scalar

    typename REGType< typename JITType< RT >::Type_t >::Type_t r_reg;
    r_reg.setup( r_jit.elem( JitDeviceLayout::Coalesced , r_idx_perm ) );

    ret_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ) = op_jit( l_reg , r_reg );   // ok: Scalar
    
    jit_get_function( function );
  }



  template<class T , class T2>
  inline void 
  function_extract_exec(JitFunction& function, multi1d<OScalar<T> >& dest, const OLattice<T2>& src, const Subset& s)
  {
#ifdef QDP_DEEP_LOG
    // function.start = s.start();
    // function.count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();
    // function.size_T = sizeof(T);
    // function.type_W = typeid(typename WordType<T>::Type_t).name();
    // function.set_dest_id( dest.getId() );
#endif
    
    if (s.numSiteTable() < 1)
      return;

    // Register the destination object with the memory cache
    int d_id = QDP_get_global_cache().registrateOwnHostMem( sizeof(T) * s.numSiteTable() , dest.slice() , nullptr );

    AddressLeaf addr_leaf(s);
    forEach(src, addr_leaf, NullCombine());

    // // For tuning
    // function.set_dest_id( dest.getId() );
    // function.set_enable_tuning();
  
    int th_count = s.numSiteTable();
		
    JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );
  
    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_th_count.get_id() );
    ids.push_back( s.getIdSiteTable() );
    ids.push_back( d_id );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
      ids.push_back( addr_leaf.ids[i] );
	  
    jit_launch(function,th_count,ids);

    // need to sync GPU
    jit_util_sync_copy();
    
    // Copy result to host
    QDP_get_global_cache().assureOnHost(d_id);

    // Sign off result
    QDP_get_global_cache().signoff( d_id );
  }

  
  

  template<class T , class T2>
  inline void 
  function_extract_build(JitFunction& function, multi1d<OScalar<T> >& dest, const OLattice<T2>& src)
  {
    if (ptx_db::db_enabled)
      {
	llvm_ptx_db( function , __PRETTY_FUNCTION__ );
	if (!function.empty())
	  return;
      }
    
    llvm_start_new_function("extract", __PRETTY_FUNCTION__ );

    ParamRef p_th_count   = llvm_add_param<int>();
    ParamRef p_site_table = llvm_add_param<int*>();

    typedef typename WordType<T>::Type_t TWT;
    ParamRef p_odata      = llvm_add_param< TWT* >();  // output array

    ParamLeaf param_leaf;

    typedef typename LeafFunctor<OLattice<T2>, ParamLeaf>::Type_t  FuncRet_t;
    FuncRet_t src_jit(forEach(src, param_leaf, TreeCombine()));

    OLatticeJIT<typename JITType<T>::Type_t> odata( p_odata );   // want scalar access later

    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );

    llvm::Value* r_idx_thread = llvm_thread_idx();

    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

    llvm::Value* r_idx = llvm_array_type_indirection( p_site_table , r_idx_thread );

    typename REGType< typename JITType<T>::Type_t >::Type_t in_data_reg;   
    in_data_reg.setup( src_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );
      
    odata.elem( JitDeviceLayout::Scalar , r_idx_thread ) = in_data_reg;

    jit_get_function( function );
  }


#if 1
  namespace
  {
    template <class T>
    void print_type()
    {
      QDPIO::cout << __PRETTY_FUNCTION__ << std::endl;
    }
  }
#endif



  

template<class T>
struct Strip
{
  typedef T Type_t;
};

template<class T,int N>
struct Strip<Reference<PSpinMatrix<T,N> > >
{
  typedef PSpinMatrix<float,N> Type_t;
};

template<class T,int N>
struct Strip<PSpinMatrix<T,N> >
{
  typedef PSpinMatrix<float,N> Type_t;
};



template<class T>
struct HasProp
{
  constexpr static bool value = false;
};

template<class Op, class A, class B>
struct HasProp< BinaryNode<Op,A,B> >
{
  constexpr static bool value = HasProp<A>::value || HasProp<B>::value;
};

template<class Op, class A>
struct HasProp< UnaryNode<Op,A> >
{
  constexpr static bool value = HasProp<A>::value;
};


template<>
struct HasProp< Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, Nc >, Ns >, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, Nc >, Ns > > > > >
{
  constexpr static bool value = true;
};



template<class A, class B, class CTag>
struct ForEach<BinaryNode<OpMultiply, A, B>, JitCreateLoopsLeaf, CTag >
{
  typedef typename ForEach<A, JitCreateLoopsLeaf, CTag>::Type_t TypeA_t;
  typedef typename ForEach<B, JitCreateLoopsLeaf, CTag>::Type_t TypeB_t;
  typedef typename Combine2<TypeA_t, TypeB_t, OpMultiply, CTag>::Type_t Type_t;
  inline static
  Type_t apply(const BinaryNode<OpMultiply, A, B> &expr, const JitCreateLoopsLeaf &f,
	       const CTag &c) 
  {
    if ( is_same<typename Strip< typename ForEach<A, EvalLeaf1, OpCombine>::Type_t >::Type_t , PSpinMatrix<float, 4> >::value &&   // float is okay
	 is_same<typename Strip< typename ForEach<B, EvalLeaf1, OpCombine>::Type_t >::Type_t , PSpinMatrix<float, 4> >::value )
      {
	QDPIO::cout << "mul(spinMat,spinMat) -> insert loop" << std::endl;
	//f.loops.push_back( JitForLoop(0,4) );
	f.loop_bounds.push_back( 4 );
      }
    
    return Combine2<TypeA_t, TypeB_t, OpMultiply, CTag>::
      combine(ForEach<A, JitCreateLoopsLeaf, CTag>::apply(expr.left(), f, c),
              ForEach<B, JitCreateLoopsLeaf, CTag>::apply(expr.right(), f, c),
	      expr.operation(), c);
  }
};


template<class A, class B, class CTag>
struct ForEach<BinaryNode<OpMultiply, A, B>, ViewSpinLeaf, CTag >
{
  typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
  typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
  typedef typename Combine2<TypeA_t, TypeB_t, OpMultiply, CTag>::Type_t Type_t;
  inline static
  Type_t apply(const BinaryNode<OpMultiply, A, B> &expr, const ViewSpinLeaf &f,
	       const CTag &c) 
  {
    ViewSpinLeaf f_left(f);
    ViewSpinLeaf f_right(f);

    typedef typename ForEach<A , JIT2BASE , TreeCombine >::Type_t A_a;
    typedef typename ForEach<B , JIT2BASE , TreeCombine >::Type_t B_a;
    
    if ( is_same<typename Strip< typename ForEach<A_a, EvalLeaf1, OpCombine>::Type_t >::Type_t , PSpinMatrix<float, 4> >::value &&
	 is_same<typename Strip< typename ForEach<B_a, EvalLeaf1, OpCombine>::Type_t >::Type_t , PSpinMatrix<float, 4> >::value )
      {
	QDPIO::cout << "mul(spinMat,spinMat) -> split index    " << std::endl;

	if (f.ind.size() != 2)
	  {
	    QDPIO::cout << "at mul(spinmat,spinmat) but not 2 indices provided (" << f.ind.size() << ")" << std::endl;
	    QDP_abort(1);
	  }

	if (f.loops.size() < f.loop_pos)
	  {
	    QDPIO::cout << "at mul(spinmat,spinmat) but not enough loops created (" << f.loops.size() << " < " << f.loop_pos << ")" << std::endl;
	    QDP_abort(1);
	  }
	
	f_left.ind[1] = f.loop_pos;
	f_left.loop_pos++;
	
	f_right.ind[0] = f.loop_pos;
	f_right.loop_pos++;
      }
    
    return Combine2<TypeA_t, TypeB_t, OpMultiply, CTag>::
      combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(), f_left, c),
              ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), f_right, c),
	      expr.operation(), c);
  }
};




template<class T,int N>
T viewSpinJit( OLatticeJIT< PSpinMatrixJIT<T,N> >& dest , const ViewSpinLeaf& v )
{
  if (v.ind.size() != 2)
    {
      QDPIO::cout << "at viewSpinJit(spinmat) but not 2 indices provided (" << v.ind.size() << ")" << std::endl;
      QDP_abort(1);
    }
  
  return dest.elem( v.getLayout() , v.getIndex() ).getJitElem( v.loops[0].index() , v.loops[1].index() );
}

template<class T>
T viewSpinJit( OLatticeJIT< PScalarJIT<T> >& dest , const ViewSpinLeaf& v )
{
  if (v.ind.size() != 0)
    {
      QDPIO::cout << "at viewSpinJit(pscalar) but not 0 indices provided (" << v.ind.size() << ")" << std::endl;
      QDP_abort(1);
    }
  
  return dest.elem( v.getLayout() , v.getIndex() ).getJitElem();
}



//


template<class T, class T1, class Op, class RHS>
typename std::enable_if_t< ! HasProp<RHS>::value >
function_build(JitFunction& function, const DynKey& key, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
{
  std::ostringstream expr;
#if 0
  printExprTreeSubset( expr , dest, op, rhs , s , key );
#else
  expr << std::string(__PRETTY_FUNCTION__) << "_key=" << key;
#endif
  
  if (ptx_db::db_enabled)
    {
      llvm_ptx_db( function , expr.str().c_str() );
      if (!function.empty())
	return;
    }
  llvm_start_new_function("eval",expr.str().c_str() );
  
  if ( key.get_offnode_comms() )
    {
      if ( s.hasOrderedRep() )
	{
	  ParamRef p_th_count   = llvm_add_param<int>();
	  ParamRef p_site_table = llvm_add_param<int*>();

	  ParamLeaf param_leaf;

	  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
	  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
	  
	  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

	  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
	  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

	  llvm::Value * r_th_count     = llvm_derefParam( p_th_count );

	  llvm::Value* r_idx_thread = llvm_thread_idx();
       
	  llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

	  llvm::Value* r_idx = llvm_array_type_indirection( p_site_table , r_idx_thread );

	  op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ), 
		  forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced , r_idx ), OpCombine()));	  
	}
      else
	{
	  QDPIO::cout << "eval with shifts on unordered subsets not supported" << std::endl;
	  QDP_abort(1);
	}
    }
  else
    {
      if ( s.hasOrderedRep() )
	{
	  ParamRef p_th_count = llvm_add_param<int>();
	  ParamRef p_start    = llvm_add_param<int>();

	  ParamLeaf param_leaf;

	  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
	  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
	  
	  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

	  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
	  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

	  //print_type<View_t>();
	  
	  llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
	  llvm::Value * r_start        = llvm_derefParam( p_start );

	  llvm::Value* r_idx_thread = llvm_thread_idx();

	  llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

	  llvm::Value* r_idx = llvm_add( r_idx_thread , r_start );

	  QDPIO::cout << "using vanilla route" << std::endl;

	  op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ), 
		  forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced , r_idx ), OpCombine()));
	  
	}
      else // unordered Subset
	{
	  ParamRef p_th_count   = llvm_add_param<int>();
	  ParamRef p_site_table = llvm_add_param<int*>();

	  ParamLeaf param_leaf;

	  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
	  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
	  
	  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

	  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
	  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

	  llvm::Value * r_th_count     = llvm_derefParam( p_th_count );

	  llvm::Value* r_idx_thread = llvm_thread_idx();
       
	  llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

	  llvm::Value* r_idx = llvm_array_type_indirection( p_site_table , r_idx_thread );

	  op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ), 
		  forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced , r_idx ), OpCombine()));	  
	}
    }
  
  jit_get_function( function );
}


template<class T, class T1, class Op, class RHS>
typename std::enable_if_t< HasProp<RHS>::value >
function_build(JitFunction& function, const DynKey& key, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
{
  std::ostringstream expr;
#if 0
  printExprTreeSubset( expr , dest, op, rhs , s , key );
#else
  expr << std::string(__PRETTY_FUNCTION__) << "_key=" << key;
#endif
  
  if (ptx_db::db_enabled)
    {
      llvm_ptx_db( function , expr.str().c_str() );
      if (!function.empty())
	return;
    }
  llvm_start_new_function("eval",expr.str().c_str() );
  
  if ( key.get_offnode_comms() )
    {
      if ( s.hasOrderedRep() )
	{
	  ParamRef p_th_count   = llvm_add_param<int>();
	  ParamRef p_site_table = llvm_add_param<int*>();

	  ParamLeaf param_leaf;

	  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
	  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
	  
	  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

	  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
	  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

	  llvm::Value * r_th_count     = llvm_derefParam( p_th_count );

	  llvm::Value* r_idx_thread = llvm_thread_idx();
       
	  llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

	  llvm::Value* r_idx = llvm_array_type_indirection( p_site_table , r_idx_thread );

	  op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ), 
		  forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced , r_idx ), OpCombine()));	  
	}
      else
	{
	  QDPIO::cout << "eval with shifts on unordered subsets not supported" << std::endl;
	  QDP_abort(1);
	}
    }
  else
    {
      if ( s.hasOrderedRep() )
	{
	  ParamRef p_th_count = llvm_add_param<int>();
	  ParamRef p_start    = llvm_add_param<int>();

	  ParamLeaf param_leaf;

	  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
	  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
	  
	  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

	  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
	  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

	  //print_type<View_t>();
	  
	  llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
	  llvm::Value * r_start        = llvm_derefParam( p_start );

	  llvm::Value* r_idx_thread = llvm_thread_idx();

	  llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

	  llvm::Value* r_idx = llvm_add( r_idx_thread , r_start );

	  
	  JitCreateLoopsLeaf jitCreateLoopsLeaf;
	  forEach( dest , jitCreateLoopsLeaf , NullCombine() );

	  int dest_rank = jitCreateLoopsLeaf.loop_bounds.size();
	  
	  jitCreateLoopsLeaf.ops_only = true;
	  forEach( rhs , jitCreateLoopsLeaf , NullCombine() );

	  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, EvalLeaf1, OpCombine>::Type_t RHS_result_t;
	  JitStackArray< typename REGType< typename RHS_result_t::Sub_t >::Type_t , 1 > stack;

	  // create loops for 'dest'
	  for ( int i = 0 ; i < dest_rank ; ++i )
	    {
	      int bound = jitCreateLoopsLeaf.loop_bounds[i];
	      jitCreateLoopsLeaf.loops.push_back( JitForLoop(0,bound) );
	    }

	  // zero-out accumulation object on stack
	  zero_rep( stack.elemJIT(0) );

	  // create loops for operations
	  for ( int i = dest_rank ; i < jitCreateLoopsLeaf.loop_bounds.size() ; ++i )
	    {
	      int bound = jitCreateLoopsLeaf.loop_bounds[i];
	      jitCreateLoopsLeaf.loops.push_back( JitForLoop(0,bound) );
	    }
	      
	  ViewSpinLeaf viewSpin( JitDeviceLayout::Coalesced , r_idx );
	  viewSpin.loops = jitCreateLoopsLeaf.loops;
	  viewSpin.loop_pos = dest_rank;
	    
	  for( int i = 0 ; i < dest_rank ; ++i )
	    {
	      viewSpin.ind.push_back( i );
	    }
	  
	  stack.elemJIT(0) += forEach( rhs_view , viewSpin , OpCombine() );

	  for( int i = jitCreateLoopsLeaf.loops.size()-1 ; dest_rank <= i ; --i )
	    {
	      QDPIO::cout << "loop[" << i << "].end();" << std::endl;
	      jitCreateLoopsLeaf.loops[i].end();
	    }

	  QDPIO::cout << "copy stack to dest, viewSpin.ind=" << std::endl;

	  for(int i = 0 ; i < viewSpin.ind.size() ; ++i )
	    QDPIO::cout << i << ": " << viewSpin.ind[i] << std::endl;
	  
	  op_jit( viewSpinJit( dest_jit , viewSpin ) , stack.elemREG(0) );

	  for( int i = dest_rank-1 ; 0 <= i ; --i )
	    {
	      QDPIO::cout << "loop[" << i << "].end();" << std::endl;
	      jitCreateLoopsLeaf.loops[i].end();
	    }
	    
	  
	}
      else // unordered Subset
	{
	  ParamRef p_th_count   = llvm_add_param<int>();
	  ParamRef p_site_table = llvm_add_param<int*>();

	  ParamLeaf param_leaf;

	  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
	  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
	  
	  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

	  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
	  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

	  llvm::Value * r_th_count     = llvm_derefParam( p_th_count );

	  llvm::Value* r_idx_thread = llvm_thread_idx();
       
	  llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

	  llvm::Value* r_idx = llvm_array_type_indirection( p_site_table , r_idx_thread );

	  op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ), 
		  forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced , r_idx ), OpCombine()));	  
	}
    }
  
  jit_get_function( function );
}




  


template<class T, class C1, class Op, class RHS>
void
function_subtype_type_build(JitFunction& function, OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,C1 >& rhs)
{
  if (ptx_db::db_enabled)
    {
      llvm_ptx_db( function , __PRETTY_FUNCTION__ );
      if (!function.empty())
	return;
    }


  llvm_start_new_function("eval_subtype_type",__PRETTY_FUNCTION__ );

  ParamRef p_th_count     = llvm_add_param<int>();
  ParamRef p_site_table   = llvm_add_param<int*>();      // subset sitetable

  ParamLeaf param_leaf;

  typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);
  typename ForEach<QDPExpr<RHS,C1 >, ParamLeaf, TreeCombine>::Type_t rhs_jit(forEach(rhs, param_leaf, TreeCombine()));
  
  llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
  llvm::Value* r_idx_thread = llvm_thread_idx();

  llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

  llvm::Value* r_idx_perm = llvm_array_type_indirection( p_site_table , r_idx_thread );

  op_jit( dest_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ), // Coalesced
	  forEach(rhs_jit, ViewLeaf( JitDeviceLayout::Coalesced , r_idx_perm ), OpCombine()));

  jit_get_function( function );
}


  
template<class T, class T1, class Op>
void
operator_type_subtype_build(JitFunction& function, OLattice<T>& dest, const Op& op, const QDPSubType<T1,OLattice<T1> >& rhs)
{
  typedef typename QDPSubType<T1,OLattice<T1>>::Subtype_t RT;
      
  if (ptx_db::db_enabled)
    {
      llvm_ptx_db( function , __PRETTY_FUNCTION__ );
      if (!function.empty())
	return;
    }


  llvm_start_new_function("eval_type_subtype",__PRETTY_FUNCTION__ );

  ParamRef p_th_count     = llvm_add_param<int>();
  ParamRef p_site_table   = llvm_add_param<int*>();      // subset sitetable

  ParamLeaf param_leaf;

  typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);
  //typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   rhs_jit(forEach(rhs, param_leaf, TreeCombine()));
  typename LeafFunctor<QDPSubType<T1,OLattice<T1> >, ParamLeaf>::Type_t   rhs_jit(forEach(rhs, param_leaf, TreeCombine()));
  //typename ForEach<QDPExpr<RHS,OSubLattice<T1> >, ParamLeaf, TreeCombine>::Type_t rhs_jit(forEach(rhs, param_leaf, TreeCombine()));

    
  llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
  llvm::Value* r_idx_thread = llvm_thread_idx();

  llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

  llvm::Value* r_idx_perm = llvm_array_type_indirection( p_site_table , r_idx_thread );

  typename REGType< typename JITType< RT >::Type_t >::Type_t rhs_reg;
  rhs_reg.setup( rhs_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ) );
  
  // op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx_thread ), // Coalesced
  // 	  forEach(rhs_jit, ViewLeaf( JitDeviceLayout::Scalar , r_idx_perm ), OpCombine()));

  op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx_perm ), rhs_reg );

  jit_get_function( function );
}


  
template<class T, class T1, class Op>
void
operator_subtype_subtype_build(JitFunction& function, OSubLattice<T>& dest, const Op& op, const QDPSubType<T1,OLattice<T1> >& rhs)
{
  typedef typename QDPSubType<T1,OLattice<T1>>::Subtype_t RT;
      
  if (ptx_db::db_enabled)
    {
      llvm_ptx_db( function , __PRETTY_FUNCTION__ );
      if (!function.empty())
	return;
    }


  llvm_start_new_function("eval_subtype_subtype",__PRETTY_FUNCTION__ );

  ParamRef p_th_count     = llvm_add_param<int>();
  ParamLeaf param_leaf;

  typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);
  typename LeafFunctor<QDPSubType<T1,OLattice<T1> >, ParamLeaf>::Type_t   rhs_jit(forEach(rhs, param_leaf, TreeCombine()));
    
  llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
  llvm::Value* r_idx_thread = llvm_thread_idx();

  llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

  typename REGType< typename JITType< RT >::Type_t >::Type_t rhs_reg;
  rhs_reg.setup( rhs_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ) );
  
  op_jit( dest_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ), rhs_reg );

  jit_get_function( function );
}


  

template<class T, class T1, class Op, class RHS>
void
function_lat_sca_exec(JitFunction& function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs, const Subset& s)
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  if (s.numSiteTable() < 1)
    return;

#ifdef QDP_DEEP_LOG
  function.start = s.start();
  function.count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();
  function.size_T = sizeof(T);
  function.type_W = typeid(typename WordType<T>::Type_t).name();
  function.set_dest_id( dest.getId() );
#endif

  int th_count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();

  AddressLeaf addr_leaf(s);

  forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  forEach(rhs, addr_leaf, NullCombine());

  JitParam jit_ordered( QDP_get_global_cache().addJitParamBool( s.hasOrderedRep() ) );
  JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );
  JitParam jit_start( QDP_get_global_cache().addJitParamInt( s.start() ) );
  JitParam jit_end( QDP_get_global_cache().addJitParamInt( s.end() ) );
  
  std::vector<QDPCache::ArgKey> ids;
  ids.push_back( jit_ordered.get_id() );
  ids.push_back( jit_th_count.get_id() );
  ids.push_back( jit_start.get_id() );
  ids.push_back( jit_end.get_id() );
  ids.push_back( s.getIdMemberTable() );
  for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
    ids.push_back( addr_leaf.ids[i] );
 
  jit_launch(function,th_count,ids);
}





  


template<class T, class T1, class Op, class RHS>
void
function_lat_sca_build(JitFunction& function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
{
  //std::cout << __PRETTY_FUNCTION__ << std::endl;
  
  if (ptx_db::db_enabled)
    {
      llvm_ptx_db( function , __PRETTY_FUNCTION__ );
      if (!function.empty())
	return;
    }


  std::vector<ParamRef> params = jit_function_preamble_param("eval_lat_sca",__PRETTY_FUNCTION__);

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

  typedef typename ForEach<QDPExpr<RHS,OScalar<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

  llvm::Value * r_idx = jit_function_preamble_get_idx( params );

  op_jit(dest_jit.elem( JitDeviceLayout::Coalesced , r_idx), 
	 forEach(rhs_view, ViewLeaf( JitDeviceLayout::Scalar , r_idx ), OpCombine()));

  jit_get_function( function );
}






template<class T, class T1, class Op, class RHS>
void
function_lat_sca_subtype_build(JitFunction& function, OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
{
  if (ptx_db::db_enabled)
    {
      llvm_ptx_db( function , __PRETTY_FUNCTION__ );
      if (!function.empty())
	return;
    }


  llvm_start_new_function("eval_lat_sca_subtype",__PRETTY_FUNCTION__);

  ParamRef p_th_count     = llvm_add_param<int>();
      
  ParamLeaf param_leaf;

  typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);
  typename ForEach<QDPExpr<RHS,OScalar<T1> >, ParamLeaf, TreeCombine>::Type_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

  llvm::Value * r_th_count  = llvm_derefParam( p_th_count );
  llvm::Value* r_idx_thread = llvm_thread_idx();

  llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

  op_jit(dest_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ), // Coalesced
	 forEach(rhs_view, ViewLeaf( JitDeviceLayout::Scalar , r_idx_thread ), OpCombine()));

  jit_get_function( function );
}

  



template<class T, class T1>
void
function_pokeSite_build( JitFunction& function, const OLattice<T>& dest , const OScalar<T1>& r  )
{
  if (ptx_db::db_enabled)
    {
      llvm_ptx_db( function , __PRETTY_FUNCTION__ );
      if (!function.empty())
	return;
    }


  llvm_start_new_function("eval_pokeSite",__PRETTY_FUNCTION__);

  ParamRef p_siteindex    = llvm_add_param<int>();

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
    
  auto op_jit = AddOpParam<OpAssign,ParamLeaf>::apply(OpAssign(),param_leaf);

  typedef typename LeafFunctor<OScalar<T1>, ParamLeaf>::Type_t  FuncRet_t1;
  FuncRet_t1 r_jit(forEach(r, param_leaf, TreeCombine()));

  llvm::Value* r_siteindex = llvm_derefParam( p_siteindex );
  llvm::Value* r_zero      = llvm_create_value(0);

  op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_siteindex ), 
	  LeafFunctor< FuncRet_t1 , ViewLeaf >::apply( r_jit , ViewLeaf( JitDeviceLayout::Scalar , r_zero ) ) );

  jit_get_function( function );
}


template<class T, class T1>
void 
function_pokeSite_exec(JitFunction& function, OLattice<T>& dest, const OScalar<T1>& rhs, const multi1d<int>& coord )
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf(all);

  forEach(dest, addr_leaf, NullCombine());
  forEach(rhs, addr_leaf, NullCombine());

  JitParam jit_siteindex( QDP_get_global_cache().addJitParamInt( Layout::linearSiteIndex(coord) ) );

  std::vector<QDPCache::ArgKey> ids;
  ids.push_back( jit_siteindex.get_id() );
  for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
    ids.push_back( addr_leaf.ids[i] );
 
  jit_launch(function,1,ids);   // 1 - thread count
}





template<class T>
void
function_zero_rep_build( JitFunction& function, OLattice<T>& dest)
{
  if (ptx_db::db_enabled)
    {
      llvm_ptx_db( function , __PRETTY_FUNCTION__ );
      if (!function.empty())
	return;
    }


  std::vector<ParamRef> params = jit_function_preamble_param("zero_rep",__PRETTY_FUNCTION__);

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  llvm::Value * r_idx = jit_function_preamble_get_idx( params );

  zero_rep( dest_jit.elem(JitDeviceLayout::Coalesced,r_idx) );

  jit_get_function( function );
}



template<class T>
void
function_zero_rep_subtype_build( JitFunction& function, OSubLattice<T>& dest)
{
  if (ptx_db::db_enabled)
    {
      llvm_ptx_db( function , __PRETTY_FUNCTION__ );
      if (!function.empty())
	return;
    }


  llvm_start_new_function("zero_rep_subtype",__PRETTY_FUNCTION__);

  ParamRef p_th_count     = llvm_add_param<int>();

  ParamLeaf param_leaf;
  typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
  
  llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
  llvm::Value* r_idx_thread = llvm_thread_idx();

  llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

  zero_rep( dest_jit.elem(JitDeviceLayout::Coalesced,r_idx_thread) );

  jit_get_function( function );
}
  






template<class T>
void
function_random_build( JitFunction& function, OLattice<T>& dest , Seed& seed_tmp)
{
  if (ptx_db::db_enabled)
    {
      llvm_ptx_db( function , __PRETTY_FUNCTION__ );
      if (!function.empty())
	return;
    }


  llvm_start_new_function("random",__PRETTY_FUNCTION__);

  // No member possible here.
  // If thread exits due to non-member
  // it possibly can't store the new seed at the end.

  ParamRef p_lo     = llvm_add_param<int>();
  ParamRef p_hi     = llvm_add_param<int>();

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  // RNG::ran_seed
  typedef typename LeafFunctor<Seed, ParamLeaf>::Type_t  SeedJIT;
  typedef typename LeafFunctor<LatticeSeed, ParamLeaf>::Type_t  LatticeSeedJIT;
  typedef typename REGType<typename SeedJIT::Subtype_t>::Type_t PSeedREG;

  SeedJIT ran_seed_jit(forEach(RNG::get_RNG_Internals()->ran_seed, param_leaf, TreeCombine()));
  SeedJIT seed_tmp_jit(forEach(seed_tmp, param_leaf, TreeCombine()));
  SeedJIT ran_mult_n_jit(forEach(RNG::get_RNG_Internals()->ran_mult_n, param_leaf, TreeCombine()));
  LatticeSeedJIT lattice_ran_mult_jit(forEach( RNG::get_RNG_Internals()->lattice_ran_mult , param_leaf, TreeCombine()));

  llvm::Value * r_lo     = llvm_derefParam( p_lo );
  llvm::Value * r_hi     = llvm_derefParam( p_hi );

  llvm::Value * r_idx_thread = llvm_thread_idx();

  llvm::Value * r_out_of_range       = llvm_gt( r_idx_thread , llvm_sub( r_hi , r_lo ) );
  llvm_cond_exit(  r_out_of_range );

  llvm::Value * r_idx = llvm_add( r_lo , r_idx_thread );

  PSeedREG seed_reg;
  PSeedREG skewed_seed_reg;
  PSeedREG ran_mult_n_reg;
  PSeedREG lattice_ran_mult_reg;

  seed_reg.setup( ran_seed_jit.elem() );

  lattice_ran_mult_reg.setup( lattice_ran_mult_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );

  skewed_seed_reg = seed_reg * lattice_ran_mult_reg;

  ran_mult_n_reg.setup( ran_mult_n_jit.elem() );

  fill_random( dest_jit.elem(JitDeviceLayout::Coalesced,r_idx) , seed_reg , skewed_seed_reg , ran_mult_n_reg );

  llvm::Value * r_save = llvm_eq( r_idx_thread , llvm_create_value(0) );

  JitIf save(r_save);
  {
    seed_tmp_jit.elem() = seed_reg;
  }
  save.end();
  
  jit_get_function( function );
}






template<class T, class T1, class RHS>
void
function_gather_build( JitFunction& function, const QDPExpr<RHS,OLattice<T1> >& rhs )
{
  if (ptx_db::db_enabled)
    {
      llvm_ptx_db( function , __PRETTY_FUNCTION__ );
      if (!function.empty())
	return;
    }


  typedef typename WordType<T1>::Type_t WT;

  llvm_start_new_function("gather",__PRETTY_FUNCTION__);

  ParamRef p_lo      = llvm_add_param<int>();
  ParamRef p_hi      = llvm_add_param<int>();
  ParamRef p_soffset = llvm_add_param<int*>();
  ParamRef p_sndbuf  = llvm_add_param<WT*>();

  ParamLeaf param_leaf;

  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view( forEach( rhs , param_leaf , TreeCombine() ) );

  typedef typename JITType< OLattice<T> >::Type_t DestView_t;
  DestView_t dest_jit( p_sndbuf );

  llvm_derefParam( p_lo );  // r_lo not used
  llvm::Value * r_hi      = llvm_derefParam( p_hi );
  llvm::Value * r_idx     = llvm_thread_idx();  

  llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

  llvm::Value * r_idx_site = llvm_array_type_indirection( p_soffset , r_idx );
  
  OpAssign()( dest_jit.elem( JitDeviceLayout::Scalar , r_idx ) , 
	      forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced , r_idx_site ) , OpCombine() ) );

  jit_get_function( function );
}





template<class T1, class RHS>
void
  function_gather_exec( JitFunction& function, int send_buf_id , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs , const Subset& subset )
{
  if (subset.numSiteTable() < 1)
    return;

  AddressLeaf addr_leaf(subset);

  forEach(rhs, addr_leaf, NullCombine());

  int hi = map.soffset(subset).size();

  JitParam jit_lo( QDP_get_global_cache().addJitParamInt( 0 ) );        // lo, leave it in
  JitParam jit_hi( QDP_get_global_cache().addJitParamInt( hi ) );

  std::vector<QDPCache::ArgKey> ids;
  ids.push_back( jit_lo.get_id() );
  ids.push_back( jit_hi.get_id() );
  ids.push_back( map.getSoffsetsId(subset) );
  ids.push_back( send_buf_id );
  for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
    ids.push_back( addr_leaf.ids[i] );
 
  jit_launch(function,hi,ids);
#if 0  
  int lo = 0;
  int hi = map.soffset(subset).size();

  int soffsetsId = map.getSoffsetsId(subset);
  void * soffsetsDev = QDP_get_global_cache().getDevicePtr( soffsetsId );

  std::vector<void*> addr;

  addr.push_back( &lo );
  addr.push_back( &hi );
  addr.push_back( &soffsetsDev );
  addr.push_back( &send_buf );
  for(unsigned i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
  }
  jit_launch(function,hi-lo,addr);
#endif
}


namespace COUNT {
  extern int count;
}


template<class T, class T1, class Op, class RHS>
void 
function_exec(JitFunction& function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
{
  //QDPIO::cout << __PRETTY_FUNCTION__ << "\n";
//  if (!s.hasOrderedRep())
//    {
//      QDPIO::cout << "no ordered rep " << __PRETTY_FUNCTION__ << std::endl;
//      QDP_abort(1);
//    }

#ifdef QDP_DEEP_LOG
  function.start = s.start();
  function.count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();
  function.size_T = sizeof(T);
  function.type_W = typeid(typename WordType<T>::Type_t).name();
  function.set_dest_id( dest.getId() );
#endif
    
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
  
  if (offnode_maps == 0)
    {
      if ( s.hasOrderedRep() )
	{
	  int th_count = s.numSiteTable();
		
	  JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );
	  JitParam jit_start( QDP_get_global_cache().addJitParamInt( s.start() ) );

	  std::vector<QDPCache::ArgKey> ids;
	  ids.push_back( jit_th_count.get_id() );
	  ids.push_back( jit_start.get_id() );
	  for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
	    ids.push_back( addr_leaf.ids[i] );
	  
	  jit_launch(function,th_count,ids);
	}
      else
	{
	  int th_count = s.numSiteTable();
      
	  JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );

	  std::vector<QDPCache::ArgKey> ids;
	  ids.push_back( jit_th_count.get_id() );
	  ids.push_back( s.getIdSiteTable() );
	  for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
	    ids.push_back( addr_leaf.ids[i] );
 
	  jit_launch(function,th_count,ids);
	}
    }
  else
    {
      // 1st. call: inner
      {
	int th_count = MasterMap::Instance().getCountInner(s,offnode_maps);
      
	JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );

	std::vector<QDPCache::ArgKey> ids;
	ids.push_back( jit_th_count.get_id() );
	ids.push_back( MasterMap::Instance().getIdInner(s,offnode_maps) );
	for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
	  ids.push_back( addr_leaf.ids[i] );
 
	jit_launch(function,th_count,ids);
      }
      
      // 2nd call: face
      {
	ShiftPhase2 phase2;
	forEach(rhs, phase2 , NullCombine());

	int th_count = MasterMap::Instance().getCountFace(s,offnode_maps);
      
	JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );

	std::vector<QDPCache::ArgKey> ids;
	ids.push_back( jit_th_count.get_id() );
	ids.push_back( MasterMap::Instance().getIdFace(s,offnode_maps) );
	for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
	  ids.push_back( addr_leaf.ids[i] );
 
	jit_launch(function,th_count,ids);
      }
      
    }
}





template<class T, class C1, class Op, class RHS>
void 
function_subtype_type_exec(JitFunction& function, OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,C1 >& rhs, const Subset& s)
{
  int th_count = s.numSiteTable();

  if (th_count < 1) {
    //QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI node\n";
    return;
  }

  AddressLeaf addr_leaf(s);
  forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  forEach(rhs, addr_leaf, NullCombine());

  JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );

  std::vector<QDPCache::ArgKey> ids;
  ids.push_back( jit_th_count.get_id() );
  ids.push_back( s.getIdSiteTable() );
  for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
    ids.push_back( addr_leaf.ids[i] );
 
  jit_launch(function,th_count,ids);
}



template<class T, class T1, class Op>
void 
operator_type_subtype_exec(JitFunction& function, OLattice<T>& dest, const Op& op, const QDPSubType<T1,OLattice<T1> >& rhs, const Subset& s)
{
  int th_count = s.numSiteTable();

  if (th_count < 1) {
    //QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI\n";
    return;
  }

  AddressLeaf addr_leaf(s);
  forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  forEach(rhs, addr_leaf, NullCombine());

  JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );

  std::vector<QDPCache::ArgKey> ids;
  ids.push_back( jit_th_count.get_id() );
  ids.push_back( s.getIdSiteTable() );
  for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
    ids.push_back( addr_leaf.ids[i] );
 
  jit_launch(function,th_count,ids);
}



template<class T, class T1, class Op>
void 
operator_subtype_subtype_exec(JitFunction& function, OSubLattice<T>& dest, const Op& op, const QDPSubType<T1,OLattice<T1> >& rhs, const Subset& s)
{
  int th_count = s.numSiteTable();

  if (th_count < 1) {
    //QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI\n";
    return;
  }

  AddressLeaf addr_leaf(s);
  forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  forEach(rhs, addr_leaf, NullCombine());

  JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );

  std::vector<QDPCache::ArgKey> ids;
  ids.push_back( jit_th_count.get_id() );
  for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
    ids.push_back( addr_leaf.ids[i] );
 
  jit_launch(function,th_count,ids);
}



  


template<class T, class T1, class Op, class RHS>
void 
function_lat_sca_subtype_exec(JitFunction& function, OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs, const Subset& s)
{
  int th_count = s.numSiteTable();

  if (th_count < 1) {
    //QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI\n";
    return;
  }

  AddressLeaf addr_leaf(s);

  forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  forEach(rhs, addr_leaf, NullCombine());

  JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );

  std::vector<QDPCache::ArgKey> ids;
  ids.push_back( jit_th_count.get_id() );
  for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
    ids.push_back( addr_leaf.ids[i] );
 
  jit_launch(function,th_count,ids);
}





template<class T>
void 
function_zero_rep_exec(JitFunction& function, OLattice<T>& dest, const Subset& s )
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  if (s.numSiteTable() < 1)
    return;

#ifdef QDP_DEEP_LOG
  function.start = s.start();
  function.count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();
  function.size_T = sizeof(T);
  function.type_W = typeid(typename WordType<T>::Type_t).name();
  function.set_dest_id( dest.getId() );
#endif
  
  AddressLeaf addr_leaf(s);
  forEach(dest, addr_leaf, NullCombine());

  int th_count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();

  JitParam jit_ordered( QDP_get_global_cache().addJitParamBool( s.hasOrderedRep() ) );
  JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );
  JitParam jit_start( QDP_get_global_cache().addJitParamInt( s.start() ) );
  JitParam jit_end( QDP_get_global_cache().addJitParamInt( s.end() ) );
  
  std::vector<QDPCache::ArgKey> ids;
  ids.push_back( jit_ordered.get_id() );
  ids.push_back( jit_th_count.get_id() );
  ids.push_back( jit_start.get_id() );
  ids.push_back( jit_end.get_id() );
  ids.push_back( s.getIdMemberTable() );
  for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
    ids.push_back( addr_leaf.ids[i] );
 
  jit_launch(function,th_count,ids);
}



template<class T>
void function_zero_rep_subtype_exec(JitFunction& function, OSubLattice<T>& dest, const Subset& s )
{
  int th_count = s.numSiteTable();

  if (th_count < 1) {
    //QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI\n";
    return;
  }

  AddressLeaf addr_leaf(s);
  forEach(dest, addr_leaf, NullCombine());

  JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );
  
  std::vector<QDPCache::ArgKey> ids;
  ids.push_back( jit_th_count.get_id() );
  for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
    ids.push_back( addr_leaf.ids[i] );
 
  jit_launch(function,th_count,ids);
}



template<class T>
void 
function_random_exec(JitFunction& function, OLattice<T>& dest, const Subset& s , Seed& seed_tmp)
{
  if (!s.hasOrderedRep())
    QDP_error_exit("random on subset with unordered representation not implemented");

  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

#ifdef QDP_DEEP_LOG
  function.start = s.start();
  function.count = s.numSiteTable();
  function.size_T = sizeof(T);
  function.type_W = typeid(typename WordType<T>::Type_t).name();
  function.set_dest_id( dest.getId() );
#endif

  // Register the seed_tmp object with the memory cache
  int seed_tmp_id = QDP_get_global_cache().registrateOwnHostMemStatus( sizeof(Seed) , seed_tmp.getF() , QDPCache::Status::undef );

  AddressLeaf addr_leaf(s);

  forEach(dest, addr_leaf, NullCombine());

  forEach(RNG::get_RNG_Internals()->ran_seed, addr_leaf, NullCombine());
  //forEach(seed_tmp, addr_leaf, NullCombine()); // Caution: ParamLeaf treats OScalar as read-only
  addr_leaf.setId( seed_tmp_id );
  forEach(RNG::get_RNG_Internals()->ran_mult_n, addr_leaf, NullCombine());
  forEach(RNG::get_RNG_Internals()->lattice_ran_mult, addr_leaf, NullCombine());

  JitParam jit_lo( QDP_get_global_cache().addJitParamInt( s.start() ) );
  JitParam jit_hi( QDP_get_global_cache().addJitParamInt( s.end() ) );
  
  std::vector<QDPCache::ArgKey> ids;
  ids.push_back( jit_lo.get_id() );
  ids.push_back( jit_hi.get_id() );
  for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
    ids.push_back( addr_leaf.ids[i] );
 
  jit_launch(function,s.numSiteTable(),ids);

  // Copy seed_tmp to host
  QDP_get_global_cache().assureOnHost(seed_tmp_id);

  // Sign off seed_tmp
  QDP_get_global_cache().signoff( seed_tmp_id );
}


}

#endif
