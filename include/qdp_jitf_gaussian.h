#ifndef QDP_JITF_GAUSS_H
#define QDP_JITF_GAUSS_H


namespace QDP {

template<class T>
JitFunction
function_gaussian_build(OLattice<T>& dest ,OLattice<T>& r1 ,OLattice<T>& r2 )
{
  if (ptx_db::db_enabled) {
    JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (!func.empty())
      return func;
  }

  std::vector<ParamRef> params = jit_function_preamble_param("gaussian",__PRETTY_FUNCTION__);

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
  FuncRet_t r1_jit(forEach(r1, param_leaf, TreeCombine()));
  FuncRet_t r2_jit(forEach(r2, param_leaf, TreeCombine()));

  llvm::Value * r_idx = jit_function_preamble_get_idx( params );

  typedef typename REGType< typename JITType<T>::Type_t >::Type_t TREG;
  TREG r1_reg;
  TREG r2_reg;
  r1_reg.setup( r1_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );
  r2_reg.setup( r2_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );

  fill_gaussian( dest_jit.elem(JitDeviceLayout::Coalesced , r_idx ) , r1_reg , r2_reg );

  return jit_get_function();
}


template<class T>
void 
function_gaussian_exec(JitFunction function, OLattice<T>& dest,OLattice<T>& r1,OLattice<T>& r2, const Subset& s )
{
  if (s.numSiteTable() < 1)
    return;

  AddressLeaf addr_leaf(s);

  forEach(dest, addr_leaf, NullCombine());
  forEach(r1, addr_leaf, NullCombine());
  forEach(r2, addr_leaf, NullCombine());

  int start = s.start();
  int end = s.end();
  bool ordered = s.hasOrderedRep();
  int th_count = ordered ? s.numSiteTable() : Layout::sitesOnNode();

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
  
  jit_launch(function,Layout::sitesOnNode(),ids);
#if 0
  void * subset_member = QDP_get_global_cache().getDevicePtr( s.getIdMemberTable() );
  
  std::vector<void*> addr;
  addr.push_back( &ordered );
  addr.push_back( &th_count );
  addr.push_back( &start );
  addr.push_back( &end );
  addr.push_back( &subset_member );
  for(unsigned i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
  }
  jit_launch(function,th_count,addr);
#endif
}



}

#endif
