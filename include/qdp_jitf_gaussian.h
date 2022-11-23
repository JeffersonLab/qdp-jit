#ifndef QDP_JITF_GAUSS_H
#define QDP_JITF_GAUSS_H


namespace QDP {

template<class T>
void
function_gaussian_build( JitFunction& function, OLattice<T>& dest ,OLattice<T>& r1 ,OLattice<T>& r2 )
{
  llvm_start_new_function("gaussian",__PRETTY_FUNCTION__);

  WorkgroupGuard workgroupGuard;
  ParamRef p_site_table = llvm_add_param<int*>();

  ParamLeafScalar param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeafScalar>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
  FuncRet_t r1_jit(forEach(r1, param_leaf, TreeCombine()));
  FuncRet_t r2_jit(forEach(r2, param_leaf, TreeCombine()));

  llvm::Value* r_idx_thread = llvm_thread_idx();
  workgroupGuard.check(r_idx_thread);

  llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );
  
  typedef typename REGType< typename ScalarType< typename JITType<T>::Type_t >::Type_t >::Type_t TREG;
  TREG r1_reg;
  TREG r2_reg;
  r1_reg.setup( r1_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );
  r2_reg.setup( r2_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );

  fill_gaussian( dest_jit.elem(JitDeviceLayout::Coalesced , r_idx ) , r1_reg , r2_reg );

  jit_get_function(function);
}


template<class T>
void 
function_gaussian_exec(JitFunction& function, OLattice<T>& dest,OLattice<T>& r1,OLattice<T>& r2, const Subset& s )
{
  if (s.numSiteTable() < 1)
    return;

#ifdef QDP_DEEP_LOG
  function.type_W = typeid(typename WordType<T>::Type_t).name();
  function.set_dest_id( dest.getId() );
  function.set_is_lat(true);
#endif

  AddressLeaf addr_leaf(s);

  forEach(dest, addr_leaf, NullCombine());
  forEach(r1, addr_leaf, NullCombine());
  forEach(r2, addr_leaf, NullCombine());

  int th_count = s.numSiteTable();
  WorkgroupGuardExec workgroupGuardExec(th_count);

  std::vector<QDPCache::ArgKey> ids;
  workgroupGuardExec.check(ids);
  ids.push_back( s.getIdSiteTable() );
  for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
    ids.push_back( addr_leaf.ids[i] );
  
  jit_launch(function,th_count,ids);

#ifdef QDP_DEEP_LOG
    jit_deep_log(function);
#endif
}



}

#endif
