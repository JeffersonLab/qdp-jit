#ifndef QDP_JITFUNC_RANDOM_H
#define QDP_JITFUNC_RANDOM_H

namespace QDP {


  template<class T>
  void
  function_random_build( JitFunction& function, OLattice<T>& dest , Seed& seed_tmp, LatticeSeed& latSeed, LatticeSeed& skewedSeed)
  {
    llvm_start_new_function("random",__PRETTY_FUNCTION__);

    WorkgroupGuard workgroupGuard;
    ParamRef p_site_table = llvm_add_param<int*>();

    ParamLeafScalar param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeafScalar>::Type_t  FuncRet_t;
    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

    // RNG::ran_seed
    typedef typename LeafFunctor<Seed, ParamLeafScalar>::Type_t  SeedJIT;
    typedef typename LeafFunctor<LatticeSeed, ParamLeafScalar>::Type_t  LatticeSeedJIT;
    typedef typename REGType<typename SeedJIT::Subtype_t>::Type_t PSeedREG;

    SeedJIT seed_tmp_jit(forEach(seed_tmp, param_leaf, TreeCombine()));
    SeedJIT ran_mult_n_jit(forEach(RNG::get_RNG_Internals()->ran_mult_n, param_leaf, TreeCombine()));
    LatticeSeedJIT lattice_ran_mult_jit(forEach( RNG::get_RNG_Internals()->lattice_ran_mult , param_leaf, TreeCombine()));
    LatticeSeedJIT skewedSeed_jit(forEach( skewedSeed , param_leaf, TreeCombine()));
    LatticeSeedJIT latSeed_jit(forEach( latSeed , param_leaf, TreeCombine()));
  
    llvm::Value * r_idx_thread = llvm_thread_idx();
    workgroupGuard.check(r_idx_thread);

    llvm::Value* r_idx = llvm_array_type_indirection( p_site_table , r_idx_thread );

    PSeedREG seed_reg;
    PSeedREG ran_mult_n_reg;
    PSeedREG lattice_ran_mult_reg;

    seed_reg.setup( latSeed_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );

    lattice_ran_mult_reg.setup( lattice_ran_mult_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );

    skewedSeed_jit.elem( JitDeviceLayout::Coalesced , r_idx ) = seed_reg * lattice_ran_mult_reg;

    ran_mult_n_reg.setup( ran_mult_n_jit.elem() );

    fill_random_jit( dest_jit.elem(JitDeviceLayout::Coalesced,r_idx) ,
		     latSeed_jit.elem( JitDeviceLayout::Coalesced , r_idx ) ,
		     skewedSeed_jit.elem( JitDeviceLayout::Coalesced , r_idx ) ,
		     ran_mult_n_reg );

    PSeedREG tmp;                     //
    tmp.setup( latSeed_jit.elem( JitDeviceLayout::Coalesced , r_idx ) ); //
    JitIf save( llvm_eq( r_idx_thread , llvm_create_value(0) ) );
    {
      seed_tmp_jit.elem() = tmp;      // seed_reg
    }
    save.end();
  
    jit_get_function( function );
  }



  template<class T>
  void 
  function_random_exec(JitFunction& function, OLattice<T>& dest, const Subset& s , Seed& seed_tmp, LatticeSeed& ranSeed, LatticeSeed& skewedSeed)
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

#ifdef QDP_DEEP_LOG
    function.type_W = typeid(typename WordType<T>::Type_t).name();
    function.set_dest_id( dest.getId() );
    function.set_is_lat(true);
#endif

    // Register the seed_tmp object with the memory cache
    int seed_tmp_id = QDP_get_global_cache().addOwnHostMemStatus( sizeof(Seed) , seed_tmp.getF() , QDPCache::Status::undef );

    AddressLeaf addr_leaf(s);

    forEach(dest, addr_leaf, NullCombine());

    addr_leaf.setId( seed_tmp_id );
    forEach(RNG::get_RNG_Internals()->ran_mult_n, addr_leaf, NullCombine());
    forEach(RNG::get_RNG_Internals()->lattice_ran_mult, addr_leaf, NullCombine());
    forEach(skewedSeed, addr_leaf, NullCombine());
    forEach(ranSeed, addr_leaf, NullCombine());

    int th_count = s.numSiteTable();
    WorkgroupGuardExec workgroupGuardExec(th_count);
  
    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
    ids.push_back( s.getIdSiteTable() );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
      ids.push_back( addr_leaf.ids[i] );
 
    jit_launch(function,s.numSiteTable(),ids);

    // Copy seed_tmp to host
    QDP_get_global_cache().assureOnHost(seed_tmp_id);

    // Sign off seed_tmp
    QDP_get_global_cache().signoff( seed_tmp_id );

#ifdef QDP_DEEP_LOG
    jit_deep_log(function);
#endif
  }

  

} //QDP

#endif
