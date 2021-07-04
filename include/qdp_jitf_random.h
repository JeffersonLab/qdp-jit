#ifndef QDP_JITFUNC_RANDOM_H
#define QDP_JITFUNC_RANDOM_H

namespace QDP {


  template<class T>
  void
  function_random_build( JitFunction& function, OLattice<T>& dest , Seed& seed_tmp, LatticeSeed& latSeed, LatticeSeed& skewedSeed)
  {
    if (ptx_db::db_enabled)
      {
	llvm_ptx_db( function , __PRETTY_FUNCTION__ );
	if (!function.empty())
	  return;
      }


    llvm_start_new_function("random",__PRETTY_FUNCTION__);

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    ParamLeaf param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

    // RNG::ran_seed
    typedef typename LeafFunctor<Seed, ParamLeaf>::Type_t  SeedJIT;
    typedef typename LeafFunctor<LatticeSeed, ParamLeaf>::Type_t  LatticeSeedJIT;
    typedef typename REGType<typename SeedJIT::Subtype_t>::Type_t PSeedREG;

    //SeedJIT ran_seed_jit(forEach(RNG::get_RNG_Internals()->ran_seed, param_leaf, TreeCombine()));
    SeedJIT seed_tmp_jit(forEach(seed_tmp, param_leaf, TreeCombine()));
    SeedJIT ran_mult_n_jit(forEach(RNG::get_RNG_Internals()->ran_mult_n, param_leaf, TreeCombine()));
    LatticeSeedJIT lattice_ran_mult_jit(forEach( RNG::get_RNG_Internals()->lattice_ran_mult , param_leaf, TreeCombine()));
    LatticeSeedJIT skewedSeed_jit(forEach( skewedSeed , param_leaf, TreeCombine()));
    LatticeSeedJIT latSeed_jit(forEach( latSeed , param_leaf, TreeCombine()));
  
    llvm::Value * r_lo     = llvm_derefParam( p_lo );
    llvm::Value * r_hi     = llvm_derefParam( p_hi );

    llvm::Value * r_idx_thread = llvm_thread_idx();

    llvm::Value * r_out_of_range       = llvm_gt( r_idx_thread , llvm_sub( r_hi , r_lo ) );
    llvm_cond_exit(  r_out_of_range );

    llvm::Value * r_idx = llvm_add( r_lo , r_idx_thread );

    PSeedREG seed_reg;
    PSeedREG ran_mult_n_reg;
    PSeedREG lattice_ran_mult_reg;

    seed_reg.setup( latSeed_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );

    lattice_ran_mult_reg.setup( lattice_ran_mult_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );

    skewedSeed_jit.elem( JitDeviceLayout::Coalesced , r_idx ) = seed_reg * lattice_ran_mult_reg;

    ran_mult_n_reg.setup( ran_mult_n_jit.elem() );

    fill_random_jit( dest_jit.elem(JitDeviceLayout::Coalesced,r_idx) , latSeed_jit.elem( JitDeviceLayout::Coalesced , r_idx ) , skewedSeed_jit.elem( JitDeviceLayout::Coalesced , r_idx ) , ran_mult_n_reg );

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

    //forEach(RNG::get_RNG_Internals()->ran_seed, addr_leaf, NullCombine());
    //forEach(seed_tmp, addr_leaf, NullCombine()); // Caution: ParamLeaf treats OScalar as read-only
    addr_leaf.setId( seed_tmp_id );
    forEach(RNG::get_RNG_Internals()->ran_mult_n, addr_leaf, NullCombine());
    forEach(RNG::get_RNG_Internals()->lattice_ran_mult, addr_leaf, NullCombine());
    forEach(skewedSeed, addr_leaf, NullCombine());
    forEach(ranSeed, addr_leaf, NullCombine());

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

  

} //QDP

#endif
