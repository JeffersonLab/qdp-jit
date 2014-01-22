#ifndef QDP_JITFUNC_H
#define QDP_JITFUNC_H

#define JIT_DO_MEMBER
//#undef JIT_DO_MEMBER


#include "qmp.h"

namespace QDP {



template<class T, class T1, class Op, class RHS>
void function_build(JitFunction& func, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs)
{
#ifdef LLVM_DEBUG
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif

  HasShift hasShift;
  int withShift = forEach(rhs, hasShift , BitOrCombine());

  QDPIO::cerr << "withShift = " << withShift << "\n";

  {
    JitMainLoop loop( getDataLayoutInnerSize() , false );  // no offnode shift

    ParamLeaf param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

    typename AddOpParam<Op,ParamLeaf>::Type_t op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

    typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
    View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

    IndexDomainVector idx = loop.getIdx();

    op_jit( dest_jit.elem( JitDeviceLayout::LayoutCoalesced , idx ), 
	    forEach(rhs_view, ViewLeaf( JitDeviceLayout::LayoutCoalesced , idx ), OpCombine()));

    loop.done();

    QDPIO::cerr << "function_build (no siteperm)\n";

    func.func().push_back( jit_function_epilogue_get("jit_eval.ptx") );
  }


  // I need to build a 2nd version of this function. This version 
  // is needed for
  //   * offnode shifts
  //   * unordered sets

  {
    // We set inner length to 1 no matter whether we use this
    // version for unordered subset or offnode shifts.

    JitMainLoop loop( getDataLayoutInnerSize() , true );

    ParamLeaf param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

    typename AddOpParam<Op,ParamLeaf>::Type_t op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

    typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
    View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

    IndexDomainVector idx = loop.getIdx();

    op_jit( dest_jit.elem( JitDeviceLayout::LayoutCoalesced , idx ), 
	    forEach(rhs_view, ViewLeaf( JitDeviceLayout::LayoutCoalesced , idx ), OpCombine()));

    loop.done();

    QDPIO::cerr << "function_build (siteperm)\n";

    func.func().push_back( jit_function_epilogue_get("jit_eval.ptx") );
  }
}




template<class T, class T1, class Op, class RHS>
void 
function_exec(const JitFunction& function, 
	      OLattice<T>& dest, 
	      const Op& op, 
	      const QDPExpr<RHS,OLattice<T1> >& rhs, 
	      const Subset& s)
{
  //QDPIO::cerr << __PRETTY_FUNCTION__ << "\n";

  //assert( s.hasOrderedRep() && "only ordered subsets are supported");

  //static int offnode_maps_previous_call = -1;
  ShiftPhase1 phase1(s);
  int offnode_maps = forEach(rhs, phase1 , BitOrCombine());

#if 0
  QDPIO::cerr << "offnode_maps_previous_call = " << offnode_maps_previous_call 
	      << "offnode_maps = " << offnode_maps << "\n";
#endif

#if 0
  if (offnode_maps_previous_call != -1) 
    if ( (offnode_maps_previous_call > 0  && offnode_maps == 0) ||
	 (offnode_maps_previous_call == 0 && offnode_maps >  0) )
      QDP_error_exit("implementation limitation: same expression template used with offnode and no offnode comms.");
  offnode_maps_previous_call = offnode_maps;
#endif
  
#ifdef LLVM_DEBUG
  QDPIO::cerr << "offnode_maps = " << offnode_maps << "\n";
#endif

  if (offnode_maps) 
    {
      int innerCount        = MasterMap::Instance().getCountInner(s,offnode_maps);
      int faceCount         = MasterMap::Instance().getCountFace(s,offnode_maps);
      const int *innerSites = MasterMap::Instance().getInnerSites(s,offnode_maps).slice();
      const int *faceSites  = MasterMap::Instance().getFaceSites(s,offnode_maps).slice();

      //QDPIO::cerr << "we have " << innerCount << " inner and " << faceCount << " face sites\n";

      // if (( innerCount % getDataLayoutInnerSize() ) ||
      // 	  ( faceCount  % getDataLayoutInnerSize() ))
      // 	QDP_error_exit("implementation limitation. innerCount=%d , faceCount=%d must be multiple of inner length=%d",
      // 		       innerCount,faceCount,getDataLayoutInnerSize());

      // QDPIO::cerr << "Inner sites: ";
      // for (int i = 0 ; i < innerCount ; ++i )
      // 	QDPIO::cerr << innerSites[i] << " ";
      // QDPIO::cerr << "\n";

      // QDPIO::cerr << "Face sites: ";
      // for (int i = 0 ; i < faceCount ; ++i )
      // 	QDPIO::cerr << faceSites[i] << " ";
      // QDPIO::cerr << "\n";

      AddressLeaf addr_leaf(s);

      AddressLeaf::Types t;
      t.ptr = const_cast<int*>( innerSites );
      addr_leaf.addr.push_back(t);

      int junk_dest = forEach(dest, addr_leaf, NullCombine());
      AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
      int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

      //QDPIO::cerr << "Calling function for inner sites\n";

      jit_dispatch(function.func().at(1),innerCount,false,0,addr_leaf); // 2nd function pointer is offnode version

      ShiftPhase2 phase2;
      forEach(rhs, phase2 , NullCombine());

      AddressLeaf addr_leaf_face(s);

      t.ptr = const_cast<int*>( faceSites );
      addr_leaf_face.addr.push_back(t);

      junk_dest = forEach(dest, addr_leaf_face, NullCombine());
      AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf_face);
      junk_rhs = forEach(rhs, addr_leaf_face , NullCombine());

      //QDPIO::cerr << "Calling function for face sites\n";

      jit_dispatch(function.func().at(1),faceCount,false,0,addr_leaf_face);
    }
  else
    {
      if (s.hasOrderedRep()) 
	{
	  AddressLeaf addr_leaf(s);

	  int junk_dest = forEach(dest, addr_leaf, NullCombine());
	  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
	  int junk_rhs = forEach(rhs, addr_leaf , NullCombine());

	  //QDPIO::cerr << "Calling function for ordered subset\n";

	  jit_dispatch(function.func().at(0),s.numSiteTable(),s.hasOrderedRep(),s.start(),addr_leaf);
	}
      else
	{
	  AddressLeaf addr_leaf(s);

	  AddressLeaf::Types t;
	  t.ptr = const_cast<int*>( s.siteTable().slice() );
	  addr_leaf.addr.push_back(t);


	  // QDPIO::cerr << "Sites: ";
	  // for (int i = 0 ; i < s.numSiteTable() ; ++i )
	  //   QDPIO::cerr << ((int*)t.ptr)[i] << " ";
	  // QDPIO::cerr << "\n";
	  

	  int junk_dest = forEach(dest, addr_leaf, NullCombine());
	  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
	  int junk_rhs = forEach(rhs, addr_leaf , NullCombine());

	  //QDPIO::cerr << "Calling function for not ordered subset\n";

	  jit_dispatch(function.func().at(1),s.numSiteTable(),s.hasOrderedRep(),s.start(),addr_leaf);
	}
    } 


#ifdef LLVM_DEBUG
  std::cout << "calling eval(Lattice,Lattice).. " << addr_leaf.addr.size() << "\n";  
#endif


}





template<class T, class T1, class Op, class RHS>
void function_lat_sca_build(JitFunction& func,OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
{
#ifdef LLVM_DEBUG
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif

  JitMainLoop loop;

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  typename AddOpParam<Op,ParamLeaf>::Type_t op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

  typedef typename ForEach<QDPExpr<RHS,OScalar<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

  IndexDomainVector idx = loop.getIdx();

  op_jit( dest_jit.elem( JitDeviceLayout::LayoutCoalesced , idx ), 
   	  forEach(rhs_view, ViewLeaf( JitDeviceLayout::LayoutScalar , idx ), OpCombine()));

  loop.done();

  func.func().push_back( jit_function_epilogue_get("jit_lat_sca.ptx") );
}




template<class T, class T1, class Op, class RHS>
void 
function_lat_sca_exec(const JitFunction& function, 
		      OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs, const Subset& s)
{
  assert( s.hasOrderedRep() );

  AddressLeaf addr_leaf(s);

  int junk_dest = forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

#ifdef LLVM_DEBUG
  std::cout << "calling eval(Lattice,Scalar)..\n";
#endif

  jit_dispatch(function.func().at(0),s.numSiteTable(),s.hasOrderedRep(),s.start(),addr_leaf);
}







template<class T>
void function_zero_rep_build(JitFunction& func,OLattice<T>& dest)
{
  JitMainLoop loop;

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  IndexDomainVector idx = loop.getIdx();

  zero_rep( dest_jit.elem(JitDeviceLayout::LayoutCoalesced,idx) );

  loop.done();

  func.func().push_back( jit_function_epilogue_get("jit_zero.ptx") );
}





template<class T>
void 
function_zero_rep_exec(const JitFunction& function, OLattice<T>& dest, const Subset& s )
{
  assert( s.hasOrderedRep() );

  AddressLeaf addr_leaf(s);

  int junk_0 = forEach(dest, addr_leaf, NullCombine());

#ifdef LLVM_DEBUG
  std::cout << "calling zero_rep(Lattice,Subset)..\n";
#endif

  jit_dispatch( function.func().at(0) , s.numSiteTable() , s.hasOrderedRep() , s.start(), addr_leaf );
}








#if 0
template<class T>
CUfunction
function_random_build(OLattice<T>& dest , Seed& seed_tmp)
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  llvm_start_new_function();

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

  SeedJIT ran_seed_jit(forEach(RNG::ran_seed, param_leaf, TreeCombine()));
  SeedJIT seed_tmp_jit(forEach(seed_tmp, param_leaf, TreeCombine()));
  SeedJIT ran_mult_n_jit(forEach(RNG::ran_mult_n, param_leaf, TreeCombine()));
  LatticeSeedJIT lattice_ran_mult_jit(forEach( *RNG::lattice_ran_mult , param_leaf, TreeCombine()));

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

  lattice_ran_mult_reg.setup( lattice_ran_mult_jit.elem( JitDeviceLayout::LayoutCoalesced , r_idx ) );

  skewed_seed_reg = seed_reg * lattice_ran_mult_reg;

  ran_mult_n_reg.setup( ran_mult_n_jit.elem() );

  fill_random( dest_jit.elem(JitDeviceLayout::LayoutCoalesced,r_idx) , seed_reg , skewed_seed_reg , ran_mult_n_reg );

  llvm::Value * r_save = llvm_eq( r_idx_thread , llvm_create_value(0) );

  llvm::BasicBlock * block_save = llvm_new_basic_block();
  llvm::BasicBlock * block_not_save = llvm_new_basic_block();
  llvm::BasicBlock * block_save_exit = llvm_new_basic_block();
  llvm_cond_branch( r_save , block_save , block_not_save );
  {
    llvm_set_insert_point(block_save);
    seed_tmp_jit.elem() = seed_reg;
    llvm_branch( block_save_exit );
  }
  {
    llvm_set_insert_point(block_not_save);
    llvm_branch( block_save_exit );
  }
  llvm_set_insert_point(block_save_exit);

  return jit_function_epilogue_get_cuf("jit_random.ptx");
}
#endif




template<class T, class T1, class RHS>
void function_gather_build( JitFunction& func , void* send_buf , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs )
{
#ifdef LLVM_DEBUG
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif

  typedef typename WordType<T1>::Type_t WT;

  JitMainLoop loop(1,false);

  ParamRef p_soffset = llvm_add_param<int*>();
  ParamRef p_sndbuf  = llvm_add_param<WT*>();

  ParamLeaf param_leaf;

  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view( forEach( rhs , param_leaf , TreeCombine() ) );

  typedef typename JITType< OLattice<T> >::Type_t DestView_t;
  DestView_t dest_jit( p_sndbuf );

  IndexDomainVector idx = loop.getIdx();

  llvm::Value * r_idx_site = llvm_array_type_indirection( p_soffset , get_index_from_index_vector(idx) );

  IndexDomainVector idx_vec_gather = get_index_vector_from_index( r_idx_site );

  OpAssign()( dest_jit.elem( JitDeviceLayout::LayoutScalar , idx ) , 
	      forEach(rhs_view, ViewLeaf( JitDeviceLayout::LayoutCoalesced , idx_vec_gather ) , OpCombine() ) );

  loop.done();

  QDPIO::cerr << "function_gather_build\n";

  func.func().push_back( jit_function_epilogue_get("jit_gather.ll") );
}








template<class T1, class RHS>
void
function_gather_exec( const JitFunction& function, 
		      void * send_buf , 
		      const Map& map , 
		      const QDPExpr<RHS,OLattice<T1> >& rhs , 
		      const Subset& subset )
{
  AddressLeaf addr_leaf(subset);

  AddressLeaf::Types t;
  t.ptr = const_cast<int*>(map.soffset(subset).slice());
  addr_leaf.addr.push_back(t);

  t.ptr = send_buf;
  addr_leaf.addr.push_back(t);

  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

  //QDP_info("gather sites into send_buf lo=%d hi=%d",lo,hi);

#ifdef LLVM_DEBUG
  QDPIO::cerr << "calling gather.. number of sites to gather: " << map.soffset(subset).size() << "\n";
#endif

  jit_dispatch( function.func().at(0) , map.soffset(subset).size() , true , 0 , addr_leaf);
}







#if 0
template<class T>
void 
function_random_exec(CUfunction function, OLattice<T>& dest, const Subset& s , Seed& seed_tmp)
{
  if (!s.hasOrderedRep())
    QDP_error_exit("random on subset with unordered representation not implemented");

  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf;

  int junk_0 = forEach(dest, addr_leaf, NullCombine());

  int junk_1 = forEach(RNG::ran_seed, addr_leaf, NullCombine());
  int junk_2 = forEach(seed_tmp, addr_leaf, NullCombine());
  int junk_3 = forEach(RNG::ran_mult_n, addr_leaf, NullCombine());
  int junk_4 = forEach(*RNG::lattice_ran_mult, addr_leaf, NullCombine());

  // lo <= idx < hi
  int lo = s.start();
  int hi = s.end();

  std::vector<void*> addr;

  addr.push_back( &lo );
  //std::cout << "addr lo = " << addr[0] << " lo=" << lo << "\n";

  addr.push_back( &hi );
  //std::cout << "addr hi = " << addr[1] << " hi=" << hi << "\n";

  int addr_dest=addr.size();
  for(int i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr = " << addr_leaf.addr[i] << "\n";
  }

  jit_launch(function,s.numSiteTable(),addr);
}
#endif


}

#endif
