#ifndef QDP_JITF_GAUSS_H
#define QDP_JITF_GAUSS_H

#include "qmp.h"

namespace QDP {

template<class T>
CUfunction
function_gaussian_build(OLattice<T>& dest ,OLattice<T>& r1 ,OLattice<T>& r2 )
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
  //  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  CUfunction func;

  jit_start_new_function();

  jit_value r_ordered      = jit_add_param(  jit_ptx_type::pred );
  jit_value r_th_count     = jit_add_param(  jit_ptx_type::s32 );
  jit_value r_start        = jit_add_param(  jit_ptx_type::s32 );
  jit_value r_end          = jit_add_param(  jit_ptx_type::s32 );

  jit_value r_idx_thread = jit_geom_get_linear_th_idx();

  jit_ins_exit( jit_ins_ge( r_idx_thread , r_th_count ) );

  jit_value r_idx = r_idx_thread;

  jit_label_t label_ordered;
  jit_label_t label_ordered_exit;
  jit_ins_branch( label_ordered , r_ordered );
  {
    jit_value r_member = jit_add_param(  jit_ptx_type::u64 );  // Subset
    jit_value r_member_addr        = jit_ins_add( r_member , r_idx );   // I don't have to multiply with wordsize, since 1
    jit_value r_ismember           = jit_ins_load ( r_member_addr , 0 , jit_ptx_type::pred );
    jit_value r_ismember_not       = jit_ins_not( r_ismember );
    jit_ins_exit( r_ismember_not );
    jit_ins_branch( label_ordered_exit );
  }
  jit_ins_label(label_ordered);
  {
    r_idx = jit_ins_add( r_idx_thread , r_start );
  }
  jit_ins_label(label_ordered_exit);



  ParamLeaf param_leaf( r_idx );

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
  FuncRet_t r1_jit(forEach(r1, param_leaf, TreeCombine()));
  FuncRet_t r2_jit(forEach(r2, param_leaf, TreeCombine()));


  typedef typename REGType< typename JITType<T>::Type_t >::Type_t TREG;
  TREG r1_reg;
  TREG r2_reg;
  r1_reg.setup( r1_jit.elem( JitDeviceLayout::Coalesced ) );
  r2_reg.setup( r2_jit.elem( JitDeviceLayout::Coalesced ) );

  fill_gaussian( dest_jit.elem(JitDeviceLayout::Coalesced) , r1_reg , r2_reg );

  return jit_get_cufunction("ptx_gaussian.ptx");
#endif
}


template<class T>
void 
function_gaussian_exec(CUfunction function, OLattice<T>& dest,OLattice<T>& r1,OLattice<T>& r2, const Subset& s )
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf;

  int junk_0 = forEach(dest, addr_leaf, NullCombine());
  int junk_1 = forEach(r1, addr_leaf, NullCombine());
  int junk_2 = forEach(r2, addr_leaf, NullCombine());

  int start = s.start();
  int end = s.end();
  bool ordered = s.hasOrderedRep();
  int th_count = ordered ? s.numSiteTable() : Layout::sitesOnNode();

  void * subset_member = QDPCache::Instance().getDevicePtr( s.getIdMemberTable() );

  std::vector<void*> addr;

  addr.push_back( &ordered );
  //std::cout << "ordered = " << ordered << "\n";

  addr.push_back( &th_count );
  //std::cout << "thread_count = " << th_count << "\n";

  addr.push_back( &start );
  //std::cout << "start        = " << start << "\n";

  addr.push_back( &end );
  //std::cout << "end          = " << end << "\n";

  addr.push_back( &subset_member );
  //std::cout << "addr idx_inner_dev = " << addr[3] << " " << idx_inner_dev << "\n";

  int addr_dest=addr.size();
  for(int i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr = " << addr_leaf.addr[i] << "\n";
  }

  static int threadsPerBlock = 0;

  if (!threadsPerBlock) {
    // Auto tuning
    threadsPerBlock = jit_autotuning(function,0,th_count,&addr[0]);
  } else {
    //QDP_info_primary("Previous auto-tuning result = %d",threadsPerBlock);
  }

  //QDP_info("Launching kernel with %d threads",hi-lo);

  kernel_geom_t now = getGeom( th_count , threadsPerBlock );

  CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock,1,1,    0, 0, &addr[0] , 0);
#endif
}



}

#endif
