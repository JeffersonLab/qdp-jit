#ifndef QDP_JITFUNC_H
#define QDP_JITFUNC_H

#include "qmp.h"

namespace QDP {


template<class T, class T1, class Op, class RHS>
CUfunction
function_build(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs)
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  CUfunction func;

  Jit function("ptxtest.ptx","func");

  ParamLeaf param_leaf(function);

  // Destination
  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  // Now the arguments for the rhs
  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

  // Automatically build the function
  // This is where the site loop would go. This version completely unrolls it.
  // Instead, want a jit generated loop

  // The site loop is implemented as CUDA thread parallelization

  //printme(rhs_view);

  op(dest_jit.elem(0), forEach(rhs_view, ViewLeaf(0), OpCombine()));

  function.write();

  CUresult ret;
  CUmodule cuModule;
  ret = cuModuleLoad(&cuModule, "ptxtest.ptx");
  if (ret) { std::cout << "Error loading CUDA module\n"; exit(1); }

  ret = cuModuleGetFunction(&func, cuModule, "func");
  if (ret) { std::cout << "Error getting function\n"; exit(1); }

  std::cout << __PRETTY_FUNCTION__ << ": exiting\n";

  return func;
}


template<class T, class T1, class Op, class RHS>
void 
function_exec(CUfunction function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs)
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf;

  int junk_dst = forEach(dest, addr_leaf, NullCombine());
  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

  std::vector<void*> addr;
  for(int i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    std::cout << "addr=" << addr_leaf.addr[i] << "\n";
  }

  // Invoke kernel
  int threadsPerBlock = 1;
  int blocksPerGrid = (Layout::sitesOnNode() + threadsPerBlock - 1) / threadsPerBlock;

  std::cout << "blocksPerGrid=" << blocksPerGrid << "  threadsPerBlock=" << threadsPerBlock << "\n";

  cuLaunchKernel(function,
  		 blocksPerGrid, 1, 1, threadsPerBlock, 1, 1,
  		 0, 0, &addr[0] , 0);

  std::cout << __PRETTY_FUNCTION__ << ": exiting\n";
}


}

#endif
