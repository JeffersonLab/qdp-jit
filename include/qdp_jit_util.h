#ifndef QDP_JITFUNCUTIL_H
#define QDP_JITFUNCUTIL_H

namespace QDP {

  llvm::Value *jit_function_preamble_get_idx( const std::vector<ParamRef>& vec );
  std::vector<ParamRef> jit_function_preamble_param();

  CUfunction jit_function_epilogue_get_cuf(const char *fname, const char* pretty);

  void jit_build_seedToFloat();
  void jit_build_seedMultiply();


  void jit_stats_lattice2dev();
  void jit_stats_lattice2host();
  void jit_stats_jitted();

  long get_jit_stats_lattice2dev();
  long get_jit_stats_lattice2host();
  long get_jit_stats_jitted();

  std::vector<llvm::Value *> llvm_seedMultiply( llvm::Value* a0 , llvm::Value* a1 , llvm::Value* a2 , llvm::Value* a3 , 
						llvm::Value* a4 , llvm::Value* a5 , llvm::Value* a6 , llvm::Value* a7 );

  llvm::Value * llvm_seedToFloat( llvm::Value* a0,llvm::Value* a1,llvm::Value* a2,llvm::Value* a3);




} // namespace

#endif
