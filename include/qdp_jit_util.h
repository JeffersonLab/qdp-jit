#ifndef QDP_JITFUNCUTIL_H
#define QDP_JITFUNCUTIL_H

namespace QDP {

  llvm::Value *jit_function_preamble_get_idx( const std::vector<ParamRef>& vec );
  std::vector<ParamRef> jit_function_preamble_param();

  //llvm::Value *jit_function_preamble_get_idx();

  CUfunction jit_function_epilogue_get_cuf(const char *);

} // namespace

#endif
