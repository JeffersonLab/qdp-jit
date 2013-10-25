#ifndef QDP_JITFUNCUTIL_H
#define QDP_JITFUNCUTIL_H

namespace QDP {

  //llvm::Value *jit_function_preamble_get_idx( const std::vector<ParamRef>& vec );

  std::vector<ParamRef> jit_function_preamble_param();

  void * jit_function_epilogue_get(const char *);

  void jit_build_seedToFloat();
  void jit_build_seedMultiply();

  std::vector<llvm::Value *> llvm_seedMultiply( llvm::Value* a0 , llvm::Value* a1 , llvm::Value* a2 , llvm::Value* a3 , 
						llvm::Value* a4 , llvm::Value* a5 , llvm::Value* a6 , llvm::Value* a7 );

  llvm::Value * llvm_seedToFloat( llvm::Value* a0,llvm::Value* a1,llvm::Value* a2,llvm::Value* a3);



  class JitMainLoop {
  public:

    JitMainLoop() {
      llvm_start_new_function();
      p_lo           = llvm_add_param<int>();
      p_hi           = llvm_add_param<int>();
    }

    llvm::Value* getIdx() {
      r_lo        = llvm_derefParam( p_lo );
      r_hi          = llvm_derefParam( p_hi );

      block_entry_point = llvm_get_insert_point();
      block_site_loop = llvm_new_basic_block();
      block_site_loop_exit = llvm_new_basic_block();
      block_end_loop_body = llvm_new_basic_block();

      llvm_branch( block_site_loop );
      llvm_set_insert_point(block_site_loop);

      r_idx      = llvm_phi( r_lo->getType() , 2 );
      return r_idx;
    }

    void done() {
      llvm_branch( block_end_loop_body );
      llvm_set_insert_point( block_end_loop_body );

      r_idx_new = llvm_add( r_idx , llvm_create_value(1) );

      r_idx->addIncoming( r_idx_new , block_end_loop_body );
      r_idx->addIncoming( r_lo , block_entry_point );

      llvm::Value * r_exit_cond = llvm_ge( r_idx_new , r_hi );

      llvm_cond_branch( r_exit_cond , block_site_loop_exit , block_site_loop );

      llvm_set_insert_point(block_site_loop_exit);
    }

  private:
    ParamRef p_lo;
    ParamRef p_hi;
    llvm::PHINode* r_idx;
    llvm::Value*   r_idx_new;
    llvm::BasicBlock * block_end_loop_body;
    llvm::BasicBlock * block_entry_point;
    llvm::BasicBlock * block_site_loop;
    llvm::BasicBlock * block_site_loop_exit;
    llvm::Value * r_lo;
    llvm::Value * r_hi;

  };


  void jit_get_empty_arguments(AddressLeaf& addr_leaf);



} // namespace

#endif
