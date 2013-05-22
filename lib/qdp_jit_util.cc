#include "qdp.h"

namespace QDP {

  CUfunction jit_function_epilogue_get_cuf(const char * fname)
  {
    llvm_exit();
    return llvm_get_cufunction( fname );
  }


  std::vector<ParamRef> jit_function_preamble_param()
  {
    llvm_start_new_function();

    ParamRef p_ordered      = llvm_add_param<bool>();
    ParamRef p_th_count     = llvm_add_param<int>();
    ParamRef p_start        = llvm_add_param<int>();
    ParamRef p_end          = llvm_add_param<int>();
    ParamRef p_member_array = llvm_add_param<bool*>();

    return { p_ordered , p_th_count , p_start , p_end , p_member_array };
  }


  llvm::Value *jit_function_preamble_get_idx( const std::vector<ParamRef>& vec )
  {
    llvm::Value * r_ordered      = llvm_derefParam( vec[0] );
    llvm::Value * r_th_count     = llvm_derefParam( vec[1] );
    llvm::Value * r_start        = llvm_derefParam( vec[2] );
    llvm::Value * r_end          = llvm_derefParam( vec[3]);
    ParamRef      p_member_array = vec[4];

    llvm::Value * r_idx_phi0 = llvm_thread_idx();

    llvm::Value * r_idx_phi1;

    llvm_cond_exit( llvm_ge( r_idx_phi0 , r_th_count ) );

    llvm::BasicBlock * block_ordered = llvm_new_basic_block();
    llvm::BasicBlock * block_not_ordered = llvm_new_basic_block();
    llvm::BasicBlock * block_ordered_exit = llvm_new_basic_block();
    llvm::BasicBlock * cond_exit;
    llvm_cond_branch( r_ordered , block_ordered , block_not_ordered );
    {
      llvm_set_insert_point(block_not_ordered);
      llvm::Value* r_ismember     = llvm_array_type_indirection( p_member_array , r_idx_phi0 );
      llvm::Value* r_ismember_not = llvm_not( r_ismember );
      cond_exit = llvm_cond_exit( r_ismember_not ); 
      llvm_branch( block_ordered_exit );
    }
    {
      llvm_set_insert_point(block_ordered);
      r_idx_phi1 = llvm_add( r_idx_phi0 , r_start );
      llvm_branch( block_ordered_exit );
    }
    llvm_set_insert_point(block_ordered_exit);

    llvm::PHINode* r_idx = llvm_phi( r_idx_phi0->getType() , 2 );

    r_idx->addIncoming( r_idx_phi0 , cond_exit );
    r_idx->addIncoming( r_idx_phi1 , block_ordered );

    return r_idx;
  }




#if 0
  llvm::Value *jit_function_preamble_get_idx() 
  {

    llvm_start_new_function();

    llvm::Value * r_ordered      = llvm_add_param<bool>();
    llvm::Value * r_th_count     = llvm_add_param<int>();
    llvm::Value * r_start        = llvm_add_param<int>();
    llvm::Value * r_end          = llvm_add_param<int>();

    llvm::Value * r_idx_phi0 = llvm_thread_idx();

    llvm::Value * r_idx_phi1;

    llvm_cond_exit( llvm_ge( r_idx_phi0 , r_th_count ) );

    llvm::BasicBlock * block_ordered = llvm_new_basic_block();
    llvm::BasicBlock * block_not_ordered = llvm_new_basic_block();
    llvm::BasicBlock * block_ordered_exit = llvm_new_basic_block();
    llvm::BasicBlock * cond_exit;
    llvm_cond_branch( r_ordered , block_ordered , block_not_ordered );
    {
      llvm_set_insert_point(block_not_ordered);
      llvm::Value* r_ismember     = llvm_array_type_indirection<bool*>( r_idx_phi0 );
      llvm::Value* r_ismember_not = llvm_not( r_ismember );
      cond_exit = llvm_cond_exit( r_ismember_not ); 
      llvm_branch( block_ordered_exit );
    }
    {
      llvm_set_insert_point(block_ordered);
      r_idx_phi1 = llvm_add( r_idx_phi0 , r_start );
      llvm_branch( block_ordered_exit );
    }
    llvm_set_insert_point(block_ordered_exit);

    llvm::PHINode* r_idx = llvm_phi( r_idx_phi0->getType() , 2 );

    r_idx->addIncoming( r_idx_phi0 , cond_exit );
    r_idx->addIncoming( r_idx_phi1 , block_ordered );

    return r_idx;
  }
#endif






} //namespace
