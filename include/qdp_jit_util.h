#ifndef QDP_JITFUNCUTIL_H
#define QDP_JITFUNCUTIL_H

namespace QDP {


  //std::vector<ParamRef> jit_function_preamble_param();

  void * jit_function_epilogue_get(const char *);

  void jit_build_seedToFloat();
  void jit_build_seedMultiply();

  std::vector<llvm::Value *> llvm_seedMultiply( llvm::Value* a0 , llvm::Value* a1 , llvm::Value* a2 , llvm::Value* a3 , 
						llvm::Value* a4 , llvm::Value* a5 , llvm::Value* a6 , llvm::Value* a7 );

  llvm::Value * llvm_seedToFloat( llvm::Value* a0,llvm::Value* a1,llvm::Value* a2,llvm::Value* a3);



  class JitMainLoop {
  private:

    void do_preamble() {
      if (done_preamble)
	return;
      done_preamble = true;
      r_lo_in        = llvm_get_arg_lo();
      r_hi_in        = llvm_get_arg_hi();
      r_thread_num   = llvm_get_arg_myId();
      r_ordered      = llvm_get_arg_ordered();
      r_start        = llvm_get_arg_start();

      r_inner = llvm_create_value( int64_t( inner ) );

      if (!do_siteperm) 
	{
	  llvm::BasicBlock * block_ordered = llvm_new_basic_block();
	  llvm::BasicBlock * block_not_ordered = llvm_new_basic_block();
	  llvm::BasicBlock * block_ordered_exit = llvm_new_basic_block();
	  llvm::Value* r_lo_added;
	  llvm::Value* r_hi_added;
	  llvm_cond_branch( r_ordered , block_ordered , block_not_ordered );
	  {
	    llvm_set_insert_point(block_not_ordered);
	    llvm_branch( block_ordered_exit );
	  }
	  {
	    llvm_set_insert_point(block_ordered);
	    r_lo_added = llvm_add( r_lo_in , r_start );
	    r_hi_added = llvm_add( r_hi_in , r_start );
	    llvm_branch( block_ordered_exit );
	  }
	  llvm_set_insert_point(block_ordered_exit);
	  
	  r_lo = llvm_phi( r_lo_in->getType() , 2 );
	  r_hi = llvm_phi( r_hi_in->getType() , 2 );
	  
	  r_lo->addIncoming( r_lo_in , block_not_ordered );
	  r_hi->addIncoming( r_hi_in , block_not_ordered );
	  
	  r_lo->addIncoming( r_lo_added , block_ordered );
	  r_hi->addIncoming( r_hi_added , block_ordered );

	  r_lo_outer = llvm_div( r_lo , r_inner );
	  r_hi_outer = llvm_div( r_hi , r_inner );
	}
      else 
	{
	  r_lo_outer = llvm_div( r_lo_in , r_inner );
	  r_hi_outer = llvm_div( r_hi_in , r_inner );
	}
    }

  public:

    JitMainLoop() {
      do_siteperm = false;
      inner = getDataLayoutInnerSize();
      llvm_start_new_function();
      done_preamble = false;
    }

    JitMainLoop(int inner_a, bool do_siteperm_a) {
      do_siteperm = do_siteperm_a;
      inner = inner_a;
      llvm_start_new_function();
      done_preamble = false;
      if (do_siteperm) {
	p_sitetable = llvm_add_param<int*>();
      }
    }


    llvm::Value * getThreadNum() {
      do_preamble();
      return r_thread_num;
    }

    llvm::Value * getLo() {
      do_preamble();
      return r_lo;
    }

    IndexDomainVector getIdx() {
      do_preamble();

      block_entry_point = llvm_get_insert_point();
      block_site_loop_inner = llvm_new_basic_block();
      block_site_loop_inner_exit = llvm_new_basic_block();
      block_site_loop_outer = llvm_new_basic_block();
      block_site_loop_outer_exit = llvm_new_basic_block();
      block_end_loop_body = llvm_new_basic_block();

      llvm_branch( block_site_loop_outer );
      llvm_set_insert_point(block_site_loop_outer);

      r_idx_outer      = llvm_phi( r_lo_outer->getType() , 2 );

      llvm_branch( block_site_loop_inner );
      llvm_set_insert_point(block_site_loop_inner);

      r_idx_inner      = llvm_phi( r_inner->getType() , 2 );

      IndexDomainVector args;
      args.push_back( make_pair( Layout::sitesOnNode()/inner , r_idx_outer ) );
      args.push_back( make_pair( inner , r_idx_inner ) );

      if (do_siteperm) {
	assert( inner == 1 && "doing siteperm and inner is not 1. makes no sense.");
	llvm::Value * r_idx_site = llvm_array_type_indirection( p_sitetable , get_index_from_index_vector(args) );
	IndexDomainVector idx_new = get_scalar_index_vector_from_index( r_idx_site );
	return idx_new;
      }

      return args;
    }


    // llvm::Value* getIdx() {
    //   r_lo_in = llvm_derefParam( p_lo );
    //   r_hi_in = llvm_derefParam( p_hi );

    //   r_lo = llvm_phi( r_lo_in->getType() , 1 );
    //   r_hi = llvm_phi( r_hi_in->getType() , 1 );

    //   block_entry_point = llvm_get_insert_point();
    //   block_site_loop = llvm_new_basic_block();
    //   block_site_loop_exit = llvm_new_basic_block();
    //   block_end_loop_body = llvm_new_basic_block();

    //   llvm_branch( block_site_loop );
    //   llvm_set_insert_point(block_site_loop);

    //   r_lo->addIncoming( r_lo_in , block_site_loop );
    //   r_hi->addIncoming( r_hi_in , block_site_loop );

    //   r_idx      = llvm_phi( r_lo->getType() , 2 );
    //   return r_idx;
    // }




    void done() {
      llvm_branch(           block_end_loop_body );
      llvm_set_insert_point( block_end_loop_body );

      r_idx_inner_new = llvm_add( r_idx_inner , llvm_create_value(int64_t(1)) );
      r_idx_inner->addIncoming( r_idx_inner_new , block_end_loop_body );
      r_idx_inner->addIncoming( llvm_create_value(int64_t(0)) , block_site_loop_outer );

      llvm::Value * r_exit_cond_inner = llvm_ge( r_idx_inner_new , llvm_create_value( inner ) );
      llvm_cond_branch( r_exit_cond_inner , block_site_loop_inner_exit , block_site_loop_inner );
      llvm_set_insert_point(block_site_loop_inner_exit);

      r_idx_outer_new = llvm_add( r_idx_outer , llvm_create_value(int64_t(1)) );
      r_idx_outer->addIncoming( r_idx_outer_new , block_site_loop_inner_exit );
      r_idx_outer->addIncoming( r_lo_outer , block_entry_point );

      llvm::Value * r_exit_cond_outer = llvm_ge( r_idx_outer_new , r_hi_outer );
      llvm_cond_branch( r_exit_cond_outer , block_site_loop_outer_exit , block_site_loop_outer );

      llvm_set_insert_point(block_site_loop_outer_exit);
    }

  private:
    llvm::PHINode* r_idx_inner;
    llvm::PHINode* r_idx_outer;
    llvm::Value*   r_idx_inner_new;
    llvm::Value*   r_idx_outer_new;
    llvm::BasicBlock * block_entry_point;
    llvm::BasicBlock * block_site_loop_inner;
    llvm::BasicBlock * block_site_loop_inner_exit;
    llvm::BasicBlock * block_site_loop_outer;
    llvm::BasicBlock * block_site_loop_outer_exit;
    llvm::BasicBlock * block_end_loop_body;
    llvm::Value * r_ordered;
    llvm::Value * r_start;
    llvm::Value * r_lo_in;
    llvm::Value * r_hi_in;
    llvm::Value * r_thread_num;
    llvm::Value * r_lo_outer;
    llvm::Value * r_hi_outer;
    llvm::PHINode * r_lo;
    llvm::PHINode * r_hi;
    llvm::Value * r_inner;

    ParamRef p_sitetable;

    int inner;
    bool done_preamble;
    bool do_siteperm;
  };




} // namespace

#endif
