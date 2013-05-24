#include "qdp.h"

namespace QDP {


  extern llvm::IRBuilder<>  *builder;
  extern llvm::Module       *Mod;


  llvm::Function *jit_build_seedToFloat() {
    assert(builder && "no builder");
    assert(Mod && "no module");

    std::vector< llvm::Type* > vecArgTypes;
    vecArgTypes.push_back( builder->getInt32Ty() );
    vecArgTypes.push_back( builder->getInt32Ty() );
    vecArgTypes.push_back( builder->getInt32Ty() );
    vecArgTypes.push_back( builder->getInt32Ty() );

    llvm::FunctionType *funcType = llvm::FunctionType::get( builder->getFloatTy(), 
							    llvm::ArrayRef<llvm::Type*>( vecArgTypes.data() , 
											 vecArgTypes.size() ) ,
							    false );
    llvm::Function* F = llvm::Function::Create(funcType, llvm::Function::InternalLinkage, "seedToFloat", Mod);

    std::vector< llvm::Value* > args;
    unsigned Idx = 0;
    for (llvm::Function::arg_iterator AI = F->arg_begin(), AE = F->arg_end() ; AI != AE ; ++AI, ++Idx) {
      AI->setName( std::string("arg") + std::to_string(Idx) );
      args.push_back(AI);
    }

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(llvm::getGlobalContext(), "entrypoint", F);
    builder->SetInsertPoint(entry);

    typedef RScalar<WordREG<int> >  T;
    PSeedREG<T> s1;
    s1.elem(0).elem().setup( args[0] );
    s1.elem(1).elem().setup( args[1] );
    s1.elem(2).elem().setup( args[2] );
    s1.elem(3).elem().setup( args[3] );

    typename UnaryReturn<PSeedREG<T>, FnSeedToFloat>::Type_t  d; // QDP::PScalarREG<QDP::RScalarREG<QDP::WordREG<float> > >
    typedef typename RealScalar<T>::Type_t  S;                                   // QDP::RScalarREG<QDP::WordREG<float> >

    S  twom11(1.0 / 2048.0);
    S  twom12(1.0 / 4096.0);
    S  fs1, fs2;

    //  recast_rep(fs1, s1.elem(0));
    fs1 = S(s1.elem(0));
    d.elem() = twom12 * S(s1.elem(0));

    //  recast_rep(fs1, s1.elem(1));
    fs1 = S(s1.elem(1));
    fs2 = fs1 + d.elem();
    d.elem() = twom12 * fs2;

    //  recast_rep(fs1, s1.elem(2));
    fs1 = S(s1.elem(2));
    fs2 = fs1 + d.elem();
    d.elem() = twom12 * fs2;

    //  recast_rep(fs1, s1.elem(3));
    fs1 = S(s1.elem(3));
    fs2 = fs1 + d.elem();
    d.elem() = twom11 * fs2;

    builder->CreateRet( d.elem().elem().get_val() );

    return F;
  }


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
