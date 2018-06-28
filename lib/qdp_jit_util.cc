#include "qdp.h"

namespace QDP {

  // extern llvm::IRBuilder<>  *builder;
  // extern llvm::Module       *Mod;


  llvm::Function    *func_seed2float;
  llvm::Function    *func_seedMultiply;

  // seedMultiply
  //
  // We build a function that takes 2 seeds (4 ints each)
  // and returns 1 seed (as a literal aggregate)
  //
  void jit_build_seedMultiply() {
    //assert(builder && "no builder");
    //assert(Mod && "no module");

    std::vector< llvm::Type* > vecArgTypes;

    vecArgTypes.push_back( llvm_get_builder()->getInt32Ty() );
    vecArgTypes.push_back( llvm_get_builder()->getInt32Ty() );
    vecArgTypes.push_back( llvm_get_builder()->getInt32Ty() );
    vecArgTypes.push_back( llvm_get_builder()->getInt32Ty() );

    vecArgTypes.push_back( llvm_get_builder()->getInt32Ty() );
    vecArgTypes.push_back( llvm_get_builder()->getInt32Ty() );
    vecArgTypes.push_back( llvm_get_builder()->getInt32Ty() );
    vecArgTypes.push_back( llvm_get_builder()->getInt32Ty() );

    llvm::Type* ret_types[] = { llvm_get_builder()->getInt32Ty(),
				llvm_get_builder()->getInt32Ty(),
				llvm_get_builder()->getInt32Ty(),
				llvm_get_builder()->getInt32Ty() };
    
    llvm::StructType* ret_type = llvm::StructType::get(llvm_get_context(), 
						       llvm::ArrayRef< llvm::Type * >( ret_types , 4 ) );

    llvm::FunctionType *funcType = llvm::FunctionType::get( ret_type , 
							    llvm::ArrayRef<llvm::Type*>( vecArgTypes.data() , 
											 vecArgTypes.size() ) ,
							    false );
    llvm::Function* F = llvm::Function::Create(funcType, llvm::Function::InternalLinkage, "seedMultiply", llvm_get_module().get() );

    std::vector< llvm::Value* > args;
    unsigned Idx = 0;
    for (llvm::Function::arg_iterator AI = F->arg_begin(), AE = F->arg_end() ; AI != AE ; ++AI, ++Idx) {
      std::ostringstream oss;
      oss << "arg" << Idx;
      AI->setName( oss.str() );
      args.push_back(&*AI);
    }

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(llvm_get_context(), "entrypoint", F);
    llvm_get_builder()->SetInsertPoint(entry);

    typedef RScalar<WordREG<int> >  T;
    PSeedREG<T> s1,s2;

    s1.elem(0).elem().setup( args[0] );
    s1.elem(1).elem().setup( args[1] );
    s1.elem(2).elem().setup( args[2] );
    s1.elem(3).elem().setup( args[3] );

    s2.elem(0).elem().setup( args[4] );
    s2.elem(1).elem().setup( args[5] );
    s2.elem(2).elem().setup( args[6] );
    s2.elem(3).elem().setup( args[7] );

    s1 = s1 * s2;

    llvm::Value* ret_val[] = { s1.elem(0).elem().get_val() ,
			       s1.elem(1).elem().get_val() ,
			       s1.elem(2).elem().get_val() ,
			       s1.elem(3).elem().get_val() };

    llvm_get_builder()->CreateAggregateRet( ret_val , 4 );

    func_seedMultiply = F;
  }


  std::vector<llvm::Value *> llvm_seedMultiply( llvm::Value* a0 , llvm::Value* a1 , llvm::Value* a2 , llvm::Value* a3 , 
						llvm::Value* a4 , llvm::Value* a5 , llvm::Value* a6 , llvm::Value* a7 ) {
    assert(a0 && "llvm_seedToFloat a0");
    assert(a1 && "llvm_seedToFloat a1");
    assert(a2 && "llvm_seedToFloat a2");
    assert(a3 && "llvm_seedToFloat a3");
    assert(a4 && "llvm_seedToFloat a4");
    assert(a5 && "llvm_seedToFloat a5");
    assert(a6 && "llvm_seedToFloat a6");
    assert(a7 && "llvm_seedToFloat a7");

    assert(func_seedMultiply && "llvm_seedMultiply func_seedMultiply");

    llvm::Value* pack[] = { a0,a1,a2,a3,a4,a5,a6,a7 };

    llvm::Value* ret_val = llvm_get_builder()->CreateCall( func_seedMultiply , llvm::ArrayRef< llvm::Value *>( pack ,  8 ) );

    std::vector<llvm::Value *> ret;
    ret.push_back( llvm_get_builder()->CreateExtractValue( ret_val , 0 ) );
    ret.push_back( llvm_get_builder()->CreateExtractValue( ret_val , 1 ) );
    ret.push_back( llvm_get_builder()->CreateExtractValue( ret_val , 2 ) );
    ret.push_back( llvm_get_builder()->CreateExtractValue( ret_val , 3 ) );

    return ret;
  }



  void jit_build_seedToFloat() {
    // assert(llvm_get_builder() && "no llvm_get_builder()");
    // assert(Mod && "no module");

    std::vector< llvm::Type* > vecArgTypes;
    vecArgTypes.push_back( llvm_get_builder()->getInt32Ty() );
    vecArgTypes.push_back( llvm_get_builder()->getInt32Ty() );
    vecArgTypes.push_back( llvm_get_builder()->getInt32Ty() );
    vecArgTypes.push_back( llvm_get_builder()->getInt32Ty() );

    llvm::FunctionType *funcType = llvm::FunctionType::get( llvm_get_builder()->getFloatTy(), 
							    llvm::ArrayRef<llvm::Type*>( vecArgTypes.data() , 
											 vecArgTypes.size() ) ,
							    false );
    llvm::Function* F = llvm::Function::Create(funcType, llvm::Function::InternalLinkage, "seedToFloat", llvm_get_module().get() );

    std::vector< llvm::Value* > args;
    unsigned Idx = 0;
    for (llvm::Function::arg_iterator AI = F->arg_begin(), AE = F->arg_end() ; AI != AE ; ++AI, ++Idx) {
      std::ostringstream oss;
      oss << "arg" << Idx;
      AI->setName( oss.str() );
      args.push_back(&*AI);
    }

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(llvm_get_context(), "entrypoint", F);
    llvm_get_builder()->SetInsertPoint(entry);

    typedef RScalar<WordREG<int> >  T;
    PSeedREG<T> s1;
    s1.elem(0).elem().setup( args[0] );
    s1.elem(1).elem().setup( args[1] );
    s1.elem(2).elem().setup( args[2] );
    s1.elem(3).elem().setup( args[3] );

     UnaryReturn<PSeedREG<T>, FnSeedToFloat>::Type_t  d; // QDP::PScalarREG<QDP::RScalarREG<QDP::WordREG<float> > >
    typedef  RealScalar<T>::Type_t  S;                                   // QDP::RScalarREG<QDP::WordREG<float> >

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

    llvm_get_builder()->CreateRet( d.elem().elem().get_val() );

    func_seed2float = F;
  }


  llvm::Value * llvm_seedToFloat( llvm::Value* a0 , llvm::Value* a1 , llvm::Value* a2 , llvm::Value* a3 ) {
    assert(a0 && "llvm_seedToFloat a0");
    assert(a1 && "llvm_seedToFloat a1");
    assert(a2 && "llvm_seedToFloat a2");
    assert(a3 && "llvm_seedToFloat a3");
    assert(func_seed2float && "llvm_seedToFloat func_seed2float");
    return llvm_get_builder()->CreateCall( func_seed2float , {a0,a1,a2,a3} );
  }


  void * jit_function_epilogue_get(const char * fname)
  {
    llvm_exit();
    return llvm_get_function( fname );
  }




} //namespace
