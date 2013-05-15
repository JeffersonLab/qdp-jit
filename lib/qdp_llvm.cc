#include "qdp.h"

namespace QDP {

  llvm::IRBuilder<> *builder;
  llvm::BasicBlock  *entry;
  llvm::Function    *mainFunc;
  llvm::Module      *Mod;

  llvm::Type* llvm_type<float>::value;
  llvm::Type* llvm_type<double>::value;
  llvm::Type* llvm_type<int>::value;
  llvm::Type* llvm_type<bool>::value;
  llvm::Type* llvm_type<float*>::value;
  llvm::Type* llvm_type<double*>::value;
  llvm::Type* llvm_type<int*>::value;
  llvm::Type* llvm_type<bool*>::value;


  void llvm_wrapper_init() {
    llvm_type<float>::value  = llvm::Type::getFloatTy(getGlobalContext());
    llvm_type<double>::value = llvm::Type::getDoubleTy(getGlobalContext());
    llvm_type<int>::value    = llvm::Type::getIntNTy(getGlobalContext(),32);
    llvm_type<bool>::value   = llvm::Type::getIntNTy(getGlobalContext(),1);
    llvm_type<float*>::value  = llvm::Type::getFloatPtrTy(getGlobalContext());
    llvm_type<double*>::value = llvm::Type::getDoublePtrTy(getGlobalContext());
    llvm_type<int*>::value    = llvm::Type::getIntNPtrTy(getGlobalContext(),32);
    llvm_type<bool*>::value   = llvm::Type::getIntNPtrTy(getGlobalContext(),1);
  }  


  llvm::PHINode * llvm_phi( llvm::Type* type, unsigned num = 0 )
  {
    return builder->CreatePHI( type , num );
  }


  llvm::Type* promote( llvm::Type* t0 , llvm::Type* t1 )
  {
    if ( t0->isFloatingPointTy() || t1->isFloatingPointTy() ) {
      outs() << "promote floating " << t0->isFloatingPointTy() << " " << t1->isFloatingPointTy() << "\n";
      if ( t0->isDoubleTy() || t1->isDoubleTy() ) {
	return llvm::Type::getDoubleTy(getGlobalContext());
      } else {
	return llvm::Type::getFloatTy(getGlobalContext());
      }
    } else {
      outs() << "promote int\n";
      unsigned upper = std::max( t0->getScalarSizeInBits() , t1->getScalarSizeInBits() );
      return llvm::Type::getIntNTy(getGlobalContext() , upper );
    }
  }


  llvm::Value* llvm_cast( llvm::Type *dest_type , llvm::Value *src )
  {
    if ( src->getType() == dest_type)
      return src;

    return builder->CreateCast( llvm::CastInst::getCastOpcode( src , true , dest_type , true ) , 
				src , dest_type , "" );
  }



  std::tuple<llvm::Value*,llvm::Value*,llvm::Type*>
  llvm_normalize_values(llvm::Value* lhs , llvm::Value* rhs)
  {
    llvm::Type* args_type = promote( lhs->getType() , rhs->getType() );
    if ( args_type != lhs->getType() ) {
      outs() << "lhs needs conversion\n";
      lhs = llvm_cast( args_type , lhs );
    }
    if ( args_type != rhs->getType() ) {
      outs() << "rhs needs conversion\n";
      rhs = llvm_cast( args_type , rhs );
    }
    return std::tie(lhs,rhs,args_type);
  }



  llvm::Value* llvm_b_op( std::function< llvm::Value *(llvm::Value *, llvm::Value *) > func_float,
			  std::function< llvm::Value *(llvm::Value *, llvm::Value *) > func_int,
			  llvm::Value* lhs , llvm::Value* rhs )
  {
    llvm::Type* args_type;
    std::tie(lhs,rhs,args_type) = llvm_normalize_values(lhs,rhs);
    if ( args_type->isFloatingPointTy() ) {
      outs() << "float binary op\n";
      return func_float( lhs , rhs );  
    }  else {
      outs() << "integer binary op\n";
      return func_int( lhs , rhs );  
    }
  }


  llvm::Value* llvm_u_op( std::function< llvm::Value *(llvm::Value *) > func_float,
			  std::function< llvm::Value *(llvm::Value *) > func_int,
			  llvm::Value* lhs )
  {
    if ( lhs->getType()->isFloatingPointTy() ) {
      outs() << "float unary op\n";
      return func_float( lhs );  
    }  else {
      outs() << "integer unary op\n";
      return func_int( lhs );  
    }
  }


  llvm::Value* llvm_xor( llvm::Value* lhs , llvm::Value* rhs ) {  
    outs() << "xor lhs = ";
    lhs->getType()->dump();
    outs() << " rhs = ";
    rhs->getType()->dump();
    outs() << "\n";
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs ) -> llvm::Value*{assert(!"Floating point Xor makes no sense.");}, 
		      [](llvm::Value* lhs , llvm::Value* rhs ) -> llvm::Value*{ return builder->CreateXor( lhs , rhs ); } , 
		      lhs , rhs ); }


  llvm::Value* llvm_mul( llvm::Value* lhs , llvm::Value* rhs ) {
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateFMul( lhs , rhs ); } , 
		      [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateMul( lhs , rhs ); } , 
		      lhs , rhs ); }

  llvm::Value* llvm_add( llvm::Value* lhs , llvm::Value* rhs ) {
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateFAdd( lhs , rhs ); } , 
		      [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateAdd( lhs , rhs ); } , 
		      lhs , rhs ); }

  llvm::Value* llvm_sub( llvm::Value* lhs , llvm::Value* rhs ) {
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateFSub( lhs , rhs ); } , 
		      [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateSub( lhs , rhs ); } , 
		      lhs , rhs ); }

  llvm::Value* llvm_div( llvm::Value* lhs , llvm::Value* rhs ) {
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateFDiv( lhs , rhs ); } , 
		      [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateSDiv( lhs , rhs ); } , 
		      lhs , rhs ); }


  llvm::Value* llvm_eq( llvm::Value* lhs , llvm::Value* rhs ) {
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateFCmpOEQ( lhs , rhs ); } , 
		      [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateICmpEQ( lhs , rhs ); } , 
		      lhs , rhs ); }

  llvm::Value* llvm_ge( llvm::Value* lhs , llvm::Value* rhs ) {
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateFCmpOGE( lhs , rhs ); } , 
		      [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateICmpSGE( lhs , rhs ); } , 
		      lhs , rhs ); }

  llvm::Value* llvm_gt( llvm::Value* lhs , llvm::Value* rhs ) {
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateFCmpOGT( lhs , rhs ); } , 
		      [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateICmpSGT( lhs , rhs ); } , 
		      lhs , rhs ); }

  llvm::Value* llvm_le( llvm::Value* lhs , llvm::Value* rhs ) {
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateFCmpOLE( lhs , rhs ); } , 
		      [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateICmpSLE( lhs , rhs ); } , 
		      lhs , rhs ); }

  llvm::Value* llvm_lt( llvm::Value* lhs , llvm::Value* rhs ) {
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateFCmpOLT( lhs , rhs ); } , 
		      [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateICmpSLT( lhs , rhs ); } , 
		      lhs , rhs ); }

  //
  // Convenience function definitions
  //
  llvm::Value* llvm_not( llvm::Value* lhs ) {
    outs() << "not\n";
    return llvm_xor( llvm_create_value(-1) , lhs );
  }



  std::string param_next()
  {
    static int counter = 0;
    std::ostringstream oss;
    oss << "arg" << counter++;
    outs() << "param_name = " << oss.str() << "\n";
    return oss.str();
  }


  template<> llvm::Argument *llvm_add_param<bool>() { 
    return new llvm::Argument( llvm::Type::getInt1Ty(getGlobalContext()) , param_next() , mainFunc ); 
  }
  template<> llvm::Argument *llvm_add_param<bool*>() { 
    return new llvm::Argument( llvm::Type::getInt1PtrTy(getGlobalContext()) , param_next() , mainFunc ); 
  }
  template<> llvm::Argument *llvm_add_param<int>() { 
    return new llvm::Argument( llvm::Type::getInt32Ty(getGlobalContext()) , param_next() , mainFunc ); 
  }
  template<> llvm::Argument *llvm_add_param<int*>() { 
    return new llvm::Argument( llvm::Type::getInt32PtrTy(getGlobalContext()) , param_next() , mainFunc ); 
  }
  template<> llvm::Argument *llvm_add_param<float>() { 
    return new llvm::Argument( llvm::Type::getFloatTy(getGlobalContext()) , param_next() , mainFunc ); 
  }
  template<> llvm::Argument *llvm_add_param<float*>() { 
    return new llvm::Argument( llvm::Type::getFloatPtrTy(getGlobalContext()) , param_next() , mainFunc ); 
  }
  template<> llvm::Argument *llvm_add_param<double>() { 
    return new llvm::Argument( llvm::Type::getDoubleTy(getGlobalContext()) , param_next() , mainFunc ); 
  }
  template<> llvm::Argument *llvm_add_param<double*>() { 
    return new llvm::Argument( llvm::Type::getDoublePtrTy(getGlobalContext()) , param_next() , mainFunc ); 
  }



  llvm::BasicBlock * llvm_new_basic_block()
  {
    static unsigned counter = 0;
    std::ostringstream oss;
    oss << "L" << counter++;
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(getGlobalContext(), oss.str() );
    mainFunc->getBasicBlockList().push_back(BB);
    return BB;
  }


  void llvm_cond_branch(llvm::Value * cond, llvm::BasicBlock * thenBB, llvm::BasicBlock * elseBB)
  {
    cond = llvm_cast( llvm_type<bool>::value , cond );
    builder->CreateCondBr( cond , thenBB, elseBB);
  }


  void llvm_branch(llvm::BasicBlock * BB)
  {
    builder->CreateBr( BB );
  }


  void llvm_set_insert_point( llvm::BasicBlock * BB )
  {
    builder->SetInsertPoint(BB);
  }

  llvm::BasicBlock * llvm_get_insert_point()
  {
    return builder->GetInsertBlock();
  }


  void llvm_exit()
  {
    builder->CreateRetVoid();
  }


  void llvm_cond_exit( llvm::Value * cond )
  {
    llvm::BasicBlock * thenBB = llvm_new_basic_block();
    llvm::BasicBlock * elseBB = llvm_new_basic_block();
    llvm_cond_branch( cond , thenBB , elseBB );
    llvm_set_insert_point(thenBB);
    llvm_exit();
    llvm_set_insert_point(elseBB);
  }



  llvm::Value * llvm_create_value( double v )
  {
    if (sizeof(REAL) == 4)
      return llvm::ConstantFP::get( llvm::Type::getFloatTy(getGlobalContext()) , v );
    else
      return llvm::ConstantFP::get( llvm::Type::getDoubleTy(getGlobalContext()) , v );
  }

  llvm::Value * llvm_create_value(int v )  {return llvm::ConstantInt::get( llvm::Type::getInt32Ty(getGlobalContext()) , v );}
  llvm::Value * llvm_create_value(size_t v){return llvm::ConstantInt::get( llvm::Type::getInt32Ty(getGlobalContext()) , v );}
  llvm::Value * llvm_create_value(bool v ) {return llvm::ConstantInt::get( llvm::Type::getInt1Ty(getGlobalContext()) , v );}


  llvm::Value * llvm_createGEP( llvm::Value * ptr , llvm::Value * idx )
  {
    return builder->CreateGEP( ptr , idx );
  }


  llvm::Value * llvm_load( llvm::Value * ptr )
  {
    return builder->CreateLoad( ptr );
  }

  void llvm_store( llvm::Value * val , llvm::Value * ptr )
  {
    builder->CreateStore( val , ptr );
  }


  llvm::Value * llvm_load_ptr_idx( llvm::Value * ptr , llvm::Value * idx )
  {
    return llvm_load( llvm_createGEP( ptr , idx ) );
  }


  void llvm_store_ptr_idx( llvm::Value * val , llvm::Value * ptr , llvm::Value * idx )
  {
    llvm_store( val , llvm_createGEP( ptr , idx ) );
  }


  template<class T>
  llvm::Value* llvm_array_type_indirection( llvm::Value* idx )
  {
    llvm::Value* base = llvm_add_param<T>();
    llvm::Value* gep = llvm_createGEP( base , idx );
    return llvm_load( gep );
  }



  llvm::Value * llvm_special( const char * name )
  {
    FunctionType *IntrinFnTy = FunctionType::get(Type::getInt32Ty(getGlobalContext()), false);

    AttrBuilder ABuilder;
    ABuilder.addAttribute(llvm::Attribute::ReadNone);

    Constant *ReadTidX = Mod->getOrInsertTargetIntrinsic( name , 
							  IntrinFnTy , 
							  AttributeSet::get(getGlobalContext(), 
									    AttributeSet::FunctionIndex, 
									    ABuilder)
							  );

    return builder->CreateCall(ReadTidX);
  }



  llvm::Value * llvm_call_special_tidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.tid.x"); }
  llvm::Value * llvm_call_special_ntidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.ntid.x"); }
  llvm::Value * llvm_call_special_ctaidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.ctaidx.x"); }


  llvm::Value * llvm_thread_idx() { 
    return llvm_add( llvm_mul( llvm_call_special_ctaidx() , 
			       llvm_call_special_ntidx() ) , 
		     llvm_call_special_tidx() );
  }



  void addKernelMetadata(llvm::Function *F) {
    llvm::Module *M = F->getParent();
    llvm::LLVMContext &Ctx = M->getContext();

    // Get "nvvm.annotations" metadata node
    llvm::NamedMDNode *MD = M->getOrInsertNamedMetadata("nvvm.annotations");

    // Create !{<func-ref>, metadata !"kernel", i32 1} node
    llvm::SmallVector<llvm::Value *, 3> MDVals;
    MDVals.push_back(F);
    MDVals.push_back(llvm::MDString::get(Ctx, "kernel"));
    MDVals.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), 1));

    // Append metadata to nvvm.annotations
    MD->addOperand(llvm::MDNode::get(Ctx, MDVals));
  }




  std::string llvm_get_ptx_kernel(const char* fname)
  {
    FunctionPassManager OurFPM( Mod );
    OurFPM.add(createBasicAliasAnalysisPass());
    OurFPM.add(createInstructionCombiningPass());
    OurFPM.add(createReassociatePass());
    OurFPM.add(createGVNPass());
    OurFPM.doInitialization();

    auto& func_list = Mod->getFunctionList();

    for(auto& x : func_list) {
      std::string tt( x.getName() );
      outs() << "running passes on: " << tt << "\n";
      OurFPM.run(x);
    }

    Mod->dump();

#if 1
    //
    // Call NVPTX
    //
    Triple TheTriple;
    TheTriple.setArch(Triple::nvptx64);
    TheTriple.setVendor(Triple::UnknownVendor);
    TheTriple.setOS(Triple::Linux);
    TheTriple.setEnvironment(Triple::ELF);

    //Mod->setTargetTriple(TheTriple);

    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget( "nvptx64", TheTriple, Error);
    if (!TheTarget) {
      errs() << "Error looking up target: " << Error;
      exit(1);
    }


    // OwningPtr<TargetMachine> target(TheTarget->createTargetMachine(TheTriple.getTriple(),
    // 								 MCPU, FeaturesStr, Options ));

    TargetOptions Options;
    OwningPtr<TargetMachine> target(TheTarget->createTargetMachine(TheTriple.getTriple(),
								   "sm_20", "ptx31", Options ));

  
    assert(target.get() && "Could not allocate target machine!");
    TargetMachine &Target = *target.get();

    // Write ptx file
    {
      std::string error;
      unsigned OpenFlags = 0;
      //  if (Binary) OpenFlags |= raw_fd_ostream::F_Binary;
      // tool_output_file *FDOut = 
    
      OwningPtr<tool_output_file> Out( new tool_output_file( fname , error, OpenFlags) );
      // (GetOutputStream(TheTarget->getName(), TheTriple.getOS(), argv[0] ));
      if (!Out) {
	errs() << "Could not create OwningPtr<tool_output_file>\n";
	return 1;
      }

      formatted_raw_ostream FOS(Out->os());

      PassManager PM;

#if 0
      // Add the target data from the target machine, if it exists, or the module.
      if (const DataLayout *TD = Target.getDataLayout()) {
	outs() << "Using target's data layout\n";
	PM.add(new DataLayout(*TD));
      }
      else {
	outs() << "Using module's data layout\n";
	PM.add(new DataLayout(Mod));
      }
#endif

      outs() << "Using module's data layout\n";
      PM.add(new DataLayout(Mod));


      // Ask the target to add backend passes as necessary.
      if (Target.addPassesToEmitFile(PM, FOS,  TargetMachine::CGFT_AssemblyFile )) {
	errs() << argv[0] << ": target does not support generation of this"
	       << " file type!\n";
	return 1;
      }

      PM.run(*Mod);
      Out->keep();

    } // Call Out's destructor
    
    std::ifstream ifs( fname );
    std::string str((std::istreambuf_iterator<char>(ifs)),
		    std::istreambuf_iterator<char>());
    
    return str;
  }



  CUfunction llvm_get_cufunction(const char* fname)
  {
    CUfunction func;
    CUresult ret;
    CUmodule cuModule;

    std::string ptx_kernel = llvm_get_ptx_kernel(fname);

    ret = cuModuleLoadDataEx( &cuModule , ptx_kernel.c_str() , 0 , 0 , 0 );
    if (ret) {
      if (Layout::primaryNode()) {
	QDP_info_primary("Error loading external data. Dumping kernel to %s.",fname);
	std::ofstream out(fname);
	out << ptx_kernel;
	out.close();
	QDP_error_exit("Abort.");
      }
    }

    ret = cuModuleGetFunction(&func, cuModule, "function");
    if (ret)
      QDP_error_exit("Error returned from cuModuleGetFunction. Abort.");

    return func;
  }



} // namespace QDP

