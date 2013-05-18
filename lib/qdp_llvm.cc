#include "qdp.h"

#include "qdp_libdevice_20.h"


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

  namespace llvm_counters {
    int param_counter;
    int label_counter;
  }



  //Imported PTX Unary operations single precision
  llvm::Function *func_sin_f32;
  llvm::Function *func_acos_f32;
  llvm::Function *func_asin_f32;
  llvm::Function *func_atan_f32;
  llvm::Function *func_ceil_f32;
  llvm::Function *func_floor_f32;
  llvm::Function *func_cos_f32;
  llvm::Function *func_cosh_f32;
  llvm::Function *func_exp_f32;
  llvm::Function *func_log_f32;
  llvm::Function *func_log10_f32;
  llvm::Function *func_sinh_f32;
  llvm::Function *func_tan_f32;
  llvm::Function *func_tanh_f32;
  llvm::Function *func_fabs_f32;
  llvm::Function *func_sqrt_f32;

  //Imported PTX Binary operations single precision
  llvm::Function *func_pow_f32;
  llvm::Function *func_atan2_f32;

  //Imported PTX Unary operations double precision
  llvm::Function *func_sin_f64;
  llvm::Function *func_acos_f64;
  llvm::Function *func_asin_f64;
  llvm::Function *func_atan_f64;
  llvm::Function *func_ceil_f64;
  llvm::Function *func_floor_f64;
  llvm::Function *func_cos_f64;
  llvm::Function *func_cosh_f64;
  llvm::Function *func_exp_f64;
  llvm::Function *func_log_f64;
  llvm::Function *func_log10_f64;
  llvm::Function *func_sinh_f64;
  llvm::Function *func_tan_f64;
  llvm::Function *func_tanh_f64;
  llvm::Function *func_fabs_f64;
  llvm::Function *func_sqrt_f64;

  //Imported PTX Binary operations double precision
  llvm::Function *func_pow_f64;
  llvm::Function *func_atan2_f64;


  llvm::Function *llvm_get_func( llvm::OwningPtr<llvm::Module>& M , const char * name )
  {
    llvm::Function *func = M->getFunction(name);
    if (!func)
      QDP_error_exit("Function %s not found.\n",name);
    QDP_info_primary("Found libdevice function: %s",name);
    return func;
  }


  void llvm_init_libdevice()
  {
    llvm::OwningPtr<llvm::Module> M;
    std::string ErrorMessage;

    llvm::outs() << "mapping " << (size_t)_usr_local_cuda_5_5_nvvm_libdevice_libdevice_compute_20_10_bc_len << " bytes\n";

    llvm::StringRef libdevice_20( (const char *)_usr_local_cuda_5_5_nvvm_libdevice_libdevice_compute_20_10_bc, 
				  (size_t)_usr_local_cuda_5_5_nvvm_libdevice_libdevice_compute_20_10_bc_len );

    M.reset( llvm::ParseBitcodeFile( llvm::MemoryBuffer::getMemBufferCopy( libdevice_20 ) ,
				     llvm::getGlobalContext() , 
				     &ErrorMessage ) 
	     );

    llvm::outs() << ErrorMessage << "\n";

    func_sin_f32 = llvm_get_func( M , "__nv_sinf" );
    func_acos_f32 = llvm_get_func( M , "__nv_acosf" );
    func_asin_f32 = llvm_get_func( M , "__nv_asinf" );
    func_atan_f32 = llvm_get_func( M , "__nv_atanf" );
    func_ceil_f32 = llvm_get_func( M , "__nv_ceilf" );
    func_floor_f32 = llvm_get_func( M , "__nv_floorf" );
    func_cos_f32 = llvm_get_func( M , "__nv_cosf" );
    func_cosh_f32 = llvm_get_func( M , "__nv_coshf" );
    func_exp_f32 = llvm_get_func( M , "__nv_expf" );
    func_log_f32 = llvm_get_func( M , "__nv_logf" );
    func_log10_f32 = llvm_get_func( M , "__nv_log10f" );
    func_sinh_f32 = llvm_get_func( M , "__nv_sinhf" );
    func_tan_f32 = llvm_get_func( M , "__nv_tanf" );
    func_tanh_f32 = llvm_get_func( M , "__nv_tanhf" );
    func_fabs_f32 = llvm_get_func( M , "__nv_fabsf" );
    func_sqrt_f32 = llvm_get_func( M , "__nv_fsqrt_rn" );


    func_pow_f32 = llvm_get_func( M , "__nv_powf" );
    func_atan2_f32 = llvm_get_func( M , "__nv_atan2f" );


    func_sin_f64 = llvm_get_func( M , "__nv_sin" );
    func_acos_f64 = llvm_get_func( M , "__nv_acos" );
    func_asin_f64 = llvm_get_func( M , "__nv_asin" );
    func_atan_f64 = llvm_get_func( M , "__nv_atan" );
    func_ceil_f64 = llvm_get_func( M , "__nv_ceil" );
    func_floor_f64 = llvm_get_func( M , "__nv_floor" );
    func_cos_f64 = llvm_get_func( M , "__nv_cos" );
    func_cosh_f64 = llvm_get_func( M , "__nv_cosh" );
    func_exp_f64 = llvm_get_func( M , "__nv_exp" );
    func_log_f64 = llvm_get_func( M , "__nv_log" );
    func_log10_f64 = llvm_get_func( M , "__nv_log10" );
    func_sinh_f64 = llvm_get_func( M , "__nv_sinh" );
    func_tan_f64 = llvm_get_func( M , "__nv_tan" );
    func_tanh_f64 = llvm_get_func( M , "__nv_tanh" );
    func_fabs_f64 = llvm_get_func( M , "__nv_fabs" );
    func_sqrt_f64 = llvm_get_func( M , "__nv_dsqrt_rn" );

    func_pow_f64 = llvm_get_func( M , "__nv_pow" );
    func_atan2_f64 = llvm_get_func( M , "__nv_atan2" );
  }


  void llvm_wrapper_init() {
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();

    llvm_type<float>::value  = llvm::Type::getFloatTy(llvm::getGlobalContext());
    llvm_type<double>::value = llvm::Type::getDoubleTy(llvm::getGlobalContext());
    llvm_type<int>::value    = llvm::Type::getIntNTy(llvm::getGlobalContext(),32);
    llvm_type<bool>::value   = llvm::Type::getIntNTy(llvm::getGlobalContext(),1);
    llvm_type<float*>::value  = llvm::Type::getFloatPtrTy(llvm::getGlobalContext());
    llvm_type<double*>::value = llvm::Type::getDoublePtrTy(llvm::getGlobalContext());
    llvm_type<int*>::value    = llvm::Type::getIntNPtrTy(llvm::getGlobalContext(),32);
    llvm_type<bool*>::value   = llvm::Type::getIntNPtrTy(llvm::getGlobalContext(),1);

    llvm_init_libdevice();
  }  


  llvm::BasicBlock * llvm_get_insert_block() {
    return builder->GetInsertBlock();
  }


  void llvm_start_new_function() {
    llvm::outs() << "Staring new LLVM function ...\n";

    Mod = new llvm::Module("module", llvm::getGlobalContext());
    builder = new llvm::IRBuilder<>(llvm::getGlobalContext());
    
    llvm::FunctionType *funcType = llvm::FunctionType::get(builder->getVoidTy(), false);
    mainFunc = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, "function", Mod);

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(llvm::getGlobalContext(), "entrypoint", mainFunc);
    builder->SetInsertPoint(entry);

    llvm_counters::param_counter = 0;
    llvm_counters::label_counter = 0;
  }


  llvm::PHINode * llvm_phi( llvm::Type* type, unsigned num )
  {
    return builder->CreatePHI( type , num );
  }


  llvm::Type* promote( llvm::Type* t0 , llvm::Type* t1 )
  {
    if ( t0->isFloatingPointTy() || t1->isFloatingPointTy() ) {
      llvm::outs() << "promote floating " << t0->isFloatingPointTy() << " " << t1->isFloatingPointTy() << "\n";
      if ( t0->isDoubleTy() || t1->isDoubleTy() ) {
	return llvm::Type::getDoubleTy(llvm::getGlobalContext());
      } else {
	return llvm::Type::getFloatTy(llvm::getGlobalContext());
      }
    } else {
      llvm::outs() << "promote int\n";
      unsigned upper = std::max( t0->getScalarSizeInBits() , t1->getScalarSizeInBits() );
      return llvm::Type::getIntNTy(llvm::getGlobalContext() , upper );
    }
  }


  llvm::Value* llvm_cast( llvm::Type *dest_type , llvm::Value *src )
  {
    assert( dest_type && "llvm_cast" );
    assert( src       && "llvm_cast" );
    dest_type->dump();
    src->dump();
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
      llvm::outs() << "lhs needs conversion\n";
      lhs = llvm_cast( args_type , lhs );
    }
    if ( args_type != rhs->getType() ) {
      llvm::outs() << "rhs needs conversion\n";
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
      llvm::outs() << "float binary op\n";
      return func_float( lhs , rhs );  
    }  else {
      llvm::outs() << "integer binary op\n";
      return func_int( lhs , rhs );  
    }
  }


  llvm::Value* llvm_u_op( std::function< llvm::Value *(llvm::Value *) > func_float,
			  std::function< llvm::Value *(llvm::Value *) > func_int,
			  llvm::Value* lhs )
  {
    if ( lhs->getType()->isFloatingPointTy() ) {
      llvm::outs() << "float unary op\n";
      return func_float( lhs );  
    }  else {
      llvm::outs() << "integer unary op\n";
      return func_int( lhs );  
    }
  }


  llvm::Value* llvm_rem( llvm::Value* lhs , llvm::Value* rhs ) {
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateFRem( lhs , rhs ); } , 
		      [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateSRem( lhs , rhs ); } , 
		      lhs , rhs ); }

  llvm::Value* llvm_shr( llvm::Value* lhs , llvm::Value* rhs ) {  
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs ) -> llvm::Value*{assert(!"Flt-pnt SHR makes no sense.");},
		      [](llvm::Value* lhs , llvm::Value* rhs ) -> llvm::Value*{ return builder->CreateAShr( lhs , rhs ); },
		      lhs , rhs ); }

  llvm::Value* llvm_shl( llvm::Value* lhs , llvm::Value* rhs ) {  
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs ) -> llvm::Value*{assert(!"Flt-pnt SHL makes no sense.");},
		      [](llvm::Value* lhs , llvm::Value* rhs ) -> llvm::Value*{ return builder->CreateShl( lhs , rhs ); },
		      lhs , rhs ); }

  llvm::Value* llvm_and( llvm::Value* lhs , llvm::Value* rhs ) {  
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs ) -> llvm::Value*{assert(!"Flt-pnt AND makes no sense.");},
		      [](llvm::Value* lhs , llvm::Value* rhs ) -> llvm::Value*{ return builder->CreateAnd( lhs , rhs ); },
		      lhs , rhs ); }

  llvm::Value* llvm_or( llvm::Value* lhs , llvm::Value* rhs ) {  
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs ) -> llvm::Value*{assert(!"Flt-pnt OR makes no sense.");},
		      [](llvm::Value* lhs , llvm::Value* rhs ) -> llvm::Value*{ return builder->CreateOr( lhs , rhs ); },
		      lhs , rhs ); }

  llvm::Value* llvm_xor( llvm::Value* lhs , llvm::Value* rhs ) {  
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs ) -> llvm::Value*{assert(!"Flt-pnt XOR makes no sense.");},
		      [](llvm::Value* lhs , llvm::Value* rhs ) -> llvm::Value*{ return builder->CreateXor( lhs , rhs ); },
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
    llvm::outs() << "not\n";
    return llvm_xor( llvm_create_value(-1) , lhs );
  }


  std::string param_next()
  {
    std::ostringstream oss;
    oss << "arg" << llvm_counters::param_counter++;
    llvm::outs() << "param_name = " << oss.str() << "\n";
    return oss.str();
  }


  llvm::Value * llvm_alloca( llvm::Type* type , int elements )
  {
    return builder->CreateAlloca( type , llvm_create_value(elements) );    // This can be a llvm::Value*
  }


  template<> llvm::Value *llvm_add_param<bool>() { 
    llvm::Argument * u8 = new llvm::Argument( llvm::Type::getInt8Ty(llvm::getGlobalContext()) , param_next() , mainFunc );
    return llvm_cast( llvm_type<bool>::value , u8 );
  }
  template<> llvm::Value *llvm_add_param<bool*>() { 
    return new llvm::Argument( llvm::Type::getInt1PtrTy(llvm::getGlobalContext()) , param_next() , mainFunc ); 
  }
  template<> llvm::Value *llvm_add_param<int>() { 
    return new llvm::Argument( llvm::Type::getInt32Ty(llvm::getGlobalContext()) , param_next() , mainFunc ); 
  }
  template<> llvm::Value *llvm_add_param<int*>() { 
    return new llvm::Argument( llvm::Type::getInt32PtrTy(llvm::getGlobalContext()) , param_next() , mainFunc ); 
  }
  template<> llvm::Value *llvm_add_param<float>() { 
    return new llvm::Argument( llvm::Type::getFloatTy(llvm::getGlobalContext()) , param_next() , mainFunc ); 
  }
  template<> llvm::Value *llvm_add_param<float*>() { 
    return new llvm::Argument( llvm::Type::getFloatPtrTy(llvm::getGlobalContext()) , param_next() , mainFunc ); 
  }
  template<> llvm::Value *llvm_add_param<double>() { 
    return new llvm::Argument( llvm::Type::getDoubleTy(llvm::getGlobalContext()) , param_next() , mainFunc ); 
  }
  template<> llvm::Value *llvm_add_param<double*>() { 
    return new llvm::Argument( llvm::Type::getDoublePtrTy(llvm::getGlobalContext()) , param_next() , mainFunc ); 
  }



  llvm::BasicBlock * llvm_new_basic_block()
  {
    std::ostringstream oss;
    oss << "L" << llvm_counters::label_counter++;
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(llvm::getGlobalContext(), oss.str() );
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


  llvm::BasicBlock * llvm_cond_exit( llvm::Value * cond )
  {
    llvm::BasicBlock * thenBB = llvm_new_basic_block();
    llvm::BasicBlock * elseBB = llvm_new_basic_block();
    llvm_cond_branch( cond , thenBB , elseBB );
    llvm_set_insert_point(thenBB);
    llvm_exit();
    llvm_set_insert_point(elseBB);
    return elseBB;
  }



  llvm::Value * llvm_create_value( double v )
  {
    if (sizeof(REAL) == 4)
      return llvm::ConstantFP::get( llvm::Type::getFloatTy(llvm::getGlobalContext()) , v );
    else
      return llvm::ConstantFP::get( llvm::Type::getDoubleTy(llvm::getGlobalContext()) , v );
  }

  llvm::Value * llvm_create_value(int v )  {return llvm::ConstantInt::get( llvm::Type::getInt32Ty(llvm::getGlobalContext()) , v );}
  llvm::Value * llvm_create_value(size_t v){return llvm::ConstantInt::get( llvm::Type::getInt32Ty(llvm::getGlobalContext()) , v );}
  llvm::Value * llvm_create_value(bool v ) {return llvm::ConstantInt::get( llvm::Type::getInt1Ty(llvm::getGlobalContext()) , v );}


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



  llvm::Value * llvm_special( const char * name )
  {
    llvm::FunctionType *IntrinFnTy = llvm::FunctionType::get(llvm::Type::getInt32Ty(llvm::getGlobalContext()), false);

    llvm::AttrBuilder ABuilder;
    ABuilder.addAttribute(llvm::Attribute::ReadNone);

    llvm::Constant *ReadTidX = Mod->getOrInsertTargetIntrinsic( name , 
								IntrinFnTy , 
								llvm::AttributeSet::get(llvm::getGlobalContext(), 
											llvm::AttributeSet::FunctionIndex, 
											ABuilder)
								);

    return builder->CreateCall(ReadTidX);
  }



  llvm::Value * llvm_call_special_tidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.tid.x"); }
  llvm::Value * llvm_call_special_ntidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.ntid.x"); }
  llvm::Value * llvm_call_special_ctaidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.ctaid.x"); }


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
#if 1
    Mod->dump();

    // Do optimizations ?
#if 1
    llvm::FunctionPassManager OurFPM( Mod );
    OurFPM.add(llvm::createBasicAliasAnalysisPass());
    OurFPM.add(llvm::createInstructionCombiningPass());
    OurFPM.add(llvm::createReassociatePass());
    OurFPM.add(llvm::createGVNPass());
    OurFPM.doInitialization();

    auto& func_list = Mod->getFunctionList();

    for(auto& x : func_list) {
      std::string tt( x.getName() );
      llvm::outs() << "running passes on: " << tt << "\n";
      OurFPM.run(x);
    }

    Mod->dump();
#endif

    //
    // Call NVPTX
    //
    llvm::Triple TheTriple;
    TheTriple.setArch(llvm::Triple::nvptx64);
    TheTriple.setVendor(llvm::Triple::UnknownVendor);
    TheTriple.setOS(llvm::Triple::Linux);
    TheTriple.setEnvironment(llvm::Triple::ELF);

    //Mod->setTargetTriple(TheTriple);

    std::string Error;
    const llvm::Target *TheTarget = llvm::TargetRegistry::lookupTarget( "nvptx64", TheTriple, Error);
    if (!TheTarget) {
      llvm::errs() << "Error looking up target: " << Error;
      exit(1);
    }


    // OwningPtr<TargetMachine> target(TheTarget->createTargetMachine(TheTriple.getTriple(),
    // 								 MCPU, FeaturesStr, Options ));

    llvm::TargetOptions Options;
    llvm::OwningPtr<llvm::TargetMachine> target(TheTarget->createTargetMachine(TheTriple.getTriple(),
									       "sm_20", "ptx31", Options ));

  
    assert(target.get() && "Could not allocate target machine!");
    llvm::TargetMachine &Target = *target.get();

    // Write ptx file
    {
      std::string error;
      unsigned OpenFlags = 0;
      //  if (Binary) OpenFlags |= raw_fd_ostream::F_Binary;
      // tool_output_file *FDOut = 
    
      llvm::OwningPtr<llvm::tool_output_file> Out( new llvm::tool_output_file( fname , error, OpenFlags) );
      // (GetOutputStream(TheTarget->getName(), TheTriple.getOS(), argv[0] ));
      if (!Out) {
	llvm::errs() << "Could not create OwningPtr<tool_output_file>\n";
	exit(1);
      }

      llvm::formatted_raw_ostream FOS(Out->os());

      llvm::PassManager PM;

#if 0
      // Add the target data from the target machine, if it exists, or the module.
      if (const DataLayout *TD = Target.getDataLayout()) {
	llvm::outs() << "Using target's data layout\n";
	PM.add(new DataLayout(*TD));
      }
      else {
	llvm::outs() << "Using module's data layout\n";
	PM.add(new DataLayout(Mod));
      }
#endif

      llvm::outs() << "Using module's data layout\n";
      PM.add(new llvm::DataLayout(Mod));


      // Ask the target to add backend passes as necessary.
      if (Target.addPassesToEmitFile(PM, FOS,  llvm::TargetMachine::CGFT_AssemblyFile )) {
	llvm::errs() << ": target does not support generation of this"
	       << " file type!\n";
	exit(1);
      }

      PM.run(*Mod);
      Out->keep();

    } // Call Out's destructor
#endif
    
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

    llvm::outs() << "Adding kernel metadata to function\n";
    addKernelMetadata( mainFunc );

    std::string ptx_kernel = llvm_get_ptx_kernel(fname);

    // llvm::outs() << "Deleting mainFunc\n";
    // delete mainFunc;
    // llvm::outs() << "Deleting builder\n";
    // delete builder;
    // llvm::outs() << "Deleting Mod\n";
    // delete Mod;

    ret = cuModuleLoadData(&cuModule, (void*)ptx_kernel.c_str());
    //ret = cuModuleLoadDataEx( &cuModule , ptx_kernel.c_str() , 0 , 0 , 0 );

    if (ret) {
      if (Layout::primaryNode()) {
	QDP_info_primary("Error loading external data. Dumping kernel to %s.",fname);
#if 1
	std::ofstream out(fname);
	out << ptx_kernel;
	out.close();
#endif
	QDP_error_exit("Abort.");
      }
    }

    ret = cuModuleGetFunction(&func, cuModule, "function");
    if (ret)
      QDP_error_exit("Error returned from cuModuleGetFunction. Abort.");

    return func;
  }

  llvm::Value* llvm_call_f32( llvm::Function* func , llvm::Value* lhs )
  {
    llvm::Value* lhs_f32 = llvm_cast( llvm_type<float>::value , lhs );
    return builder->CreateCall(func,lhs_f32);
  }

  llvm::Value* llvm_call_f32( llvm::Function* func , llvm::Value* lhs , llvm::Value* rhs )
  {
    llvm::Value* lhs_f32 = llvm_cast( llvm_type<float>::value , lhs );
    llvm::Value* rhs_f32 = llvm_cast( llvm_type<float>::value , rhs );
    return builder->CreateCall2(func,lhs_f32,rhs_f32);
  }

  llvm::Value* llvm_call_f64( llvm::Function* func , llvm::Value* lhs )
  {
    llvm::Value* lhs_f64 = llvm_cast( llvm_type<double>::value , lhs );
    return builder->CreateCall(func,lhs_f64);
  }

  llvm::Value* llvm_call_f64( llvm::Function* func , llvm::Value* lhs , llvm::Value* rhs )
  {
    llvm::Value* lhs_f64 = llvm_cast( llvm_type<float>::value , lhs );
    llvm::Value* rhs_f64 = llvm_cast( llvm_type<float>::value , rhs );
    return builder->CreateCall2(func,lhs_f64,rhs_f64);
  }

  llvm::Value* llvm_sin_f32( llvm::Value* lhs ) { return llvm_call_f32( func_sin_f32 , lhs ); }
  llvm::Value* llvm_acos_f32( llvm::Value* lhs ) { return llvm_call_f32( func_acos_f32 , lhs ); }
  llvm::Value* llvm_asin_f32( llvm::Value* lhs ) { return llvm_call_f32( func_asin_f32 , lhs ); }
  llvm::Value* llvm_atan_f32( llvm::Value* lhs ) { return llvm_call_f32( func_atan_f32 , lhs ); }
  llvm::Value* llvm_ceil_f32( llvm::Value* lhs ) { return llvm_call_f32( func_ceil_f32 , lhs ); }
  llvm::Value* llvm_floor_f32( llvm::Value* lhs ) { return llvm_call_f32( func_floor_f32 , lhs ); }
  llvm::Value* llvm_cos_f32( llvm::Value* lhs ) { return llvm_call_f32( func_cos_f32 , lhs ); }
  llvm::Value* llvm_cosh_f32( llvm::Value* lhs ) { return llvm_call_f32( func_cosh_f32 , lhs ); }
  llvm::Value* llvm_exp_f32( llvm::Value* lhs ) { return llvm_call_f32( func_exp_f32 , lhs ); }
  llvm::Value* llvm_log_f32( llvm::Value* lhs ) { return llvm_call_f32( func_log_f32 , lhs ); }
  llvm::Value* llvm_log10_f32( llvm::Value* lhs ) { return llvm_call_f32( func_log10_f32 , lhs ); }
  llvm::Value* llvm_sinh_f32( llvm::Value* lhs ) { return llvm_call_f32( func_sinh_f32 , lhs ); }
  llvm::Value* llvm_tan_f32( llvm::Value* lhs ) { return llvm_call_f32( func_tan_f32 , lhs ); }
  llvm::Value* llvm_tanh_f32( llvm::Value* lhs ) { return llvm_call_f32( func_tanh_f32 , lhs ); }
  llvm::Value* llvm_fabs_f32( llvm::Value* lhs ) { return llvm_call_f32( func_fabs_f32 , lhs ); }
  llvm::Value* llvm_sqrt_f32( llvm::Value* lhs ) { return llvm_call_f32( func_sqrt_f32 , lhs ); }

  llvm::Value* llvm_pow_f32( llvm::Value* lhs, llvm::Value* rhs ) { return llvm_call_f32( func_pow_f32 , lhs , rhs ); }
  llvm::Value* llvm_atan2_f32( llvm::Value* lhs, llvm::Value* rhs ) { return llvm_call_f32( func_atan2_f32 , lhs , rhs ); }

  llvm::Value* llvm_sin_f64( llvm::Value* lhs ) { return llvm_call_f64( func_sin_f64 , lhs ); }
  llvm::Value* llvm_acos_f64( llvm::Value* lhs ) { return llvm_call_f64( func_acos_f64 , lhs ); }
  llvm::Value* llvm_asin_f64( llvm::Value* lhs ) { return llvm_call_f64( func_asin_f64 , lhs ); }
  llvm::Value* llvm_atan_f64( llvm::Value* lhs ) { return llvm_call_f64( func_atan_f64 , lhs ); }
  llvm::Value* llvm_ceil_f64( llvm::Value* lhs ) { return llvm_call_f64( func_ceil_f64 , lhs ); }
  llvm::Value* llvm_floor_f64( llvm::Value* lhs ) { return llvm_call_f64( func_floor_f64 , lhs ); }
  llvm::Value* llvm_cos_f64( llvm::Value* lhs ) { return llvm_call_f64( func_cos_f64 , lhs ); }
  llvm::Value* llvm_cosh_f64( llvm::Value* lhs ) { return llvm_call_f64( func_cosh_f64 , lhs ); }
  llvm::Value* llvm_exp_f64( llvm::Value* lhs ) { return llvm_call_f64( func_exp_f64 , lhs ); }
  llvm::Value* llvm_log_f64( llvm::Value* lhs ) { return llvm_call_f64( func_log_f64 , lhs ); }
  llvm::Value* llvm_log10_f64( llvm::Value* lhs ) { return llvm_call_f64( func_log10_f64 , lhs ); }
  llvm::Value* llvm_sinh_f64( llvm::Value* lhs ) { return llvm_call_f64( func_sinh_f64 , lhs ); }
  llvm::Value* llvm_tan_f64( llvm::Value* lhs ) { return llvm_call_f64( func_tan_f64 , lhs ); }
  llvm::Value* llvm_tanh_f64( llvm::Value* lhs ) { return llvm_call_f64( func_tanh_f64 , lhs ); }
  llvm::Value* llvm_fabs_f64( llvm::Value* lhs ) { return llvm_call_f64( func_fabs_f64 , lhs ); }
  llvm::Value* llvm_sqrt_f64( llvm::Value* lhs ) { return llvm_call_f64( func_sqrt_f64 , lhs ); }

  llvm::Value* llvm_pow_f64( llvm::Value* lhs, llvm::Value* rhs ) { return llvm_call_f64( func_pow_f64 , lhs , rhs ); }
  llvm::Value* llvm_atan2_f64( llvm::Value* lhs, llvm::Value* rhs ) { return llvm_call_f64( func_atan2_f64 , lhs , rhs ); }



} // namespace QDP

