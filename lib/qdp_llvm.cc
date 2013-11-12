#include "qdp.h"


#include "llvm/Support/raw_ostream.h"

namespace QDP {

  llvm::IRBuilder<> *builder;
  llvm::BasicBlock  *entry;
  llvm::Function    *mainFunc;
  llvm::Function    *mainFunc_extern;
  llvm::Module      *Mod;
  llvm::ExecutionEngine *TheExecutionEngine;

  llvm::TargetMachine *targetMachine;

  llvm::FunctionPassManager *TheFPM;

  std::string mcjit_error;

  void * fptr_mainFunc_extern;

  bool function_started;
  bool function_created;


  std::vector< llvm::Type* > vecParamType;
  std::vector< llvm::Value* > vecArgument;

  llvm::OwningPtr<llvm::Module> module_libdevice;

  llvm::Type* llvm_type<float>::value;
  llvm::Type* llvm_type<double>::value;
  llvm::Type* llvm_type<int>::value;
  llvm::Type* llvm_type<bool>::value;
  llvm::Type* llvm_type<float*>::value;
  llvm::Type* llvm_type<double*>::value;
  llvm::Type* llvm_type<int*>::value;
  llvm::Type* llvm_type<bool*>::value;

  namespace llvm_counters {
    int label_counter;
  }


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

  llvm::Function *func_pow_f32;
  llvm::Function *func_atan2_f32;

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

  llvm::Function *func_pow_f64;
  llvm::Function *func_atan2_f64;


  llvm::Function *llvm_get_func_float( const char * name )
  {
    return llvm::Function::Create( 
           llvm::FunctionType::get( 
           builder->getFloatTy(),llvm::ArrayRef<llvm::Type*>( builder->getFloatTy() ) , false) , 
           llvm::Function::ExternalLinkage, name , Mod );
  }

  llvm::Function *llvm_get_func_float_float( const char * name )
  {
    std::vector< llvm::Type* > args;
    args.push_back(builder->getFloatTy());
    args.push_back(builder->getFloatTy());
    return llvm::Function::Create( 
           llvm::FunctionType::get( 
           builder->getFloatTy(),llvm::ArrayRef<llvm::Type*>( args.data() , 2 ) , false) , 
           llvm::Function::ExternalLinkage, name , Mod );
  }

  llvm::Function *llvm_get_func_double( const char * name )
  {
    return llvm::Function::Create( 
           llvm::FunctionType::get( 
           builder->getDoubleTy(),llvm::ArrayRef<llvm::Type*>( builder->getDoubleTy() ) , false) , 
           llvm::Function::ExternalLinkage, name , Mod );
  }

  llvm::Function *llvm_get_func_double_double( const char * name )
  {
    std::vector< llvm::Type* > args;
    args.push_back(builder->getDoubleTy());
    args.push_back(builder->getDoubleTy());
    return llvm::Function::Create( 
           llvm::FunctionType::get( 
           builder->getDoubleTy(),llvm::ArrayRef<llvm::Type*>( args.data() , 2 ) , false) , 
           llvm::Function::ExternalLinkage, name , Mod );
  }




  void llvm_setup_math_functions() 
  {
    QDPIO::cerr << "Initializing math functions\n";

    func_sin_f32 = llvm_get_func_float( "sinf" );
    func_acos_f32 = llvm_get_func_float( "acosf" );
    func_asin_f32 = llvm_get_func_float( "asinf" );
    func_atan_f32 = llvm_get_func_float( "atanf" );
    func_ceil_f32 = llvm_get_func_float( "ceilf" );
    func_floor_f32 = llvm_get_func_float( "floorf" );
    func_cos_f32 = llvm_get_func_float( "cosf" );
    func_cosh_f32 = llvm_get_func_float( "coshf" );
    func_exp_f32 = llvm_get_func_float( "expf" );
    func_log_f32 = llvm_get_func_float( "logf" );
    func_log10_f32 = llvm_get_func_float( "log10f" );
    func_sinh_f32 = llvm_get_func_float( "sinhf" );
    func_tan_f32 = llvm_get_func_float( "tanf" );
    func_tanh_f32 = llvm_get_func_float( "tanhf" );
    func_fabs_f32 = llvm_get_func_float( "fabsf" );
    func_sqrt_f32 = llvm_get_func_float( "sqrtf" );

    func_pow_f32 = llvm_get_func_float_float( "powf" );
    func_atan2_f32 = llvm_get_func_float_float( "atan2f" );

    func_sin_f64 = llvm_get_func_double( "sin" );
    func_acos_f64 = llvm_get_func_double( "acos" );
    func_asin_f64 = llvm_get_func_double( "asin" );
    func_atan_f64 = llvm_get_func_double( "atan" );
    func_ceil_f64 = llvm_get_func_double( "ceil" );
    func_floor_f64 = llvm_get_func_double( "floor" );
    func_cos_f64 = llvm_get_func_double( "cos" );
    func_cosh_f64 = llvm_get_func_double( "cosh" );
    func_exp_f64 = llvm_get_func_double( "exp" );
    func_log_f64 = llvm_get_func_double( "log" );
    func_log10_f64 = llvm_get_func_double( "log10" );
    func_sinh_f64 = llvm_get_func_double( "sinh" );
    func_tan_f64 = llvm_get_func_double( "tan" );
    func_tanh_f64 = llvm_get_func_double( "tanh" );
    func_fabs_f64 = llvm_get_func_double( "fabs" );
    func_sqrt_f64 = llvm_get_func_double( "sqrt" );

    func_pow_f64 = llvm_get_func_double_double( "pow" );
    func_atan2_f64 = llvm_get_func_double_double( "atan2" );
  }


  void llvm_wrapper_init() {
    // llvm::InitializeAllTargets();
    // llvm::InitializeAllTargetMCs();

    // "-print-machineinstrs"

    char * argument = new char[128];
    const char *SetTinyVectorThreshold[] = {"program",argument};

    QDPIO::cerr << "Setting loop vectorizer minimum trip count to " << (int)getDataLayoutInnerSize() << "\n";
    sprintf( argument , "-vectorizer-min-trip-count=%d" , (int)getDataLayoutInnerSize() );
    llvm::cl::ParseCommandLineOptions(2, SetTinyVectorThreshold);

    QDPIO::cerr << "Forcing loop vectorizer vector width to " << (int)getDataLayoutInnerSize() << "\n";
    sprintf( argument , "-force-vector-width=%d" , (int)getDataLayoutInnerSize() );
    llvm::cl::ParseCommandLineOptions(2, SetTinyVectorThreshold);

    delete[] argument;

    llvm::InitializeNativeTarget();

    llvm::InitializeNativeTargetAsmPrinter(); // MCJIT
    llvm::InitializeNativeTargetAsmParser(); // MCJIT


    function_created = false;
    function_started = false;


    // llvm::InitializeAllAsmPrinters();
    // llvm::InitializeAllAsmParsers();

    llvm_type<float>::value  = llvm::Type::getFloatTy(llvm::getGlobalContext());
    llvm_type<double>::value = llvm::Type::getDoubleTy(llvm::getGlobalContext());
    llvm_type<int>::value    = llvm::Type::getIntNTy(llvm::getGlobalContext(),32);
    llvm_type<bool>::value   = llvm::Type::getIntNTy(llvm::getGlobalContext(),1);
    llvm_type<float*>::value  = llvm::Type::getFloatPtrTy(llvm::getGlobalContext());
    llvm_type<double*>::value = llvm::Type::getDoublePtrTy(llvm::getGlobalContext());
    llvm_type<int*>::value    = llvm::Type::getIntNPtrTy(llvm::getGlobalContext(),32);
    llvm_type<bool*>::value   = llvm::Type::getIntNPtrTy(llvm::getGlobalContext(),1);

  }  


  llvm::BasicBlock * llvm_get_insert_block() {
    return builder->GetInsertBlock();
  }


  void llvm_start_new_function() {
    assert( !function_started && "Function already started");
    function_started = true;
    function_created = false;

    QDP_info_primary( "Creating new module ...");

    Mod = new llvm::Module("module", llvm::getGlobalContext());

    // llvm::Triple TheTriple;
    // TheTriple.setArch(llvm::Triple::x86_64);
    // TheTriple.setVendor(llvm::Triple::UnknownVendor);
    // TheTriple.setOS(llvm::Triple::Linux);
    // TheTriple.setEnvironment(llvm::Triple::ELF);

    // Mod->setTargetTriple(TheTriple.getTriple());

    Mod->setTargetTriple(llvm::sys::getProcessTriple());

    llvm::EngineBuilder engineBuilder(Mod);
    engineBuilder.setMCPU(llvm::sys::getHostCPUName());
    engineBuilder.setEngineKind(llvm::EngineKind::JIT);
    engineBuilder.setOptLevel(llvm::CodeGenOpt::Aggressive);
    engineBuilder.setErrorStr(&mcjit_error);

    TheExecutionEngine = engineBuilder.setUseMCJIT(true).create(); // MCJIT
    
    assert(TheExecutionEngine && "failed to create LLVM ExecutionEngine with error");

    targetMachine = engineBuilder.selectTarget();



    QDP_info_primary( "Staring new LLVM function ...");

    builder = new llvm::IRBuilder<>(llvm::getGlobalContext());

    llvm_setup_math_functions();


    // jit_build_seedToFloat();
    // jit_build_seedMultiply();

    vecParamType.clear();
    vecArgument.clear();

    //llvm_setup_math_functions();

    // llvm::outs() << "------------------------- linked module\n";
    // llvm_print_module(Mod,"ir_linked.ll");
    //Mod->dump();
  }



  void llvm_create_function() {
    assert(!function_created && "Function already created");
    assert(function_started && "Function not started");
    assert(vecParamType.size()>0 && "vecParamType.size()>0");


    // Create the main function

    llvm::FunctionType *funcType = 
      llvm::FunctionType::get( builder->getVoidTy() , 
			       llvm::ArrayRef<llvm::Type*>( vecParamType.data() , vecParamType.size() ) , 
			       false); // no vararg
    mainFunc = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, "main", Mod);

    // Set argument names in 'main'

    //std::cout << "setting argument names in 'main'\n";

    unsigned Idx = 0;
    for (llvm::Function::arg_iterator AI = mainFunc->arg_begin(), AE = mainFunc->arg_end() ; AI != AE ; ++AI, ++Idx) {
      AI->setName( std::string("arg")+std::to_string(Idx) );

      if ( vecParamType.at(Idx)->isPointerTy() ) {
      	llvm::AttrBuilder B;
      	B.addAttribute(llvm::Attribute::NoAlias);
      	AI->addAttr( llvm::AttributeSet::get( llvm::getGlobalContext() , 0 ,  B ) );
      }

      vecArgument.push_back( AI );
    }

    llvm::BasicBlock* entry_main = llvm::BasicBlock::Create(llvm::getGlobalContext(), "entrypoint", mainFunc);
    builder->SetInsertPoint(entry_main);

    // if (Layout::primaryNode())
    //   mainFunc->dump();

    llvm_counters::label_counter = 0;
    function_created = true;

  }



  llvm::Value * llvm_derefParam( ParamRef r ) {
    if (!function_created)
      llvm_create_function();
    assert( vecArgument.size() > (int)r && "derefParam out of range");
    return vecArgument.at(r);
  }


  llvm::Value* llvm_array_type_indirection( ParamRef p , llvm::Value* idx )
  {
    llvm::Value* base = llvm_derefParam( p );
    llvm::Value* gep = llvm_createGEP( base , idx );
    return llvm_load( gep );
  }


  llvm::SwitchInst * llvm_switch( llvm::Value* val , llvm::BasicBlock* bb_default ) 
  {
    return builder->CreateSwitch( val , bb_default );
  }


  llvm::PHINode * llvm_phi( llvm::Type* type, unsigned num )
  {
    return builder->CreatePHI( type , num );
  }


  llvm::Type* promote( llvm::Type* t0 , llvm::Type* t1 )
  {
    if ( t0->isFloatingPointTy() || t1->isFloatingPointTy() ) {
      //llvm::outs() << "promote floating " << t0->isFloatingPointTy() << " " << t1->isFloatingPointTy() << "\n";
      if ( t0->isDoubleTy() || t1->isDoubleTy() ) {
	return llvm::Type::getDoubleTy(llvm::getGlobalContext());
      } else {
	return llvm::Type::getFloatTy(llvm::getGlobalContext());
      }
    } else {
      //llvm::outs() << "promote int\n";
      unsigned upper = std::max( t0->getScalarSizeInBits() , t1->getScalarSizeInBits() );
      return llvm::Type::getIntNTy(llvm::getGlobalContext() , upper );
    }
  }


  llvm::Value* llvm_cast( llvm::Type *dest_type , llvm::Value *src )
  {
    assert( dest_type && "llvm_cast" );
    assert( src       && "llvm_cast" );

    // llvm::outs() << "\ncast: dest_type  = "; dest_type->dump();
    // llvm::outs() << "\ncast: src->getType  = "; src->getType()->dump();
    
    if ( src->getType() == dest_type)
      return src;

    // llvm::outs() << "\ncast: dest_type is array = " << dest_type->isArrayTy() << "\n";
    // if (dest_type->isArrayTy()) {
    //   llvm::outs() << "\ncast: dest_type->getArrayElementTy() = "; 
    //   dest_type->getArrayElementType()->dump();
    // }

    if ( dest_type->isArrayTy() )
      if ( dest_type->getArrayElementType() == src->getType() )
	return src;

    if (!llvm::CastInst::isCastable( src->getType() , dest_type ))
      QDP_error_exit("not castable");

    llvm::Value* ret = builder->CreateCast( llvm::CastInst::getCastOpcode( src , true , dest_type , true ) , 
				src , dest_type , "" );
    return ret;
  }



  std::tuple<llvm::Value*,llvm::Value*,llvm::Type*>
  llvm_normalize_values(llvm::Value* lhs , llvm::Value* rhs)
  {
    llvm::Type* args_type = promote( lhs->getType() , rhs->getType() );
    if ( args_type != lhs->getType() ) {
      //llvm::outs() << "lhs needs conversion\n";
      lhs = llvm_cast( args_type , lhs );
    }
    if ( args_type != rhs->getType() ) {
      //llvm::outs() << "rhs needs conversion\n";
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
      //llvm::outs() << "float binary op\n";
      return func_float( lhs , rhs );  
    }  else {
      //llvm::outs() << "integer binary op\n";
      return func_int( lhs , rhs );  
    }
  }


  llvm::Value* llvm_u_op( std::function< llvm::Value *(llvm::Value *) > func_float,
			  std::function< llvm::Value *(llvm::Value *) > func_int,
			  llvm::Value* lhs )
  {
    if ( lhs->getType()->isFloatingPointTy() ) {
      return func_float( lhs );  
    }  else {
      return func_int( lhs );  
    }
  }


  llvm::Value* llvm_neg( llvm::Value* src ) {
    return llvm_b_op( [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateFSub( lhs , rhs ); } , 
		      [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateSub( lhs , rhs ); } , 
		      llvm_create_value(0) , src ); }


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
		      [](llvm::Value* lhs , llvm::Value* rhs) -> llvm::Value*{ return builder->CreateNSWAdd( lhs , rhs ); } , 
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
    //llvm::outs() << "not\n";
    return llvm_xor( llvm_create_value(-1) , lhs );
  }


  // std::string param_next()
  // {
  //   std::ostringstream oss;
  //   oss << "arg" << llvm_counters::param_counter++;
  //   llvm::outs() << "param_name = " << oss.str() << "\n";
  //   return oss.str();
  // }

  llvm::Value* llvm_get_shared_ptr( llvm::Type *ty ) {

    //

    llvm::GlobalVariable *gv = new llvm::GlobalVariable ( *Mod , 
							  llvm::ArrayType::get(ty,0) ,
							  false , 
							  llvm::GlobalVariable::ExternalLinkage, 
							  0, 
							  "shared_buffer", 
							  0, //GlobalVariable *InsertBefore=0, 
							  llvm::GlobalVariable::NotThreadLocal, //ThreadLocalMode=NotThreadLocal
							  3, // unsigned AddressSpace=0, 
							  false); //bool isExternallyInitialized=false)
    return builder->CreatePointerCast(gv, llvm::PointerType::get(ty,3) );
    //return builder->CreatePointerCast(gv,llvm_type<double*>::value);
    //return gv;
  }



  llvm::Value * llvm_alloca( llvm::Type* type , int elements )
  {
    return builder->CreateAlloca( type , llvm_create_value(elements) );    // This can be a llvm::Value*
  }


  template<> ParamRef llvm_add_param<bool>() { 
    vecParamType.push_back( llvm::Type::getInt1Ty(llvm::getGlobalContext()) );
    return vecParamType.size()-1;
    // llvm::Argument * u8 = new llvm::Argument( llvm::Type::getInt8Ty(llvm::getGlobalContext()) , param_next() , mainFunc );
    // return llvm_cast( llvm_type<bool>::value , u8 );
  }
  template<> ParamRef llvm_add_param<bool*>() { 
    vecParamType.push_back( llvm::Type::getInt1PtrTy(llvm::getGlobalContext()) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<std::int64_t>() { 
    vecParamType.push_back( llvm::Type::getInt64Ty(llvm::getGlobalContext()) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<int>() { 
    vecParamType.push_back( llvm::Type::getInt32Ty(llvm::getGlobalContext()) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<int*>() { 
    vecParamType.push_back( llvm::Type::getInt32PtrTy(llvm::getGlobalContext()) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<float>() { 
    vecParamType.push_back( llvm::Type::getFloatTy(llvm::getGlobalContext()) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<float*>() { 
    vecParamType.push_back( llvm::Type::getFloatPtrTy(llvm::getGlobalContext()) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<double>() { 
    vecParamType.push_back( llvm::Type::getDoubleTy(llvm::getGlobalContext()) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<double*>() { 
    vecParamType.push_back( llvm::Type::getDoublePtrTy(llvm::getGlobalContext()) );
    return vecParamType.size()-1;
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


  llvm::ConstantInt * llvm_create_const_int(int i) {
    return llvm::ConstantInt::getSigned( llvm::Type::getIntNTy(llvm::getGlobalContext(),32) , i );
  }

  llvm::Value * llvm_create_value( double v )
  {
    if (sizeof(REAL) == 4)
      return llvm::ConstantFP::get( llvm::Type::getFloatTy(llvm::getGlobalContext()) , v );
    else
      return llvm::ConstantFP::get( llvm::Type::getDoubleTy(llvm::getGlobalContext()) , v );
  }

  llvm::Value * llvm_create_value(std::int64_t v )  {return llvm::ConstantInt::get( llvm::Type::getInt64Ty(llvm::getGlobalContext()) , v );}
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
    assert(ptr->getType()->isPointerTy() && "llvm_store: not a pointer type");
    llvm::Value * val_cast = llvm_cast( ptr->getType()->getPointerElementType() , val );
    // llvm::outs() << "\nstore: val_cast  = "; val_cast->dump();
    // llvm::outs() << "\nstore: ptr  = "; ptr->dump();
    builder->CreateStore( val_cast , ptr );
  }


  llvm::Value * llvm_load_ptr_idx( llvm::Value * ptr , llvm::Value * idx )
  {
    return llvm_load( llvm_createGEP( ptr , idx ) );
  }


  void llvm_store_ptr_idx( llvm::Value * val , llvm::Value * ptr , llvm::Value * idx )
  {
    // llvm::outs() << "\nstore_ptr: val->getType  = "; val->getType()->dump();
    // llvm::outs() << "\nstore_ptr: ptr->getType  = "; ptr->getType()->dump();
    // llvm::outs() << "\nstore_ptr: idx->getType  = "; idx->getType()->dump();

    llvm_store( val , llvm_createGEP( ptr , idx ) );
  }



  void llvm_bar_sync()
  {
    llvm::FunctionType *IntrinFnTy = llvm::FunctionType::get(llvm::Type::getVoidTy(llvm::getGlobalContext()), false);

    llvm::AttrBuilder ABuilder;
    ABuilder.addAttribute(llvm::Attribute::ReadNone);

    llvm::Constant *Bar = Mod->getOrInsertFunction( "llvm.nvvm.barrier0" , 
						    IntrinFnTy , 
						    llvm::AttributeSet::get(llvm::getGlobalContext(), 
									    llvm::AttributeSet::FunctionIndex, 
									    ABuilder)
						    );

    builder->CreateCall(Bar);
  }


  


  llvm::Value * llvm_special( const char * name )
  {
    llvm::FunctionType *IntrinFnTy = llvm::FunctionType::get(llvm::Type::getInt32Ty(llvm::getGlobalContext()), false);

    llvm::AttrBuilder ABuilder;
    ABuilder.addAttribute(llvm::Attribute::ReadNone);

    llvm::Constant *ReadTidX = Mod->getOrInsertFunction( name , 
							 IntrinFnTy , 
							 llvm::AttributeSet::get(llvm::getGlobalContext(), 
										 llvm::AttributeSet::FunctionIndex, 
										 ABuilder)
							 );

    return builder->CreateCall(ReadTidX);
  }



  // llvm::Value * llvm_call_special_tidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.tid.x"); }
  // llvm::Value * llvm_call_special_ntidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.ntid.x"); }
  // llvm::Value * llvm_call_special_ctaidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.ctaid.x"); }




#if 0
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
#endif


  void llvm_print_module( llvm::Module* m , const char * fname ) {
    std::string ErrorMsg;
    llvm::raw_fd_ostream outfd( fname ,ErrorMsg);
    llvm::outs() << ErrorMsg << "\n";
    std::string banner;
    {
      llvm::PassManager PM;
      PM.add( llvm::createPrintModulePass( &outfd, false, banner ) ); 
      PM.run( *m );
    }
  }


  std::string get_PTX_from_Module_using_llvm( llvm::Module *Mod )
  {
    llvm::FunctionPassManager OurFPM( Mod );
    //OurFPM.add(llvm::createCFGSimplificationPass());  // skip this for now. causes problems with CUDA generic pointers
    //OurFPM.add(llvm::createBasicAliasAnalysisPass());
    //OurFPM.add(llvm::createInstructionCombiningPass()); // this causes problems. Not anymore!
    //OurFPM.add(llvm::createReassociatePass());
    OurFPM.add(llvm::createGVNPass());
    OurFPM.doInitialization();

    auto& func_list = Mod->getFunctionList();

    QDP_info_primary("Running optimization passes on functions");

    for(auto& x : func_list) {
      //QDP_info("Running all optimization passes on function %s",x.getName());
      OurFPM.run(x);
    }



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


    std::string str;
    llvm::raw_string_ostream rss(str);
    llvm::formatted_raw_ostream FOS(rss);

    llvm::PassManager PMTM;


#if 0
    // Add the target data from the target machine, if it exists, or the module.
    if (const DataLayout *TD = Target.getDataLayout()) {
      QDP_info_primary( "Using targets's data layout" );
      PMTM.add(new DataLayout(*TD));
    }
    else {
      QDP_info_primary( "Using module's data layout" );
      PMTM.add(new DataLayout(Mod));
    }
#else
    QDP_info_primary( "Using module's data layout" );
    PMTM.add(new llvm::DataLayout(Mod));
#endif


    // Ask the target to add backend passes as necessary.
    if (Target.addPassesToEmitFile(PMTM, FOS,  llvm::TargetMachine::CGFT_AssemblyFile )) {
      llvm::errs() << ": target does not support generation of this"
		   << " file type!\n";
      exit(1);
    }

    QDP_info_primary("PTX code generation");
    PMTM.run(*Mod);
    FOS.flush();

    return str;
  }


  void str_replace(std::string& str, const std::string& oldStr, const std::string& newStr)
  {
    size_t pos = 0;
    while((pos = str.find(oldStr, pos)) != std::string::npos)
      {
	str.replace(pos, oldStr.length(), newStr);
	pos += newStr.length();
      }
  }

  std::map<std::string,std::string> mapAttr;
  std::map<std::string,std::string>::iterator mapAttrIter;

  bool find_attr(std::string& str)
  {
    mapAttr.clear();
    size_t pos = 0;
    while((pos = str.find("attributes #", pos)) != std::string::npos)
      {
	size_t pos_space = str.find(" ", pos+12);
	std::string num = str.substr(pos+12,pos_space-pos-12);
	num = " #"+num;
	std::cout << "# num found = " << num << "()\n";
	size_t pos_open = str.find("{", pos_space);
	size_t pos_close = str.find("}", pos_open);
	std::string val = str.substr(pos_open+1,pos_close-pos_open-1);
	std::cout << "# val found = " << val << "\n";
	str.replace(pos, pos_close-pos+1, "");
	if (mapAttr.count(num) > 0)
	  QDP_error_exit("unexp.");
	mapAttr[num]=val;
      }
  }



#if 0
  std::string get_PTX_from_Module_using_nvvm( llvm::Module *Mod )
  {
    Mod->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");

    llvm::PassManager PMTM;
#if 0
    // Add the target data from the target machine, if it exists, or the module.
    if (const DataLayout *TD = Target.getDataLayout()) {
      QDP_info_primary( "Using targets's data layout" );
      PMTM.add(new DataLayout(*TD));
    }
    else {
      QDP_info_primary( "Using module's data layout" );
      PMTM.add(new DataLayout(Mod));
    }
#else
    QDP_info_primary( "Using module's data layout" );
    PMTM.add(new llvm::DataLayout(Mod));
#endif
    QDP_info_primary("Adding data layout");
    PMTM.run(*Mod);

#if 1
    std::string str;
    llvm::raw_string_ostream rsos(str);
    llvm::formatted_raw_ostream fros(rsos);

    Mod->print(fros,NULL);

    // std::cout << "Do we need the ostream in binary mode?\n";
    // llvm::WriteBitcodeToFile(Mod,fros);
    fros.flush();

    find_attr(str);
    for (mapAttrIter=mapAttr.begin(); mapAttrIter!=mapAttr.end(); ++mapAttrIter)
      str_replace(str, mapAttrIter->first, mapAttrIter->second );

    str_replace( str , "!nvvm.internalize.after.link = !{}" , "" );
#endif


#if 0
    std::string error;
    unsigned OpenFlags = 0;
    OpenFlags |= llvm::raw_fd_ostream::F_Binary;
    llvm::OwningPtr<llvm::tool_output_file> Out( new llvm::tool_output_file( "test.bc" , error, OpenFlags) );
    if (!Out) {
      llvm::errs() << "Could not create OwningPtr<tool_output_file>\n";
      exit(1);
    }
    llvm::formatted_raw_ostream fros(Out->os());
    llvm::WriteBitcodeToFile(Mod,fros);
    fros.flush();
    // open the file:
    std::streampos fileSize;
    std::ifstream file("test.bc", std::ios::binary);
    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    // read the data:
    std::vector<char> buffer(fileSize);
    file.read((char*) &buffer[0], fileSize);
    file.close();
#endif


#if 0
    std::ifstream input( "test.bc", std::ios::binary );
    // copies all data into buffer
    std::vector<unsigned char> buffer( std::istreambuf_iterator<unsigned char>(input) ,  
				       std::istreambuf_iterator<unsigned char>() );
#endif

    //exit(1);

    std::cout << str << "\n";

    nvvmResult result;
    nvvmProgram program;
    size_t PTXSize;
    char *PTX = NULL;

    result = nvvmCreateProgram(&program);
    if (result != NVVM_SUCCESS) {
      fprintf(stderr, "nvvmCreateProgram: Failed\n");
      exit(1); 
    }

    result = nvvmAddModuleToProgram(program, str.c_str() , str.size() , "module" );
    //result = nvvmAddModuleToProgram(program, (const char*)buffer.data() , buffer.size() , "module" );
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmAddModuleToProgram: Failed\n");
        exit(-1);
    }

    std::stringstream ss;
    ss << "-arch=compute_" << DeviceParams::Instance().getMajor() << DeviceParams::Instance().getMinor();

    std::string sm_str = ss.str();

    const char * arch = sm_str.c_str();
    const char * opt_val[] = {arch};

    result = nvvmCompileProgram(program,  1, opt_val );
    if (result != NVVM_SUCCESS) {
        char *Msg = NULL;
        size_t LogSize;
        fprintf(stderr, "nvvmCompileProgram: Failed\n");
        nvvmGetProgramLogSize(program, &LogSize);
        Msg = (char*)malloc(LogSize);
        nvvmGetProgramLog(program, Msg);
        fprintf(stderr, "%s\n", Msg);
        free(Msg);
        exit(-1);
    }
    
    result = nvvmGetCompiledResultSize(program, &PTXSize);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmGetCompiledResultSize: Failed\n");
        exit(-1);
    }
    
    PTX = (char*)malloc(PTXSize);
    result = nvvmGetCompiledResult(program, PTX);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmGetCompiledResult: Failed\n");
        free(PTX);
        exit(-1);
    }
    
    result = nvvmDestroyProgram(&program);
    if (result != NVVM_SUCCESS) {
      fprintf(stderr, "nvvmDestroyProgram: Failed\n");
      free(PTX);
      exit(-1);
    }

    std::string ret(PTX);
    std::cout << ret << "\n";

    return ret;
    //
  }
#endif




#if 0
  std::string llvm_get_ptx_kernel(const char* fname)
  {
    QDP_info_primary("Internalizing module");

    const char *ExportList[] = { "main" };

    llvm::StringMap<int> Mapping;
    Mapping["__CUDA_FTZ"] = 0;

    std::string banner;

    llvm::PassManager OurPM;
    OurPM.add( llvm::createInternalizePass( llvm::ArrayRef<const char *>(ExportList, 1)));
    OurPM.add( llvm::createNVVMReflectPass(Mapping));
    OurPM.run( *Mod );


    QDP_info_primary("Running optimization passes on module");

    llvm::PassManager PM;
    PM.add( llvm::createGlobalDCEPass() );
    PM.run( *Mod );

    llvm_print_module(Mod,"ir_internalized_reflected_globalDCE.ll");

    std::string str = get_PTX_from_Module_using_nvvm( Mod );

#if 1
    // Write PTX string to file
    std::ofstream ptxfile;
    ptxfile.open ( fname );
    ptxfile << str << "\n";
    ptxfile.close();
#endif


#if 0
    // Read PTX string from file
    std::ifstream ptxfile(fname);
    std::stringstream buffer;
    buffer << ptxfile.rdbuf();
    ptxfile.close();
    str = buffer.str();
#endif

    //llvm::outs() << str << "\n";

    return str;
  }
#endif




  void * llvm_get_function(const char* fname)
  {
    assert(function_created && "Function not created");
    assert(function_started && "Function not started");

    //addKernelMetadata( mainFunc );

    // QDPIO::cerr << "LLVM IR function (before passes)\n";
    // if (Layout::primaryNode())
    //   mainFunc->dump();

    QDPIO::cerr << "Verifying main function\n";
    llvm::verifyFunction(*mainFunc);
    //QDPIO::cerr << "Verifying main_extern function\n";
    //assert( mainFunc_extern && "mainFunc_extern is NULL");
    //mainFunc_extern->dump();
    //llvm::verifyFunction(*mainFunc_extern);

    QDPIO::cerr << "running passes ...\n";

    static llvm::FunctionPassManager *functionPassManager = NULL;
    if (functionPassManager == NULL) {
      llvm::PassRegistry &registry = *llvm::PassRegistry::getPassRegistry();
      initializeScalarOpts(registry);

      functionPassManager = new llvm::FunctionPassManager(Mod);
      functionPassManager->add(llvm::createVerifierPass(llvm::PrintMessageAction));
      targetMachine->addAnalysisPasses(*functionPassManager);
      functionPassManager->add(new llvm::TargetLibraryInfo(llvm::Triple(Mod->getTargetTriple())));
      functionPassManager->add(new llvm::DataLayout(Mod));
      functionPassManager->add(llvm::createBasicAliasAnalysisPass());
      functionPassManager->add(llvm::createLICMPass());
      functionPassManager->add(llvm::createGVNPass());
      functionPassManager->add(llvm::createPromoteMemoryToRegisterPass());
      functionPassManager->add(llvm::createLoopVectorizePass());
      functionPassManager->add(llvm::createEarlyCSEPass());
      functionPassManager->add(llvm::createInstructionCombiningPass());
      functionPassManager->add(llvm::createCFGSimplificationPass());

    }
    llvm::DebugFlag = true;
    llvm::setCurrentDebugType("loop-vectorize");

    functionPassManager->run(*mainFunc);

    // QDPIO::cerr << "LLVM IR function (after passes)\n";
    // if (Layout::primaryNode())
    //   mainFunc->dump();


    // Right now a trampoline function which calls the main function
    // is necessary. For the auto-vectorizer we need the arguments to
    // to be noalias. Adding this attribute to a pointer is only possible
    // to function arguments. Since from host code I can only call
    // functions with a static signature, this cannot be done in one
    // step.

    // Create the 'trampoline' function

    llvm::FunctionType *funcType = 
      llvm::FunctionType::get( builder->getVoidTy() , 
			       llvm::PointerType::get( 
						 llvm::ArrayType::get( llvm::Type::getInt8Ty(llvm::getGlobalContext()) , 
								       8 ) , 0  ),
			       false); // no vararg

    llvm::Function *mainFunc_extern = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, "main_extern", Mod);

    // Convert Parameter to Argument

    llvm::Argument* arg_ptr = mainFunc_extern->arg_begin();
    arg_ptr->setName( "arg_ptr" );

    // Create entry basic block

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(llvm::getGlobalContext(), "entrypoint", mainFunc_extern);
    builder->SetInsertPoint(entry);

    // Extract each parameter

    std::vector<llvm::Value*> vecCallArgument;

    //std::cout << "Building trampoline with the following arguments:\n";

    int i=0;
    for( std::vector< llvm::Type* >::const_iterator param_type = vecParamType.begin() ; 
    	 param_type != vecParamType.end() ; 
    	 param_type++,i++ ) {
      //(*param_type)->dump(); std::cout << "\n";
      llvm::Value* gep = builder->CreateGEP( arg_ptr , llvm_create_value(i) );
      llvm::Type* param_ptr_type = llvm::PointerType::get( *param_type , 0  );
      llvm::Value* ptr_to_arg = builder->CreatePointerCast( gep , param_ptr_type );
      llvm::Value* arg = builder->CreateLoad( ptr_to_arg );
      vecCallArgument.push_back( arg );      
    }

    // Call 'main' from 'main_extern'
    
    builder->CreateCall( mainFunc , llvm::ArrayRef<llvm::Value*>( vecCallArgument.data() , vecCallArgument.size() ) );
    builder->CreateRetVoid();

    //mainFunc_extern->dump();

    std::cout << "in get function: Verifying main_extern function\n";
    llvm::verifyFunction(*mainFunc_extern);

    std::cout << "finalizing the module\n";
    TheExecutionEngine->finalizeObject();  // MCJIT

    std::cout << "in get function: JIT compiling main_extern ...\n";
    fptr_mainFunc_extern = TheExecutionEngine->getPointerToFunction( mainFunc_extern );

    function_created = false;
    function_started = false;

    return fptr_mainFunc_extern; 
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
    llvm::Value* lhs_f64 = llvm_cast( llvm_type<double>::value , lhs );
    llvm::Value* rhs_f64 = llvm_cast( llvm_type<double>::value , rhs );
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

