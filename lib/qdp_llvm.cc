#include "qdp.h"

#include "llvm-c/Core.h"
#include "llvm/Pass.h"

#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Vectorize.h"

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"

#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachine.h"

//#include <memory>


namespace QDP {

  const char* getCurrentFunctionExternName();


  namespace {
    std::vector<std::string>  vec_mattr;
  }


  class qdpjit_t
  {
    std::unique_ptr<llvm::TargetMachine> TM;
    llvm::DataLayout DL;
    llvm::orc::RTDyldObjectLinkingLayer ObjectLayer;
    llvm::orc::IRCompileLayer<decltype(ObjectLayer), llvm::orc::SimpleCompiler> CompileLayer;
  public:    
    qdpjit_t():
      TM( configureTarget() ),
      DL( TM->createDataLayout() ),
      ObjectLayer( []() { return std::make_shared<llvm::SectionMemoryManager>(); } ),
      CompileLayer( ObjectLayer, llvm::orc::SimpleCompiler(*TM) )
    {
      llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
      std::cout << "qdp-jit initialized, dynamic symbols loaded.\n";
    }

    llvm::DataLayout getDL() const
    {
      return DL;
    }

    llvm::TargetMachine* getTM() const
    {
      return TM.get();
    }
    
    llvm::TargetMachine* configureTarget() const
    {
      llvm::InitializeAllTargets();
      llvm::InitializeAllTargetMCs();

      char * argument = new char[128];
      sprintf( argument , "-vectorizer-min-trip-count=%d" , (int)getDataLayoutInnerSize() );

      QDPIO::cout << "Using inner lattice size of " << (int)getDataLayoutInnerSize() << "\n";
      QDPIO::cout << "Setting loop vectorizer minimum trip count to " << (int)getDataLayoutInnerSize() << "\n";
    
      const char *SetTinyVectorThreshold[] = {"program",argument};
      llvm::cl::ParseCommandLineOptions(2, SetTinyVectorThreshold);

      delete[] argument;

      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter(); // MCJIT
      llvm::InitializeNativeTargetAsmParser(); // MCJIT
      
      llvm::EngineBuilder engineBuilder;
      engineBuilder.setMCPU(llvm::sys::getHostCPUName());
      if (vec_mattr.size() > 0) 
	engineBuilder.setMAttrs( vec_mattr );
      engineBuilder.setEngineKind(llvm::EngineKind::JIT);
      engineBuilder.setOptLevel(llvm::CodeGenOpt::Aggressive);
      std::string mcjit_error;
      engineBuilder.setErrorStr(&mcjit_error);
      llvm::TargetOptions targetOptions;
      targetOptions.AllowFPOpFusion = llvm::FPOpFusion::Fast;
      engineBuilder.setTargetOptions( targetOptions );

      return engineBuilder.selectTarget();
    }


    //ModuleHandle addModule(std::unique_ptr<Module> M) {
    void addModule(std::unique_ptr<llvm::Module> M) {
      // Build our symbol resolver:
      // Lambda 1: Look back into the JIT itself to find symbols that are part of
      //           the same "logical dylib".
      // Lambda 2: Search for external symbols in the host process.
      auto Resolver = llvm::orc::createLambdaResolver(
						      [&](const std::string &Name) {
							if (auto Sym = CompileLayer.findSymbol(Name, false))
							  return Sym;
							return llvm::JITSymbol(nullptr);
						      },
						      [](const std::string &Name) {
							if (auto SymAddr =
							    llvm::RTDyldMemoryManager::getSymbolAddressInProcess(Name))
							  return llvm::JITSymbol(SymAddr, llvm::JITSymbolFlags::Exported);
							return llvm::JITSymbol(nullptr);
						      });
      
      // Add the set to the JIT with the resolver we created above and a newly
      // created SectionMemoryManager.
      cantFail(CompileLayer.addModule(std::move(M),
				      std::move(Resolver)));
    }

    
    void* getAddr()
    {
      std::string MangledName;
      llvm::raw_string_ostream MangledNameStream(MangledName);
      llvm::Mangler::getNameWithPrefix(MangledNameStream, getCurrentFunctionExternName() , DL);

      if (auto Sym = CompileLayer.findSymbol(MangledNameStream.str(),true))   // ,true
	{
	  //QDPIO::cout << "extern function found\n";
	  void* fptr = (void *)cantFail(Sym.getAddress());

	  return fptr;
	}
      else
	{
	  QDPIO::cout << std::string(getCurrentFunctionExternName()) << "\n";
	  QDP_error_exit("extern function not found!");
	  return NULL;
	}
    }
  };



  namespace {

    qdpjit_t& get_qdpjit() {
      static std::unique_ptr<qdpjit_t> qdpjit_ptr;

      if (!qdpjit_ptr) {
	QDPIO::cout << "Creating JIT\n";
	qdpjit_ptr = llvm::make_unique<qdpjit_t>();
	QDPIO::cout << "JIT successfully created\n";
      }

      return *qdpjit_ptr;
    }
    
    static llvm::LLVMContext context;
    static std::unique_ptr<llvm::Module> module;
    static std::unique_ptr<llvm::IRBuilder<> > builder;

    //llvm::BasicBlock  *entry;
    llvm::BasicBlock  *entry_main;
    llvm::Function    *mainFunc;
    //llvm::Function    *mainFunc_extern;


    //llvm::ExecutionEngine *TheExecutionEngine;

    //llvm::legacy::FunctionPassManager *TheFPM;

    std::string mcjit_error;

    //void * fptr_mainFunc_extern;

    bool function_started;
    bool function_created;

    std::vector< llvm::Type* > vecParamType;
    std::vector< llvm::Value* > vecArgument;

    llvm::Value *r_arg_lo;
    llvm::Value *r_arg_hi;
    llvm::Value *r_arg_myId;
    llvm::Value *r_arg_ordered;
    llvm::Value *r_arg_start;

    static int fcount = 0;
    static std::string fname;
    
  } // namespace
  

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

  namespace llvm_debug {
    bool debug_func_build      = false;
    bool debug_func_dump       = false;
    bool debug_func_dump_asm   = false;
    bool debug_func_write      = false;
    bool debug_loop_vectorizer = false;
    std::string name_pretty;
    std::string name_additional;
  }


  const char* getCurrentFunctionName() {
    fname = std::string("main") + std::to_string( fcount );
    return fname.c_str();
  }

  const char* getCurrentFunctionExternName() {
    fname = std::string("main") + std::to_string( fcount ) + std::string("_extern");
    return fname.c_str();
  }

  void nextFunctionName() {
    fcount++;
  }


  llvm::LLVMContext& llvm_get_context() {
    return context;
  }

  std::unique_ptr<llvm::IRBuilder<> >& llvm_get_builder() {
    return builder;
  }

  std::unique_ptr<llvm::Module>& llvm_get_module() {
    return module;
  }
  
  
  void llvm_set_debug( const char * c_str ) {
    std::string str(c_str);
    if (str.find("loop-vectorize") != string::npos) {
      llvm_debug::debug_loop_vectorizer = true;
      return;
    }
    if (str.find("function-builder") != string::npos) {
      llvm_debug::debug_func_build = true;
      return;
    }
    if (str.find("function-dump-ir") != string::npos) {
      llvm_debug::debug_func_dump = true;
      return;
    }
    if (str.find("function-write") != string::npos) {
      llvm_debug::debug_func_write = true;
      return;
    }
    if (str.find("function-dump-asm") != string::npos) {
      llvm_debug::debug_func_dump_asm = true;
      return;
    }
    QDP_error_exit("unknown debug argument: %s",c_str);
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



  void llvm_debug_write_set_name( const char* pretty, const char* additional )
  {
    llvm_debug::name_pretty = std::string(pretty);
    llvm_debug::name_additional = std::string(additional);
  }


  void llvm_append_mattr( const char * attr )
  {
    vec_mattr.push_back(attr);
  }



  void llvm_create_function();

  llvm::Value *llvm_get_arg_lo() {     
    if (!function_created)
      llvm_create_function();
    return r_arg_lo; 
  }

  llvm::Value *llvm_get_arg_hi() { 
    if (!function_created)
      llvm_create_function();
    return r_arg_hi; 
  }

  llvm::Value *llvm_get_arg_myId() { 
    if (!function_created)
      llvm_create_function();
    return r_arg_myId; 
  }

  llvm::Value *llvm_get_arg_ordered() { 
    if (!function_created)
      llvm_create_function();
    return r_arg_ordered; 
  }

  llvm::Value *llvm_get_arg_start() { 
    if (!function_created)
      llvm_create_function();
    return r_arg_start; 
  }


  llvm::Function *llvm_get_func_float( const char * name )
  {
    return llvm::Function::Create( 
				  llvm::FunctionType::get( 
							  builder->getFloatTy(),llvm::ArrayRef<llvm::Type*>( builder->getFloatTy() ) , false) , 
				  llvm::Function::ExternalLinkage, name , module.get() );
  }

  llvm::Function *llvm_get_func_float_float( const char * name )
  {
    std::vector< llvm::Type* > args;
    args.push_back(builder->getFloatTy());
    args.push_back(builder->getFloatTy());
    return llvm::Function::Create( 
           llvm::FunctionType::get( 
           builder->getFloatTy(),llvm::ArrayRef<llvm::Type*>( args.data() , 2 ) , false) , 
           llvm::Function::ExternalLinkage, name , module.get() );
  }

  llvm::Function *llvm_get_func_double( const char * name )
  {
    return llvm::Function::Create( 
           llvm::FunctionType::get( 
           builder->getDoubleTy(),llvm::ArrayRef<llvm::Type*>( builder->getDoubleTy() ) , false) , 
           llvm::Function::ExternalLinkage, name , module.get() );
  }

  llvm::Function *llvm_get_func_double_double( const char * name )
  {
    std::vector< llvm::Type* > args;
    args.push_back(builder->getDoubleTy());
    args.push_back(builder->getDoubleTy());
    return llvm::Function::Create( 
           llvm::FunctionType::get( 
           builder->getDoubleTy(),llvm::ArrayRef<llvm::Type*>( args.data() , 2 ) , false) , 
           llvm::Function::ExternalLinkage, name , module.get() );
  }




  void llvm_setup_math_functions() 
  {
    if (llvm_debug::debug_func_build) {
      QDPIO::cerr << "    initializing math functions\n";
    }

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
    function_created = false;
    function_started = false;

    llvm_type<float>::value  = llvm::Type::getFloatTy(context);
    llvm_type<double>::value = llvm::Type::getDoubleTy(context);
    llvm_type<int>::value    = llvm::Type::getIntNTy(context,32);
    llvm_type<bool>::value   = llvm::Type::getIntNTy(context,1);
    llvm_type<float*>::value  = llvm::Type::getFloatPtrTy(context);
    llvm_type<double*>::value = llvm::Type::getDoublePtrTy(context);
    llvm_type<int*>::value    = llvm::Type::getIntNPtrTy(context,32);
    llvm_type<bool*>::value   = llvm::Type::getIntNPtrTy(context,1);
  }  


  llvm::BasicBlock * llvm_get_insert_block() {
    return builder->GetInsertBlock();
  }


  void llvm_start_new_function() {
    assert( !function_started && "Function already started");
    function_started = true;
    function_created = false;

    if (llvm_debug::debug_func_build) {
      QDPIO::cerr << "Creating new module ...\n";
    }

    module = llvm::make_unique<llvm::Module>("module", context);
    module->setDataLayout( get_qdpjit().getDL() );
    builder = llvm::make_unique< llvm::IRBuilder<> >(context);
    
    if (llvm_debug::debug_func_build) {
      if (vec_mattr.size() > 0) {
	QDPIO::cerr << "    MCPU attributes: ";
	for ( int i = 0 ; i < (int)vec_mattr.size() ; i++ )
	  QDPIO::cerr << vec_mattr.at(i) << " ";
	QDPIO::cerr << "\n";
      }
    }

    if (llvm_debug::debug_func_build) {
      QDPIO::cerr << "    staring new LLVM function ...\n";
    }

    llvm_setup_math_functions();

    vecParamType.clear();
    vecArgument.clear();
  }



  void llvm_create_function() {
    nextFunctionName();

    assert(!function_created && "Function already created");
    assert(function_started && "Function not started");
    assert(vecParamType.size()>0 && "vecParamType.size()>0");


    // Make a local copy of the argument vector
    std::vector<llvm::Type*> vecPT;

    // Push back lo,hi,myId
    vecPT.push_back( llvm::Type::getInt64Ty(context) ); // lo
    vecPT.push_back( llvm::Type::getInt64Ty(context) ); // hi
    vecPT.push_back( llvm::Type::getInt64Ty(context) ); // myId
    vecPT.push_back( llvm::Type::getInt1Ty(context) );  // ordered
    vecPT.push_back( llvm::Type::getInt64Ty(context) ); // start

    vecPT.insert( vecPT.end() , vecParamType.begin() , vecParamType.end() );

    // Create the main function

    llvm::FunctionType *funcType = 
      llvm::FunctionType::get( builder->getVoidTy() , 
			       llvm::ArrayRef<llvm::Type*>( vecPT.data() , vecPT.size() ) , 
			       false); // no vararg
    mainFunc = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, getCurrentFunctionName() , module.get() );

    // Set argument names in 'main'

    llvm::Function::arg_iterator AI = mainFunc->arg_begin();
    llvm::Function::arg_iterator AE = mainFunc->arg_end();
    AI->setName("lo"); 
    r_arg_lo = &*AI; 
    AI++;
    
    AI->setName("hi"); 
    r_arg_hi = &*AI;
    AI++;
    
    AI->setName("myId"); 
    r_arg_myId = &*AI;
    AI++;

    AI->setName("ordered");
    r_arg_ordered = &*AI;
    AI++;

    AI->setName("start"); 
    r_arg_start = &*AI;
    AI++;

    unsigned Idx = 0;
    for ( ; AI != AE ; ++AI, ++Idx) {
      std::ostringstream oss;
      oss << "arg" << Idx;
      AI->setName( oss.str() );

      if ( vecParamType.at(Idx)->isPointerTy() ) {
      	llvm::AttrBuilder B;
      	B.addAttribute(llvm::Attribute::NoAlias);

	// We assume a FP pointer type coming from an OLattice which uses the default QDP allocator
	if (vecParamType.at(Idx)->getPointerElementType()->isFloatingPointTy()) {
	  B.addAlignmentAttr( QDP_ALIGNMENT_SIZE );
	}

	AI->addAttrs(  B  );
      }

      vecArgument.push_back( &*AI );
    }

    entry_main = llvm::BasicBlock::Create(context, "entrypoint", mainFunc);
    builder->SetInsertPoint(entry_main);

    // if (Layout::primaryNode())
    //   mainFunc->dump();

    llvm_counters::label_counter = 0;
    function_created = true;
  }



  llvm::Value * llvm_derefParam( ParamRef r ) {
    if (!function_created)
      llvm_create_function();
    assert( (int)vecArgument.size() > (int)r && "derefParam out of range");
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
	return llvm::Type::getDoubleTy(context);
      } else {
	return llvm::Type::getFloatTy(context);
      }
    } else {
      //llvm::outs() << "promote int\n";
      unsigned upper = std::max( t0->getScalarSizeInBits() , t1->getScalarSizeInBits() );
      return llvm::Type::getIntNTy(context , upper );
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



  llvm::Type* llvm_normalize_values(llvm::Value*& lhs , llvm::Value*& rhs)
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
    return args_type;
  }



  llvm::Value* llvm_neg( llvm::Value* rhs ) {
    llvm::Value* lhs = llvm_create_value(0);
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFSub( lhs , rhs );
    else
      return builder->CreateSub( lhs , rhs );
  }


  llvm::Value* llvm_rem( llvm::Value* lhs , llvm::Value* rhs ) {
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFRem( lhs , rhs );
    else
      return builder->CreateSRem( lhs , rhs );
  }


  llvm::Value* llvm_shr( llvm::Value* lhs , llvm::Value* rhs ) {  
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    assert( !args_type->isFloatingPointTy() );
    return builder->CreateAShr( lhs , rhs );
  }


  llvm::Value* llvm_shl( llvm::Value* lhs , llvm::Value* rhs ) {  
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    assert( !args_type->isFloatingPointTy() );
    return builder->CreateShl( lhs , rhs );
  }


  llvm::Value* llvm_and( llvm::Value* lhs , llvm::Value* rhs ) {  
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    assert( !args_type->isFloatingPointTy() );
    return builder->CreateAnd( lhs , rhs );
  }


  llvm::Value* llvm_or( llvm::Value* lhs , llvm::Value* rhs ) {  
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    assert( !args_type->isFloatingPointTy() );
    return builder->CreateOr( lhs , rhs );
  }


  llvm::Value* llvm_xor( llvm::Value* lhs , llvm::Value* rhs ) {  
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    assert( !args_type->isFloatingPointTy() );
    return builder->CreateXor( lhs , rhs );
  }


  llvm::Value* llvm_mul( llvm::Value* lhs , llvm::Value* rhs ) {
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFMul( lhs , rhs );
    else
      return builder->CreateMul( lhs , rhs );
  }


  llvm::Value* llvm_add( llvm::Value* lhs , llvm::Value* rhs ) {
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFAdd( lhs , rhs );
    else
      return builder->CreateNSWAdd( lhs , rhs );
  }


  llvm::Value* llvm_sub( llvm::Value* lhs , llvm::Value* rhs ) {
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFSub( lhs , rhs );
    else
      return builder->CreateSub( lhs , rhs );
  }


  llvm::Value* llvm_div( llvm::Value* lhs , llvm::Value* rhs ) {
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFDiv( lhs , rhs );
    else 
      return builder->CreateSDiv( lhs , rhs );
  }


  llvm::Value* llvm_eq( llvm::Value* lhs , llvm::Value* rhs ) {
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFCmpOEQ( lhs , rhs );
    else
      return builder->CreateICmpEQ( lhs , rhs );
  }


  llvm::Value* llvm_ge( llvm::Value* lhs , llvm::Value* rhs ) {
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFCmpOGE( lhs , rhs );
    else
      return builder->CreateICmpSGE( lhs , rhs );
  }


  llvm::Value* llvm_gt( llvm::Value* lhs , llvm::Value* rhs ) {
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFCmpOGT( lhs , rhs );
    else
      return builder->CreateICmpSGT( lhs , rhs );
  }


  llvm::Value* llvm_le( llvm::Value* lhs , llvm::Value* rhs ) {
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFCmpOLE( lhs , rhs );
    else
      return builder->CreateICmpSLE( lhs , rhs );
  }


  llvm::Value* llvm_lt( llvm::Value* lhs , llvm::Value* rhs ) {
    llvm::Type* args_type = llvm_normalize_values(lhs,rhs);
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFCmpOLT( lhs , rhs );
    else 
      return builder->CreateICmpSLT( lhs , rhs );
  }


  //
  // Convenience function definitions
  //
  llvm::Value* llvm_not( llvm::Value* lhs ) {
    //llvm::outs() << "not\n";
    return llvm_xor( llvm_create_value(-1) , lhs );
  }



  llvm::Value * llvm_alloca( llvm::Type* type , int elements )
  {
    llvm::BasicBlock *insert_block = builder->GetInsertBlock();
    llvm::BasicBlock::iterator insert_point = builder->GetInsertPoint();
    builder->SetInsertPoint( entry_main , entry_main->begin() );
    llvm::Value *ptr = builder->CreateAlloca( type , llvm_create_value(elements) );    // This can be a llvm::Value*
    builder->SetInsertPoint( insert_block , insert_point );
    return ptr;
  }


  template<> ParamRef llvm_add_param<bool>() { 
    vecParamType.push_back( llvm::Type::getInt1Ty(context) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<bool*>() { 
    vecParamType.push_back( llvm::Type::getInt1PtrTy(context) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<int64_t>() { 
    vecParamType.push_back( llvm::Type::getInt64Ty(context) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<int>() { 
    vecParamType.push_back( llvm::Type::getInt32Ty(context) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<int*>() { 
    vecParamType.push_back( llvm::Type::getInt32PtrTy(context) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<float>() { 
    vecParamType.push_back( llvm::Type::getFloatTy(context) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<float*>() { 
    vecParamType.push_back( llvm::Type::getFloatPtrTy(context) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<double>() { 
    vecParamType.push_back( llvm::Type::getDoubleTy(context) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<double*>() { 
    vecParamType.push_back( llvm::Type::getDoublePtrTy(context) );
    return vecParamType.size()-1;
  }



  llvm::BasicBlock * llvm_new_basic_block()
  {
    std::ostringstream oss;
    oss << "L" << llvm_counters::label_counter++;
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(context, oss.str() );
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
    return llvm::ConstantInt::getSigned( llvm::Type::getIntNTy(context,32) , i );
  }

  llvm::Value * llvm_create_value( double v )
  {
    if (sizeof(REAL) == 4)
      return llvm::ConstantFP::get( llvm::Type::getFloatTy(context) , v );
    else
      return llvm::ConstantFP::get( llvm::Type::getDoubleTy(context) , v );
  }

  llvm::Value * llvm_create_value(int64_t v )  {return llvm::ConstantInt::get( llvm::Type::getInt64Ty(context) , v );}
  llvm::Value * llvm_create_value(int v )  {return llvm::ConstantInt::get( llvm::Type::getInt32Ty(context) , v );}
  llvm::Value * llvm_create_value(size_t v){return llvm::ConstantInt::get( llvm::Type::getInt32Ty(context) , v );}
  llvm::Value * llvm_create_value(bool v ) {return llvm::ConstantInt::get( llvm::Type::getInt1Ty(context) , v );}


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
    builder->CreateStore( val_cast , ptr );
  }


  llvm::Value * llvm_load_ptr_idx( llvm::Value * ptr , llvm::Value * idx )
  {
    return llvm_load( llvm_createGEP( ptr , idx ) );
  }


  void llvm_store_ptr_idx( llvm::Value * val , llvm::Value * ptr , llvm::Value * idx )
  {
    llvm_store( val , llvm_createGEP( ptr , idx ) );
  }



  void llvm_print_module( llvm::Module* m , const char * fname ) {
    std::error_code EC;
    llvm::raw_fd_ostream outfd( fname , EC, llvm::sys::fs::OpenFlags::F_Text);
    //ASSERT_FALSE(outfd.has_error());
    std::string banner;
    {
      llvm::outs() << "llvm_print_module ni\n";
    }
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



  void * llvm_get_function(const char* fname)
  {
    assert(function_created && "Function not created");
    assert(function_started && "Function not started");

    if (llvm_debug::debug_func_dump) {
      if (Layout::primaryNode()) {
#ifdef QDP_LLVM_DEBUG_BUILD
	QDPIO::cerr << "LLVM IR function (before passes)\n";
	mainFunc->dump();
	//module->dump();
#endif
      }
    }

    if (llvm_debug::debug_func_build) {
      QDPIO::cerr << "    verifying: main\n";
    }
    llvm::verifyFunction(*mainFunc);

    if (llvm_debug::debug_func_build) {
      QDPIO::cerr << "    optimizing ...\n";
    }

    llvm::legacy::FunctionPassManager *functionPassManager = new llvm::legacy::FunctionPassManager(module.get());

    llvm::PassRegistry &registry = *llvm::PassRegistry::getPassRegistry();
    initializeScalarOpts(registry);

    functionPassManager->add(createTargetTransformInfoWrapperPass(get_qdpjit().getTM()->getTargetIRAnalysis()));

    functionPassManager->add( new llvm::TargetLibraryInfoWrapperPass(llvm::TargetLibraryInfoImpl(get_qdpjit().getTM()->getTargetTriple())) );
    functionPassManager->add(llvm::createBasicAAWrapperPass());
    functionPassManager->add(llvm::createLICMPass());
    functionPassManager->add(llvm::createGVNPass());
    functionPassManager->add(llvm::createPromoteMemoryToRegisterPass());
    functionPassManager->add(llvm::createLoopVectorizePass());
    functionPassManager->add(llvm::createEarlyCSEPass());
    functionPassManager->add(llvm::createInstructionCombiningPass());
    functionPassManager->add(llvm::createCFGSimplificationPass());
    functionPassManager->add(llvm::createLoopUnrollPass() );  // LLVM 3.8
    functionPassManager->add(llvm::createCFGSimplificationPass());  // join BB of vectorized loop with header
    functionPassManager->add(llvm::createGVNPass()); // eliminate redundant index instructions

    if (llvm_debug::debug_loop_vectorizer) {
      if (Layout::primaryNode()) {
#ifdef QDP_LLVM_DEBUG_BUILD	
	llvm::DebugFlag = true;
	llvm::setCurrentDebugType("loop-vectorize");
#endif
      }
    }

    functionPassManager->run(*mainFunc);

    if (llvm_debug::debug_func_dump) {
      if (Layout::primaryNode()) {
#ifdef QDP_LLVM_DEBUG_BUILD
	QDPIO::cerr << "LLVM IR function (after passes)\n";
	mainFunc->dump();
	//Mod->dump();
#endif
      }
    }

    if (llvm_debug::debug_func_write) {
      if (Layout::primaryNode()) {
	std::string str;

	llvm::raw_string_ostream rss(str);
	module->print(rss,new llvm::AssemblyAnnotationWriter());

	char* fname = new char[100];
	sprintf(fname,"module_XXXXXX");
	mkstemp(fname);
	QDPIO::cerr << fname << "\n";

	size_t start_pos = str.find("main");
	if(start_pos == std::string::npos)
	  QDP_error_exit("main not found in IR");
	str.replace(start_pos, 4, fname);

	str.insert(0,llvm_debug::name_additional);
	str.insert(0,llvm_debug::name_pretty);
	str.insert(0,";");

	ofstream myfile;
	myfile.open (fname);
	myfile << str;
	myfile.close();

	delete[] fname;
      }
    }



#if 1 
    // Write assembly
    if (llvm_debug::debug_func_dump_asm) {
      if (Layout::primaryNode()) {
	//llvm::legacy::FunctionPassManager *functionPassManager = new llvm::legacy::FunctionPassManager(module.get());
	llvm::legacy::PassManager PM;

	llvm::SmallString<128> Str;
	llvm::raw_svector_ostream dest(Str); 

	//if (targetmachine->addPassesToEmitFile( PM , dest , llvm::TargetMachine::CGFT_AssemblyFile ) ) {
	if (get_qdpjit().getTM()->addPassesToEmitFile( PM , dest , llvm::TargetMachine::CGFT_AssemblyFile ) ) {
	  std::cout << "addPassesToEmitFile failed\n";
	  exit(1);
	}
	PM.run(*module.get());
	std::cerr << "Assembly:\n";
	std::cerr << std::string( Str.c_str() ) << "\n";
	std::cerr << "end assembly!\n";
      }
    }
#endif



#if 1
    // Right now a trampoline function which calls the main function
    // is necessary. For the auto-vectorizer we need the arguments to
    // to be noalias. Adding this attribute to a pointer is only possible
    // to function arguments. Since from host code I can only call
    // functions with a static signature, this cannot be done in one
    // step.

    // Create the 'trampoline' function

    std::vector< llvm::Type* > vecArgs;

    // Push front lo,hi,myId
    vecArgs.push_back( llvm::Type::getInt64Ty(context) ); // lo
    vecArgs.push_back( llvm::Type::getInt64Ty(context) ); // hi
    vecArgs.push_back( llvm::Type::getInt64Ty(context) ); // myId
    vecArgs.push_back( llvm::Type::getInt1Ty(context)  ); // ordered
    vecArgs.push_back( llvm::Type::getInt64Ty(context) ); // start
    vecArgs.push_back( llvm::PointerType::get( 
					       llvm::ArrayType::get( llvm::Type::getInt8Ty(context) , 
								     8 ) , 0  ) );
    llvm::FunctionType *funcType = 
      llvm::FunctionType::get( builder->getVoidTy() , 
			       llvm::ArrayRef<llvm::Type*>( vecArgs.data() , vecArgs.size() ) , 
			       false); // no vararg

    llvm::Function *mainFunc_extern = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, getCurrentFunctionExternName() , module.get());

    std::vector<llvm::Value*> vecCallArgument;

    // Convert Parameter to Argument
    // Push back lo,hi,myId

    llvm::Function::arg_iterator AI = mainFunc_extern->arg_begin();

    AI->setName( "lo" );
    vecCallArgument.push_back( &*AI );
    AI++;

    AI->setName( "hi" );
    vecCallArgument.push_back( &*AI );
    AI++;

    AI->setName( "myId" );
    vecCallArgument.push_back( &*AI );
    AI++;

    AI->setName( "ordered" );
    vecCallArgument.push_back( &*AI );
    AI++;

    AI->setName( "start" );
    vecCallArgument.push_back( &*AI );
    AI++;

    AI->setName( "arg_ptr" );

    // Create entry basic block

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(context, "entrypoint", mainFunc_extern);
    builder->SetInsertPoint(entry);

    int i=0;
    for( std::vector< llvm::Type* >::const_iterator param_type = vecParamType.begin() ; 
    	 param_type != vecParamType.end() ; 
    	 param_type++,i++ ) {
      //(*param_type)->dump(); std::cout << "\n";
      llvm::Value* gep = builder->CreateGEP( &*AI , llvm_create_value(i) );
      llvm::Type* param_ptr_type = llvm::PointerType::get( *param_type , 0  );
      llvm::Value* ptr_to_arg = builder->CreatePointerCast( gep , param_ptr_type );
      llvm::Value* arg = builder->CreateLoad( ptr_to_arg );
      vecCallArgument.push_back( arg );      
    }

    builder->CreateCall( mainFunc , llvm::ArrayRef<llvm::Value*>( vecCallArgument.data() , vecCallArgument.size() ) );
    builder->CreateRetVoid();

    //mainFunc_extern->dump();
#endif
    
    if (llvm_debug::debug_func_build) {
      QDPIO::cerr << "    verifying: main_extern\n";
    }
    llvm::verifyFunction(*mainFunc_extern);

    if (llvm_debug::debug_func_dump) {
      if (Layout::primaryNode()) {
#ifdef QDP_LLVM_DEBUG_BUILD
	QDPIO::cerr << "LLVM IR function (before passes)\n";
	mainFunc_extern->dump();
#endif
      }
    }
    
    if (llvm_debug::debug_func_build) {
      QDPIO::cerr << "    finalizing the module\n";
      //Mod->dump();
    }

    get_qdpjit().addModule( std::move(module) );

    function_created = false;
    function_started = false;
    
    return get_qdpjit().getAddr();
    
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
    return builder->CreateCall(func,{lhs_f32,rhs_f32});
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
    return builder->CreateCall(func,{lhs_f64,rhs_f64});
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

