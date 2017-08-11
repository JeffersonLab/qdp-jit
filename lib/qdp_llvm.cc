#include "qdp.h"

#include "qdp_libdevice.h"
//#include "nvvm.h"

#include "llvm/IR/DataLayout.h"
//#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Linker/Linker.h"

#include "llvm/Support/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/CodeGen/CommandFlags.h"

#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"

#include "llvm/Transforms/Utils/Cloning.h"

#include <memory>

namespace QDP {

  llvm::LLVMContext TheContext;

  llvm::IRBuilder<> *builder;
  llvm::BasicBlock  *entry;
  llvm::Function    *mainFunc;
  llvm::Module      *Mod;
  //llvm::Value      *mainFunc;

  std::map<CUfunction,std::string> mapCUFuncPTX;

  std::string getPTXfromCUFunc(CUfunction f) {
    return mapCUFuncPTX[f];
  }

  bool function_created;

  std::vector< llvm::Type* > vecParamType;
  std::vector< llvm::Value* > vecArgument;

  llvm::Value *r_arg_lo;
  llvm::Value *r_arg_hi;
  llvm::Value *r_arg_myId;
  llvm::Value *r_arg_ordered;
  llvm::Value *r_arg_start;

  std::unique_ptr<llvm::Module> module_libdevice;

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
    bool debug_func_write      = false;
    bool debug_loop_vectorizer = false;
    std::string name_pretty;
    std::string name_additional;
  }

  namespace llvm_opt {
    int opt_level   = 3;   // opt -O level
    int nvptx_FTZ   = 0;   // NVPTX Flush subnormals to zero
    bool DisableInline = false;
    bool UnitAtATime = false;
    bool DisableLoopUnrolling = false;
    bool DisableLoopVectorization = false;
    bool DisableSLPVectorization = false;
  }

  void llvm_set_opt( const char * c_str ) {
    std::string str(c_str);
    if (str.find("DisableInline") != string::npos) {
      llvm_opt::DisableInline = true;
      return;
    }
    if (str.find("UnitAtATime") != string::npos) {
      llvm_opt::UnitAtATime = true;
      return;
    }
    if (str.find("DisableLoopUnrolling") != string::npos) {
      llvm_opt::DisableLoopUnrolling = true;
      return;
    }
    if (str.find("DisableLoopVectorization") != string::npos) {
      llvm_opt::DisableLoopVectorization = true;
      return;
    }
    if (str.find("DisableSLPVectorization") != string::npos) {
      llvm_opt::DisableSLPVectorization = true;
      return;
    }
    if (str.find("O0") != string::npos) {
      llvm_opt::opt_level = 0;
      return;
    }
    if (str.find("O1") != string::npos) {
      llvm_opt::opt_level = 1;
      return;
    }
    if (str.find("O2") != string::npos) {
      llvm_opt::opt_level = 2;
      return;
    }
    if (str.find("O3") != string::npos) {
      llvm_opt::opt_level = 3;
      return;
    }
    if (str.find("FTZ0") != string::npos) {
      llvm_opt::nvptx_FTZ = 0;
      return;
    }
    if (str.find("FTZ1") != string::npos) {
      llvm_opt::nvptx_FTZ = 1;
      return;
    }
    QDP_error_exit("unknown llvm-opt argument: %s",c_str);
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
    if (str.find("function-dump") != string::npos) {
      llvm_debug::debug_func_dump = true;
      return;
    }
    if (str.find("function-write") != string::npos) {
      llvm_debug::debug_func_write = true;
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


  llvm::Function *llvm_get_func( const char * name )
  {
    llvm::Function *func = Mod->getFunction(name);
    if (!func)
      QDP_error_exit("Function %s not found.\n",name);
    return func;
  }


  void llvm_init_libdevice()
  {
    auto major = DeviceParams::Instance().getMajor();
    auto minor = DeviceParams::Instance().getMinor();

    //QDPIO::cout << "Loading CUDA libdevice for compute capability " << major << minor << "\n";
    
    std::string ErrorMessage;

    if ( QDP::LIBDEVICE::map_sm_lib.find( major*10 + minor ) == QDP::LIBDEVICE::map_sm_lib.end() )
      {
	QDPIO::cout << "Compute capability " << major*10 + minor << " not found in libdevice libmap\n";
	QDP_abort(1);
      }
    if ( QDP::LIBDEVICE::map_sm_len.find( major*10 + minor ) == QDP::LIBDEVICE::map_sm_len.end() )
      {
	QDPIO::cout << "Compute capability " << major*10 + minor << " not found in libdevice lenmap\n";
	QDP_abort(1);
      }

    llvm::StringRef libdevice_bc( (const char *) QDP::LIBDEVICE::map_sm_lib[ major*10 + minor ], 
				  (size_t) QDP::LIBDEVICE::map_sm_len[ major*10 + minor ] );

    {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer> > BufferOrErr =
	llvm::MemoryBuffer::getMemBufferCopy(libdevice_bc );
      
      if (std::error_code ec = BufferOrErr.getError())
	ErrorMessage = ec.message();
      else {
	std::unique_ptr<llvm::MemoryBuffer> &BufferPtr = BufferOrErr.get();
	
	llvm::Expected<std::unique_ptr<llvm::Module> > ModuleOrErr = llvm::parseBitcodeFile(BufferPtr.get()->getMemBufferRef(), TheContext);
	
	if (llvm::Error Err = ModuleOrErr.takeError()) {
	  ErrorMessage = llvm::toString(std::move(Err));
	}
	else
	  {
	    module_libdevice.reset( ModuleOrErr.get().release() );
	  }
      }
    }

    if (!module_libdevice) {
      if (ErrorMessage.size())
	llvm::errs() << ErrorMessage << "\n";
      else
	llvm::errs() << "libdevice bitcode didn't read correctly.\n";
      QDP_abort( 1 );
    }
  }


  void llvm_setup_math_functions() 
  {
    //QDPIO::cout << "Setup math functions..\n";

    // Cloning a module takes more time than creating the module from scratch
    // So, I am creating the libdevice module from the embedded bitcode.
    //
#if 0
    std::unique_ptr<llvm::Module> libdevice_clone( CloneModule( module_libdevice.get() ) );

    std::string ErrorMsg;
    if (llvm::Linker::linkModules( *Mod , std::move( libdevice_clone ) )) {  // llvm::Linker::PreserveSource
      QDP_error_exit("Linking libdevice failed: %s",ErrorMsg.c_str());
    }
#else
    llvm_init_libdevice();

    std::string ErrorMsg;
    if (llvm::Linker::linkModules( *Mod , std::move( module_libdevice ) )) {  // llvm::Linker::PreserveSource
      QDP_error_exit("Linking libdevice failed: %s",ErrorMsg.c_str());
    }
#endif    

    func_sin_f32 = llvm_get_func( "__nv_sinf" );
    func_acos_f32 = llvm_get_func( "__nv_acosf" );
    func_asin_f32 = llvm_get_func( "__nv_asinf" );
    func_atan_f32 = llvm_get_func( "__nv_atanf" );
    func_ceil_f32 = llvm_get_func( "__nv_ceilf" );
    func_floor_f32 = llvm_get_func( "__nv_floorf" );
    func_cos_f32 = llvm_get_func( "__nv_cosf" );
    func_cosh_f32 = llvm_get_func( "__nv_coshf" );
    func_exp_f32 = llvm_get_func( "__nv_expf" );
    func_log_f32 = llvm_get_func( "__nv_logf" );
    func_log10_f32 = llvm_get_func( "__nv_log10f" );
    func_sinh_f32 = llvm_get_func( "__nv_sinhf" );
    func_tan_f32 = llvm_get_func( "__nv_tanf" );
    func_tanh_f32 = llvm_get_func( "__nv_tanhf" );
    func_fabs_f32 = llvm_get_func( "__nv_fabsf" );
    func_sqrt_f32 = llvm_get_func( "__nv_fsqrt_rn" );


    func_pow_f32 = llvm_get_func( "__nv_powf" );
    func_atan2_f32 = llvm_get_func( "__nv_atan2f" );


    func_sin_f64 = llvm_get_func( "__nv_sin" );
    func_acos_f64 = llvm_get_func( "__nv_acos" );
    func_asin_f64 = llvm_get_func( "__nv_asin" );
    func_atan_f64 = llvm_get_func( "__nv_atan" );
    func_ceil_f64 = llvm_get_func( "__nv_ceil" );
    func_floor_f64 = llvm_get_func( "__nv_floor" );
    func_cos_f64 = llvm_get_func( "__nv_cos" );
    func_cosh_f64 = llvm_get_func( "__nv_cosh" );
    func_exp_f64 = llvm_get_func( "__nv_exp" );
    func_log_f64 = llvm_get_func( "__nv_log" );
    func_log10_f64 = llvm_get_func( "__nv_log10" );
    func_sinh_f64 = llvm_get_func( "__nv_sinh" );
    func_tan_f64 = llvm_get_func( "__nv_tan" );
    func_tanh_f64 = llvm_get_func( "__nv_tanh" );
    func_fabs_f64 = llvm_get_func( "__nv_fabs" );
    func_sqrt_f64 = llvm_get_func( "__nv_dsqrt_rn" );

    func_pow_f64 = llvm_get_func( "__nv_pow" );
    func_atan2_f64 = llvm_get_func( "__nv_atan2" );
  }


  void llvm_wrapper_init() {
    function_created = false;

    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();

    llvm::PassRegistry *Registry = llvm::PassRegistry::getPassRegistry();
    llvm::initializeCore(*Registry);
    llvm::initializeCodeGen(*Registry);
    llvm::initializeLoopStrengthReducePass(*Registry);
    llvm::initializeLowerIntrinsicsPass(*Registry);
    llvm::initializeCountingFunctionInserterPass(*Registry);
    llvm::initializeUnreachableBlockElimLegacyPassPass(*Registry);
    llvm::initializeConstantHoistingLegacyPassPass(*Registry);

    llvm_type<float>::value  = llvm::Type::getFloatTy(TheContext);
    llvm_type<double>::value = llvm::Type::getDoubleTy(TheContext);
    llvm_type<int>::value    = llvm::Type::getIntNTy(TheContext,32);
    llvm_type<bool>::value   = llvm::Type::getIntNTy(TheContext,1);
    llvm_type<float*>::value  = llvm::Type::getFloatPtrTy(TheContext);
    llvm_type<double*>::value = llvm::Type::getDoublePtrTy(TheContext);
    llvm_type<int*>::value    = llvm::Type::getIntNPtrTy(TheContext,32);
    llvm_type<bool*>::value   = llvm::Type::getIntNPtrTy(TheContext,1);

    QDPIO::cout << "LLVM optimization level : " << llvm_opt::opt_level << "\n";
    QDPIO::cout << "NVPTX Flush to zero     : " << llvm_opt::nvptx_FTZ << "\n";

    //
    // I initialize libdevice in math_setup
    //
    // llvm_init_libdevice();
  }  


  llvm::BasicBlock * llvm_get_insert_block() {
    return builder->GetInsertBlock();
  }


  void llvm_start_new_function() {
    //QDPIO::cout << "Starting new LLVM function..\n";

    Mod = new llvm::Module("module", TheContext);
    builder = new llvm::IRBuilder<>(TheContext);

    jit_build_seedToFloat();
    jit_build_seedMultiply();

    vecParamType.clear();
    vecArgument.clear();
    function_created = false;

    llvm_setup_math_functions();

    // llvm::outs() << "------------------------- linked module\n";
    // llvm_print_module(Mod,"ir_linked.ll");
    //Mod->dump();
  }


  void llvm_create_function() {
    assert(!function_created && "Function already created");
    assert(vecParamType.size()>0 && "vecParamType.size()>0");
    llvm::FunctionType *funcType = 
      llvm::FunctionType::get( builder->getVoidTy() , 
			       llvm::ArrayRef<llvm::Type*>( vecParamType.data() , vecParamType.size() ) , 
			       false); // no vararg
    mainFunc = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, "main", Mod);

    unsigned Idx = 0;
    for (llvm::Function::arg_iterator AI = mainFunc->arg_begin(), AE = mainFunc->arg_end() ; AI != AE ; ++AI, ++Idx) {
      AI->setName( std::string("arg")+std::to_string(Idx) );
      vecArgument.push_back( &*AI );
    }

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(TheContext, "entrypoint", mainFunc);
    builder->SetInsertPoint(entry);

    llvm_counters::label_counter = 0;
    function_created = true;
  }



  llvm::Value * llvm_derefParam( ParamRef r ) {
    if (!function_created)
      llvm_create_function();
    assert( vecArgument.size() > (unsigned)r && "derefParam out of range");
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
	return llvm::Type::getDoubleTy(TheContext);
      } else {
	return llvm::Type::getFloatTy(TheContext);
      }
    } else {
      //llvm::outs() << "promote int\n";
      unsigned upper = std::max( t0->getScalarSizeInBits() , t1->getScalarSizeInBits() );
      return llvm::Type::getIntNTy(TheContext , upper );
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
    vecParamType.push_back( llvm::Type::getInt1Ty(TheContext) );
    return vecParamType.size()-1;
    // llvm::Argument * u8 = new llvm::Argument( llvm::Type::getInt8Ty(TheContext) , param_next() , mainFunc );
    // return llvm_cast( llvm_type<bool>::value , u8 );
  }
  template<> ParamRef llvm_add_param<bool*>() { 
    vecParamType.push_back( llvm::Type::getInt1PtrTy(TheContext) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<int64_t>() { 
    vecParamType.push_back( llvm::Type::getInt64Ty(TheContext) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<int>() { 
    vecParamType.push_back( llvm::Type::getInt32Ty(TheContext) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<int*>() { 
    vecParamType.push_back( llvm::Type::getInt32PtrTy(TheContext) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<float>() { 
    vecParamType.push_back( llvm::Type::getFloatTy(TheContext) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<float*>() { 
    vecParamType.push_back( llvm::Type::getFloatPtrTy(TheContext) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<double>() { 
    vecParamType.push_back( llvm::Type::getDoubleTy(TheContext) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<double*>() { 
    vecParamType.push_back( llvm::Type::getDoublePtrTy(TheContext) );
    return vecParamType.size()-1;
  }



  llvm::BasicBlock * llvm_new_basic_block()
  {
    std::ostringstream oss;
    oss << "L" << llvm_counters::label_counter++;
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(TheContext, oss.str() );
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
    return llvm::ConstantInt::getSigned( llvm::Type::getIntNTy(TheContext,32) , i );
  }

  llvm::Value * llvm_create_value( double v )
  {
    if (sizeof(REAL) == 4)
      return llvm::ConstantFP::get( llvm::Type::getFloatTy(TheContext) , v );
    else
      return llvm::ConstantFP::get( llvm::Type::getDoubleTy(TheContext) , v );
  }

  llvm::Value * llvm_create_value(int64_t v )  {return llvm::ConstantInt::get( llvm::Type::getInt64Ty(TheContext) , v );}
  llvm::Value * llvm_create_value(int v )  {return llvm::ConstantInt::get( llvm::Type::getInt32Ty(TheContext) , v );}
  llvm::Value * llvm_create_value(size_t v){return llvm::ConstantInt::get( llvm::Type::getInt32Ty(TheContext) , v );}
  llvm::Value * llvm_create_value(bool v ) {return llvm::ConstantInt::get( llvm::Type::getInt1Ty(TheContext) , v );}


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
    llvm::FunctionType *IntrinFnTy = llvm::FunctionType::get(llvm::Type::getVoidTy(TheContext), false);

    llvm::AttrBuilder ABuilder;
    ABuilder.addAttribute(llvm::Attribute::ReadNone);

    llvm::Constant *Bar = Mod->getOrInsertFunction( "llvm.nvvm.barrier0" , 
						    IntrinFnTy , 
						    llvm::AttributeSet::get(TheContext, 
									    llvm::AttributeSet::FunctionIndex, 
									    ABuilder)
						    );

    builder->CreateCall(Bar);
  }


  


  llvm::Value * llvm_special( const char * name )
  {
    llvm::FunctionType *IntrinFnTy = llvm::FunctionType::get(llvm::Type::getInt32Ty(TheContext), false);

    llvm::AttrBuilder ABuilder;
    ABuilder.addAttribute(llvm::Attribute::ReadNone);

    llvm::Constant *ReadTidX = Mod->getOrInsertFunction( name , 
							 IntrinFnTy , 
							 llvm::AttributeSet::get(TheContext, 
										 llvm::AttributeSet::FunctionIndex, 
										 ABuilder)
							 );

    return builder->CreateCall(ReadTidX);
  }



  llvm::Value * llvm_call_special_tidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.tid.x"); }
  llvm::Value * llvm_call_special_ntidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.ntid.x"); }
  llvm::Value * llvm_call_special_ctaidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.ctaid.x"); }
  llvm::Value * llvm_call_special_nctaidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.nctaid.x"); }
  llvm::Value * llvm_call_special_ctaidy() { return llvm_special("llvm.nvvm.read.ptx.sreg.ctaid.y"); }


  llvm::Value * llvm_thread_idx() { 
    llvm::Value * tidx = llvm_call_special_tidx();
    llvm::Value * ntidx = llvm_call_special_ntidx();
    llvm::Value * ctaidx = llvm_call_special_ctaidx();
    llvm::Value * ctaidy = llvm_call_special_ctaidy();
    llvm::Value * nctaidx = llvm_call_special_nctaidx();
    return llvm_add( llvm_mul( llvm_add( llvm_mul( ctaidy , nctaidx ) , ctaidx ) , ntidx ) , tidx );
  }
  


  void addKernelMetadata(llvm::Function *F) {
#if 0
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
#else
    llvm::Module *M = F->getParent();
    llvm::LLVMContext &Ctx = M->getContext();

    // Get "nvvm.annotations" metadata node
    llvm::NamedMDNode *MD = M->getOrInsertNamedMetadata("nvvm.annotations");

    // Create !{<func-ref>, metadata !"kernel", i32 1} node
    llvm::SmallVector<llvm::Metadata *, 3> MDVals;
    MDVals.push_back(llvm::ValueAsMetadata::get(F));
    MDVals.push_back(llvm::MDString::get(Ctx, "kernel"));
    MDVals.push_back(llvm::ValueAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), 1)));    //ConstantAsMetadata::get

    // Append metadata to nvvm.annotations
    MD->addOperand(llvm::MDNode::get(Ctx, MDVals));
#endif
  }


    void llvm_print_module( llvm::Module* m , const char * fname ) {
    std::error_code EC;
    llvm::raw_fd_ostream outfd( fname , EC, llvm::sys::fs::OpenFlags::F_Text);
    //ASSERT_FALSE(outfd.has_error());
    std::string banner;
    {
      llvm::outs() << "llvm_print_module ni\n";
#if 0
      llvm::PassManager PM;
      PM.add( llvm::createPrintModulePass( &outfd, false, banner ) ); 
      PM.run( *m );
#endif
    }
  }


  
  /// This routine adds optimization passes based on selected optimization level,
  /// OptLevel.
  ///
  /// OptLevel - Optimization Level
  static void AddOptimizationPasses(legacy::PassManagerBase &MPM,
				    legacy::FunctionPassManager &FPM,
				    TargetMachine *TM, unsigned OptLevel,
				    unsigned SizeLevel)
  {
    //QDPIO::cout << " adding opt passes..\n";

    const bool DisableInline = llvm_opt::DisableInline;
    const bool UnitAtATime = llvm_opt::UnitAtATime;
    const bool DisableLoopUnrolling = llvm_opt::DisableLoopUnrolling;
    const bool DisableLoopVectorization = llvm_opt::DisableLoopVectorization;
    const bool DisableSLPVectorization = llvm_opt::DisableSLPVectorization;
      
    FPM.add(createVerifierPass()); // Verify that input is correct

    PassManagerBuilder Builder;
    Builder.OptLevel = OptLevel;
    Builder.SizeLevel = SizeLevel;

    if (DisableInline) {
      // No inlining pass
    } else if (OptLevel > 1) {
      Builder.Inliner = createFunctionInliningPass(OptLevel, SizeLevel);
    } else {
      Builder.Inliner = createAlwaysInlinerLegacyPass();
    }
    Builder.DisableUnitAtATime = !UnitAtATime;
    Builder.DisableUnrollLoops = DisableLoopUnrolling;

    // This is final, unless there is a #pragma vectorize enable
    if (DisableLoopVectorization)
      Builder.LoopVectorize = false;
    else 
      Builder.LoopVectorize = OptLevel > 1 && SizeLevel < 2;

    // When #pragma vectorize is on for SLP, do the same as above
    Builder.SLPVectorize = DisableSLPVectorization ? false : OptLevel > 1 && SizeLevel < 2;

    // Add target-specific passes that need to run as early as possible.
    if (TM)
      Builder.addExtension(
			   llvm::PassManagerBuilder::EP_EarlyAsPossible,
			   [&](const llvm::PassManagerBuilder &, llvm::legacy::PassManagerBase &PM) {
			     TM->addEarlyAsPossiblePasses(PM);
			   });

    Builder.populateFunctionPassManager(FPM);
    Builder.populateModulePassManager(MPM);
  }


  void optimize_module( std::unique_ptr< llvm::TargetMachine >& TM )
  {
    //QDPIO::cout << "optimize module...\n";
    
    llvm::legacy::PassManager Passes;

    llvm::Triple ModuleTriple(Mod->getTargetTriple());

    llvm::TargetLibraryInfoImpl TLII(ModuleTriple);

    Passes.add(new llvm::TargetLibraryInfoWrapperPass(TLII));

    Passes.add(createTargetTransformInfoWrapperPass( TM->getTargetIRAnalysis() ) );

    std::unique_ptr<llvm::legacy::FunctionPassManager> FPasses;

    FPasses.reset(new llvm::legacy::FunctionPassManager(Mod));
    FPasses->add(createTargetTransformInfoWrapperPass( TM->getTargetIRAnalysis() ) );

    AddOptimizationPasses(Passes, *FPasses, TM.get(), llvm_opt::opt_level , 0);

    if (FPasses) {
      FPasses->doInitialization();
      for (Function &F : *Mod)
	FPasses->run(F);
      FPasses->doFinalization();
    }

    Passes.add(createVerifierPass());

    Passes.run(*Mod);
  }
  

  std::string get_PTX_from_Module_using_llvm( llvm::Module *Mod )
  {
    //QDPIO::cout << "get PTX using NVPTC..\n";

    llvm::Triple triple("nvptx64-nvidia-cuda");
      
    std::string Error;
    const llvm::Target *TheTarget = llvm::TargetRegistry::lookupTarget( "", triple, Error);
    if (!TheTarget) {
      llvm::errs() << "Error looking up target: " << Error;
      exit(1);
    }

    //QDPIO::cout << "target name: " << TheTarget->getName() << "\n";

    //llvm::Optional<llvm::Reloc::Model> relocModel;
    // if (m_generatePIC) 
    // relocModel = llvm::Reloc::PIC_;


    //llvm::TargetOptions Options;

    llvm::TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
    // Options.DisableIntegratedAS = llvm::NoIntegratedAssembler;
    // Options.MCOptions.ShowMCEncoding = llvm::ShowMCEncoding;
    // Options.MCOptions.MCUseDwarfDirectory = llvm::EnableDwarfDirectory;
    // Options.MCOptions.AsmVerbose = llvm::AsmVerbose;
    // Options.MCOptions.PreserveAsmComments = llvm::PreserveComments;
    // Options.MCOptions.IASSearchPaths = llvm::IncludeDirs;

    
    //std::unique_ptr<llvm::TargetMachine> target(TheTarget->createTargetMachine(TheTriple.getTriple(),"sm_50", "ptx50", Options , relocModel ));

    auto major = DeviceParams::Instance().getMajor();
    auto minor = DeviceParams::Instance().getMinor();

    std::ostringstream oss;
    oss << "sm_" << major * 10 + minor;

    std::string compute = oss.str();

    //QDPIO::cout << "create target machine for compute capability " << compute << "\n";
    
    std::unique_ptr<llvm::TargetMachine> target_machine(TheTarget->createTargetMachine(
										       "nvptx64-nvidia-cuda",
										       compute,
										       "",
										       llvm::TargetOptions(),
										       getRelocModel(),
										       llvm::CodeModel::Default,
										       llvm::CodeGenOpt::Aggressive ));

    assert(target_machine.get() && "Could not allocate target machine!");

    //QDPIO::cout << "target machine cpu:     " << target_machine->getTargetCPU().str() << "\n";
    //QDPIO::cout << "target machine feature: " << target_machine->getTargetFeatureString().str() << "\n";
 
    //llvm::TargetMachine &Target = *target.get();


    //std::string str;
    //llvm::raw_string_ostream rss(str);
    //llvm::formatted_raw_ostream FOS(rss);

    llvm::legacy::PassManager PM;
    //FOS <<  "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64\";\n";
    Mod->setTargetTriple( "nvptx64-nvidia-cuda" );

    llvm::TargetLibraryInfoImpl TLII(Triple(Mod->getTargetTriple()));
    PM.add(new TargetLibraryInfoWrapperPass(TLII));
    //PM.add(new llvm::TargetLibraryInfoWrapperPass(llvm::Triple(Mod->getTargetTriple())));

    Mod->setDataLayout(target_machine->createDataLayout());
    //Mod->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");

    //setFunctionAttributes("sm_30", "", *Mod);  // !!!!!

    //QDPIO::cout << "BEFORE OPT ---------------\n";
    //Mod->dump();
    
    optimize_module( target_machine );

    //QDPIO::cout << "AFTER OPT ---------------\n";
    //Mod->dump();

    
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
    //QDP_info_primary( "Using module's data layout" );
    //PMTM.add(new llvm::DataLayoutPass(Mod));
#endif

    std::string str;
    llvm::raw_string_ostream rss(str);
    llvm::buffer_ostream bos(rss);



    
    // Ask the target to add backend passes as necessary.
    if (target_machine->addPassesToEmitFile(PM, bos ,  llvm::TargetMachine::CGFT_AssemblyFile )) {
      llvm::errs() << ": target does not support generation of this"
		   << " file type!\n";
      exit(1);
    }


    //Mod->dump();
    //QDPIO::cout << "(module right before PTX codegen)------\n";
	
    //QDPIO::cout << "PTX code generation\n";
    PM.run(*Mod);
    //bos.flush();

    //QDPIO::cout << "PTX generated2: " << bos.str().str() << " (end)\n";

    return bos.str().str();
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

  void find_attr(std::string& str)
  {
    mapAttr.clear();
    size_t pos = 0;
    while((pos = str.find("attributes #", pos)) != std::string::npos)
    {
	  size_t pos_space = str.find(" ", pos+12);
	  std::string num = str.substr(pos+12,pos_space-pos-12);
	  num = " #"+num;
	  //QDPIO::cout << "# num found = " << num << "()\n";
	  size_t pos_open = str.find("{", pos_space);
	  size_t pos_close = str.find("}", pos_open);
	  std::string val = str.substr(pos_open+1,pos_close-pos_open-1);
	  //QDPIO::cout << "# val found = " << val << "\n";
	  str.replace(pos, pos_close-pos+1, "");
	  if (mapAttr.count(num) > 0)
	    QDP_error_exit("unexp.");
	  mapAttr[num]=val;
    }
  }





  // LLVM 4.0
  bool all_but_main(const llvm::GlobalValue & gv)
  {
    return gv.getName().str() == "main";
  }


  std::string llvm_get_ptx_kernel(const char* fname)
  {
    //QDPIO::cout << "get PTX..\n";
    //QDPIO::cout << "enter get_ptx_kernel------\n";
    //Mod->dump();
    //QDP_info_primary("Internalizing module");

    //const char *ExportList[] = { "main" };

    llvm::StringMap<int> Mapping;
    Mapping["__CUDA_FTZ"] = llvm_opt::nvptx_FTZ;

    std::string banner;

    llvm::legacy::PassManager OurPM;
    OurPM.add( llvm::createInternalizePass( all_but_main ) );
    OurPM.add( llvm::createNVVMReflectPass(Mapping));
    OurPM.run( *Mod );


    //QDP_info_primary("Running optimization passes on module");

    llvm::legacy::PassManager PM;
    PM.add( llvm::createGlobalDCEPass() );
    PM.run( *Mod );


    //QDPIO::cout << "------------------------------------------------ new module\n";
    //Mod->dump();
    //QDPIO::cout << "--------------------------------------------------------------\n";

    //Mod->dump();


    //llvm_print_module(Mod,"ir_internalized_reflected_globalDCE.ll");

    //std::string str = get_PTX_from_Module_using_nvvm( Mod );
    std::string str = get_PTX_from_Module_using_llvm( Mod );

#if 0
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





  CUfunction llvm_get_cufunction(const char* fname)
  {
    CUfunction func;
    CUresult ret;
    CUmodule cuModule;

    addKernelMetadata( mainFunc );

    // llvm::FunctionType *funcType = mainFunc->getFunctionType();
    // funcType->dump();

    std::string ptx_kernel = llvm_get_ptx_kernel(fname);

    //QDP_info_primary("Loading PTX kernel with the CUDA driver");

    //QDPIO::cout << ptx_kernel << "\n";
    
    ret = cuModuleLoadData(&cuModule, (void*)ptx_kernel.c_str());
    //ret = cuModuleLoadDataEx( &cuModule , ptx_kernel.c_str() , 0 , 0 , 0 );

    if (ret) {
      if (Layout::primaryNode()) {
	QDP_info_primary("Error loading external data. Dumping kernel to %s.",fname);
#if 1
	std::ofstream out(fname);
	out << ptx_kernel;
	out.close();

	//Mod->dump();
#endif
	QDP_error_exit("Abort.");
      }
    }

    ret = cuModuleGetFunction(&func, cuModule, "main");
    if (ret)
      QDP_error_exit("Error returned from cuModuleGetFunction. Abort.");

    mapCUFuncPTX[func] = ptx_kernel;

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

