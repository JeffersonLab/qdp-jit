#include "qdp_config.h"
#include "qdp_params.h"
#include "qdp_forward_jit.h"
#include "qdp_llvm_includes.h"
#include "qdp_llvm.h"

#include <system_error>
#include <memory>
#include <unistd.h>
#include <signal.h>

#include "qdp_jit.h"


using namespace llvm;

using namespace QDP;


namespace JIT {


  void CompilerHost::create_llvm_kernel_function(llvm::FunctionType *funcType)
  {
    mainFunc = llvm::Function::Create(funcType, llvm::Function::PrivateLinkage, str_kernel_name + "_intern" , Mod.get());
  }


  
  void CompilerHost::setup( CompilerFactory* fac , string ftype , string pretty )
  {
    Compiler::setup(fac,ftype,pretty);

    // This specific to host jit
    Mod->setDataLayout(globals::TheJIT->getDataLayout());
    
    // Count it
    llvm_jit_stats_jitted(false);
  }



  llvm::Value * CompilerHost::call_special_tidx()    { QDP_error_exit("shouldnt be here"); return nullptr; }
  llvm::Value * CompilerHost::call_special_ntidx()   { QDP_error_exit("shouldnt be here"); return nullptr; }
  llvm::Value * CompilerHost::call_special_ctaidx()  { QDP_error_exit("shouldnt be here"); return nullptr; }
  llvm::Value * CompilerHost::call_special_nctaidx() { QDP_error_exit("shouldnt be here"); return nullptr; }
  llvm::Value * CompilerHost::call_special_ctaidy()  { QDP_error_exit("shouldnt be here"); return nullptr; }
  void          CompilerHost::bar_sync()             { QDP_error_exit("shouldnt be here");}
  llvm::Value*  CompilerHost::get_shared_ptr( llvm::Type *ty , int n ) { QDP_error_exit("shouldnt be here"); return nullptr; }

  


  namespace {
    TargetInfo make_target( const CompileOptions& options )
    {
      TargetInfo t;
      t.global_addrspace = 0;
      t.local_addrspace  = 0;

      auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
      if (!jtmb)
	{
	  QDPIO::cerr << "host JIT: detecting the host target failed: " << llvm::toString( jtmb.takeError() ) << "\n";
	  QDP_abort(1);
	}

      t.triple = jtmb->getTargetTriple();

      if (options.verbose)
	{
	  QDPIO::cout << "Target triple: " << t.triple.str() << std::endl;
	}

      auto tm = jtmb->createTargetMachine();
      if (!tm)
	{
	  QDPIO::cerr << "host JIT: creating the target machine failed: " << llvm::toString( tm.takeError() ) << "\n";
	  QDP_abort(1);
	}
      t.machine = std::move( *tm );

      QDPIO::cout << "LLVM initialization (host jit)" << std::endl;
      QDPIO::cout << "  Target machine CPU                  : " << t.machine->getTargetCPU().str() << "\n";
      QDPIO::cout << "  Target triple                       : " << t.machine->getTargetTriple().str() << "\n";


      t.math["sin_f32"]="sinf";
      t.math["acos_f32"]="acosf";
      t.math["asin_f32"]="asinf";
      t.math["atan_f32"]="atanf";
      t.math["ceil_f32"]="ceilf";
      t.math["floor_f32"]="floorf";
      t.math["cos_f32"]="cosf";
      t.math["cosh_f32"]="coshf";
      t.math["exp_f32"]="expf";
      t.math["log_f32"]="logf";
      t.math["log10_f32"]="log10f";
      t.math["sinh_f32"]="sinhf";
      t.math["tan_f32"]="tanf";
      t.math["tanh_f32"]="tanhf";
      t.math["fabs_f32"]="fabsf";
      t.math["sqrt_f32"]="sqrtf";

      t.math["pow_f32"]="powf";
      t.math["atan2_f32"]="atan2f";

      t.math["sin_f64"]="sin";
      t.math["acos_f64"]="acos";
      t.math["asin_f64"]="asin";
      t.math["atan_f64"]="atan";
      t.math["ceil_f64"]="ceil";
      t.math["floor_f64"]="floor";
      t.math["cos_f64"]="cos";
      t.math["cosh_f64"]="cosh";
      t.math["exp_f64"]="exp";
      t.math["log_f64"]="log";
      t.math["log10_f64"]="log10";
      t.math["sinh_f64"]="sinh";
      t.math["tan_f64"]="tan";
      t.math["tanh_f64"]="tanh";
      t.math["fabs_f64"]="fabs";
      t.math["sqrt_f64"]="sqrt";

      t.math["pow_f64"]="pow";
      t.math["atan2_f64"]="atan2";

      return t;
    }
  }


  const TargetInfo& CompilerHost::target_info()
  {
    static TargetInfo t = make_target( options() );
    return t;
  }




  CodeObject CompilerHost::build_function_backend(JitFunction& func)
  {
    func.setIsGPU(false);

    StopWatch swatch(false);

    func.time_math = 0;


    std::vector< llvm::Type* > vecArgs;

    vecArgs.push_back( llvm::PointerType::get(*get_context(), 0) );
    llvm::FunctionType *funcType = 
      llvm::FunctionType::get( builder->getVoidTy() , 
			       llvm::ArrayRef<llvm::Type*>( vecArgs.data() , vecArgs.size() ) , 
			       false); // no vararg

    llvm::Function *mainFunc_extern = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, str_kernel_name , Mod.get());

    
    std::vector<llvm::Value*> vecCallArgument;
    llvm::Function::arg_iterator AI = mainFunc_extern->arg_begin();


    AI->setName( "arg_ptr" );

    // Create entry basic block
    llvm::BasicBlock* entry = llvm::BasicBlock::Create( *get_context(), "entrypoint", mainFunc_extern);
    builder->SetInsertPoint(entry);

    int i=0;
    for( std::vector< llvm::Type* >::const_iterator param_type = vecParamType.begin();  // + 1;
    	 param_type != vecParamType.end() ; 
    	 param_type++,i++ ) {

      llvm::Value* gep = builder->CreateGEP( llvm::PointerType::get(*get_context(), 0) , &*AI , create_value(i) );
      

      llvm::Value* arg = builder->CreateLoad( *param_type , gep ); // ptr_to_arg->getType()->getPointerElementType()
      
      vecCallArgument.push_back( arg );
    }

    builder->CreateCall( mainFunc , llvm::ArrayRef<llvm::Value*>( vecCallArgument.data() , vecCallArgument.size() ) );
    builder->CreateRetVoid();

    if (options().verbose)
      {
	QDPIO::cout << "\n\n";
	QDPIO::cout << str_pretty << std::endl;
	
	if (options().is_primary)
	  {
	    module_dump();
	  }
      }


    if ( options().opt_level != -1 )
      {
	opt( true );
      }

    
    swatch.reset();
    swatch.start();


    if (auto Err = globals::TheJIT->addIRModule(llvm::orc::ThreadSafeModule(std::move(Mod), *globals::TSC)))
      {
	QDPIO::cerr << "error adding IR module" << std::endl;
	QDP_abort(1);
      }
    
    
    ExitOnError ExitOnErr;
    
    auto EntrySym = ExitOnErr( globals::TheJIT->lookup( str_kernel_name ) );


    // Cast the entry point address to a function pointer.
    void* Entry = EntrySym.toPtr<void*>();

    
    func.set_kernel_name( str_kernel_name );
    func.set_pretty( str_pretty );
    func.set_function( Entry );

    
    swatch.stop();
    func.time_codegen = swatch.getTimeInMicroseconds();

    func.time_dynload = 0.0;

    return CodeObject();
  }




  Compiler* CompilerCreatorHost::create( CompilerFactory* fac , string ftype , string pretty )
  {
    CompilerHost* c = new CompilerHost;
    c->setup( fac , ftype , pretty );
    return c;
  }

  
  


  
} // JIT
