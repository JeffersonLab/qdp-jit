#include "qdp_config.h"
#include "qdp_params.h"

#include "qdp_llvm_includes.h"

#include "qdp_forward_jit.h"
#include "qdp_jit.h"

#include <system_error>
#include <memory>
#include <unistd.h>

#include <signal.h>

using namespace llvm;
using namespace llvm::codegen;

namespace JIT {

  namespace globals {
    std::unique_ptr< llvm::orc::LLJIT >             TheJIT;
  }
  
  CompilerFactory::CompilerFactory( bool _isPrimary , int _nodeNumber)
  {
    isPrimary  = _isPrimary;
    nodeNumber = _nodeNumber;

    JIT::globals::TSC = std::make_unique<ThreadSafeContext>( std::move( std::make_unique<LLVMContext>() ) );

    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();

    llvm::PassRegistry *Registry = llvm::PassRegistry::getPassRegistry();
    llvm::initializeCore(*Registry);
    llvm::initializeCodeGen(*Registry);
    llvm::initializeLoopStrengthReducePass(*Registry);
    llvm::initializeLowerIntrinsicsPass(*Registry);
    //    llvm::initializeCountingFunctionInserterPass(*Registry);
    llvm::initializeUnreachableBlockElimLegacyPassPass(*Registry);
    llvm::initializeConstantHoistingLegacyPassPass(*Registry);

#if defined (QDP_HOST_COMPILER)
    if (qdp_jit_config_get_hostjit())
      {
	QDPIO::cout << "Creating host JIT .." << endl;

	ExitOnError ExitOnErr;
	globals::TheJIT = ExitOnErr( llvm::orc::LLJITBuilder().create() );

	QDPIO::cout << ".. successful" << std::endl;
      }
#endif
    
  }


  Compiler* CompilerFactory::createGPU( string ftype , string pretty )
  {
    return creatorGPU->create(this,ftype,pretty);
  }

  Compiler* CompilerFactory::createHost( string ftype , string pretty )
  {
#if defined (QDP_HOST_COMPILER)
    return creatorHost->create(this,ftype,pretty);
#else
    return nullptr;
#endif
  }



} // JIT
