#include "qdp_llvm.h"
#include "qdp_config.h"
#include "qdp_params.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"

#if defined (QDP_LLVM14) || defined (QDP_LLVM15) || defined (QDP_LLVM16)
#include "llvm/MC/TargetRegistry.h"
#else
#include "llvm/Support/TargetRegistry.h"
#endif

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"

#ifdef QDP_BACKEND_AVX
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
//#include "llvm/ExecutionEngine/Orc/TargetProcessControl.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
using namespace llvm::orc;
#endif

#ifdef QDP_BACKEND_ROCM
#include "lld/Common/Driver.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#if ! defined(QDP_ROCM_PRE)
#include "lld/Common/CommonLinkerContext.h"
#endif
#endif

#ifdef QDP_BACKEND_L0
#include "LLVMSPIRVLib/LLVMSPIRVLib.h"
#endif

#include <system_error>
#include <memory>
#include <unistd.h>

#include <signal.h>

    


using namespace llvm;
using namespace llvm::codegen;



namespace llvm {
  ModulePass *createNVVMReflectPass(unsigned int);
}


#ifdef QDP_BACKEND_AVX
#if defined (QDP_LLVM14) || defined (QDP_LLVM15) || defined (QDP_LLVM16)
class KaleidoscopeJIT {
private:
  std::unique_ptr<ExecutionSession> ES;

  DataLayout DL;
  MangleAndInterner Mangle;

  RTDyldObjectLinkingLayer ObjectLayer;
  IRCompileLayer CompileLayer;

  JITDylib &MainJD;

public:
  KaleidoscopeJIT(std::unique_ptr<ExecutionSession> ES,
                  JITTargetMachineBuilder JTMB, DataLayout DL)
      : ES(std::move(ES)), DL(std::move(DL)), Mangle(*this->ES, this->DL),
        ObjectLayer(*this->ES,
                    []() { return std::make_unique<SectionMemoryManager>(); }),
        CompileLayer(*this->ES, ObjectLayer,
                     std::make_unique<ConcurrentIRCompiler>(std::move(JTMB))),
        MainJD(this->ES->createBareJITDylib("<main>")) {
    MainJD.addGenerator(
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
            DL.getGlobalPrefix())));
  }

  ~KaleidoscopeJIT() {
    if (auto Err = ES->endSession())
      ES->reportError(std::move(Err));
  }

  static Expected<std::unique_ptr<KaleidoscopeJIT>> Create() {
    auto EPC = SelfExecutorProcessControl::Create();
    if (!EPC)
      return EPC.takeError();

    auto ES = std::make_unique<ExecutionSession>(std::move(*EPC));

    JITTargetMachineBuilder JTMB(
        ES->getExecutorProcessControl().getTargetTriple());

    auto DL = JTMB.getDefaultDataLayoutForTarget();
    if (!DL)
      return DL.takeError();

    return std::make_unique<KaleidoscopeJIT>(std::move(ES), std::move(JTMB),
                                             std::move(*DL));
  }

  const DataLayout &getDataLayout() const { return DL; }

  JITDylib &getMainJITDylib() { return MainJD; }

  Error addModule(ThreadSafeModule TSM, ResourceTrackerSP RT = nullptr) {
    if (!RT)
      RT = MainJD.getDefaultResourceTracker();
    return CompileLayer.add(RT, std::move(TSM));
  }

  Expected<JITEvaluatedSymbol> lookup(StringRef Name) {
    return ES->lookup({&MainJD}, Mangle(Name.str()));
  }
};
#else
class KaleidoscopeJIT {
private:
  std::unique_ptr<TargetProcessControl> TPC;
  std::unique_ptr<ExecutionSession> ES;

  DataLayout DL;
  MangleAndInterner Mangle;

  RTDyldObjectLinkingLayer ObjectLayer;
  IRCompileLayer CompileLayer;

  JITDylib &MainJD;

public:
  KaleidoscopeJIT(std::unique_ptr<TargetProcessControl> TPC,
                  std::unique_ptr<ExecutionSession> ES,
                  JITTargetMachineBuilder JTMB, DataLayout DL)
      : TPC(std::move(TPC)), ES(std::move(ES)), DL(std::move(DL)),
        Mangle(*this->ES, this->DL),
        ObjectLayer(*this->ES,
                    []() { return std::make_unique<SectionMemoryManager>(); }),
        CompileLayer(*this->ES, ObjectLayer,
                     std::make_unique<ConcurrentIRCompiler>(std::move(JTMB))),
        MainJD(this->ES->createBareJITDylib("<main>")) {
    MainJD.addGenerator(
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
            DL.getGlobalPrefix())));
  }

  ~KaleidoscopeJIT() {
    if (auto Err = ES->endSession())
      ES->reportError(std::move(Err));
  }

  static Expected<std::unique_ptr<KaleidoscopeJIT>> Create() {
    auto SSP = std::make_shared<SymbolStringPool>();
    auto TPC = SelfTargetProcessControl::Create(SSP);
    if (!TPC)
      return TPC.takeError();

    auto ES = std::make_unique<ExecutionSession>(std::move(SSP));

    JITTargetMachineBuilder JTMB((*TPC)->getTargetTriple());

    llvm::outs() << "feature string: " << JTMB.getFeatures().getString() << "\n";
	
    llvm::outs() << "adding features...\n";
    JTMB.addFeatures({"+avx2"});
    
    llvm::outs() << "feature string: " << JTMB.getFeatures().getString() << "\n";
    

    
    auto DL = JTMB.getDefaultDataLayoutForTarget();
    if (!DL)
      return DL.takeError();

    return std::make_unique<KaleidoscopeJIT>(std::move(*TPC), std::move(ES),
                                             std::move(JTMB), std::move(*DL));
  }

  const DataLayout &getDataLayout() const { return DL; }

  JITDylib &getMainJITDylib() { return MainJD; }

  Error addModule(ThreadSafeModule TSM, ResourceTrackerSP RT = nullptr) {
    if (!RT)
      RT = MainJD.getDefaultResourceTracker();
    return CompileLayer.add(RT, std::move(TSM));
  }

  Expected<JITEvaluatedSymbol> lookup(StringRef Name) {
    return ES->lookup({&MainJD}, Mangle(Name.str()));
  }
};
#endif


namespace QDP
{
  ExitOnError ExitOnErr;

  std::unique_ptr<KaleidoscopeJIT> TheJIT;
}
#endif



#ifdef QDP_BACKEND_ROCM
namespace {
  int lldMain(int argc, const char **argv, llvm::raw_ostream &stdoutOS,
	      llvm::raw_ostream &stderrOS, bool exitEarly = true)
  {
    bool ret;
    std::vector<const char *> args(argv, argv + argc);

#if defined (QDP_LLVM15) || defined (QDP_LLVM16)
    ret = lld::elf::link(args, stdoutOS, stderrOS, exitEarly, false);
#else
    ret = lld::elf::link(args, exitEarly, stdoutOS, stderrOS);
#endif

#if ! defined(QDP_ROCM_PRE)
    // Cleanup
    lld::CommonLinkerContext::destroy();
#endif
    
    return ret ? 0 : 1;
  }
}
#endif


namespace QDP
{

  namespace JITSTATS {
    long lattice2dev  = 0;   // changing lattice data layout to device format
    long lattice2host = 0;   // changing lattice data layout to host format
    long jitted       = 0;   // functions not in DB, thus jit-built
  }
  
  void jit_stats_lattice2dev()  { ++JITSTATS::lattice2dev; }
  void jit_stats_lattice2host() { ++JITSTATS::lattice2host; }
  void jit_stats_jitted()       { ++JITSTATS::jitted; }
  
  long get_jit_stats_lattice2dev()  { return JITSTATS::lattice2dev; }
  long get_jit_stats_lattice2host() { return JITSTATS::lattice2host; }
  long get_jit_stats_jitted()       { return JITSTATS::jitted; }

  enum jitprec { i32 , f32 , f64 };

  namespace
  {
    StopWatch swatch_builder(false);
    
    std::string user_libdevice_path;
    std::string user_libdevice_name;

    bool clang_codegen = false;
    std::string clang_opt;

    //
    // Default search locations for libdevice
    // "ARCH"      gets replaced by "gfx908" (ROCm) or "70" (CUDA)
    // "CUDAPATH"  gets replaced by the envvar CUDAPATH or CUDA_PATH or NVIDIA_PATH
    // "CUDAVERSION"  gets replaced by the CUDA version, like "11.4"
    //
#ifdef QDP_BACKEND_ROCM
    std::vector<std::string> vec_str_libdevice_path = { ROCM_DIR };
    std::vector<std::string> vec_str_libdevice_path_append = { "llvm/lib/libdevice/" , "llvm/lib/" };
    std::vector<std::string> vec_str_libdevice_name = { "libm-amdgcn-ARCH.bc" , "libomptarget-amdgcn-ARCH.bc" , "libomptarget-new-amdpu-ARCH.bc" };
#elif QDP_BACKEND_CUDA
    std::vector<std::string> vec_str_libdevice_path = { "CUDAPATH" , "/usr/local/cuda/" , "/usr/lib/nvidia-cuda-toolkit/" };
    std::vector<std::string> vec_str_libdevice_path_append = { "nvvm/libdevice/" , "libdevice/" , "cuda/nvvm/libdevice/" , "cuda/CUDAVERSION/nvvm/libdevice/"};
    std::vector<std::string> vec_str_libdevice_name = { "libdevice.10.bc" , "libdevice.compute_ARCH.10.bc" };
#else
#endif
    
    llvm::Triple TheTriple;
    std::unique_ptr<llvm::TargetMachine> TargetMachine;
    
    llvm::BasicBlock  *bb_stack;
    llvm::BasicBlock  *bb_afterstack;

    BasicBlock::iterator it_stack;
    
    llvm::Function    *mainFunc;

    std::unique_ptr< llvm::LLVMContext > TheContext;
    std::unique_ptr< llvm::Module >      Mod;
    std::unique_ptr< llvm::Module >      module_libdevice;
#ifdef QDP_BACKEND_ROCM
    std::vector<std::unique_ptr< llvm::Module > >     module_ocml;
#endif

    std::unique_ptr< llvm::IRBuilder<> > builder;
    bool function_created;

    std::string str_func_type;
    std::string str_pretty;
    std::map<std::string,int> map_func_counter;
    std::string str_kernel_name;
    std::string str_arch;
    
    std::vector< llvm::Type* > vecParamType;
    std::vector< llvm::Value* > vecArgument;

    llvm::Value *r_arg_lo;
    llvm::Value *r_arg_hi;
    llvm::Value *r_arg_myId;
    llvm::Value *r_arg_ordered;
    llvm::Value *r_arg_start;

    std::map< std::string , std::string > mapMath;

    std::map< jitprec , std::map< std::string , int > > math_declarations;
  }

#ifdef QDP_BACKEND_ROCM
  namespace AMDspecific  {
    ParamRef __threads_per_group;
    ParamRef __grid_size_x;
  }
#elif QDP_BACKEND_AVX
  namespace AVXspecific  {
    ParamRef thread_num;
  }
#endif

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


  llvm::Value* llvm_builder_CreateLoad( llvm::Type* ty , llvm::Value* ptr )
  {
    return builder->CreateLoad( ty , ptr );
  }


  void         llvm_builder_CreateStore( llvm::Type* ty , llvm::Value* val , llvm::Value* ptr )
  {
    builder->CreateStore( val , ptr );
  }


  llvm::Value* llvm_builder_CreateGEP( llvm::Type*  ty  , llvm::Value* ptr , llvm::Value* idx )
  {
    return builder->CreateGEP( ty , ptr , idx );
  }

  

  void llvm_set_clang_codegen()
  {
    clang_codegen = true;
  }

  void llvm_set_clang_opt(const char* opt)
  {
    clang_opt = opt;
  }
      
  void llvm_set_libdevice_path(const char* path)
  {
    user_libdevice_path = std::string(path);
    if (user_libdevice_path.back() != '/')
      user_libdevice_path.append("/");
  }

  void llvm_set_libdevice_name(const char* name)
  {
    user_libdevice_name = std::string(name);
  }


  llvm::LLVMContext& llvm_get_context()
  {
    return *TheContext;
  }

  
  llvm::IRBuilder<>* llvm_get_builder()
  {
    return builder.get();
  }

  llvm::Module* llvm_get_module()
  {
    return Mod.get();
  }


  template<> llvm::Type* llvm_get_type<void>()       { return llvm::Type::getVoidTy(*TheContext); }
  template<> llvm::Type* llvm_get_type<jit_half_t>() { return llvm::Type::getHalfTy(*TheContext); }
  template<> llvm::Type* llvm_get_type<float>()      { return llvm::Type::getFloatTy(*TheContext); }
  template<> llvm::Type* llvm_get_type<double>()     { return llvm::Type::getDoubleTy(*TheContext); }
  template<> llvm::Type* llvm_get_type<int>()        { return llvm::Type::getIntNTy(*TheContext,32); }
  template<> llvm::Type* llvm_get_type<bool>()       { return llvm::Type::getIntNTy(*TheContext,8); }
  template<> llvm::Type* llvm_get_type<size_t>()     { return llvm::Type::getIntNTy(*TheContext,64); }

  
#if defined (QDP_LLVM15) || defined (QDP_LLVM16)
  template<> llvm::Type* llvm_get_type<jit_half_t*>() { return llvm::PointerType::get(*TheContext , qdp_jit_config_get_global_addrspace()); }
  template<> llvm::Type* llvm_get_type<float*>()      { return llvm::PointerType::get(*TheContext , qdp_jit_config_get_global_addrspace()); }
  template<> llvm::Type* llvm_get_type<double*>()     { return llvm::PointerType::get(*TheContext , qdp_jit_config_get_global_addrspace()); }
  template<> llvm::Type* llvm_get_type<int*>()        { return llvm::PointerType::get(*TheContext , qdp_jit_config_get_global_addrspace()); }
  template<> llvm::Type* llvm_get_type<bool*>()       { return llvm::PointerType::get(*TheContext , qdp_jit_config_get_global_addrspace()); }
#else
  template<> llvm::Type* llvm_get_type<jit_half_t*>() { return llvm::Type::getHalfPtrTy(*TheContext); }
  template<> llvm::Type* llvm_get_type<float*>()      { return llvm::Type::getFloatPtrTy(*TheContext); }
  template<> llvm::Type* llvm_get_type<double*>()     { return llvm::Type::getDoublePtrTy(*TheContext); }
  template<> llvm::Type* llvm_get_type<int*>()        { return llvm::Type::getIntNPtrTy(*TheContext,32); }
  template<> llvm::Type* llvm_get_type<bool*>()       { return llvm::Type::getIntNPtrTy(*TheContext,8); }
#endif


  
#if defined (QDP_CODEGEN_VECTOR)
  template<> llvm::Type* llvm_get_vectype<float>()   { return llvm::FixedVectorType::get( llvm::Type::getFloatTy(*TheContext) , Layout::virtualNodeNumber() );  }
  template<> llvm::Type* llvm_get_vectype<double>()  { return llvm::FixedVectorType::get( llvm::Type::getDoubleTy(*TheContext) , Layout::virtualNodeNumber() );  }
  template<> llvm::Type* llvm_get_vectype<bool>()    { return llvm::FixedVectorType::get( llvm::Type::getIntNTy(*TheContext,8) , Layout::virtualNodeNumber() );  }
#else
  template<> llvm::Type* llvm_get_vectype<float>()  { return llvm::Type::getFloatTy(*TheContext); }
  template<> llvm::Type* llvm_get_vectype<double>() { return llvm::Type::getDoubleTy(*TheContext); }
  template<> llvm::Type* llvm_get_vectype<bool>()   { return llvm::Type::getIntNTy(*TheContext,8); }
#endif
  


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



  
  llvm::Function *llvm_get_func( std::string name , jitprec p_out , jitprec p_in , int num_args )
  {
    if (math_declarations[p_in][name] == 1)
      {
	//QDPIO::cout << "math function declaration " << name << " found.\n";
	llvm::Function *func = Mod->getFunction(name.c_str());
	if (!func)
	  {
	    QDPIO::cerr << "Function " << name << " not found.\n";
	    QDP_abort(1);
	  }
	return func;
      }
    else
      {
	math_declarations[p_in][name] = 1;
	
	llvm::Type* type_in;
	switch(p_in)
	  {
	  case jitprec::i32:
	    type_in = llvm::Type::getInt32Ty(*TheContext);
	    break;
	  case jitprec::f32:
	    type_in = llvm::Type::getFloatTy(*TheContext);
	    break;
	  case jitprec::f64:
	    type_in = llvm::Type::getDoubleTy(*TheContext);
	    break;
	  }

	llvm::Type* type_out;
	switch(p_out)
	  {
	  case jitprec::i32:
	    type_out = llvm::Type::getInt32Ty(*TheContext);
	    break;
	  case jitprec::f32:
	    type_out = llvm::Type::getFloatTy(*TheContext);
	    break;
	  case jitprec::f64:
	    type_out = llvm::Type::getDoubleTy(*TheContext);
	    break;
	  }
	
	std::vector< llvm::Type* > Args( num_args , type_in );
	llvm::FunctionType *funcType = llvm::FunctionType::get( type_out , Args , false); // no vararg
	return llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, name.c_str() , Mod.get());
      }
  }

  

  namespace {
    std::string append_slash(std::string tmp)
    {
      if (tmp.back() != '/')
	tmp.append("/");
      return tmp;
    }
  }


#ifdef QDP_BACKEND_ROCM
  void llvm_init_ocml()
  {
    std::string arch = str_arch;
    auto index = arch.find("gfx", 0);
    if (index != std::string::npos)
      {
	arch.replace(index, 3, "" ); // Remove 
      }

    std::vector<std::string> libs;
    libs.push_back(std::string(ROCM_DIR) + "/amdgcn/bitcode/ocml.bc");
    libs.push_back(std::string(ROCM_DIR) + "/amdgcn/bitcode/oclc_finite_only_off.bc");
    libs.push_back(std::string(ROCM_DIR) + "/amdgcn/bitcode/oclc_isa_version_" + arch + ".bc");
    libs.push_back(std::string(ROCM_DIR) + "/amdgcn/bitcode/oclc_unsafe_math_off.bc");
    libs.push_back(std::string(ROCM_DIR) + "/amdgcn/bitcode/oclc_daz_opt_off.bc");
    libs.push_back(std::string(ROCM_DIR) + "/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc");
  
    libs.insert( libs.end() , jit_config_get_extra_lib().begin() , jit_config_get_extra_lib().end() );
    
    module_ocml.clear();
    
    for( int i = 0 ; i < libs.size() ; ++i )
      {
	std::string FileName = libs[i];

	if (jit_config_get_verbose_output())
	  {
	    QDPIO::cout << "Reading bitcode from " << FileName << "\n";
	  }
	
	std::ifstream ftmp(FileName.c_str());
	if (!ftmp.good())
	  {
	    QDPIO::cerr << "file not found:" << FileName << ". Skipping instead of aborting\n";
	  }
	else
	  {
	    ErrorOr<std::unique_ptr<MemoryBuffer>> mb = MemoryBuffer::getFile(FileName);
	    if (std::error_code ec = mb.getError()) {
	      errs() << ec.message();
	      QDP_abort(1);
	    }
  
	    llvm::Expected<std::unique_ptr<llvm::Module>> m = llvm::parseBitcodeFile(mb->get()->getMemBufferRef(), *TheContext);
	    if (std::error_code ec = errorToErrorCode(m.takeError()))
	      {
		errs() << "Error reading bitcode from " << FileName << ": " << ec.message() << "\n";
		QDP_abort(1);
	      }

	    module_ocml.push_back( std::move( m.get() ) );
	  }
      }
  }
#endif


#if defined(QDP_BACKEND_CUDA) || defined(QDP_BACKEND_ROCM)
  void llvm_init_libdevice()
  {
    static std::string FileName;

    if (FileName.empty())
      {
	std::vector<std::string> all;

	if (!user_libdevice_path.empty())
	  {
	    vec_str_libdevice_path.resize(0);
	    vec_str_libdevice_path.push_back( user_libdevice_path );
	    vec_str_libdevice_path_append.resize(0);
	    vec_str_libdevice_path_append.push_back( "" );
	  }
	
	if (!user_libdevice_name.empty())
	  {
	    vec_str_libdevice_name.resize(0);
	    vec_str_libdevice_name.push_back( user_libdevice_name );
	  }

	//
	// Replace ARCH with architecture string
	//
	for( auto name = vec_str_libdevice_name.begin() ; name != vec_str_libdevice_name.end() ; ++name )
	  {
	    std::string arch = str_arch;
	    auto index = arch.find("sm_", 0);
	    if (index != std::string::npos)
	      {
		arch.replace(index, 3, "" ); // Remove 
	      }
	    
	    index = name->find("ARCH", 0);
	    if (index == std::string::npos) continue;

	    name->replace(index, 4, arch );
	  }

#ifdef QDP_BACKEND_CUDA
	//
	// Replace CUDAVERSION with CUDA version string
	//
	for( auto pathappend = vec_str_libdevice_path_append.begin() ; pathappend != vec_str_libdevice_path_append.end() ; ++pathappend )
	  {
	    std::string version_str = std::to_string( gpu_SDK_version_major() ) + "." + std::to_string( gpu_SDK_version_minor() );

	    auto index = pathappend->find("CUDAVERSION", 0);
	    if (index != std::string::npos)
	      pathappend->replace(index, 11, version_str );
	  }
#endif
	
	//
	// Replace CUDAPATH with endvar
	//
	for( auto path = vec_str_libdevice_path.begin() ; path != vec_str_libdevice_path.end() ; ++path )
	  {
	    char *env = getenv( "CUDAPATH" );
	    if (!env)
	      env = getenv( "CUDA_PATH" );
	    if (!env)
	      env = getenv( "CUDA_HOME" );
	    if (!env)
	      env = getenv( "NVIDIA_PATH" );
	    if (env)
	      {
		std::string ENV(env);
		if (ENV.back() != '/')
		  ENV.append("/");

		auto index = path->find("CUDAPATH", 0);
		if (index != std::string::npos)
		  {
		    path->replace(index, 8, ENV );
		  }
	      }
	  }


	
	
	for( auto path = vec_str_libdevice_path.begin() ; path != vec_str_libdevice_path.end() ; ++path )
	  for( auto append = vec_str_libdevice_path_append.begin() ; append != vec_str_libdevice_path_append.end() ; ++append )
	    for( auto name = vec_str_libdevice_name.begin() ; name != vec_str_libdevice_name.end() ; ++name )
	      {
		std::string norm_path = append_slash( *path );
		all.push_back( norm_path + *append + *name );
	      }

	for( auto fname = all.begin() ; fname != all.end() ; ++fname )
	  {
	    if (jit_config_get_verbose_output())
	      {
		QDPIO::cout << "trying: " << *fname << std::endl;
	      }
	    
	    std::ifstream ftmp(fname->c_str());
	    if (ftmp.good())
	      {
		if (jit_config_get_verbose_output())
		  {
		    QDPIO::cout << "libdevice found.\n";
		  }

		FileName = *fname;
		break;
	      }
	  }
      }

    std::ifstream ftmp(FileName.c_str());
    if (!ftmp.good())
      {
	QDPIO::cerr << "libdevice not found:" << FileName << "\n";
	QDP_abort(1);
      }

    
    ErrorOr<std::unique_ptr<MemoryBuffer>> mb = MemoryBuffer::getFile(FileName);
    if (std::error_code ec = mb.getError()) {
      errs() << ec.message();
      QDP_abort(1);
    }
  
    llvm::Expected<std::unique_ptr<llvm::Module>> m = llvm::parseBitcodeFile(mb->get()->getMemBufferRef(), *TheContext);
    if (std::error_code ec = errorToErrorCode(m.takeError()))
      {
	errs() << "Error reading bitcode: " << ec.message() << "\n";
	QDP_abort(1);
      }

    module_libdevice.reset( m.get().release() );

    if (!module_libdevice) {
      QDPIO::cerr << "libdevice bitcode didn't read correctly.\n";
      QDP_abort(1);
    }
  }
#endif


  
#if defined (QDP_BACKEND_ROCM)
  void llvm_backend_init_rocm() {
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
    //    llvm::initializeCountingFunctionInserterPass(*Registry);
    llvm::initializeUnreachableBlockElimLegacyPassPass(*Registry);
    llvm::initializeConstantHoistingLegacyPassPass(*Registry);


    // Get the GPU arch, e.g.
    // sm_50 (CUDA)
    // gfx908 (ROCM)
    str_arch = gpu_get_arch();

    
    TheTriple.setArch (llvm::Triple::ArchType::amdgcn);
    TheTriple.setVendor (llvm::Triple::VendorType::AMD);
    TheTriple.setOS (llvm::Triple::OSType::AMDHSA);

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "triple set\n";
      }
    
    std::string Error;
    const llvm::Target *TheTarget = llvm::TargetRegistry::lookupTarget( TheTriple.str() , Error );
    if (!TheTarget) {
      std::cout << Error;
      QDPIO::cerr << "Something went wrong setting the target\n";
      QDP_abort(1);
    }


    
    llvm::TargetOptions Options;

    std::string FeaturesStr;
    
    TargetMachine.reset(TheTarget->createTargetMachine(
						       TheTriple.getTriple(), 
						       str_arch,
						       FeaturesStr, 
						       Options,
						       llvm::Reloc::PIC_
						       )
			);

    QDPIO::cout << "LLVM initialization" << std::endl;
    QDPIO::cout << "  Target machine CPU                  : " << TargetMachine->getTargetCPU().str() << "\n";
    QDPIO::cout << "  Target triple                       : " << TargetMachine->getTargetTriple().str() << "\n";


    mapMath["sin_f32"]="sinf";
    mapMath["acos_f32"]="acosf";
    mapMath["asin_f32"]="asinf";
    mapMath["atan_f32"]="atanf";
    mapMath["ceil_f32"]="ceilf";
    mapMath["floor_f32"]="floorf";
    mapMath["cos_f32"]="cosf";
    mapMath["cosh_f32"]="coshf";
    mapMath["exp_f32"]="expf";
    mapMath["log_f32"]="logf";
    mapMath["log10_f32"]="log10f";
    mapMath["sinh_f32"]="sinhf";
    mapMath["tan_f32"]="tanf";
    mapMath["tanh_f32"]="tanhf";
    mapMath["fabs_f32"]="fabsf";
    mapMath["sqrt_f32"]="sqrtf";
    mapMath["isfinite_f32"]="__finitef";
    // mapMath["isinf_f32"]="isinfdf";
    // mapMath["isnan_f32"]="isnandf";
    
    mapMath["pow_f32"]="powf";
    mapMath["atan2_f32"]="atan2f";
    
    mapMath["sin_f64"]="sin";
    mapMath["acos_f64"]="acos";
    mapMath["asin_f64"]="asin";
    mapMath["atan_f64"]="atan";
    mapMath["ceil_f64"]="ceil";
    mapMath["floor_f64"]="floor";
    mapMath["cos_f64"]="cos";
    mapMath["cosh_f64"]="cosh";
    mapMath["exp_f64"]="exp";
    mapMath["log_f64"]="log";
    mapMath["log10_f64"]="log10";
    mapMath["sinh_f64"]="sinh";
    mapMath["tan_f64"]="tan";
    mapMath["tanh_f64"]="tanh";
    mapMath["fabs_f64"]="fabs";
    mapMath["sqrt_f64"]="sqrt";
    mapMath["isfinite_f64"]="__finite";
    // mapMath["isinf_f64"]="isinfd";
    // mapMath["isnan_f64"]="isnand";
    
    mapMath["pow_f64"]="pow";
    mapMath["atan2_f64"]="atan2";
    
    //
    // libdevice is initialized in math_setup
    //
    //llvm_init_libdevice();
  }  
#endif
  

#if defined (QDP_BACKEND_CUDA)
  void llvm_backend_init_cuda() {
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
    //    llvm::initializeCountingFunctionInserterPass(*Registry);
    llvm::initializeUnreachableBlockElimLegacyPassPass(*Registry);
    llvm::initializeConstantHoistingLegacyPassPass(*Registry);


    // Get the GPU arch, e.g.
    // sm_50 (CUDA)
    // gfx908 (ROCM)
    str_arch = gpu_get_arch();


    std::string str_triple("nvptx64-nvidia-cuda");

    TheTriple.setTriple(str_triple);
      
    std::string Error;
    
    const llvm::Target *TheTarget = llvm::TargetRegistry::lookupTarget( "", TheTriple, Error);
    if (!TheTarget) {
      llvm::errs() << "Error looking up target: " << Error;
      exit(1);
    }


    llvm::TargetOptions options;

    TargetMachine.reset ( TheTarget->createTargetMachine(
							 TheTriple.str(),
							 str_arch,
							 "",
							 options,
							 Reloc::PIC_));
    
    if (!TargetMachine)
      {
	QDPIO::cerr << "Could not create LLVM target machine\n";
	QDP_abort(1);
      }

    QDPIO::cout << "LLVM initialization" << std::endl;
    QDPIO::cout << "  Target machine CPU                  : " << TargetMachine->getTargetCPU().str() << "\n";
    QDPIO::cout << "  Target triple                       : " << TargetMachine->getTargetTriple().str() << "\n";
    


    mapMath["sin_f32"]="__nv_sinf";
    mapMath["acos_f32"]="__nv_acosf";
    mapMath["asin_f32"]="__nv_asinf";
    mapMath["atan_f32"]="__nv_atanf";
    mapMath["ceil_f32"]="__nv_ceilf";
    mapMath["floor_f32"]="__nv_floorf";
    mapMath["cos_f32"]="__nv_cosf";
    mapMath["cosh_f32"]="__nv_coshf";
    mapMath["exp_f32"]="__nv_expf";
    mapMath["log_f32"]="__nv_logf";
    mapMath["log10_f32"]="__nv_log10f";
    mapMath["sinh_f32"]="__nv_sinhf";
    mapMath["tan_f32"]="__nv_tanf";
    mapMath["tanh_f32"]="__nv_tanhf";
    mapMath["fabs_f32"]="__nv_fabsf";
    mapMath["sqrt_f32"]="__nv_fsqrt_rn";
    mapMath["isfinite_f32"]="__nv_finitef";
    // mapMath["isinf_f32"]="__nv_isinfdf";
    // mapMath["isnan_f32"]="__nv_isnandf";
    
    mapMath["pow_f32"]="__nv_powf";
    mapMath["atan2_f32"]="__nv_atan2f";
    

    mapMath["sin_f64"]="__nv_sin";
    mapMath["acos_f64"]="__nv_acos";
    mapMath["asin_f64"]="__nv_asin";
    mapMath["atan_f64"]="__nv_atan";
    mapMath["ceil_f64"]="__nv_ceil";
    mapMath["floor_f64"]="__nv_floor";
    mapMath["cos_f64"]="__nv_cos";
    mapMath["cosh_f64"]="__nv_cosh";
    mapMath["exp_f64"]="__nv_exp";
    mapMath["log_f64"]="__nv_log";
    mapMath["log10_f64"]="__nv_log10";
    mapMath["sinh_f64"]="__nv_sinh";
    mapMath["tan_f64"]="__nv_tan";
    mapMath["tanh_f64"]="__nv_tanh";
    mapMath["fabs_f64"]="__nv_fabs";
    mapMath["sqrt_f64"]="__nv_dsqrt_rn";
    mapMath["isfinite_f64"]="__nv_isfinited";
    // mapMath["isinf_f64"]="__nv_isinfd";
    // mapMath["isnan_f64"]="__nv_isnand";
    
    mapMath["pow_f64"]="__nv_pow";
    mapMath["atan2_f64"]="__nv_atan2";
  }  
#endif


#if defined (QDP_BACKEND_AVX)
  void llvm_backend_init_avx() {
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
    //    llvm::initializeCountingFunctionInserterPass(*Registry);
    llvm::initializeUnreachableBlockElimLegacyPassPass(*Registry);
    llvm::initializeConstantHoistingLegacyPassPass(*Registry);


    QDPIO::cout << "Creating JIT" << std::endl;
    TheJIT = ExitOnErr(KaleidoscopeJIT::Create());
    QDPIO::cout << "Creating JIT successful" << std::endl;
    //InitializeModule();


#if 1
    // TheTriple.setArch (llvm::Triple::ArchType::amdgcn);
    // TheTriple.setVendor (llvm::Triple::VendorType::AMD);
    // TheTriple.setOS (llvm::Triple::OSType::AMDHSA);

    TheTriple.setTriple(sys::getDefaultTargetTriple());

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "TheTriple: " << TheTriple.str() << std::endl;
      }

    std::string Error;
    const llvm::Target *TheTarget = llvm::TargetRegistry::lookupTarget( TheTriple.str() , Error );
    if (!TheTarget) {
      std::cout << Error;
      QDPIO::cerr << "Something went wrong setting the target\n";
      QDP_abort(1);
    }

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "Target lookup successful: " << TheTarget->getName() << std::endl;
      }

    llvm::TargetOptions Options;

    std::string FeaturesStr;
    
    TargetMachine.reset(TheTarget->createTargetMachine(
						       TheTriple.getTriple(), 
						       str_arch,
						       FeaturesStr, 
						       Options,
						       llvm::Reloc::PIC_
						       )
			);

    
    QDPIO::cout << "LLVM initialized" << std::endl;
#endif

    // QDPIO::cout << "LLVM initialization" << std::endl;
    // QDPIO::cout << "  Target machine CPU                  : " << TargetMachine->getTargetCPU().str() << "\n";
    // QDPIO::cout << "  Target triple                       : " << TargetMachine->getTargetTriple().str() << "\n";

    mapMath["sin_f32"]="sinf";
    mapMath["acos_f32"]="acosf";
    mapMath["asin_f32"]="asinf";
    mapMath["atan_f32"]="atanf";
    mapMath["ceil_f32"]="ceilf";
    mapMath["floor_f32"]="floorf";
    mapMath["cos_f32"]="cosf";
    mapMath["cosh_f32"]="coshf";
    mapMath["exp_f32"]="expf";
    mapMath["log_f32"]="logf";
    mapMath["log10_f32"]="log10f";
    mapMath["sinh_f32"]="sinhf";
    mapMath["tan_f32"]="tanf";
    mapMath["tanh_f32"]="tanhf";
    mapMath["fabs_f32"]="fabsf";
    mapMath["sqrt_f32"]="sqrtf";
    
    mapMath["pow_f32"]="powf";
    mapMath["atan2_f32"]="atan2f";
    
    mapMath["sin_f64"]="sin";
    mapMath["acos_f64"]="acos";
    mapMath["asin_f64"]="asin";
    mapMath["atan_f64"]="atan";
    mapMath["ceil_f64"]="ceil";
    mapMath["floor_f64"]="floor";
    mapMath["cos_f64"]="cos";
    mapMath["cosh_f64"]="cosh";
    mapMath["exp_f64"]="exp";
    mapMath["log_f64"]="log";
    mapMath["log10_f64"]="log10";
    mapMath["sinh_f64"]="sinh";
    mapMath["tan_f64"]="tan";
    mapMath["tanh_f64"]="tanh";
    mapMath["fabs_f64"]="fabs";
    mapMath["sqrt_f64"]="sqrt";
    
    mapMath["pow_f64"]="pow";
    mapMath["atan2_f64"]="atan2";
  }
#endif



#if defined (QDP_BACKEND_L0)
  void llvm_backend_init_levelzero() {
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
    //    llvm::initializeCountingFunctionInserterPass(*Registry);
    llvm::initializeUnreachableBlockElimLegacyPassPass(*Registry);
    llvm::initializeConstantHoistingLegacyPassPass(*Registry);

    //std::string str_triple("spir64-unknown-unknown");
    std::string str_triple("spir64");
    TheTriple.setTriple(str_triple);

    QDPIO::cout << "LLVM initialization" << std::endl;
    QDPIO::cout << "  Target triple                       : " << str_triple << "\n";

    mapMath["sin_f32"]="_Z3sinf";
    mapMath["acos_f32"]="_Z4acosf";
    mapMath["asin_f32"]="_Z4asinf";
    mapMath["atan_f32"]="_Z4atanf";
    mapMath["ceil_f32"]="_Z4ceilf";
    mapMath["floor_f32"]="_Z4floorf";
    mapMath["cos_f32"]="_Z3cosf";
    mapMath["cosh_f32"]="_Z4coshf";
    mapMath["exp_f32"]="_Z3expf";
    mapMath["log_f32"]="_Z3logf";
    mapMath["log10_f32"]="_Z5log10f";
    mapMath["sinh_f32"]="_Z4sinhf";
    mapMath["tan_f32"]="_Z3tanf";
    mapMath["tanh_f32"]="_Z4tanhf";
    mapMath["fabs_f32"]="_Z4fabsf";
    mapMath["sqrt_f32"]="_Z4sqrtf";
    mapMath["isfinite_f32"]="_Z8isfinitef";
    // mapMath["isinf_f32"]="isinfdf";
    // mapMath["isnan_f32"]="isnandf";
    
    mapMath["pow_f32"]="_Z3powff";
    mapMath["atan2_f32"]="_Z5atan2ff";
    
    mapMath["sin_f64"]="_Z3sind";
    mapMath["acos_f64"]="_Z4acosd";
    mapMath["asin_f64"]="_Z4asind";
    mapMath["atan_f64"]="_Z4atand";
    mapMath["ceil_f64"]="_Z4ceild";
    mapMath["floor_f64"]="_Z4floord";
    mapMath["cos_f64"]="_Z3cosd";
    mapMath["cosh_f64"]="_Z4coshd";
    mapMath["exp_f64"]="_Z3expd";
    mapMath["log_f64"]="_Z3logd";
    mapMath["log10_f64"]="_Z5log10d";
    mapMath["sinh_f64"]="_Z4sinhd";
    mapMath["tan_f64"]="_Z3tand";
    mapMath["tanh_f64"]="_Z4tanhd";
    mapMath["fabs_f64"]="_Z4fabsd";
    mapMath["sqrt_f64"]="_Z4sqrtd";
    mapMath["isfinite_f64"]="_Z8isfinited";
    
    mapMath["pow_f64"]="_Z3powdd";
    mapMath["atan2_f64"]="_Z5atan2dd";
  }
#endif

  


  void llvm_backend_init()
  {
    TheContext = std::make_unique<LLVMContext>();

#ifdef QDP_BACKEND_ROCM
    llvm_backend_init_rocm();
#elif QDP_BACKEND_CUDA
    llvm_backend_init_cuda();
#elif QDP_BACKEND_AVX
    llvm_backend_init_avx();
#elif QDP_BACKEND_L0
    llvm_backend_init_levelzero();
#else
#error "No LLVM backend specified."
#endif
  }


  
  llvm::BasicBlock * llvm_get_insert_block() {
    return builder->GetInsertBlock();
  }


  void llvm_start_new_function( const char* ftype , const char* pretty )
  {
    // std::cout << ftype << "\n";
    // std::cout << pretty << "\n";
    
    swatch_builder.reset();
    swatch_builder.start();
    
    math_declarations.clear();
    
    str_func_type = ftype;
    str_pretty = pretty;
    str_kernel_name = str_func_type + std::to_string( map_func_counter[str_func_type]++ );
    
    // Count it
    jit_stats_jitted();
    
    //QDPIO::cout << "Starting new LLVM function..\n";
#if defined(QDP_BACKEND_AVX)
    TheContext.reset();
    TheContext = std::make_unique<LLVMContext>();
#endif
    
    Mod.reset();
    
    Mod = std::make_unique<llvm::Module>("module", *TheContext);

    builder.reset( new llvm::IRBuilder<>( *TheContext ) );

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "setting module data layout\n";
      }

#if defined(QDP_BACKEND_CUDA) || defined(QDP_BACKEND_ROCM)
    Mod->setDataLayout(TargetMachine->createDataLayout());
#elif defined(QDP_BACKEND_AVX)
    //QDPIO::cout << "setting module data layout\n";
    Mod->setDataLayout(TheJIT->getDataLayout());
#elif defined(QDP_BACKEND_L0)
    std::string dl = "";
    if (0) // 32 bit
      dl += "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:"
	"64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:"
	"32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:"
	"256:256-v256:256:256-v512:512:512-v1024:1024:1024";
    else
      dl += "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024";
	// "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:"
	// "64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:"
	// "32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:"
	// "256:256-v256:256:256-v512:512:512-v1024:1024:1024";

    Mod->setDataLayout(dl);
    Mod->setTargetTriple(TheTriple.getTriple());
#else
#error "No LLVM backend set"
#endif
    
    vecParamType.clear();
    vecArgument.clear();
    function_created = false;

#if defined (QDP_BACKEND_ROCM)
    AMDspecific::__threads_per_group = llvm_add_param<int>();
    AMDspecific::__grid_size_x       = llvm_add_param<int>();
#elif defined (QDP_BACKEND_AVX)
    AVXspecific::thread_num = llvm_add_param<int>();
#else
#endif
  }


  void llvm_create_function() {
    assert( !function_created );
    assert( vecParamType.size() > 0 );

    llvm::FunctionType *funcType = 
      llvm::FunctionType::get( builder->getVoidTy() , 
			       llvm::ArrayRef<llvm::Type*>( vecParamType.data() , vecParamType.size() ) , 
			       false); // no vararg
#if defined QDP_BACKEND_AVX
    mainFunc = llvm::Function::Create(funcType, llvm::Function::PrivateLinkage, str_kernel_name + "_intern" , Mod.get());
    //mainFunc = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, str_kernel_name , Mod.get());
#else
    mainFunc = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, str_kernel_name             , Mod.get());
#endif
    
#if defined QDP_BACKEND_ROCM
    mainFunc->setCallingConv( llvm::CallingConv::AMDGPU_KERNEL );
#elif defined QDP_BACKEND_L0
    mainFunc->setCallingConv( llvm::CallingConv::SPIR_KERNEL );
#else
    ;
#endif
    
    unsigned Idx = 0;
    for (llvm::Function::arg_iterator AI = mainFunc->arg_begin(), AE = mainFunc->arg_end() ; AI != AE ; ++AI, ++Idx) {
      AI->setName( std::string("arg")+std::to_string(Idx) );
      vecArgument.push_back( &*AI );
    }

    bb_stack = llvm::BasicBlock::Create(*TheContext, "stack", mainFunc);
    builder->SetInsertPoint(bb_stack);
    it_stack = builder->GetInsertPoint(); // probly bb_stack.begin()
    
    bb_afterstack = llvm::BasicBlock::Create(*TheContext, "afterstack" );
#if defined (QDP_LLVM16)
    mainFunc->insert(mainFunc->end(), bb_afterstack);
#else
    mainFunc->getBasicBlockList().push_back(bb_afterstack);
#endif

    builder->SetInsertPoint(bb_afterstack);

    llvm_counters::label_counter = 0;
    function_created = true;
  }



  llvm::Value * llvm_derefParam( ParamRef r ) {
    if (!function_created)
      llvm_create_function();
    assert( vecArgument.size() > (unsigned)r && "derefParam out of range");
    return vecArgument.at(r);
  }



  llvm::SwitchInst * llvm_switch_create( llvm::Value* val , llvm::BasicBlock* bb_default ) 
  {
    return builder->CreateSwitch( val , bb_default );
  }

  void llvm_switch_add_case( llvm::SwitchInst * SI , int val , llvm::BasicBlock* bb )
  {
    SI->addCase( builder->getInt32(val) , bb );
  }
  

  llvm::Value * llvm_phi( llvm::Type* type, unsigned num )
  {
    return builder->CreatePHI( type , num );
  }


  llvm::Type* promote_scalar( llvm::Type* t0 , llvm::Type* t1 )
  {
    if ( t0->isFloatingPointTy() || t1->isFloatingPointTy() ) {
      if ( t0->isDoubleTy() || t1->isDoubleTy() ) {
	return llvm::Type::getDoubleTy(*TheContext);
      } else {
	return llvm::Type::getFloatTy(*TheContext);
      }
    } else {
      unsigned upper = std::max( t0->getScalarSizeInBits() , t1->getScalarSizeInBits() );
      return llvm::Type::getIntNTy(*TheContext , upper );
    }
  }


  llvm::Type* promote( llvm::Type* t0 , llvm::Type* t1 )
  {
    if (t0->isVectorTy() != t1->isVectorTy())
      {
	raise(SIGSEGV);
	QDP_error_exit("interal error");
      }

    if (t0->isVectorTy() && t1->isVectorTy())
      {
	if (cast<llvm::VectorType>(t0)->getElementCount() != cast<llvm::VectorType>(t1)->getElementCount() )
	  {
	    llvm::outs() << "promote error: trying to promote vectors with not matching lengths." << "\n";
	    QDP_error_exit("interal error");
	  }
      }
    
    if (t0->isVectorTy())
      {
	return llvm::VectorType::get( promote_scalar( t0->getScalarType() , t1->getScalarType() ) , cast<llvm::VectorType>(t0)->getElementCount() );
      }
    else
      {
	return promote_scalar(t0,t1);
      }
  }

  

  llvm::Value* llvm_cast( llvm::Type *dest_type , llvm::Value *src )
  {
    assert( dest_type && "llvm_cast" );
    assert( src       && "llvm_cast" );
    
    if ( src->getType() == dest_type)
      return src;

    if ( dest_type->isArrayTy() )
      if ( dest_type->getArrayElementType() == src->getType() )
	return src;

    llvm::Value* ret = builder->CreateCast( llvm::CastInst::getCastOpcode( src , true , dest_type , true ) , 
					    src , dest_type , "" );
    return ret;
  }




  



  std::pair<llvm::Value*,llvm::Value*> llvm_normalize_values(llvm::Value* lhs , llvm::Value* rhs)
  {
    llvm::Value* lhs_new = lhs;
    llvm::Value* rhs_new = rhs;
    llvm::Type* args_type = promote( lhs->getType() , rhs->getType() );
    if ( args_type != lhs->getType() ) {
      lhs_new = llvm_cast( args_type , lhs );
    }
    if ( args_type != rhs->getType() ) {
      rhs_new = llvm_cast( args_type , rhs );
    }
    return std::make_pair(lhs_new,rhs_new);
  }
  



  llvm::Value* llvm_neg( llvm::Value* val ) {
    if ( val->getType()->getScalarType()->isFloatingPointTy() )
      return builder->CreateFNeg( val );
    else
      return builder->CreateNeg( val );
  }


  llvm::Value* llvm_rem( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFRem( vals.first , vals.second );
    else
      return builder->CreateSRem( vals.first , vals.second );
  }


  llvm::Value* llvm_shr( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    //   llvm::Type* args_type = vals.first->getType();
    //   assert( !args_type->isFloatingPointTy() );

    assert( ! ( vals.first->getType()->getScalarType()->isFloatingPointTy() ) );
    return builder->CreateAShr( vals.first , vals.second );
  }


  llvm::Value* llvm_shl( llvm::Value* lhs , llvm::Value* rhs ) {  
    auto vals = llvm_normalize_values(lhs,rhs);
    //  llvm::Type* args_type = vals.first->getType();
    //  assert( !args_type->isFloatingPointTy() );

    assert( ! ( vals.first->getType()->getScalarType()->isFloatingPointTy()  ) );
    return builder->CreateShl( vals.first , vals.second );
  }


  llvm::Value* llvm_and( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    // llvm::Type* args_type = vals.first->getType();
    // assert( !args_type->isFloatingPointTy() );
    assert( ! ( vals.first->getType()->getScalarType()->isFloatingPointTy()  ) );
    return builder->CreateAnd( vals.first , vals.second );
  }


  llvm::Value* llvm_or( llvm::Value* lhs , llvm::Value* rhs ) {  
    auto vals = llvm_normalize_values(lhs,rhs);
    //  llvm::Type* args_type = vals.first->getType();
    // assert( !args_type->isFloatingPointTy() );
    assert( ! ( vals.first->getType()->getScalarType()->isFloatingPointTy()  ) );

    return builder->CreateOr( vals.first , vals.second );
  }


  llvm::Value* llvm_xor( llvm::Value* lhs , llvm::Value* rhs ) {  
    auto vals = llvm_normalize_values(lhs,rhs);
    //    llvm::Type* args_type = vals.first->getType();
    //    assert( !args_type->isFloatingPointTy() );

    assert( ! ( vals.first->getType()->getScalarType()->isFloatingPointTy()  ) );

    return builder->CreateXor( vals.first , vals.second );
  }


  llvm::Value* llvm_mul( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    if ( vals.first->getType()->getScalarType()->isFloatingPointTy() )
      return builder->CreateFMul( vals.first , vals.second );
    else
      return builder->CreateMul( vals.first , vals.second );
  }


  llvm::Value* llvm_add( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFAdd( vals.first , vals.second );
    else
      return builder->CreateNSWAdd( vals.first , vals.second );
  }


  llvm::Value* llvm_sub( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFSub( vals.first , vals.second );
    else
      return builder->CreateSub( vals.first , vals.second );
  }


  llvm::Value* llvm_div( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFDiv( vals.first , vals.second );
    else 
      return builder->CreateSDiv( vals.first , vals.second );
  }


  llvm::Value* llvm_eq( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFCmpOEQ( vals.first , vals.second );
    else
      return builder->CreateICmpEQ( vals.first , vals.second );
  }

  
  llvm::Value* llvm_ne( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFCmpONE( vals.first , vals.second );
    else
      return builder->CreateICmpNE( vals.first , vals.second );
  }


  llvm::Value* llvm_ge( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFCmpOGE( vals.first , vals.second );
    else
      return builder->CreateICmpSGE( vals.first , vals.second );
  }


  llvm::Value* llvm_gt( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFCmpOGT( vals.first , vals.second );
    else
      return builder->CreateICmpSGT( vals.first , vals.second );
  }


  llvm::Value* llvm_le( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFCmpOLE( vals.first , vals.second );
    else
      return builder->CreateICmpSLE( vals.first , vals.second );
  }


  llvm::Value* llvm_lt( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFCmpOLT( vals.first , vals.second );
    else 
      return builder->CreateICmpSLT( vals.first , vals.second );
  }


  //
  // Convenience function definitions
  //
  llvm::Value* llvm_not( llvm::Value* lhs ) {
    llvm::Value* tr = llvm::ConstantInt::getTrue( llvm::Type::getInt1Ty(*TheContext) );

#if defined (QDP_CODEGEN_VECTOR)  
    if (lhs->getType()->isVectorTy())
      {
	llvm::Type* ty = llvm::FixedVectorType::get( llvm::Type::getInt1Ty(*TheContext) , Layout::virtualNodeNumber() );
	tr = llvm::ConstantInt::getTrue( ty );
      }
#endif
    
    return llvm_xor( llvm_trunc_i1( lhs ) , tr );
  }


#if defined(QDP_BACKEND_CUDA) || defined(QDP_BACKEND_ROCM)
  llvm::Value* llvm_get_shared_ptr( llvm::Type *ty , int n ) {

    llvm::GlobalVariable *gv = new llvm::GlobalVariable ( *Mod , 
							  llvm::ArrayType::get(ty,0) ,
							  false , 
							  llvm::GlobalVariable::ExternalLinkage, 
							  0, 
							  "shared_buffer", 
							  0, //GlobalVariable *InsertBefore=0, 
							  llvm::GlobalVariable::NotThreadLocal, //ThreadLocalMode=NotThreadLocal
							  qdp_jit_config_get_local_addrspace(), // unsigned AddressSpace=0, 
							  false); //bool isExternallyInitialized=false)
    return builder->CreatePointerCast(gv, llvm::PointerType::get( ty , qdp_jit_config_get_local_addrspace() ) );
  }
#else
  llvm::Value* llvm_get_shared_ptr( llvm::Type *ty , int n ) {

    llvm::GlobalVariable *gv = new llvm::GlobalVariable ( *Mod , 
							  llvm::ArrayType::get( ty , n ) ,
							  false , 
							  llvm::GlobalVariable::InternalLinkage, 
							  nullptr, 
							  "shared_buffer", 
							  0, //GlobalVariable *InsertBefore=0, 
							  llvm::GlobalVariable::NotThreadLocal, //ThreadLocalMode=NotThreadLocal
							  qdp_jit_config_get_local_addrspace(), // unsigned AddressSpace=0, 
							  false); //bool isExternallyInitialized=false)

    return builder->CreateGEP( gv->getType()->getElementType() , gv , { llvm_create_value(0) , llvm_create_value(0) } );
  }
#endif
  

#if defined (QDP_CODEGEN_VECTOR)  
  llvm::Value* llvm_insert_element( llvm::Value* vec , llvm::Value* val , int pos )
  {
    return builder->CreateInsertElement ( vec, val , pos );
  }

  llvm::Value* llvm_extract_element( llvm::Value* vec , int pos )
  {
    return builder->CreateExtractElement ( vec , pos );
  }

  
  llvm::Value* llvm_cast_to_vector( llvm::Value* val )
  {
    if (!isa<PointerType>(val->getType()))
      {
	QDPIO::cout << "internal error\n";
	QDP_abort(1);
      }

    llvm::Type * vec_type =  llvm::FixedVectorType::get( cast<PointerType>(val->getType())->getElementType() , Layout::virtualNodeNumber() );

    return builder->CreatePointerCast( val , llvm::PointerType::get( vec_type , qdp_jit_config_get_global_addrspace() ) );
  }

  llvm::Value* llvm_fill_vector( llvm::Value* val )
  {
    llvm::Type * vec_type =  llvm::FixedVectorType::get( val->getType() , Layout::virtualNodeNumber() );

    llvm::Value *vec = Constant::getNullValue(vec_type);
    
    for ( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      vec = llvm_insert_element( vec, val , i ); // builder->getInt32(i)
 
    return vec;
  }

  llvm::Value* llvm_get_zero_vector( llvm::Value* type_from_val )
  {
    llvm::Type* vec_type =  llvm::FixedVectorType::get( type_from_val->getType()->getScalarType() , Layout::virtualNodeNumber() );

    return Constant::getNullValue(vec_type);
  }
    
  
  

  bool llvm_is_ptr_to_vector( llvm::Value * ptr )
  {
    if (ptr->getType()->isPointerTy())
      {
	return ptr->getType()->getPointerElementType()->isVectorTy();
      }
    else
      return false;
  }

  
  llvm::Value * llvm_vecload_ptr_idx( llvm::Value * ptr , llvm::Value * idx )
  {
    if (llvm_is_ptr_to_vector(ptr))
      return llvm_load( llvm_createGEP( ptr , idx ) );
    else
      return llvm_load( llvm_cast_to_vector( llvm_createGEP( ptr , idx ) ) );
  }


  
  void llvm_vecstore_ptr_idx( llvm::Value * val , llvm::Value * ptr , llvm::Value * idx )
  {
    if (llvm_is_ptr_to_vector(ptr))
      llvm_store( val , llvm_createGEP( ptr , idx ) );
    else
      llvm_store( val , llvm_cast_to_vector( llvm_createGEP( ptr , idx ) ) );
  }


  llvm::Value* llvm_veccast( llvm::Type *dest_type , llvm::Value *src )
  {
    if (! src->getType()->isVectorTy())
      {
	llvm::outs() << "llvm_veccast: src value not a vector " << *src << "\n";
	QDP_abort(1);
      }
    
    if ( src->getType()->getScalarType() == dest_type)
      return src;

    llvm::Type* Dest = llvm::FixedVectorType::get( dest_type , Layout::virtualNodeNumber() );
    
    llvm::Value* ret = builder->CreateCast( llvm::CastInst::getCastOpcode( src , true , Dest , true ) , 
					    src , Dest , "" );
    return ret;
  }

  
#endif
  
  

  llvm::Value * llvm_alloca( llvm::Type* type , int elements )
  {
    auto it_save = builder->GetInsertPoint();
    auto bb_save = builder->GetInsertBlock();
    
    builder->SetInsertPoint(bb_stack, it_stack);

    auto DL = Mod->getDataLayout();
    unsigned AddrSpace = DL.getAllocaAddrSpace();

    //QDPIO::cout << "Alloca: using address space : " << AddrSpace << "\n";
    
    llvm::Value* ret = builder->CreateAlloca( type , AddrSpace , llvm_create_value(elements) );    // This can be a llvm::Value*

    it_stack = builder->GetInsertPoint();
    
    builder->SetInsertPoint(bb_save,it_save);

    return ret;
  }


  int llvm_get_last_param_count()
  {
    if (vecParamType.size() == 0)
      {
	QDPIO::cerr << "Internal error: llvm_get_last_param_count" << std::endl;
	QDP_abort(1);
      }
    return vecParamType.size() - 1 ;
  }

  template<> ParamRef llvm_add_param<bool>() { 
    vecParamType.push_back( llvm::Type::getInt8Ty(*TheContext) );
    return vecParamType.size()-1;
    // llvm::Argument * u8 = new llvm::Argument( llvm::Type::getInt8Ty(*TheContext) , param_next() , mainFunc );
    // return llvm_cast( llvm_type<bool>::value , u8 );
  }
  template<> ParamRef llvm_add_param<bool*>() { 
    vecParamType.push_back( llvm::Type::getInt8PtrTy(*TheContext,qdp_jit_config_get_global_addrspace()) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<int64_t>() { 
    vecParamType.push_back( llvm::Type::getInt64Ty(*TheContext) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<int>() { 
    vecParamType.push_back( llvm::Type::getInt32Ty(*TheContext) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<int*>() { 
    vecParamType.push_back( llvm::Type::getInt32PtrTy(*TheContext,qdp_jit_config_get_global_addrspace()) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<jit_half_t>() { 
    vecParamType.push_back( llvm::Type::getHalfTy(*TheContext) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<float>() { 
    vecParamType.push_back( llvm::Type::getFloatTy(*TheContext) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<jit_half_t*>() { 
    vecParamType.push_back( llvm::Type::getHalfPtrTy(*TheContext,qdp_jit_config_get_global_addrspace()) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<float*>() { 
    vecParamType.push_back( llvm::Type::getFloatPtrTy(*TheContext,qdp_jit_config_get_global_addrspace()) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<double>() { 
    vecParamType.push_back( llvm::Type::getDoubleTy(*TheContext) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<double*>() { 
    vecParamType.push_back( llvm::Type::getDoublePtrTy(*TheContext,qdp_jit_config_get_global_addrspace()) );
    return vecParamType.size()-1;
  }

  template<> ParamRef llvm_add_param<int**>() {
    vecParamType.push_back( llvm::PointerType::get( llvm::Type::getInt32PtrTy(*TheContext , qdp_jit_config_get_global_addrspace() ) , qdp_jit_config_get_global_addrspace() ) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<float**>() {
    vecParamType.push_back( llvm::PointerType::get( llvm::Type::getFloatPtrTy(*TheContext , qdp_jit_config_get_global_addrspace() ) , qdp_jit_config_get_global_addrspace() ) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<double**>() {
    vecParamType.push_back( llvm::PointerType::get( llvm::Type::getDoublePtrTy(*TheContext , qdp_jit_config_get_global_addrspace() ) , qdp_jit_config_get_global_addrspace() ) );
    return vecParamType.size()-1;
  }


  

  llvm::BasicBlock * llvm_new_basic_block()
  {
    std::ostringstream oss;
    oss << "L" << llvm_counters::label_counter++;
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(*TheContext, oss.str() );
#if defined (QDP_LLVM16)
    mainFunc->insert(mainFunc->end(), BB);
#else
    mainFunc->getBasicBlockList().push_back(BB);
#endif
    return BB;
  }


  void llvm_cond_branch(llvm::Value * cond, llvm::BasicBlock * thenBB, llvm::BasicBlock * elseBB)
  {
    builder->CreateCondBr( llvm_trunc_i1( cond ) , thenBB, elseBB);
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


  llvm::Type* llvm_val_type( llvm::Value* l )
  {
    return l->getType();
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
    return llvm::ConstantInt::getSigned( llvm::Type::getIntNTy(*TheContext,32) , i );
  }

  llvm::Value * llvm_create_value( double v )
  {
    if (sizeof(REAL) == 4)
      return llvm::ConstantFP::get( llvm::Type::getFloatTy(*TheContext) , v );
    else
      return llvm::ConstantFP::get( llvm::Type::getDoubleTy(*TheContext) , v );
  }

  llvm::Value * llvm_create_value(int64_t v ) {return llvm::ConstantInt::get( llvm::Type::getInt64Ty(*TheContext) , v );}
  llvm::Value * llvm_create_value(int v )     {return llvm::ConstantInt::get( llvm::Type::getInt32Ty(*TheContext) , v );}
  llvm::Value * llvm_create_value(size_t v)   {return llvm::ConstantInt::get( llvm::Type::getInt32Ty(*TheContext) , v );}
  llvm::Value * llvm_create_value(bool v )    {return llvm::ConstantInt::get( llvm::Type::getInt8Ty(*TheContext) , v );}


  
  llvm::Value* llvm_trunc_i1( llvm::Value* val )
  {
    llvm::Type* Dest = llvm::Type::getInt1Ty(*TheContext);
    
#ifdef QDP_CODEGEN_VECTOR
    if (val->getType()->isVectorTy())
      {
	Dest = llvm::FixedVectorType::get( llvm::Type::getInt1Ty(*TheContext) , Layout::virtualNodeNumber() );
      }
#endif
    
    return builder->CreateTrunc( val , Dest );
  }
  


  void llvm_add_incoming( llvm::Value* phi , llvm::Value* val , llvm::BasicBlock* bb )
  {
    dyn_cast<llvm::PHINode>(phi)->addIncoming( val , bb );
  }
  


  llvm::Value * llvm_special( const char * name , llvm::Type* ret , std::vector<llvm::Type*> param_types , std::vector<llvm::Value*> param_values )
  {
    llvm::FunctionType *FT = llvm::FunctionType::get( ret ,
    						      llvm::ArrayRef<llvm::Type*>( param_types.data() , param_types.size() ) , 
    						      false );

#if (defined (QDP_LLVM14) && (!defined (QDP_ROCM5FIX))) || defined (QDP_LLVM15) || defined (QDP_LLVM16)
    llvm::AttrBuilder ABuilder(*TheContext);
#else
    llvm::AttrBuilder ABuilder;
#endif
    
    //ABuilder.addAttribute(llvm::Attribute::ReadNone);
    ABuilder.addAttribute(llvm::Attribute::Convergent);

  
    auto F = Mod->getOrInsertFunction( name , 
					      FT , 
					      llvm::AttributeList::get(*TheContext, 
								       llvm::AttributeList::FunctionIndex, 
								       ABuilder)
					      );

#if defined (QDP_BACKEND_L0)
    if (auto FF = dyn_cast<llvm::Function>(F.getCallee())) {
      FF->setCallingConv( llvm::CallingConv::SPIR_FUNC );
    }
#endif    
    
    return builder->CreateCall(F,param_values);
    //return builder->CreateCall(F);
  }



#if defined (QDP_BACKEND_L0)
  void llvm_bar_sync()
  {
    // CLK_LOCAL_MEM_FENCE  == 1
    // CLK_GLOBAL_MEM_FENCE == 2
    llvm_special( "_Z7barrierj" , llvm_get_type<void>() , { llvm_get_type<int>() } , { llvm_create_value(3) } );
    //llvm_special( "_Z18work_group_barrierj" , llvm_get_type<void>() , { llvm_get_type<int>() } , { llvm_create_value(3) } );
    //llvm_special( "_Z18work_group_barrierj12memory_scope" , llvm_get_type<void>() , { llvm_get_type<int>() , llvm_get_type<int>() } , { llvm_create_value(3) , llvm_create_value(1) } );
    //llvm_special( "_Z18work_group_barrierj12memory_scope" , llvm_get_type<void>() , { llvm_get_type<int>() , llvm_get_type<int>() } , { llvm_create_value(3) , llvm_create_value(2) } );
  }
#else
  void llvm_bar_sync()
  {
    llvm::FunctionType *IntrinFnTy = llvm::FunctionType::get(llvm::Type::getVoidTy(*TheContext), false);

#if (defined (QDP_LLVM14) && (!defined (QDP_ROCM5FIX))) || defined (QDP_LLVM15) || defined (QDP_LLVM16)
    llvm::AttrBuilder ABuilder(*TheContext);
#else
    llvm::AttrBuilder ABuilder;
#endif
    
    ABuilder.addAttribute(llvm::Attribute::ReadNone);

#ifdef QDP_BACKEND_ROCM
    std::string bar_name("llvm.amdgcn.s.barrier");
#else
    std::string bar_name("llvm.nvvm.barrier0");
#endif
    
    auto Bar = Mod->getOrInsertFunction( bar_name.c_str() , 
					 IntrinFnTy , 
					 llvm::AttributeList::get(*TheContext, 
								  llvm::AttributeList::FunctionIndex, 
								  ABuilder) );

    builder->CreateCall(Bar);
  }
#endif

  

  

#if defined (QDP_BACKEND_ROCM)
  llvm::Value * llvm_call_special_workitem_x()  { return llvm_special("llvm.amdgcn.workitem.id.x" , llvm_get_type<int>() , {} , {} ); }
  llvm::Value * llvm_call_special_workgroup_x() { return llvm_special("llvm.amdgcn.workgroup.id.x", llvm_get_type<int>() , {} , {} ); }
  llvm::Value * llvm_call_special_workgroup_y() { return llvm_special("llvm.amdgcn.workgroup.id.y", llvm_get_type<int>() , {} , {} ); }

  llvm::Value * llvm_call_special_tidx() { return llvm_call_special_workitem_x(); }
  llvm::Value * llvm_call_special_ntidx() { return llvm_derefParam( AMDspecific::__threads_per_group ); }
  llvm::Value * llvm_call_special_ctaidx() { return llvm_call_special_workgroup_x(); }
  llvm::Value * llvm_call_special_nctaidx() { return llvm_derefParam( AMDspecific::__grid_size_x ); }
  llvm::Value * llvm_call_special_ctaidy() { return llvm_call_special_workgroup_y();  }
#elif defined (QDP_BACKEND_CUDA)
  llvm::Value * llvm_call_special_tidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.tid.x"      ,llvm_get_type<int>() , {} , {} ); }
  llvm::Value * llvm_call_special_ntidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.ntid.x"    ,llvm_get_type<int>() , {} , {} ); }
  llvm::Value * llvm_call_special_ctaidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.ctaid.x"  ,llvm_get_type<int>() , {} , {} ); }
  llvm::Value * llvm_call_special_nctaidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.nctaid.x",llvm_get_type<int>() , {} , {} ); }
  llvm::Value * llvm_call_special_ctaidy() { return llvm_special("llvm.nvvm.read.ptx.sreg.ctaid.y"  ,llvm_get_type<int>() , {} , {} ); }
#elif defined (QDP_BACKEND_L0)
  llvm::Value * llvm_call_special_get_global_id() { return llvm_special( "_Z13get_global_idj" ,	llvm_get_type<size_t>() , { llvm_get_type<int>() } , { llvm_create_value(0) } ); }
  llvm::Value * llvm_call_special_tidx()          { return llvm_cast( llvm_get_type<int>() , llvm_special( "_Z12get_local_idj" ,	llvm_get_type<size_t>() , { llvm_get_type<int>() } , { llvm_create_value(0) } )); }
  llvm::Value * llvm_call_special_ntidx()         { return llvm_cast( llvm_get_type<int>() , llvm_special( "_Z14get_local_sizej" ,llvm_get_type<size_t>() , { llvm_get_type<int>() } , { llvm_create_value(0) } )); }
  llvm::Value * llvm_call_special_ctaidx()        { return llvm_cast( llvm_get_type<int>() , llvm_special( "_Z12get_group_idj"   ,llvm_get_type<size_t>() , { llvm_get_type<int>() } , { llvm_create_value(0) } )); }
  llvm::Value * llvm_call_special_nctaidx()       { return llvm_cast( llvm_get_type<int>() , llvm_special( "_Z14get_num_groupsj" ,llvm_get_type<size_t>() , { llvm_get_type<int>() } , { llvm_create_value(0) } )); }
  llvm::Value * llvm_call_special_ctaidy()        { return llvm_cast( llvm_get_type<int>() , llvm_special( "_Z12get_group_idj"   ,llvm_get_type<size_t>() , { llvm_get_type<int>() } , { llvm_create_value(1) } )); }
#else
#endif

  
  
#if defined (QDP_BACKEND_ROCM) || defined (QDP_BACKEND_CUDA) || defined (QDP_BACKEND_L0)
  llvm::Value * llvm_thread_idx()
  {
    if (!function_created)
      llvm_create_function();
    llvm::Value * tidx = llvm_call_special_tidx();
    llvm::Value * ntidx = llvm_call_special_ntidx();
    llvm::Value * ctaidx = llvm_call_special_ctaidx();
    llvm::Value * ctaidy = llvm_call_special_ctaidy();
    llvm::Value * nctaidx = llvm_call_special_nctaidx();
    return llvm_add( llvm_mul( llvm_add( llvm_mul( ctaidy , nctaidx ) , ctaidx ) , ntidx ) , tidx );
  }
#elif defined (QDP_BACKEND_AVX)
  llvm::Value * llvm_thread_idx()
  { 
    return llvm_derefParam( AVXspecific::thread_num );
  }
#else
#endif


  void addKernelMetadata(llvm::Function *F) {
    auto i32_t = llvm::Type::getInt32Ty(*TheContext);
    
    llvm::Metadata *md_args[] = {
				 llvm::ValueAsMetadata::get(F),
				 MDString::get(*TheContext, "kernel"),
				 llvm::ValueAsMetadata::get(ConstantInt::get(i32_t, 1))};

    MDNode *md_node = MDNode::get(*TheContext, md_args);

    Mod->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md_node);
  }


  void llvm_print_module( llvm::Module* m , const char * fname )
  {
  }




  namespace {
    bool all_but_kernel_name(const llvm::GlobalValue & gv)
    {
      return gv.getName().str() == str_kernel_name;
    }
  }



  std::string get_ptx()
  {
    llvm::legacy::PassManager PM;

    std::string str;
    llvm::raw_string_ostream rss(str);
    llvm::buffer_ostream bos(rss);
    
    if (TargetMachine->addPassesToEmitFile(PM, bos , nullptr ,  llvm::CGFT_AssemblyFile )) {
      llvm::errs() << ": target does not support generation of this"
		   << " file type!\n";
      QDP_abort(1);
    }
    
    PM.run(*Mod);

    std::string ptx = bos.str().str();

    return ptx;
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







  void llvm_module_dump()
  {
    QDPIO::cout << "--------------------------  Module dump...\n";
    Mod->print(llvm::errs(), nullptr);
    QDPIO::cout << "--------------------------\n";
  }



  
  void llvm_opt(JitFunction& func)
  {
    StopWatch swatch(false);

    swatch.reset();
    swatch.start();

    llvm::legacy::PassManager PM2;

    if ( jit_config_get_instcombine() )
      PM2.add( llvm::createInstructionCombiningPass() );
    if ( jit_config_get_inline() )
    PM2.add( llvm::createFunctionInliningPass() );

  
    if (jit_config_get_verbose_output())
      {
	if ( jit_config_get_instcombine() )
	  QDPIO::cout << "LLVM opt instcombine\n";
	if ( jit_config_get_inline() )
	  QDPIO::cout << "LLVM opt inline\n";
      }

    PM2.run(*Mod);
    
    //llvm_module_dump();

    swatch.stop();
    func.time_passes = swatch.getTimeInMicroseconds();
  }

  
  

#ifdef QDP_BACKEND_CUDA
  void llvm_build_function_cuda(JitFunction& func)
  {
    addKernelMetadata( mainFunc );

    StopWatch swatch(false);
    swatch.start();
    
    if (math_declarations.size() > 0)
      {
	if (jit_config_get_verbose_output())
	  {
	    QDPIO::cout << "adding math function definitions ...\n";
	  }
	llvm_init_libdevice();
    
	if (jit_config_get_verbose_output())
	  {
	    QDPIO::cout << "link modules ...\n";
	  }
	std::string ErrorMsg;

	Mod->setDataLayout("");
	
	if (llvm::Linker::linkModules( *Mod , std::move( module_libdevice ) )) {  // llvm::Linker::PreserveSource
	  QDPIO::cerr << "Linking libdevice failed: " << ErrorMsg.c_str() << "\n";
	  QDP_abort(1);
	}
      }

    swatch.stop();
    func.time_math = swatch.getTimeInMicroseconds();
    swatch.reset();
    swatch.start();

    //QDPIO::cout << "setting module data layout\n";
    Mod->setDataLayout(TargetMachine->createDataLayout());

    uint32_t NVPTX_CUDA_FTZ = jit_config_get_CUDA_FTZ();

    Mod->addModuleFlag( llvm::Module::ModFlagBehavior::Override, "nvvm-reflect-ftz" , NVPTX_CUDA_FTZ );

    llvm::legacy::PassManager PM2;
    PM2.add( llvm::createInternalizePass( all_but_kernel_name ) );
#if 1
    unsigned int sm_gpu = gpu_getMajor() * 10 + gpu_getMinor();
    PM2.add( llvm::createNVVMReflectPass( sm_gpu ));
#endif
    PM2.add( llvm::createGlobalDCEPass() );

#if 0
    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "\n\n";
	QDPIO::cout << str_pretty << std::endl;
	
	if (Layout::primaryNode())
	  {
	    llvm_module_dump();
	  }

	std::string module_name = "module_" + str_kernel_name + ".bc";
	QDPIO::cout << "write code to " << module_name << "\n";
	std::error_code EC;
	llvm::raw_fd_ostream OS(module_name, EC, llvm::sys::fs::F_None);
	llvm::WriteBitcodeToFile(*Mod, OS);
	OS.flush();
      }
#endif

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "internalize and remove dead code ...\n";
      }
    PM2.run(*Mod);
    
    swatch.stop();
    func.time_passes = swatch.getTimeInMicroseconds();

    
    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "\n\n";
	QDPIO::cout << str_pretty << std::endl;
	
	if (Layout::primaryNode())
	  {
	    llvm_module_dump();
	  }

	std::string module_name = "module_" + str_kernel_name + ".bc";
	QDPIO::cout << "write code to " << module_name << "\n";
	std::error_code EC;

	llvm::raw_fd_ostream OS(module_name, EC, llvm::sys::fs::OF_None);

	llvm::WriteBitcodeToFile(*Mod, OS);
	OS.flush();
      }

    
    swatch.reset();
    swatch.start();
    std::string ptx_kernel = get_ptx();
    swatch.stop();
    func.time_codegen = swatch.getTimeInMicroseconds();

    
    swatch.reset();
    swatch.start();
    get_jitf( func , ptx_kernel , str_kernel_name , str_pretty , str_arch );
    swatch.stop();
    func.time_dynload = swatch.getTimeInMicroseconds();
  }
#endif

  
#ifdef QDP_BACKEND_ROCM
  void build_function_rocm_codegen( JitFunction& func , const std::string& shared_path)
  {
#if 0
    {
      QDPIO::cout << "write code to module.bc ...\n";
      std::error_code EC;
      llvm::raw_fd_ostream OS("module.bc", EC, llvm::sys::fs::F_None);
      llvm::WriteBitcodeToFile(*Mod, OS);
      OS.flush();
    }
#endif

    // previous location for setting the datalayout

    StopWatch swatch(false);
    swatch.start();
    
    if (math_declarations.size() > 0)
      {
	llvm_init_libdevice();

	llvm_init_ocml();
    
	std::string ErrorMsg;
	if (llvm::Linker::linkModules( *Mod , std::move( module_libdevice ) )) {  // llvm::Linker::PreserveSource
	  QDPIO::cerr << "Linking libdevice failed: " << ErrorMsg.c_str() << "\n";
	  QDP_abort(1);
	}

	for ( int i = 0 ; i < module_ocml.size() ; ++i )
	  {
	    if (jit_config_get_verbose_output())
	      {
		QDPIO::cout << "linking in additional library " << i << "\n";
	      }
	    if (llvm::Linker::linkModules( *Mod , std::move( module_ocml[i] ) )) {  // llvm::Linker::PreserveSource
	      QDPIO::cerr << "Linking additional library failed: " << ErrorMsg.c_str() << "\n";
	      QDP_abort(1);
	    }
	  }
      }

    swatch.stop();
    func.time_math = swatch.getTimeInMicroseconds();
    
#if 0
    {
      QDPIO::cout << "write code to module_linked.bc ...\n";
      std::error_code EC;
      llvm::raw_fd_ostream OS("module_linked.bc", EC, llvm::sys::fs::F_None);
      llvm::WriteBitcodeToFile(*Mod, OS);
      OS.flush();
    }
#endif

    swatch.reset();
    swatch.start();

    llvm::legacy::PassManager PM2;
    
    PM2.add( llvm::createInternalizePass( all_but_kernel_name ) );
    PM2.add( llvm::createGlobalDCEPass() );

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "internalize and remove dead code ...\n";
      }
    PM2.run(*Mod);
    
    //llvm_module_dump();

    swatch.stop();
    func.time_passes = swatch.getTimeInMicroseconds();

#if 0
    {
      QDPIO::cout << "write code to module_internal_dce.bc ...\n";
      std::error_code EC;
      llvm::raw_fd_ostream OS("module_internal_dce.bc", EC, llvm::sys::fs::F_None);
      llvm::WriteBitcodeToFile(*Mod, OS);
      OS.flush();
    }
#endif
    
    swatch.reset();
    swatch.start();
    
    std::string clang_name;
    if (clang_codegen)
      {
	clang_name =
	  jit_config_get_prepend_path() +
	  "module_" + str_kernel_name +
	  "_node_" + std::to_string(Layout::nodeNumber()) +
	  "_pid_" + std::to_string(::getpid()) +
	  ".bc";
	QDPIO::cout << "write code to " << clang_name << "\n";
	std::error_code EC;

	llvm::raw_fd_ostream OS(clang_name, EC, llvm::sys::fs::OF_None);

	llvm::WriteBitcodeToFile(*Mod, OS);
	OS.flush();
      }

    std::string isabin_path =
      jit_config_get_prepend_path() +
      "module_" + str_kernel_name +
      "_node_" + std::to_string(Layout::nodeNumber()) +
      "_pid_" + std::to_string(::getpid()) +
      ".o";

    if (clang_codegen)
      {	
	std::string clang_path = std::string(ROCM_DIR) + "/llvm/bin/clang";
	std::string command = clang_path + " -c " + clang_opt + " -target amdgcn-amd-amdhsa -mcpu=" + str_arch + " " + clang_name + " -o " + isabin_path;
	
	std::cout << "System: " << command.c_str() << "\n";
    
	system( command.c_str() );

	if (! jit_config_get_keepfiles() )
	  {
	    if (std::remove(clang_name.c_str()))
	      {
		QDPIO::cout << "Error removing file: " << clang_name << std::endl;
		QDP_abort(1);
	      }
	  }
      }
    else
      {
	//legacy::FunctionPassManager PerFunctionPasses(Mod.get());
	//PerFunctionPasses.add( createTargetTransformInfoWrapperPass( TargetMachine->getTargetIRAnalysis() ) );
    
	llvm::legacy::PassManager PM;

	llvm::TargetLibraryInfoImpl TLII( TheTriple );
	//TLII.addVectorizableFunctionsFromVecLib(TargetLibraryInfoImpl::Accelerate);
	//TLII.addVectorizableFunctionsFromVecLib(TargetLibraryInfoImpl::SVML);
	//TLII.addVectorizableFunctionsFromVecLib(TargetLibraryInfoImpl::MASSV);
	PM.add(new llvm::TargetLibraryInfoWrapperPass(TLII));

	//
	//
	// This PMBuilder.populateModulePassManager is essential
	PassManagerBuilder PMBuilder;
	PMBuilder.OptLevel = jit_config_get_codegen_opt();

	//PMBuilder.populateFunctionPassManager(PerFunctionPasses);
	PMBuilder.populateModulePassManager(PM);

	// New stuff
	//PMBuilder.Inliner = createAlwaysInlinerLegacyPass(true);
	//
	
#if 0
	QDPIO::cout << "Running function passes..\n";
	PerFunctionPasses.doInitialization();
	for (Function &F : *Mod)
	  if (!F.isDeclaration())
	    PerFunctionPasses.run(F);
	PerFunctionPasses.doFinalization();
	QDPIO::cout << "..done\n";
#endif


	if (jit_config_get_verbose_output())
	  {
	    QDPIO::cout << "running module passes ...\n";
	  }
	PM.run(*Mod);


#if 0
	{
	  QDPIO::cout << "write code to module.bc ...\n";
	  std::error_code EC;
	  llvm::raw_fd_ostream OS("module.bc", EC, llvm::sys::fs::F_None);
	  llvm::WriteBitcodeToFile(*Mod, OS);
	  OS.flush();
	}
#endif



	// ------------------- CODE GEN ----------------------
    
	llvm::legacy::PassManager CodeGenPasses;
    
	llvm::LLVMTargetMachine &LLVMTM = static_cast<llvm::LLVMTargetMachine &>(*TargetMachine);

	std::error_code ec;

	{
	  std::unique_ptr<llvm::raw_fd_ostream> isabin_fs( new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::OF_Text));
	  
	  if (TargetMachine->addPassesToEmitFile(CodeGenPasses, 
						 *isabin_fs,
						 nullptr,
						 llvm::CodeGenFileType::CGFT_ObjectFile ))

	    {
	      QDPIO::cerr << "target does not support generation of object file type!\n";
	      QDP_abort(1);
	    }

	  if (jit_config_get_verbose_output())
	    {
	      QDPIO::cout << "running code gen ...\n";
	    }
	    
	  CodeGenPasses.run(*Mod);
	}
      }

    swatch.stop();
    func.time_codegen = swatch.getTimeInMicroseconds();

    //
    // Call linker as a library
    //    
    int argc=5;
    const char *argv[] = { "ld" , "-shared" , isabin_path.c_str() , "-o" , shared_path.c_str() };

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "Library call to ld.lld: ";
	for ( int i = 0 ; i < argc ; ++i )
	  QDPIO::cout << argv[i] << " ";
	QDPIO::cout << std::endl;
      }

    if (lldMain(argc, argv, llvm::outs(), llvm::errs(), false))
      {
	QDPIO::cout << "Linker invocation unsuccessful" << std::endl;
	QDP_error_exit("calling ld.lld failed");
      }
    
    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "Linker invocation successful" << std::endl;
      }

    if (! jit_config_get_keepfiles() )
      {
	if (std::remove(isabin_path.c_str()))
	  {
	    QDPIO::cout << "Error removing file: " << shared_path << std::endl;
	    QDP_abort(1);
	  }
      }
    
    swatch.stop();
    func.time_linking = swatch.getTimeInMicroseconds();
  }



  
  void llvm_build_function_rocm(JitFunction& func)
  {
    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "\n\n";
	QDPIO::cout << str_pretty << std::endl;
	
	if (Layout::primaryNode())
	  {
	    llvm_module_dump();
	  }
      }
    
    std::string shared_path =
      jit_config_get_prepend_path() +
      "module_" + str_kernel_name +
      "_node_" + std::to_string(Layout::nodeNumber()) +
      "_pid_" + std::to_string(::getpid()) +
      ".so";

    // call codegen
    build_function_rocm_codegen( func , shared_path );
    
    std::ostringstream sstream;
    std::ifstream fin(shared_path, ios::binary);
    sstream << fin.rdbuf();
    std::string shared(sstream.str());

    if (! jit_config_get_keepfiles() )
      {
	if (std::remove(shared_path.c_str()))
	  {
	    QDPIO::cout << "Error removing file: " << shared_path << std::endl;
	    QDP_abort(1);
	  }
      }

    
    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "shared object file read back in. size = " << shared.size() << "\n";
      }

    StopWatch swatch(false);
    swatch.start();

    if (!get_jitf( func , shared , str_kernel_name , str_pretty , str_arch ))
      {
	// Something went wrong loading the module or finding the kernel
	// Print some diagnostics about the module
	QDPIO::cout << "Module declarations:" << std::endl;
	auto F = Mod->begin();
	while ( F != Mod->end() )
	  {
	    if (F->isDeclaration())
	      {
		QDPIO::cout << F->getName().str() << std::endl;
	      }
	    F++;
	  }
	sleep(1);
      }

    swatch.stop();
    func.time_dynload = swatch.getTimeInMicroseconds();
  }
#endif



#ifdef QDP_BACKEND_AVX
  void llvm_build_function_avx(JitFunction& func)
  {
    StopWatch swatch(false);

    func.time_math = 0;

#if 1
    // Right now a trampoline function which calls the main function
    // is necessary. For the auto-vectorizer we need the arguments to
    // to be noalias. Adding this attribute to a pointer is only possible
    // to function arguments. Since from host code I can only call
    // functions with a static signature, this cannot be done in one
    // step.

    // Create the 'trampoline' function

    std::vector< llvm::Type* > vecArgs;

    // omp_num_thread, or thread index
    vecArgs.push_back( llvm::Type::getInt32Ty( *TheContext ) ); 

    vecArgs.push_back( llvm::PointerType::get( llvm::ArrayType::get( llvm::Type::getInt8Ty(*TheContext) , 8 ) , 0  ) );
    llvm::FunctionType *funcType = 
      llvm::FunctionType::get( builder->getVoidTy() , 
			       llvm::ArrayRef<llvm::Type*>( vecArgs.data() , vecArgs.size() ) , 
			       false); // no vararg

    llvm::Function *mainFunc_extern = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, str_kernel_name , Mod.get());

    
    std::vector<llvm::Value*> vecCallArgument;
    llvm::Function::arg_iterator AI = mainFunc_extern->arg_begin();

    AI->setName( "idx" );
    vecCallArgument.push_back( &*AI );
    AI++;

    AI->setName( "arg_ptr" );

    // Create entry basic block
    llvm::BasicBlock* entry = llvm::BasicBlock::Create( *TheContext, "entrypoint", mainFunc_extern);
    builder->SetInsertPoint(entry);

    int i=0;
    for( std::vector< llvm::Type* >::const_iterator param_type = vecParamType.begin()+1 ; 
    	 param_type != vecParamType.end() ; 
    	 param_type++,i++ ) {
      //(*param_type)->dump(); std::cout << "\n";
      llvm::Value* gep = builder->CreateGEP( AI->getType()->getPointerElementType() , &*AI , llvm_create_value(i) );

      llvm::Type* param_ptr_type = llvm::PointerType::get( *param_type , 0  );
      llvm::Value* ptr_to_arg = builder->CreatePointerCast( gep , param_ptr_type );

      llvm::Value* arg = builder->CreateLoad( ptr_to_arg->getType()->getPointerElementType() , ptr_to_arg );

      vecCallArgument.push_back( arg );      
    }

    builder->CreateCall( mainFunc , llvm::ArrayRef<llvm::Value*>( vecCallArgument.data() , vecCallArgument.size() ) );
    builder->CreateRetVoid();

    //mainFunc_extern->dump();
#endif

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "\n\n";
	QDPIO::cout << str_pretty << std::endl;
	
	if (Layout::primaryNode())
	  {
	    llvm_module_dump();
	  }
      }

    
    // passes
    llvm_opt(func);

    
    if (0)
    {
      std::string module_name = "module_" + str_kernel_name + ".bc";
      QDPIO::cout << "write code to " << module_name << "\n";
      std::error_code EC;

      llvm::raw_fd_ostream OS(module_name, EC, llvm::sys::fs::OF_None);

      llvm::WriteBitcodeToFile(*Mod, OS);
    }

#if 0
    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "\n\n";
	QDPIO::cout << str_pretty << std::endl;
	
	if (Layout::primaryNode())
	  {
	    llvm_module_dump();
	  }
      }
#endif
    
    swatch.reset();
    swatch.start();

#if 0
    QDPIO::cout << "Print asm\n";
    {
      llvm::legacy::PassManager PM;

      std::string str;
      llvm::raw_string_ostream rss(str);
      llvm::buffer_ostream bos(rss);
    
      if (TargetMachine->addPassesToEmitFile(PM, bos , nullptr ,  llvm::CGFT_AssemblyFile )) {
	llvm::errs() << ": target does not support generation of this"
		     << " file type!\n";
	QDP_abort(1);
      }
    
      PM.run(*Mod);

      std::cout << bos.str().str() << std::endl;
    }
#endif

    //QDPIO::cout << "Add module\n";

    
    // Add module
    auto RT = TheJIT->getMainJITDylib().createResourceTracker();
    auto TSM = ThreadSafeModule( std::move(Mod) , std::move(TheContext) );
    ExitOnErr(TheJIT->addModule(std::move(TSM), RT));
    
    //QDPIO::cout << "Lookup: " << str_kernel_name << "\n";

    // Lookup 
    auto Sym = ExitOnErr(TheJIT->lookup( str_kernel_name ));

    func.set_kernel_name( str_kernel_name );
    func.set_pretty( str_pretty );
    func.set_function( (void*)Sym.getAddress() );

    
    swatch.stop();
    func.time_codegen = swatch.getTimeInMicroseconds();

    //    
    func.time_dynload = 0.0;
  }
#endif

  


#ifdef QDP_BACKEND_L0
  void llvm_build_function_levelzero_codegen(JitFunction& func, const std::string& spirv_path)
  {
#if 0
    SmallVector<llvm::Metadata *, 8> addressQuals;
    SmallVector<llvm::Metadata *, 8> accessQuals;
    SmallVector<llvm::Metadata *, 8> argTypeNames;
    SmallVector<llvm::Metadata *, 8> argTypeQuals;

    unsigned Idx = 0;
    for (llvm::Function::arg_iterator AI = mainFunc->arg_begin(), AE = mainFunc->arg_end() ; AI != AE ; ++AI, ++Idx)
      {
	llvm::Type* ty = AI->getType();
	
	if (ty->isPointerTy())
	  {
	    std::cout << "pointer: " << AI->getName().str() << "\n";
	    
	    addressQuals.push_back( llvm::ConstantAsMetadata::get(builder->getInt32( dyn_cast<llvm::PointerType>(ty)->getAddressSpace() )));

	    llvm::Type* ety = dyn_cast<llvm::PointerType>(ty)->getElementType();
	    std::string type_str;
	    if (ety == llvm::Type::getFloatTy(*TheContext))
	      type_str = "float*";
	    else if (ety == llvm::Type::getDoubleTy(*TheContext))
	      type_str = "double*";
	    else if (ety == llvm::Type::getInt8Ty(*TheContext))
	      type_str = "bool*";
	    else if (ety == llvm::Type::getInt32Ty(*TheContext))
	      type_str = "int*";
	    else
	      {
		QDPIO::cout << "ptr type not recognized\n";
		QDP_abort(1);
	      }
	    argTypeNames.push_back(llvm::MDString::get(*TheContext, type_str ));

	  }
	else
	  {
	    std::cout << "non-pointer: " << AI->getName().str() << "\n";
	    addressQuals.push_back( llvm::ConstantAsMetadata::get(builder->getInt32( 0 )));

	    std::string type_str;
	    if (ty == llvm::Type::getFloatTy(*TheContext))
	      type_str = "float";
	    else if (ty == llvm::Type::getDoubleTy(*TheContext))
	      type_str = "double";
	    else if (ty == llvm::Type::getInt8Ty(*TheContext))
	      type_str = "bool";
	    else if (ty == llvm::Type::getInt32Ty(*TheContext))
	      type_str = "int";
	    else
	      {
		QDPIO::cout << "type not recognized\n";
		QDP_abort(1);
	      }
	    argTypeNames.push_back(llvm::MDString::get(*TheContext, type_str ));

	  }
	accessQuals.push_back(llvm::MDString::get(*TheContext, "none"));

	argTypeQuals.push_back(llvm::MDString::get(*TheContext, "" ));
      }

    
    mainFunc->setMetadata("kernel_arg_addr_space" , llvm::MDNode::get(*TheContext, addressQuals));
    mainFunc->setMetadata("kernel_arg_access_qual", llvm::MDNode::get(*TheContext, accessQuals));
    mainFunc->setMetadata("kernel_arg_type",        llvm::MDNode::get(*TheContext, argTypeNames));
    mainFunc->setMetadata("kernel_arg_base_type",   llvm::MDNode::get(*TheContext, argTypeNames));
    mainFunc->setMetadata("kernel_arg_type_qual",   llvm::MDNode::get(*TheContext, argTypeQuals));
#endif
    
    
    
    StopWatch swatch(false);
    func.time_math = 0.;
    swatch.reset();
    swatch.start();

    //QDPIO::cout << "setting module data layout\n";
    //Mod->setDataLayout(TargetMachine->createDataLayout());

    if (jit_config_get_verbose_output())
      {
	//QDPIO::cout << "internalize and remove dead code ...\n";
      }
    
    swatch.stop();
    func.time_passes = swatch.getTimeInMicroseconds();

    
    if (jit_config_get_verbose_output() || jit_config_get_keepfiles())
      {
	// QDPIO::cout << "\n\n";
	// QDPIO::cout << str_pretty << std::endl;
	
	// if (Layout::primaryNode())
	//   {
	//     llvm_module_dump();
	//   }

	std::string module_name = "module_" + str_kernel_name + ".bc";
	QDPIO::cout << "write code to " << module_name << "\n";
	std::error_code EC;

	llvm::raw_fd_ostream OS(module_name, EC, llvm::sys::fs::OF_None);

	llvm::WriteBitcodeToFile(*Mod, OS);
	OS.flush();
      }

    
    swatch.reset();
    swatch.start();

    //SPIRV::TranslatorOpts Opts;

    {
      std::string Err;
      std::ofstream OutFile( spirv_path , std::ios::binary );

      if ( ! llvm::writeSpirv( Mod.get() , OutFile, Err) )
	{
	  QDPIO::cout << "Error writing SPIRV file\n\n";
	  QDPIO::cout << str_pretty << std::endl;
	  QDP_abort(1);
	}
    }

    swatch.stop();
    func.time_codegen = swatch.getTimeInMicroseconds();

    
    swatch.reset();
    swatch.start();
    
    get_jitf( func , spirv_path , str_kernel_name , str_pretty , str_arch );
    swatch.stop();
    func.time_dynload = swatch.getTimeInMicroseconds();
  }

  
  void llvm_build_function_levelzero(JitFunction& func)
  {
    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "\n\n";
	QDPIO::cout << str_pretty << std::endl;
	
	if (Layout::primaryNode())
	  {
	    llvm_module_dump();
	  }
      }
    
    std::string spirv_path =
	    "module_" + str_kernel_name +
	    "_node_" + std::to_string(Layout::nodeNumber()) +
	    "_pid_" + std::to_string(::getpid()) +
	    ".spv";

    // call codegen
    llvm_build_function_levelzero_codegen( func , spirv_path );

#if 0
    std::ostringstream sstream;
    std::ifstream fin(shared_path, ios::binary);
    sstream << fin.rdbuf();
    std::string shared(sstream.str());

    if (! jit_config_get_keepfiles() )
      {
	if (std::remove(shared_path.c_str()))
	  {
	    QDPIO::cout << "Error removing file: " << shared_path << std::endl;
	    QDP_abort(1);
	  }
      }

    
    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "shared object file read back in. size = " << shared.size() << "\n";
      }

    StopWatch swatch(false);
    swatch.start();

    if (!get_jitf( func , shared , str_kernel_name , str_pretty , str_arch ))
      {
	// Something went wrong loading the module or finding the kernel
	// Print some diagnostics about the module
	QDPIO::cout << "Module declarations:" << std::endl;
	auto F = Mod->begin();
	while ( F != Mod->end() )
	  {
	    if (F->isDeclaration())
	      {
		QDPIO::cout << F->getName().str() << std::endl;
	      }
	    F++;
	  }
	sleep(1);
      }

    swatch.stop();
    func.time_dynload = swatch.getTimeInMicroseconds();
#endif
  }
#endif


  

  void llvm_build_function(JitFunction& func)
  {
    builder->SetInsertPoint(bb_stack,it_stack);
    builder->CreateBr( bb_afterstack );

    swatch_builder.stop();
    func.time_builder = swatch_builder.getTimeInMicroseconds();
    
#ifdef QDP_BACKEND_ROCM
    llvm_build_function_rocm(func);
#elif QDP_BACKEND_CUDA
    llvm_build_function_cuda(func);
#elif QDP_BACKEND_AVX
    llvm_build_function_avx(func);
#elif QDP_BACKEND_L0
    llvm_build_function_levelzero(func);
#else
#error "No LLVM backend specified."
#endif
  }


  

  llvm::Value* llvm_call_f32( llvm::Function* func , llvm::Value* lhs )
  {
    llvm::Value* lhs_f32 = llvm_cast( llvm_get_type<float>() , lhs );
    return builder->CreateCall(func,lhs_f32);
  }

  llvm::Value* llvm_call_f32( llvm::Function* func , llvm::Value* lhs , llvm::Value* rhs )
  {
    llvm::Value* lhs_f32 = llvm_cast( llvm_get_type<float>() , lhs );
    llvm::Value* rhs_f32 = llvm_cast( llvm_get_type<float>() , rhs );
    return builder->CreateCall(func,{lhs_f32,rhs_f32});
  }

  llvm::Value* llvm_call_f64( llvm::Function* func , llvm::Value* lhs )
  {
    llvm::Value* lhs_f64 = llvm_cast( llvm_get_type<double>() , lhs );
    return builder->CreateCall(func,lhs_f64);
  }

  llvm::Value* llvm_call_f64( llvm::Function* func , llvm::Value* lhs , llvm::Value* rhs )
  {
    llvm::Value* lhs_f64 = llvm_cast( llvm_get_type<double>() , lhs );
    llvm::Value* rhs_f64 = llvm_cast( llvm_get_type<double>() , rhs );
    return builder->CreateCall(func,{lhs_f64,rhs_f64});
  }



  llvm::Value* llvm_sin_f32( llvm::Value* lhs )      { return llvm_call_f32( llvm_get_func( mapMath.at("sin_f32")      , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_acos_f32( llvm::Value* lhs )     { return llvm_call_f32( llvm_get_func( mapMath.at("acos_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_asin_f32( llvm::Value* lhs )     { return llvm_call_f32( llvm_get_func( mapMath.at("asin_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_atan_f32( llvm::Value* lhs )     { return llvm_call_f32( llvm_get_func( mapMath.at("atan_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_ceil_f32( llvm::Value* lhs )     { return llvm_call_f32( llvm_get_func( mapMath.at("ceil_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_floor_f32( llvm::Value* lhs )    { return llvm_call_f32( llvm_get_func( mapMath.at("floor_f32")    , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_cos_f32( llvm::Value* lhs )      { return llvm_call_f32( llvm_get_func( mapMath.at("cos_f32")      , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_cosh_f32( llvm::Value* lhs )     { return llvm_call_f32( llvm_get_func( mapMath.at("cosh_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_exp_f32( llvm::Value* lhs )      { return llvm_call_f32( llvm_get_func( mapMath.at("exp_f32")      , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_log_f32( llvm::Value* lhs )      { return llvm_call_f32( llvm_get_func( mapMath.at("log_f32")      , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_log10_f32( llvm::Value* lhs )    { return llvm_call_f32( llvm_get_func( mapMath.at("log10_f32")    , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_sinh_f32( llvm::Value* lhs )     { return llvm_call_f32( llvm_get_func( mapMath.at("sinh_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_tan_f32( llvm::Value* lhs )      { return llvm_call_f32( llvm_get_func( mapMath.at("tan_f32")      , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_tanh_f32( llvm::Value* lhs )     { return llvm_call_f32( llvm_get_func( mapMath.at("tanh_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_fabs_f32( llvm::Value* lhs )     { return llvm_call_f32( llvm_get_func( mapMath.at("fabs_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_sqrt_f32( llvm::Value* lhs )     { return llvm_call_f32( llvm_get_func( mapMath.at("sqrt_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_isfinite_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("isfinite_f32") , jitprec::i32 , jitprec::f32 , 1 )  , lhs ); }

  llvm::Value* llvm_pow_f32( llvm::Value* lhs, llvm::Value* rhs )   { return llvm_call_f32( llvm_get_func( mapMath.at("pow_f32")   , jitprec::f32 , jitprec::f32 , 2 )  , lhs , rhs ); }
  llvm::Value* llvm_atan2_f32( llvm::Value* lhs, llvm::Value* rhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("atan2_f32") , jitprec::f32 , jitprec::f32 , 2 )  , lhs , rhs ); }

  
  llvm::Value* llvm_sin_f64( llvm::Value* lhs )      { return llvm_call_f64( llvm_get_func( mapMath.at("sin_f64")      , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_acos_f64( llvm::Value* lhs )     { return llvm_call_f64( llvm_get_func( mapMath.at("acos_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_asin_f64( llvm::Value* lhs )     { return llvm_call_f64( llvm_get_func( mapMath.at("asin_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_atan_f64( llvm::Value* lhs )     { return llvm_call_f64( llvm_get_func( mapMath.at("atan_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_ceil_f64( llvm::Value* lhs )     { return llvm_call_f64( llvm_get_func( mapMath.at("ceil_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_floor_f64( llvm::Value* lhs )    { return llvm_call_f64( llvm_get_func( mapMath.at("floor_f64")    , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_cos_f64( llvm::Value* lhs )      { return llvm_call_f64( llvm_get_func( mapMath.at("cos_f64")      , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_cosh_f64( llvm::Value* lhs )     { return llvm_call_f64( llvm_get_func( mapMath.at("cosh_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_exp_f64( llvm::Value* lhs )      { return llvm_call_f64( llvm_get_func( mapMath.at("exp_f64")      , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_log_f64( llvm::Value* lhs )      { return llvm_call_f64( llvm_get_func( mapMath.at("log_f64")      , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_log10_f64( llvm::Value* lhs )    { return llvm_call_f64( llvm_get_func( mapMath.at("log10_f64")    , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_sinh_f64( llvm::Value* lhs )     { return llvm_call_f64( llvm_get_func( mapMath.at("sinh_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_tan_f64( llvm::Value* lhs )      { return llvm_call_f64( llvm_get_func( mapMath.at("tan_f64")      , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_tanh_f64( llvm::Value* lhs )     { return llvm_call_f64( llvm_get_func( mapMath.at("tanh_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_fabs_f64( llvm::Value* lhs )     { return llvm_call_f64( llvm_get_func( mapMath.at("fabs_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_sqrt_f64( llvm::Value* lhs )     { return llvm_call_f64( llvm_get_func( mapMath.at("sqrt_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_isfinite_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("isfinite_f64") , jitprec::i32 , jitprec::f64 , 1 )  , lhs ); }

  llvm::Value* llvm_pow_f64( llvm::Value* lhs, llvm::Value* rhs )   { return llvm_call_f64( llvm_get_func( mapMath.at("pow_f64")   , jitprec::f64 , jitprec::f64 , 2 )  , lhs , rhs ); }
  llvm::Value* llvm_atan2_f64( llvm::Value* lhs, llvm::Value* rhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("atan2_f64") , jitprec::f64 , jitprec::f64 , 2 )  , lhs , rhs ); }












 




  std::string jit_util_get_static_dynamic_string( const std::string& pretty )
  {
    std::ostringstream oss;
    
    oss << gpu_get_arch() << "_";

    for ( int i = 0 ; i < Nd ; ++i )
      oss << Layout::subgridLattSize()[i] << "_";

    oss << pretty;

    return oss.str();
  }

  
  
  
} // namespace QDP

