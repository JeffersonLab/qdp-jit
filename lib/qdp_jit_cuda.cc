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
//using namespace std;

using namespace QDP;

namespace JIT {


  void CompilerCUDA::setup( CompilerFactory* fac , string ftype , string pretty )
  {
    Compiler::setup(fac,ftype,pretty);

    // Set data layout
    Mod->setDataLayout(target().machine->createDataLayout());

    // Count it
    llvm_jit_stats_jitted(true);
  }


  void CompilerCUDA::init_libdevice()
  {
    static std::unique_ptr< llvm::MemoryBuffer > libdevice;

    if (libdevice)
      {
	module_libdevice = load_lazy( *libdevice );
	return;
      }

    std::vector<std::string> tried;
    std::string FileName = find_device_library( { "CUDAPATH" , "/usr/local/cuda/" , "/usr/lib/nvidia-cuda-toolkit/" } ,
						{ "nvvm/libdevice/" , "libdevice/" , "cuda/nvvm/libdevice/" , "cuda/CUDAVERSION/nvvm/libdevice/" } ,
						{ "libdevice.10.bc" , "libdevice.compute_ARCH.10.bc" } ,
						[]( std::vector<std::string>& paths , std::vector<std::string>& appends ) {
      std::string version_str = std::to_string( gpu_SDK_version_major() ) + "." + std::to_string( gpu_SDK_version_minor() );
      for ( std::string& append : appends )
	{
	  auto index = append.find("CUDAVERSION", 0);
	  if (index != std::string::npos)
	    append.replace(index, 11, version_str );
	}

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

	  for ( std::string& path : paths )
	    {
	      auto index = path.find("CUDAPATH", 0);
	      if (index != std::string::npos)
		path.replace(index, 8, ENV );
	    }
	}
    } , &tried );

    if (FileName.empty())
      abort_device_library_not_found( tried );

    
    ErrorOr<std::unique_ptr<MemoryBuffer>> mb = MemoryBuffer::getFile(FileName);
    if (std::error_code ec = mb.getError()) {
      errs() << ec.message();
      QDP_abort(1);
    }

    libdevice = std::move( mb.get() );

    module_libdevice = load_lazy( *libdevice );
  }


  std::unique_ptr< llvm::Module > CompilerCUDA::load_lazy( const llvm::MemoryBuffer& bc )
  {
    llvm::Expected<std::unique_ptr<llvm::Module>> m = llvm::getLazyBitcodeModule( bc.getMemBufferRef() , *get_context() );
    if (std::error_code ec = errorToErrorCode(m.takeError()))
      {
	errs() << "Error reading bitcode: " << ec.message() << "\n";
	QDP_abort(1);
      }
    return std::move( m.get() );
  }



  namespace {
    TargetInfo make_target()
    {
      TargetInfo t;
      t.global_addrspace = 0;
      t.local_addrspace  = 3;

      t.arch = gpu_get_arch();

      std::string str_triple("nvptx64-nvidia-cuda");

      t.triple.setTriple(str_triple);

      std::string Error;

      const llvm::Target *TheTarget = llvm::TargetRegistry::lookupTarget( "", t.triple, Error);
      if (!TheTarget) {
	llvm::errs() << "Error looking up target: " << Error;
	exit(1);
      }

      llvm::TargetOptions options;

      t.machine.reset ( TheTarget->createTargetMachine(
							 t.triple,
							 t.arch,
							 "",
							 options,
							 Reloc::PIC_));

      if (!t.machine)
	{
	QDPIO::cerr << "Could not create LLVM target machine\n";
	QDP_abort(1);
	}


      QDPIO::cout << "LLVM initialization" << std::endl;
      QDPIO::cout << "  Target machine CPU                  : " << t.machine->getTargetCPU().str() << "\n";
      QDPIO::cout << "  Target triple                       : " << t.machine->getTargetTriple().str() << "\n";

      t.math["sin_f32"]="__nv_sinf";
      t.math["acos_f32"]="__nv_acosf";
      t.math["asin_f32"]="__nv_asinf";
      t.math["atan_f32"]="__nv_atanf";
      t.math["ceil_f32"]="__nv_ceilf";
      t.math["floor_f32"]="__nv_floorf";
      t.math["cos_f32"]="__nv_cosf";
      t.math["cosh_f32"]="__nv_coshf";
      t.math["exp_f32"]="__nv_expf";
      t.math["log_f32"]="__nv_logf";
      t.math["log10_f32"]="__nv_log10f";
      t.math["sinh_f32"]="__nv_sinhf";
      t.math["tan_f32"]="__nv_tanf";
      t.math["tanh_f32"]="__nv_tanhf";
      t.math["fabs_f32"]="__nv_fabsf";
      t.math["sqrt_f32"]="__nv_fsqrt_rn";
      t.math["isfinite_f32"]="__nv_finitef";

      t.math["pow_f32"]="__nv_powf";
      t.math["atan2_f32"]="__nv_atan2f";


      t.math["sin_f64"]="__nv_sin";
      t.math["acos_f64"]="__nv_acos";
      t.math["asin_f64"]="__nv_asin";
      t.math["atan_f64"]="__nv_atan";
      t.math["ceil_f64"]="__nv_ceil";
      t.math["floor_f64"]="__nv_floor";
      t.math["cos_f64"]="__nv_cos";
      t.math["cosh_f64"]="__nv_cosh";
      t.math["exp_f64"]="__nv_exp";
      t.math["log_f64"]="__nv_log";
      t.math["log10_f64"]="__nv_log10";
      t.math["sinh_f64"]="__nv_sinh";
      t.math["tan_f64"]="__nv_tan";
      t.math["tanh_f64"]="__nv_tanh";
      t.math["fabs_f64"]="__nv_fabs";
      t.math["sqrt_f64"]="__nv_dsqrt_rn";
      t.math["isfinite_f64"]="__nv_isfinited";

      t.math["pow_f64"]="__nv_pow";
      t.math["atan2_f64"]="__nv_atan2";

      return t;
    }
  }


  const TargetInfo& CompilerCUDA::target_info()
  {
    static TargetInfo t = make_target();
    return t;
  }


  llvm::Value* CompilerCUDA::get_shared_ptr(llvm::Type* ty, int n)
  {
    const unsigned addrspace = target().local_addrspace;

    llvm::GlobalVariable* gv =
      new llvm::GlobalVariable(
			       *Mod,
			       llvm::ArrayType::get(ty, 0),
			       false,
			       llvm::GlobalVariable::ExternalLinkage,
			       nullptr,
			       "shared_buffer",
			       nullptr,
			       llvm::GlobalVariable::NotThreadLocal,
			       addrspace,
			       false
			       );

    return builder->CreatePointerCast(
				      gv,
				      llvm::PointerType::get(Mod->getContext(), addrspace)
				      );
  }


  void CompilerCUDA::bar_sync()
  {
    builder->CreateIntrinsic(
			     llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_all,
			     {},
			     {builder->getInt32(0)});
  }




  llvm::Value * CompilerCUDA::call_special_tidx()    { return special("llvm.nvvm.read.ptx.sreg.tid.x"    ,get_type<int>() , {} , {} ); }
  llvm::Value * CompilerCUDA::call_special_ntidx()   { return special("llvm.nvvm.read.ptx.sreg.ntid.x"   ,get_type<int>() , {} , {} ); }
  llvm::Value * CompilerCUDA::call_special_ctaidx()  { return special("llvm.nvvm.read.ptx.sreg.ctaid.x"  ,get_type<int>() , {} , {} ); }
  llvm::Value * CompilerCUDA::call_special_nctaidx() { return special("llvm.nvvm.read.ptx.sreg.nctaid.x" ,get_type<int>() , {} , {} ); }
  llvm::Value * CompilerCUDA::call_special_ctaidy()  { return special("llvm.nvvm.read.ptx.sreg.ctaid.y"  ,get_type<int>() , {} , {} ); }

  
  void CompilerCUDA::addKernelMetadata(llvm::Function *F)
  {
    auto i32_t = llvm::Type::getInt32Ty(*get_context());
    
    llvm::Metadata *md_args[] = {
      llvm::ValueAsMetadata::get(F),
      MDString::get(*get_context(), "kernel"),
      llvm::ValueAsMetadata::get(ConstantInt::get(i32_t, 1))};

    MDNode *md_node = MDNode::get(*get_context(), md_args);

    Mod->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md_node);
  }
  

  std::string CompilerCUDA::get_ptx()
  {
    llvm::legacy::PassManager PM;

    std::string str;
    llvm::raw_string_ostream rss(str);
    llvm::buffer_ostream bos(rss);
    
    if (target().machine->addPassesToEmitFile(PM, bos , nullptr ,  llvm::CodeGenFileType::AssemblyFile ))
    {
      llvm::errs() << ": target does not support generation of this"
		   << " file type!\n";
      QDP_abort(1);
    }
    
    PM.run(*Mod);

    std::string ptx = bos.str().str();

    return ptx;
  }

  


  CodeObject CompilerCUDA::build_function_backend(JitFunction& func)
  {
    func.setIsGPU(true);

    addKernelMetadata( mainFunc );

    StopWatch swatch(false);
    swatch.start();
    
    if (math_declarations.size() > 0)
      {
	if (options().verbose)
	  {
	    QDPIO::cout << "adding math function definitions ...\n";
	  }
	init_libdevice();
    
	if (options().verbose)
	  {
	    QDPIO::cout << "link modules ...\n";
	  }
	std::string ErrorMsg;

	Mod->setDataLayout("");
	
	if (llvm::Linker::linkModules( *Mod , std::move( module_libdevice ) , llvm::Linker::Flags::LinkOnlyNeeded )) {
	  QDPIO::cerr << "Linking libdevice failed: " << ErrorMsg.c_str() << "\n";
	  QDP_abort(1);
	}
      }

    swatch.stop();
    func.time_math = swatch.getTimeInMicroseconds();
    swatch.reset();
    swatch.start();

    //QDPIO::cout << "setting module data layout\n";
    Mod->setDataLayout(target().machine->createDataLayout());

    uint32_t NVPTX_CUDA_FTZ = options().cuda_ftz;

    Mod->addModuleFlag( llvm::Module::ModFlagBehavior::Override, "nvvm-reflect-ftz" , NVPTX_CUDA_FTZ );

    if (options().verbose)
      {
	QDPIO::cout << "running passes ...\n";
      }

    internalize_and_dce();

    if (options().verbose)
      {
	QDPIO::cout << "internalize and remove dead code ...\n";
      }
    
    swatch.stop();
    func.time_passes = swatch.getTimeInMicroseconds();

    if ( options().opt_level != -1 )
      {
	opt( true );
      }

    if (options().verbose)
      {
	QDPIO::cout << "\n\n";
	QDPIO::cout << str_pretty << std::endl;

	if (options().is_primary)
	  {
	    module_dump();
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
    CodeObject obj;
    obj.image = get_ptx();
    swatch.stop();
    func.time_codegen = swatch.getTimeInMicroseconds();

    return obj;
  }



  Compiler* CompilerCreatorGPU::create( CompilerFactory* fac , string ftype , string pretty )
  {
    CompilerCUDA* c = new CompilerCUDA;
    c->setup( fac , ftype , pretty );
    return c;
  }


} // JIT
