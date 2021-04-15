#include "qdp.h"
#include "qdp_config.h"

#include "llvm/InitializePasses.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LegacyPassManager.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Linker/Linker.h"

#include "llvm/Support/TargetRegistry.h"
#include "llvm/PassRegistry.h"

#include "llvm/IR/InstrTypes.h"
#include "llvm/Bitcode/BitcodeWriter.h"

#include "llvm/CodeGen/CommandFlags.h"
using namespace llvm;
using namespace llvm::codegen;

#include "llvm/Support/CommandLine.h"

#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"

#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/Target/TargetMachine.h"

#include <memory>
#include <unistd.h>

namespace QDP {

  enum jitprec { f32 , f64 };

  namespace
  {
    std::string user_libdevice_path;
    std::string user_libdevice_name;

    bool clang_codegen = false;
    std::string clang_opt;
    
    //
    // Default search locations for libdevice
    // "ARCH"      gets replaced by "gfx908" (ROCm) or "70" (CUDA)
    // "CUDAPATH"  gets replaced by the envvar CUDAPATH or CUDA_PATH
    //
#ifdef QDP_BACKEND_ROCM
    std::vector<std::string> vec_str_libdevice_path = { ROCM_DIR };
    std::vector<std::string> vec_str_libdevice_path_append = { "llvm/lib/libdevice/" };
    std::vector<std::string> vec_str_libdevice_name = { "libm-amdgcn-ARCH.bc" };
#else
    std::vector<std::string> vec_str_libdevice_path = { "CUDAPATH" , "/usr/local/cuda/" , "/usr/lib/nvidia-cuda-toolkit/" };
    std::vector<std::string> vec_str_libdevice_path_append = { "nvvm/libdevice/" , "libdevice/" };
    std::vector<std::string> vec_str_libdevice_name = { "libdevice.10.bc" , "libdevice.compute_ARCH.10.bc" };
#endif
    
    
    llvm::LLVMContext TheContext;

    llvm::Triple TheTriple;
    std::unique_ptr<llvm::TargetMachine> TargetMachine;
    
    llvm::BasicBlock  *bb_stack;
    llvm::BasicBlock  *bb_afterstack;

    BasicBlock::iterator it_stack;
    
    llvm::Function    *mainFunc;

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

  namespace AMDspecific {
    ParamRef __threads_per_group;
    ParamRef __grid_size_x;
  }


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



  namespace ptx_db {
    bool db_enabled = false;
    std::string dbname = "dummy.dat";
    typedef std::map< std::string , std::pair< std::string , std::string > > DBType; // pretty+other stuff --> function name , PTX string
    DBType db;
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
    return TheContext;
  }

  
  llvm::IRBuilder<>* llvm_get_builder()
  {
    return builder.get();
  }

  llvm::Module* llvm_get_module()
  {
    return Mod.get();
  }


  template<> llvm::Type* llvm_get_type<jit_half_t>()  { return llvm::Type::getHalfTy(TheContext); }
  template<> llvm::Type* llvm_get_type<float>()  { return llvm::Type::getFloatTy(TheContext); }
  template<> llvm::Type* llvm_get_type<double>() { return llvm::Type::getDoubleTy(TheContext); }
  template<> llvm::Type* llvm_get_type<int>()    { return llvm::Type::getIntNTy(TheContext,32); }
  template<> llvm::Type* llvm_get_type<bool>()   { return llvm::Type::getIntNTy(TheContext,1); }

  template<> llvm::Type* llvm_get_type<jit_half_t*>()  { return llvm::Type::getHalfPtrTy(TheContext); }
  template<> llvm::Type* llvm_get_type<float*>()  { return llvm::Type::getFloatPtrTy(TheContext); }
  template<> llvm::Type* llvm_get_type<double*>() { return llvm::Type::getDoublePtrTy(TheContext); }
  template<> llvm::Type* llvm_get_type<int*>()    { return llvm::Type::getIntNPtrTy(TheContext,32); }
  template<> llvm::Type* llvm_get_type<bool*>()   { return llvm::Type::getIntNPtrTy(TheContext,1); }


  
  std::string get_ptx_db_fname() {
    return ptx_db::dbname;
  }
  bool get_ptx_db_enabled() {
    return ptx_db::db_enabled;
  }
  int get_ptx_db_size() {
    return ptx_db::db.size();
  }

  



  void llvm_ptx_db( JitFunction& f , const char * pretty )
  {
    std::string id = jit_util_get_static_dynamic_string( pretty );
    
    ptx_db::DBType::iterator it = ptx_db::db.find( id );

    if ( it != ptx_db::db.end() )
      {
	//return get_fptr_from_ptx( "generic.ptx" , it->second );
	get_jitf( f , it->second.second , it->second.first , pretty , str_arch );
      }
  }



  void llvm_set_ptxdb( const char * c_str ) {
    ptx_db::db_enabled = true;
    ptx_db::dbname = std::string( c_str );
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



  
  llvm::Function *llvm_get_func( std::string name , jitprec p , int num_args )
  {
    if (math_declarations[p][name] == 1)
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
	math_declarations[p][name] = 1;
	
	llvm::Type* type;
	switch(p)
	  {
	  case jitprec::f32:
	    type = llvm::Type::getFloatTy(TheContext);
	    break;
	  case jitprec::f64:
	    type = llvm::Type::getDoubleTy(TheContext);
	    break;
	  }

	std::vector< llvm::Type* > Args( num_args , type );
	llvm::FunctionType *funcType = llvm::FunctionType::get( type , Args , false); // no vararg
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
    libs.push_back("/amdgcn/bitcode/ocml.bc");
    libs.push_back("/amdgcn/bitcode/oclc_finite_only_on.bc");
    libs.push_back("/amdgcn/bitcode/oclc_isa_version_" + arch + ".bc");
    libs.push_back("/amdgcn/bitcode/oclc_unsafe_math_on.bc");
    libs.push_back("/amdgcn/bitcode/oclc_daz_opt_on.bc");

    module_ocml.clear();
    
    for( int i = 0 ; i < libs.size() ; ++i )
      {
	std::string FileName = std::string(ROCM_DIR) + libs[i];

	QDPIO::cout << "Reading bitcode from " << FileName << "\n";
	
	std::ifstream ftmp(FileName.c_str());
	if (!ftmp.good())
	  {
	    QDPIO::cerr << "file not found:" << FileName << "\n";
	    QDP_abort(1);
	  }
    
	ErrorOr<std::unique_ptr<MemoryBuffer>> mb = MemoryBuffer::getFile(FileName);
	if (std::error_code ec = mb.getError()) {
	  errs() << ec.message();
	  QDP_abort(1);
	}
  
	llvm::Expected<std::unique_ptr<llvm::Module>> m = llvm::parseBitcodeFile(mb->get()->getMemBufferRef(), TheContext);
	if (std::error_code ec = errorToErrorCode(m.takeError()))
	  {
	    errs() << "Error reading bitcode from " << FileName << ": " << ec.message() << "\n";
	    QDP_abort(1);
	  }

	module_ocml.push_back( std::move( m.get() ) );
      }
  }
#endif


  
  void llvm_init_libdevice()
  {
    static std::string FileName;

    if (FileName.empty())
      {
	// std::vector<std::string> vec_str_cuda_path = { "/usr/local/cuda" , "/usr/lib/nvidia-cuda-toolkit" };
	// std::vector<std::string> vec_str_cuda_path_append = { "/nvvm/libdevice" , "/libdevice" };
	// std::vector<std::string> vec_str_libdevice_name = { "libdevice.10.bc" };

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

	//
	// Replace CUDAPATH with endvar
	//
	for( auto path = vec_str_libdevice_path.begin() ; path != vec_str_libdevice_path.end() ; ++path )
	  {
	    char *env = getenv( "CUDAPATH" );
	    if (!env)
	      env = getenv( "CUDA_PATH" );
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
#ifdef QDP_BACKEND_ROCM
	    QDPIO::cout << "trying: " << *fname << std::endl;
#endif
	    
	    std::ifstream ftmp(fname->c_str());
	    if (ftmp.good())
	      {
#ifdef QDP_BACKEND_ROCM
		QDPIO::cout << "libdevice found.\n";
#endif
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
  
    llvm::Expected<std::unique_ptr<llvm::Module>> m = llvm::parseBitcodeFile(mb->get()->getMemBufferRef(), TheContext);
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
							 Reloc::PIC_,
							 None,
							 llvm::CodeGenOpt::Aggressive, true ));
    
    if (!TargetMachine)
      {
	QDPIO::cerr << "Could not create LLVM target machine\n";
	QDP_abort(1);
      }

    QDPIO::cout << "LLVM initialization" << std::endl;
    QDPIO::cout << "  Target machine CPU                  : " << TargetMachine->getTargetCPU().str() << "\n";
    QDPIO::cout << "  Target triple                       : " << TargetMachine->getTargetTriple().str() << "\n";
    

    if (ptx_db::db_enabled) {
      // Load DB
      QDPIO::cout << "PTX DB" << std::endl;
      QDPIO::cout << "  database file name                  : " << ptx_db::dbname << "\n";
      std::ifstream f(ptx_db::dbname , ios::in | ios::binary );
      if (f.good())
	{
	  int count = 0;
	  while ( ! f.eof() )
	    {
	      int size1;
	      f >> size1;
	      if (f.eof())
		break;
	      char* buf1 = new char[size1];
	      f.read( buf1 , size1 );
	      if (f.eof())
		break;

	      int size2;
	      f >> size2;
	      if (f.eof())
		break;
	      char* buf2 = new char[size2];
	      f.read( buf2 , size2 );
	      if (f.eof())
		break;

	      int size3;
	      f >> size3;
	      if (f.eof())
		break;
	      char* buf3 = new char[size3];
	      f.read( buf3 , size3 );
	      if (f.eof())
		break;

	      // QDPIO::cout << "ptx_db: read "
	      // 		  << " key_size=" << size1
	      // 		  << " name_size=" << size2
	      // 		  << " ptx_size=" << size3 << "\n";

	      std::string key(buf1,size1);
	      std::string name(buf2,size2);
	      std::string ptx(buf3,size3);
	      ptx_db::db.insert( std::make_pair( key , std::make_pair( name , ptx ) ) );
	      
	      ++count;
	      
	      delete[] buf1;
	      delete[] buf2;
	      delete[] buf3;
	    }
	  QDPIO::cout << "  number of records read in           : " << count << "\n";
	}

    } // ptx db


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



  void llvm_backend_init()
  {
#ifdef QDP_BACKEND_ROCM
    llvm_backend_init_rocm();
#else
    llvm_backend_init_cuda();
#endif
  }


  
  llvm::BasicBlock * llvm_get_insert_block() {
    return builder->GetInsertBlock();
  }


  void llvm_start_new_function( const char* ftype , const char* pretty )
  {
    math_declarations.clear();
    
    str_func_type = ftype;
    str_pretty = pretty;
    str_kernel_name = str_func_type + std::to_string( map_func_counter[str_func_type]++ );
    
    // Count it
    jit_stats_jitted();
    
    //QDPIO::cout << "Starting new LLVM function..\n";

    Mod.reset( new llvm::Module( "module", TheContext) );

    builder.reset( new llvm::IRBuilder<>( TheContext ) );

    jit_build_seedToFloat();
    jit_build_seedMultiply();

    vecParamType.clear();
    vecArgument.clear();
    function_created = false;

#ifdef QDP_BACKEND_ROCM
    AMDspecific::__threads_per_group = llvm_add_param<int>();
    AMDspecific::__grid_size_x       = llvm_add_param<int>();
#endif
  }


  void llvm_create_function() {
    assert( !function_created );
    assert( vecParamType.size() > 0 );

    llvm::FunctionType *funcType = 
      llvm::FunctionType::get( builder->getVoidTy() , 
			       llvm::ArrayRef<llvm::Type*>( vecParamType.data() , vecParamType.size() ) , 
			       false); // no vararg
    mainFunc = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, str_kernel_name , Mod.get());

#ifdef QDP_BACKEND_ROCM
    mainFunc->setCallingConv( llvm::CallingConv::AMDGPU_KERNEL );
#endif
    
    unsigned Idx = 0;
    for (llvm::Function::arg_iterator AI = mainFunc->arg_begin(), AE = mainFunc->arg_end() ; AI != AE ; ++AI, ++Idx) {
      AI->setName( std::string("arg")+std::to_string(Idx) );
      vecArgument.push_back( &*AI );
    }

    bb_stack = llvm::BasicBlock::Create(TheContext, "stack", mainFunc);
    builder->SetInsertPoint(bb_stack);
    it_stack = builder->GetInsertPoint(); // probly bb_stack.begin()
    
    bb_afterstack = llvm::BasicBlock::Create(TheContext, "afterstack" );
    mainFunc->getBasicBlockList().push_back(bb_afterstack);
    
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


  llvm::Value* llvm_array_type_indirection( ParamRef p , llvm::Value* idx )
  {
    llvm::Value* base = llvm_derefParam( p );
    llvm::Value* gep = llvm_createGEP( base , idx );
    return llvm_load( gep );
  }

  
  llvm::Value* llvm_array_type_indirection( llvm::Value* base , llvm::Value* idx )
  {
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
      //llvm::outs() << "promote int " << t0->getScalarSizeInBits() << " " << t1->getScalarSizeInBits() << "\n";
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

#if defined (QDP_LLVM12)
#else
    if (!llvm::CastInst::isCastable( src->getType() , dest_type ))
      QDP_error_exit("not castable");
#endif

    //llvm::outs() << "cast instruction: dest type = " << dest_type << "   from " << src->getType() << "\n";
    
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
      //llvm::outs() << "lhs needs conversion\n";
      lhs_new = llvm_cast( args_type , lhs );
    }
    if ( args_type != rhs->getType() ) {
      //llvm::outs() << "rhs needs conversion\n";
      rhs_new = llvm_cast( args_type , rhs );
    }
    return std::make_pair(lhs_new,rhs_new);
  }
  



  llvm::Value* llvm_neg( llvm::Value* rhs ) {
    llvm::Value* lhs = llvm_create_value(0);
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFSub( vals.first , vals.second );
    else
      return builder->CreateSub( vals.first , vals.second );
  }


  llvm::Value* llvm_rem( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFRem( vals.first , vals.second );
    else
      return builder->CreateSRem( vals.first , vals.second );
  }


  llvm::Value* llvm_shr( llvm::Value* lhs , llvm::Value* rhs ) {  
    auto vals = llvm_normalize_values(lhs,rhs);
 //   llvm::Type* args_type = vals.first->getType();
 //   assert( !args_type->isFloatingPointTy() );

    assert( ! ( vals.first->getType()->isFloatingPointTy() ) );
    return builder->CreateAShr( vals.first , vals.second );
  }


  llvm::Value* llvm_shl( llvm::Value* lhs , llvm::Value* rhs ) {  
    auto vals = llvm_normalize_values(lhs,rhs);
  //  llvm::Type* args_type = vals.first->getType();
  //  assert( !args_type->isFloatingPointTy() );

    assert( ! ( vals.first->getType()->isFloatingPointTy()  ) );
    return builder->CreateShl( vals.first , vals.second );
  }


  llvm::Value* llvm_and( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    // llvm::Type* args_type = vals.first->getType();
    // assert( !args_type->isFloatingPointTy() );
    assert( ! ( vals.first->getType()->isFloatingPointTy()  ) );
    return builder->CreateAnd( vals.first , vals.second );
  }


  llvm::Value* llvm_or( llvm::Value* lhs , llvm::Value* rhs ) {  
    auto vals = llvm_normalize_values(lhs,rhs);
  //  llvm::Type* args_type = vals.first->getType();
   // assert( !args_type->isFloatingPointTy() );
    assert( ! ( vals.first->getType()->isFloatingPointTy()  ) );

    return builder->CreateOr( vals.first , vals.second );
  }


  llvm::Value* llvm_xor( llvm::Value* lhs , llvm::Value* rhs ) {  
    auto vals = llvm_normalize_values(lhs,rhs);
//    llvm::Type* args_type = vals.first->getType();
//    assert( !args_type->isFloatingPointTy() );

    assert( ! ( vals.first->getType()->isFloatingPointTy()  ) );

    return builder->CreateXor( vals.first , vals.second );
  }


  llvm::Value* llvm_mul( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    if ( vals.first->getType()->isFloatingPointTy() )
      return builder->CreateFMul( vals.first , vals.second );
    else
      return builder->CreateMul( vals.first , vals.second );
  }


  llvm::Value* llvm_add( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFAdd( vals.first , vals.second );
    else
      return builder->CreateNSWAdd( vals.first , vals.second );
  }


  llvm::Value* llvm_sub( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFSub( vals.first , vals.second );
    else
      return builder->CreateSub( vals.first , vals.second );
  }


  llvm::Value* llvm_div( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFDiv( vals.first , vals.second );
    else 
      return builder->CreateSDiv( vals.first , vals.second );
  }


  llvm::Value* llvm_eq( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFCmpOEQ( vals.first , vals.second );
    else
      return builder->CreateICmpEQ( vals.first , vals.second );
  }


  llvm::Value* llvm_ge( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFCmpOGE( vals.first , vals.second );
    else
      return builder->CreateICmpSGE( vals.first , vals.second );
  }


  llvm::Value* llvm_gt( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFCmpOGT( vals.first , vals.second );
    else
      return builder->CreateICmpSGT( vals.first , vals.second );
  }


  llvm::Value* llvm_le( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFCmpOLE( vals.first , vals.second );
    else
      return builder->CreateICmpSLE( vals.first , vals.second );
  }


  llvm::Value* llvm_lt( llvm::Value* lhs , llvm::Value* rhs ) {
    auto vals = llvm_normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFCmpOLT( vals.first , vals.second );
    else 
      return builder->CreateICmpSLT( vals.first , vals.second );
  }


  //
  // Convenience function definitions
  //
  llvm::Value* llvm_not( llvm::Value* lhs ) {
    //llvm::outs() << "not\n";
    return llvm_xor( llvm_create_value(-1) , lhs );
  }



  llvm::Value* llvm_get_shared_ptr( llvm::Type *ty ) {

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
  }



  llvm::Value * llvm_alloca( llvm::Type* type , int elements )
  {
    auto it_save = builder->GetInsertPoint();
    auto bb_save = builder->GetInsertBlock();
    
    builder->SetInsertPoint(bb_stack, it_stack);

    auto DL = Mod->getDataLayout();
    unsigned AddrSpace = DL.getAllocaAddrSpace();

    //QDPIO::cout << "using address space : " << AddrSpace << "\n";
    
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
  template<> ParamRef llvm_add_param<jit_half_t>() { 
    vecParamType.push_back( llvm::Type::getHalfTy(TheContext) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<float>() { 
    vecParamType.push_back( llvm::Type::getFloatTy(TheContext) );
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<jit_half_t*>() { 
    vecParamType.push_back( llvm::Type::getHalfPtrTy(TheContext) );
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


  template<> ParamRef llvm_add_param<int**>() {
    vecParamType.push_back( llvm::PointerType::get( llvm::Type::getInt32PtrTy(TheContext) , 0 ) );  // AddressSpace = 0 ??
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<float**>() {
    vecParamType.push_back( llvm::PointerType::get( llvm::Type::getFloatPtrTy(TheContext) , 0 ) );  // AddressSpace = 0 ??
    return vecParamType.size()-1;
  }
  template<> ParamRef llvm_add_param<double**>() {
    vecParamType.push_back( llvm::PointerType::get( llvm::Type::getDoublePtrTy(TheContext) , 0 ) );  // AddressSpace = 0 ??
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
    cond = llvm_cast( llvm_get_type<bool>() , cond );
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

  llvm::Value * llvm_create_value(int64_t v ) {return llvm::ConstantInt::get( llvm::Type::getInt64Ty(TheContext) , v );}
  llvm::Value * llvm_create_value(int v )     {return llvm::ConstantInt::get( llvm::Type::getInt32Ty(TheContext) , v );}
  llvm::Value * llvm_create_value(size_t v)   {return llvm::ConstantInt::get( llvm::Type::getInt32Ty(TheContext) , v );}
  llvm::Value * llvm_create_value(bool v )    {return llvm::ConstantInt::get( llvm::Type::getInt1Ty(TheContext) , v );}


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

#ifdef QDP_BACKEND_ROCM
    std::string bar_name("llvm.amdgcn.s.barrier");
#else
    std::string bar_name("llvm.nvvm.barrier0");
#endif
    
    auto Bar = Mod->getOrInsertFunction( bar_name.c_str() , 
					 IntrinFnTy , 
					 llvm::AttributeList::get(TheContext, 
								  llvm::AttributeList::FunctionIndex, 
								  ABuilder) );

    builder->CreateCall(Bar);
  }


  


  llvm::Value * llvm_special( const char * name )
  {
    llvm::FunctionType *IntrinFnTy = llvm::FunctionType::get(llvm::Type::getInt32Ty(TheContext), false);

    llvm::AttrBuilder ABuilder;
    ABuilder.addAttribute(llvm::Attribute::ReadNone);

    auto ReadTidX = Mod->getOrInsertFunction( name , 
					      IntrinFnTy , 
					      llvm::AttributeList::get(TheContext, 
								       llvm::AttributeList::FunctionIndex, 
								       ABuilder)
					      );

    return builder->CreateCall(ReadTidX);
  }


  
#ifdef QDP_BACKEND_ROCM
  llvm::Value * llvm_call_special_workitem_x() {     return llvm_special("llvm.amdgcn.workitem.id.x"); }
  llvm::Value * llvm_call_special_workgroup_x() {     return llvm_special("llvm.amdgcn.workgroup.id.x"); }
  llvm::Value * llvm_call_special_workgroup_y() {     return llvm_special("llvm.amdgcn.workgroup.id.y"); }

  llvm::Value * llvm_call_special_tidx() { return llvm_call_special_workitem_x(); }
  llvm::Value * llvm_call_special_ntidx() { return llvm_derefParam( AMDspecific::__threads_per_group ); }
  llvm::Value * llvm_call_special_ctaidx() { return llvm_call_special_workgroup_x(); }
  llvm::Value * llvm_call_special_nctaidx() { return llvm_derefParam( AMDspecific::__grid_size_x ); }
  llvm::Value * llvm_call_special_ctaidy() { return llvm_call_special_workgroup_y();  }
#else
  llvm::Value * llvm_call_special_tidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.tid.x"); }
  llvm::Value * llvm_call_special_ntidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.ntid.x"); }
  llvm::Value * llvm_call_special_ctaidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.ctaid.x"); }
  llvm::Value * llvm_call_special_nctaidx() { return llvm_special("llvm.nvvm.read.ptx.sreg.nctaid.x"); }
  llvm::Value * llvm_call_special_ctaidy() { return llvm_special("llvm.nvvm.read.ptx.sreg.ctaid.y"); }
#endif

  
  

  llvm::Value * llvm_thread_idx() { 
    llvm::Value * tidx = llvm_call_special_tidx();
    llvm::Value * ntidx = llvm_call_special_ntidx();
    llvm::Value * ctaidx = llvm_call_special_ctaidx();
    llvm::Value * ctaidy = llvm_call_special_ctaidy();
    llvm::Value * nctaidx = llvm_call_special_nctaidx();
    return llvm_add( llvm_mul( llvm_add( llvm_mul( ctaidy , nctaidx ) , ctaidx ) , ntidx ) , tidx );
  }
  


  void addKernelMetadata(llvm::Function *F) {
    auto i32_t = llvm::Type::getInt32Ty(TheContext);
    
    llvm::Metadata *md_args[] = {
      llvm::ValueAsMetadata::get(F),
      MDString::get(TheContext, "kernel"),
      llvm::ValueAsMetadata::get(ConstantInt::get(i32_t, 1))};

    MDNode *md_node = MDNode::get(TheContext, md_args);

    Mod->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md_node);
  }


  void llvm_print_module( llvm::Module* m , const char * fname ) {
    std::error_code EC;
    llvm::raw_fd_ostream outfd( fname , EC, llvm::sys::fs::OpenFlags::F_Text);

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




  namespace {
    bool all_but_kernel_name(const llvm::GlobalValue & gv)
    {
      return gv.getName().str() == str_kernel_name;
    }
  }



  std::string get_ptx()
  {
    llvm::legacy::PassManager PM;
    PM.add( llvm::createInternalizePass( all_but_kernel_name ) );
#if 0
    unsigned int sm_gpu = gpu_getMajor() * 10 + gpu_getMinor();
    PM.add( llvm::createNVVMReflectPass( sm_gpu ));
#endif
    PM.add( llvm::createGlobalDCEPass() );
    
    PM.run(*Mod);

    
    //QDPIO::cout << "AFTER OPT ---------------\n";
    //llvm_module_dump();

    
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


  

  
  void llvm_build_function_cuda(JitFunction& func)
  {
    addKernelMetadata( mainFunc );

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
	if (llvm::Linker::linkModules( *Mod , std::move( module_libdevice ) )) {  // llvm::Linker::PreserveSource
	  QDPIO::cerr << "Linking libdevice failed: " << ErrorMsg.c_str() << "\n";
	  QDP_abort(1);
	}
      }

    //QDPIO::cout << "setting module data layout\n";
    Mod->setDataLayout(TargetMachine->createDataLayout());


    llvm::legacy::PassManager PM2;
    
    PM2.add( llvm::createInternalizePass( all_but_kernel_name ) );
    PM2.add( llvm::createGlobalDCEPass() );

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "internalize and remove dead code ...\n";
      }
    PM2.run(*Mod);
    

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "\n\n";
	QDPIO::cout << str_pretty << std::endl;
	
	if (Layout::primaryNode())
	  {
	    llvm_module_dump();
	  }
      }

    
    
    std::string ptx_kernel = get_ptx();

    //JitFunction func = get_fptr_from_ptx( fname , ptx_kernel );
    get_jitf( func , ptx_kernel , str_kernel_name , str_pretty , str_arch );

    if ( ptx_db::db_enabled ) {

      if (Layout::primaryNode())
	{
	  std::string id = jit_util_get_static_dynamic_string( str_pretty );

	  if ( ptx_db::db.find( id ) != ptx_db::db.end() ) {
	    QDPIO::cout << "internal error: key already exists in DB but wasn't found earlier\n" << id << "\n";
	    QDP_abort(1);
	  }

	  ptx_db::db[ id ] = std::make_pair( str_kernel_name , ptx_kernel );

	  std::ofstream db;
	  db.open ( ptx_db::dbname , ios::out | ios::binary );
	  for ( ptx_db::DBType::iterator it = ptx_db::db.begin() ; it != ptx_db::db.end() ; ++it ) 
	    {
	      int size1 = it->first.size();
	      db << size1;
	      db.write( it->first.c_str() , size1 );

	      int size2 = it->second.first.size();
	      db << size2;
	      db.write( it->second.first.c_str() , size2 );

	      int size3 = it->second.second.size();
	      db << size3;
	      db.write( it->second.second.c_str() , size3 );
	    }
	  db.close();
	}
    } // ptx db
  }



  
#ifdef QDP_BACKEND_ROCM

  void build_function_codegen(const std::string& shared_path)
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

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "setting module data layout\n";
      }
    Mod->setDataLayout(TargetMachine->createDataLayout());

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
	    QDPIO::cout << "linking in additional library " << i << "\n";
	    if (llvm::Linker::linkModules( *Mod , std::move( module_ocml[i] ) )) {  // llvm::Linker::PreserveSource
	      QDPIO::cerr << "Linking additional library failed: " << ErrorMsg.c_str() << "\n";
	      QDP_abort(1);
	    }
	  }
      }

#if 0
    {
      QDPIO::cout << "write code to module_linked.bc ...\n";
      std::error_code EC;
      llvm::raw_fd_ostream OS("module_linked.bc", EC, llvm::sys::fs::F_None);
      llvm::WriteBitcodeToFile(*Mod, OS);
      OS.flush();
    }
#endif
    
    llvm::legacy::PassManager PM2;
    
    PM2.add( llvm::createInternalizePass( all_but_kernel_name ) );
    PM2.add( llvm::createGlobalDCEPass() );

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "internalize and remove dead code ...\n";
      }
    PM2.run(*Mod);
    
    //llvm_module_dump();

#if 0
    {
      QDPIO::cout << "write code to module_internal_dce.bc ...\n";
      std::error_code EC;
      llvm::raw_fd_ostream OS("module_internal_dce.bc", EC, llvm::sys::fs::F_None);
      llvm::WriteBitcodeToFile(*Mod, OS);
      OS.flush();
    }
#endif

    std::string clang_name;
    if (clang_codegen)
      {
	clang_name = "module_" + str_kernel_name + ".bc";
	QDPIO::cout << "write code to " << clang_name << "\n";
	std::error_code EC;
	llvm::raw_fd_ostream OS(clang_name, EC, llvm::sys::fs::F_None);
	llvm::WriteBitcodeToFile(*Mod, OS);
	OS.flush();
      }

    std::string isabin_path = "module_" + str_kernel_name + ".o";

    if (clang_codegen)
      {	
	std::string clang_path = std::string(ROCM_DIR) + "/llvm/bin/clang";
	std::string command = clang_path + " -c " + clang_opt + " -target amdgcn-amd-amdhsa -mcpu=" + str_arch + " " + clang_name + " -o " + isabin_path;
	
	std::cout << "System: " << command.c_str() << "\n";
    
	system( command.c_str() );
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
	  std::unique_ptr<llvm::raw_fd_ostream> isabin_fs( new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::F_Text));

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

	  isabin_fs->flush();
	}
      }


    
    std::string lld_path = std::string(ROCM_DIR) + "/llvm/bin/ld.lld";
    std::string command = lld_path + " -shared " + isabin_path + " -o " + shared_path;

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "System: " << command.c_str() << "\n";
      }
    
    system( command.c_str() );
  }

  
  void llvm_build_function_rocm(JitFunction& func)
  {
    if (Layout::primaryNode())
      {
	//llvm_module_dump();
      }
    
    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << str_pretty << "\n";
      }
    
    std::string shared_path = "module_" + str_kernel_name + ".so";

    if (Layout::primaryNode())
      {
	build_function_codegen( shared_path );
      }

    //
    // All nodes wait until primary node has finished LLVM codegen
    QMP_barrier();
    
    std::ostringstream sstream;
    std::ifstream fin(shared_path, ios::binary);
    sstream << fin.rdbuf();
    std::string shared(sstream.str());
    
    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "shared object file read back in. size = " << shared.size() << "\n";
      }
    
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
  }
#endif


  

  void llvm_build_function(JitFunction& func)
  {
    builder->SetInsertPoint(bb_stack,it_stack);
    builder->CreateBr( bb_afterstack );

    
#ifdef QDP_BACKEND_ROCM
    llvm_build_function_rocm(func);
#else
    llvm_build_function_cuda(func);
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


#if 0
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
  llvm::Value* llvm_isfinite_f32( llvm::Value* lhs ) { return llvm_call_f32( func_isfinite_f32 , lhs ); }

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
  llvm::Value* llvm_isfinite_f64( llvm::Value* lhs ) { return llvm_call_f64( func_isfinite_f64 , lhs ); }

  llvm::Value* llvm_pow_f64( llvm::Value* lhs, llvm::Value* rhs ) { return llvm_call_f64( func_pow_f64 , lhs , rhs ); }
  llvm::Value* llvm_atan2_f64( llvm::Value* lhs, llvm::Value* rhs ) { return llvm_call_f64( func_atan2_f64 , lhs , rhs ); }
#endif

  llvm::Value* llvm_sin_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("sin_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_acos_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("acos_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_asin_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("asin_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_atan_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("atan_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_ceil_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("ceil_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_floor_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("floor_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_cos_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("cos_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_cosh_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("cosh_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_exp_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("exp_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_log_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("log_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_log10_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("log10_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_sinh_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("sinh_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_tan_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("tan_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_tanh_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("tanh_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_fabs_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("fabs_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_sqrt_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("sqrt_f32") , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* llvm_isfinite_f32( llvm::Value* lhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("isfinite_f32") , jitprec::f32 , 1 )  , lhs ); }

  llvm::Value* llvm_pow_f32( llvm::Value* lhs, llvm::Value* rhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("pow_f32") , jitprec::f32 , 2 )  , lhs , rhs ); }
  llvm::Value* llvm_atan2_f32( llvm::Value* lhs, llvm::Value* rhs ) { return llvm_call_f32( llvm_get_func( mapMath.at("atan2_f32") , jitprec::f32 , 2 )  , lhs , rhs ); }

  
  llvm::Value* llvm_sin_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("sin_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_acos_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("acos_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_asin_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("asin_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_atan_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("atan_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_ceil_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("ceil_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_floor_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("floor_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_cos_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("cos_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_cosh_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("cosh_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_exp_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("exp_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_log_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("log_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_log10_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("log10_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_sinh_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("sinh_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_tan_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("tan_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_tanh_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("tanh_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_fabs_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("fabs_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_sqrt_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("sqrt_f64") , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* llvm_isfinite_f64( llvm::Value* lhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("isfinite_f64") , jitprec::f64 , 1 )  , lhs ); }

  llvm::Value* llvm_pow_f64( llvm::Value* lhs, llvm::Value* rhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("pow_f64") , jitprec::f64 , 2 )  , lhs , rhs ); }
  llvm::Value* llvm_atan2_f64( llvm::Value* lhs, llvm::Value* rhs ) { return llvm_call_f64( llvm_get_func( mapMath.at("atan2_f64") , jitprec::f64 , 2 )  , lhs , rhs ); }

  
} // namespace QDP

