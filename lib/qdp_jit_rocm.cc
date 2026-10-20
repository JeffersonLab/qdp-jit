#include "qdp_config.h"
#include "qdp_params.h"
#include "qdp_forward_jit.h"
#include "qdp_llvm_includes.h"
#include "qdp_llvm.h"

#include <system_error>
#include <memory>
#include <unistd.h>
#include <signal.h>
#include <set>

#include "qdp_jit.h"

#ifdef QDP_JIT_ROCM_COMGR_LINK
#if __has_include(<amd_comgr/amd_comgr.h>)
#include <amd_comgr/amd_comgr.h>
#else
#include <amd_comgr.h>
#endif
#endif


using namespace llvm;
using namespace QDP;


namespace JIT {

#ifdef QDP_JIT_ROCM_COMGR_LINK
  namespace {

    struct ComgrData {
      amd_comgr_data_t h = {0};
      ~ComgrData() { if (h.handle) amd_comgr_release_data(h); }
    };

    struct ComgrDataSet {
      amd_comgr_data_set_t h = {0};
      ~ComgrDataSet() { if (h.handle) amd_comgr_destroy_data_set(h); }
    };

    struct ComgrActionInfo {
      amd_comgr_action_info_t h = {0};
      ~ComgrActionInfo() { if (h.handle) amd_comgr_destroy_action_info(h); }
    };

    bool comgr_ok( const char* what , amd_comgr_status_t status )
    {
      if (status == AMD_COMGR_STATUS_SUCCESS)
	return true;

      const char* str = nullptr;
      if (amd_comgr_status_string( status , &str ) != AMD_COMGR_STATUS_SUCCESS || !str)
	str = "unknown status";
      QDPIO::cerr << what << " failed: " << str << "\n";
      return false;
    }

    bool comgr_get_bytes( amd_comgr_data_t data , std::vector<char>& bytes )
    {
      size_t size = 0;
      if (!comgr_ok( "amd_comgr_get_data" , amd_comgr_get_data( data , &size , nullptr ) ))
	return false;
      bytes.resize( size );
      return comgr_ok( "amd_comgr_get_data" , amd_comgr_get_data( data , &size , bytes.data() ) );
    }

    void comgr_print_log( amd_comgr_data_set_t set )
    {
      size_t count = 0;
      if (amd_comgr_action_data_count( set , AMD_COMGR_DATA_KIND_LOG , &count ) != AMD_COMGR_STATUS_SUCCESS)
	return;

      for ( size_t i = 0 ; i < count ; ++i )
	{
	  ComgrData log;
	  std::vector<char> bytes;
	  if (amd_comgr_action_data_get_data( set , AMD_COMGR_DATA_KIND_LOG , i , &log.h ) == AMD_COMGR_STATUS_SUCCESS &&
	      comgr_get_bytes( log.h , bytes ) && !bytes.empty())
	    {
	      QDPIO::cerr << "comgr log:\n" << std::string( bytes.begin() , bytes.end() ) << "\n";
	    }
	}
    }

    bool comgr_link( const char* obj , size_t size , const std::string& isa , std::vector<char>& exe )
    {
      ComgrData input;
      ComgrDataSet input_set;
      ComgrDataSet output_set;
      ComgrActionInfo info;

      if (!comgr_ok( "amd_comgr_create_data" , amd_comgr_create_data( AMD_COMGR_DATA_KIND_RELOCATABLE , &input.h ) ) ||
	  !comgr_ok( "amd_comgr_set_data" , amd_comgr_set_data( input.h , size , obj ) ) ||
	  !comgr_ok( "amd_comgr_set_data_name" , amd_comgr_set_data_name( input.h , "qdpjit.o" ) ) ||
	  !comgr_ok( "amd_comgr_create_data_set" , amd_comgr_create_data_set( &input_set.h ) ) ||
	  !comgr_ok( "amd_comgr_data_set_add" , amd_comgr_data_set_add( input_set.h , input.h ) ) ||
	  !comgr_ok( "amd_comgr_create_data_set" , amd_comgr_create_data_set( &output_set.h ) ) ||
	  !comgr_ok( "amd_comgr_create_action_info" , amd_comgr_create_action_info( &info.h ) ) ||
	  !comgr_ok( "amd_comgr_action_info_set_isa_name" , amd_comgr_action_info_set_isa_name( info.h , isa.c_str() ) ) ||
	  !comgr_ok( "amd_comgr_action_info_set_logging" , amd_comgr_action_info_set_logging( info.h , true ) ))
	return false;

      if (!comgr_ok( "amd_comgr_do_action(LINK_RELOCATABLE_TO_EXECUTABLE)" ,
		     amd_comgr_do_action( AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE , info.h , input_set.h , output_set.h ) ))
	{
	  comgr_print_log( output_set.h );
	  return false;
	}

      size_t count = 0;
      if (!comgr_ok( "amd_comgr_action_data_count" , amd_comgr_action_data_count( output_set.h , AMD_COMGR_DATA_KIND_EXECUTABLE , &count ) ))
	return false;

      if (count != 1)
	{
	  QDPIO::cerr << "comgr link produced " << count << " executables, expected 1\n";
	  comgr_print_log( output_set.h );
	  return false;
	}

      ComgrData output;
      if (!comgr_ok( "amd_comgr_action_data_get_data" , amd_comgr_action_data_get_data( output_set.h , AMD_COMGR_DATA_KIND_EXECUTABLE , 0 , &output.h ) ))
	return false;

      return comgr_get_bytes( output.h , exe );
    }

  } // namespace


  std::vector<char> link_amdgpu_object_comgr( const char* obj , size_t size , const std::string& isa )
  {
    std::vector<char> exe;
    if (!comgr_link( obj , size , isa , exe ))
      {
	QDP_error_exit("linking with amd_comgr failed (ISA %s)", isa.c_str());
      }
    return exe;
  }
#endif


  void CompilerROCM::setup( CompilerFactory* fac , string ftype , string pretty )
  {
    Compiler::setup(fac,ftype,pretty);

    Mod->setDataLayout(target().machine->createDataLayout());

    Mod->addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version", 500);

    // Count it
    llvm_jit_stats_jitted(true);
  }


  void CompilerROCM::bar_sync()
  {
    llvm::FunctionType *IntrinFnTy = llvm::FunctionType::get(llvm::Type::getVoidTy(*get_context()), false);

    llvm::AttrBuilder ABuilder(*get_context());
    
    ABuilder.addAttribute(llvm::Attribute::ReadNone);

    std::string bar_name("llvm.amdgcn.s.barrier");

    auto Bar = Mod->getOrInsertFunction( bar_name.c_str() , 
					 IntrinFnTy , 
					 llvm::AttributeList::get(*get_context(), 
								  llvm::AttributeList::FunctionIndex, 
								  ABuilder) );

    builder->CreateCall(Bar);
  }

  

  llvm::Value* CompilerROCM::get_shared_ptr( llvm::Type *ty , int n )
  {
    const unsigned addrspace = target().local_addrspace;

    llvm::GlobalVariable *gv = new llvm::GlobalVariable ( *Mod , 
							  llvm::ArrayType::get(ty,0) ,
							  false , 
							  llvm::GlobalVariable::ExternalLinkage, 
							  0, 
							  "shared_buffer", 
							  0, //GlobalVariable *InsertBefore=0, 
							  llvm::GlobalVariable::NotThreadLocal, //ThreadLocalMode=NotThreadLocal
							  target().local_addrspace, // unsigned AddressSpace=0, 
							  false); //bool isExternallyInitialized=false)

    return builder->CreatePointerCast(
				      gv,
				      llvm::PointerType::get(Mod->getContext(), addrspace)
				      );
  }


  
  namespace {
    TargetInfo make_target()
    {
      TargetInfo t;
      t.global_addrspace = 0;
      t.local_addrspace  = 3;

      t.arch = gpu_get_arch();

      t.triple.setArch (llvm::Triple::ArchType::amdgcn);
      t.triple.setVendor (llvm::Triple::VendorType::AMD);
      t.triple.setOS (llvm::Triple::OSType::AMDHSA);

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

      t.math["sin_f32"]="__ocml_sin_f32";
      t.math["acos_f32"]="__ocml_acos_f32";
      t.math["asin_f32"]="__ocml_asin_f32";
      t.math["atan_f32"]="__ocml_atan_f32";
      t.math["ceil_f32"]="__ocml_ceil_f32";
      t.math["floor_f32"]="__ocml_floor_f32";
      t.math["cos_f32"]="__ocml_cos_f32";
      t.math["cosh_f32"]="__ocml_cosh_f32";
      t.math["exp_f32"]="__ocml_exp_f32";
      t.math["log_f32"]="__ocml_log_f32";
      t.math["log10_f32"]="__ocml_log10_f32";
      t.math["sinh_f32"]="__ocml_sinh_f32";
      t.math["tan_f32"]="__ocml_tan_f32";
      t.math["tanh_f32"]="__ocml_tanh_f32";
      t.math["fabs_f32"]="__ocml_fabs_f32";
      t.math["sqrt_f32"]="__ocml_sqrt_f32";
      t.math["isfinite_f32"]="__ocml_isfinite_f32";

      t.math["pow_f32"]="__ocml_pow_f32";
      t.math["atan2_f32"]="__ocml_atan2_f32";

      t.math["sin_f64"]="__ocml_sin_f64";
      t.math["acos_f64"]="__ocml_acos_f64";
      t.math["asin_f64"]="__ocml_asin_f64";
      t.math["atan_f64"]="__ocml_atan_f64";
      t.math["ceil_f64"]="__ocml_ceil_f64";
      t.math["floor_f64"]="__ocml_floor_f64";
      t.math["cos_f64"]="__ocml_cos_f64";
      t.math["cosh_f64"]="__ocml_cosh_f64";
      t.math["exp_f64"]="__ocml_exp_f64";
      t.math["log_f64"]="__ocml_log_f64";
      t.math["log10_f64"]="__ocml_log10_f64";
      t.math["sinh_f64"]="__ocml_sinh_f64";
      t.math["tan_f64"]="__ocml_tan_f64";
      t.math["tanh_f64"]="__ocml_tanh_f64";
      t.math["fabs_f64"]="__ocml_fabs_f64";
      t.math["sqrt_f64"]="__ocml_sqrt_f64";
      t.math["isfinite_f64"]="__ocml_isfinite_f64";

      t.math["pow_f64"]="__ocml_pow_f64";
      t.math["atan2_f64"]="__ocml_atan2_f64";

      return t;
    }
  }


  const TargetInfo& CompilerROCM::target_info()
  {
    static TargetInfo t = make_target();
    return t;
  }


  llvm::Value * CompilerROCM::call_special_tidx()    { return special("llvm.amdgcn.workitem.id.x" , get_type<int>() , {} , {} ); }
  llvm::Value * CompilerROCM::call_special_ctaidx()  { return special("llvm.amdgcn.workgroup.id.x", get_type<int>() , {} , {} ); }

  llvm::Value * CompilerROCM::load_dispatch_field( unsigned offset , llvm::Type* ty )
  {
    llvm::Value* dp = special("llvm.amdgcn.dispatch.ptr", llvm::PointerType::get( *get_context() , 4 ) , {} , {} );
    llvm::Value* p  = builder->CreateConstInBoundsGEP1_64( builder->getInt8Ty() , dp , offset );
    llvm::LoadInst* ld = builder->CreateAlignedLoad( ty , p , llvm::Align( 4 ) );
    ld->setMetadata( llvm::LLVMContext::MD_invariant_load , llvm::MDNode::get( *get_context() , {} ) );
    return ld;
  }

  llvm::Value * CompilerROCM::call_special_ntidx()
  {
    return builder->CreateZExt( load_dispatch_field( 4 , builder->getInt16Ty() ) , get_type<int>() );
  }

  llvm::Value * CompilerROCM::call_special_nctaidx()
  {
    return builder->CreateUDiv( load_dispatch_field( 12 , get_type<int>() ) , call_special_ntidx() );
  }
  llvm::Value * CompilerROCM::call_special_ctaidy()  { return special("llvm.amdgcn.workgroup.id.y", get_type<int>() , {} , {} ); }


  const std::vector< std::pair< std::string , std::unique_ptr< llvm::MemoryBuffer > > >& CompilerROCM::device_libraries()
  {
    static std::vector< std::pair< std::string , std::unique_ptr< llvm::MemoryBuffer > > > libs;
    static bool done = false;

    if (!done)
      {
	done = true;

	std::unique_ptr< llvm::MemoryBuffer > libdevice = read_libdevice();
	if (libdevice)
	  libs.emplace_back( "libdevice" , std::move( libdevice ) );

	read_ocml( libs );
      }

    return libs;
  }


  namespace {
    std::set< std::string > undefined_symbols( const llvm::Module& M )
    {
      std::set< std::string > s;
      for ( const llvm::GlobalValue& gv : M.global_values() )
	if (gv.isDeclaration())
	  s.insert( gv.getName().str() );
      return s;
    }
  }


  void CompilerROCM::link_device_libraries()
  {
    const auto& libs = device_libraries();

    std::set< std::string > undefined = undefined_symbols( *Mod );

    while (true)
      {
	for ( const auto& lib : libs )
	  {
	    if (options().verbose)
	      {
		QDPIO::cout << "linking needed definitions from " << lib.first << "\n";
	      }

	    llvm::Expected<std::unique_ptr<llvm::Module>> m = llvm::getLazyBitcodeModule( lib.second->getMemBufferRef() , *get_context() );
	    if (std::error_code ec = errorToErrorCode(m.takeError()))
	      {
		errs() << "Error reading bitcode from " << lib.first << ": " << ec.message() << "\n";
		QDP_abort(1);
	      }

	    if (llvm::Linker::linkModules( *Mod , std::move( m.get() ) , llvm::Linker::Flags::LinkOnlyNeeded ))
	      {
		QDPIO::cerr << "Linking " << lib.first << " failed\n";
		QDP_abort(1);
	      }
	  }

	std::set< std::string > now = undefined_symbols( *Mod );
	if (now == undefined)
	  break;
	undefined = std::move( now );
      }
  }


  void CompilerROCM::read_ocml( std::vector< std::pair< std::string , std::unique_ptr< llvm::MemoryBuffer > > >& buffers )
  {
    std::string arch = target().arch;
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
    libs.push_back(std::string(ROCM_DIR) + "/amdgcn/bitcode/oclc_abi_version_500.bc");
  
    libs.insert( libs.end() , options().extra_libs.begin() , options().extra_libs.end() );
    
    for( int i = 0 ; i < libs.size() ; ++i )
      {
	std::string FileName = libs[i];

	if (options().verbose)
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

	    buffers.emplace_back( FileName , std::move( mb.get() ) );
	  }
      }
  }


  std::unique_ptr< llvm::MemoryBuffer > CompilerROCM::read_libdevice()
  {
    std::vector<std::string> tried;
    std::string FileName = find_device_library( { ROCM_DIR } ,
						{ "llvm/lib/libdevice/" , "llvm/lib/" } ,
						{ "libm-amdgcn-ARCH.bc" , "libomptarget-amdgcn-ARCH.bc" , "libomptarget-new-amdgpu-ARCH.bc" , "libomptarget-old-amdgpu-ARCH.bc" , "libomptarget-amdgpu-ARCH.bc" } ,
						nullptr , &tried );

    if (FileName.empty())
      {
	if (!llvm_get_libdevice_path().empty() || !llvm_get_libdevice_name().empty())
	  abort_device_library_not_found( tried );

	if (options().verbose)
	  {
	    QDPIO::cout << "libdevice not found, linking ocml only\n";
	  }
	return nullptr;
      }

    
    ErrorOr<std::unique_ptr<MemoryBuffer>> mb = MemoryBuffer::getFile(FileName);
    if (std::error_code ec = mb.getError()) {
      errs() << ec.message();
      QDP_abort(1);
    }

    return std::move( mb.get() );
  }



  void CompilerROCM::build_function_rocm_codegen( JitFunction& func , const std::string& shared_path , std::string& shared )
  {
    StopWatch swatch(false);
    swatch.start();
    
    if (math_declarations.size() > 0)
      {
	link_device_libraries();
      }

    swatch.stop();
    func.time_math = swatch.getTimeInMicroseconds();
    
    swatch.reset();
    swatch.start();


    if (options().verbose)
      {
	QDPIO::cout << "running passes ...\n";
      }
 
    if (options().verbose)
      {
	QDPIO::cout << "internalize and remove dead code ...\n";
      }

    internalize_and_dce();

    swatch.stop();
    func.time_passes = swatch.getTimeInMicroseconds();

    if ( options().workgroup_bound > 0 && !func.get_explicit_geom() )
      {
	if (llvm::Function* kernel = Mod->getFunction( str_kernel_name ))
	  {
	    kernel->addFnAttr( "amdgpu-flat-work-group-size" , "1," + std::to_string( options().workgroup_bound ) );
	    func.set_workgroup_bound( options().workgroup_bound );
	  }
      }

    if ( options().opt_prepass && options().opt_level > 0 )
      {
	run_pipeline( target().machine.get() ,
		      "function(sroa,instcombine<no-verify-fixpoint>,simplifycfg,"
		      "loop(loop-rotate,indvars,loop-unroll-full),"
		      "sroa,instcombine<no-verify-fixpoint>,simplifycfg)" );
      }
    else if ( options().opt_level != -1 )
      {
	opt();
      }

    swatch.reset();
    swatch.start();

    //node number
    int nodeNumber = options().node;
    
    std::string isabin_path =
      options().prepend_path +
      "module_" + str_kernel_name +
      "_node_" + std::to_string(nodeNumber) +
      "_pid_" + std::to_string(::getpid()) +
      ".o";

#ifdef QDP_JIT_ROCM_COMGR_LINK
    llvm::SmallVector<char, 0> isabin;
#endif

    if (options().opt_level > 0)
      {
	OptimizationLevel opt = OptimizationLevel::O0;
	switch (options().opt_level)
	  {
	  case 1: opt = OptimizationLevel::O1; break;
	  case 2: opt = OptimizationLevel::O2; break;
	  case 3: opt = OptimizationLevel::O3; break;
	  default: break;
	  }

	if (options().verbose)
	  {
	    QDPIO::cout << "running optimization passes ...\n";
	  }

	run_passes( target().machine.get() , opt );
      }


    // ------------------- CODE GEN ----------------------

    llvm::legacy::PassManager CodeGenPasses;

    std::error_code ec;

    {
#ifdef QDP_JIT_ROCM_COMGR_LINK
      llvm::raw_svector_ostream isabin_os(isabin);
#else
      std::unique_ptr<llvm::raw_fd_ostream> isabin_fs( new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::OF_Text));
      llvm::raw_pwrite_stream& isabin_os = *isabin_fs;
#endif

      if (target().machine->addPassesToEmitFile(CodeGenPasses,
					     isabin_os,
					     nullptr,
					     llvm::CodeGenFileType::ObjectFile ))
	{
	  QDPIO::cerr << "target does not support generation of object file type!\n";
	  QDP_abort(1);
	}

      if (options().verbose)
	{
	  QDPIO::cout << "running code gen ...\n";
	}

      CodeGenPasses.run(*Mod);
    }

    swatch.stop();
    func.time_codegen = swatch.getTimeInMicroseconds();

    swatch.reset();
    swatch.start();

#ifdef QDP_JIT_ROCM_COMGR_LINK
    if (options().keepfiles)
      {
	std::ofstream fout(isabin_path, ios::binary);
	fout.write( isabin.data() , isabin.size() );
      }

    std::string isa = "amdgcn-amd-amdhsa--" + target().arch;

    if (options().verbose)
      {
	QDPIO::cout << "Linking with amd_comgr, ISA " << isa << ", object size " << isabin.size() << "\n";
      }

    std::vector<char> exe = link_amdgpu_object_comgr( isabin.data() , isabin.size() , isa );
    shared.assign( exe.begin() , exe.end() );

    if (options().keepfiles)
      {
	std::ofstream fout(shared_path, ios::binary);
	fout.write( shared.data() , shared.size() );
      }

    if (options().verbose)
      {
	QDPIO::cout << "Linker invocation successful" << std::endl;
      }
#else
    //
    // Call linker as a library
    //    
    int argc=5;
    const char *argv[] = { "ld" , "-shared" , isabin_path.c_str() , "-o" , shared_path.c_str() };

    if (options().verbose)
      {
	QDPIO::cout << "Library call to ld.lld: ";
	for ( int i = 0 ; i < argc ; ++i )
	  QDPIO::cout << argv[i] << " ";
	QDPIO::cout << std::endl;
      }

    std::string command =
      std::string(QDP_ROCM_LD_LLD) +
      " -shared " +
      isabin_path +
      " -o " +
      shared_path;

    if (options().verbose) {
      QDPIO::cout << "System linker call: " << command << "\n";
    }

    int rc = system(command.c_str());
    if (rc != 0) {
      QDP_error_exit("calling ROCm ld.lld failed");
    }


    
    if (options().verbose)
      {
	QDPIO::cout << "Linker invocation successful" << std::endl;
      }

    if (! options().keepfiles )
      {
	if (std::remove(isabin_path.c_str()))
	  {
	    QDPIO::cout << "Error removing file: " << shared_path << std::endl;
	    QDP_abort(1);
	  }
      }
#endif
    
    swatch.stop();
    func.time_linking = swatch.getTimeInMicroseconds();
  }


  CodeObject CompilerROCM::build_function_backend(JitFunction& func)
  {
    func.setIsGPU(true);

    if (options().verbose)
      {
	QDPIO::cout << "\n\n";
	QDPIO::cout << str_pretty << std::endl;
	
	if (options().is_primary)
	  {
	    module_dump();
	  }
      }

    //node number
    int nodeNumber = options().node;
    
    std::string shared_path =
      options().prepend_path +
      "module_" + str_kernel_name +
      "_node_" + std::to_string(nodeNumber) +
      "_pid_" + std::to_string(::getpid()) +
      ".so";

    // call codegen
    std::string shared;
    build_function_rocm_codegen( func , shared_path , shared );

#ifndef QDP_JIT_ROCM_COMGR_LINK
    std::ostringstream sstream;
    std::ifstream fin(shared_path, ios::binary);
    sstream << fin.rdbuf();
    shared = sstream.str();

    if (! options().keepfiles )
      {
	if (std::remove(shared_path.c_str()))
	  {
	    QDPIO::cout << "Error removing file: " << shared_path << std::endl;
	    QDP_abort(1);
	  }
      }
#endif

    
    if (options().verbose)
      {
	QDPIO::cout << "shared object file read back in. size = " << shared.size() << "\n";
      }

    CodeObject obj;
    obj.image = std::move( shared );
    return obj;
  }



  Compiler* CompilerCreatorGPU::create( CompilerFactory* fac , string ftype , string pretty )
  {
    CompilerROCM* c = new CompilerROCM;
    c->setup( fac , ftype , pretty );
    return c;
  }

  
  
} // JIT
