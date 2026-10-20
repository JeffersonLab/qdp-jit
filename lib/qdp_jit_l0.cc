#include "qdp_config.h"
#include "qdp_params.h"
#include "qdp_forward_jit.h"
#include "qdp_llvm_includes.h"
#include "qdp_llvm.h"

#include <system_error>
#include <memory>
#include <unistd.h>
#include <signal.h>
#include <system_error>
#include <memory>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>

#include "qdp_jit.h"


using namespace llvm;
using namespace QDP;


namespace JIT {

  
  void CompilerL0::setup( CompilerFactory* fac , string ftype , string pretty )
  {
    Compiler::setup(fac,ftype,pretty);

    std::string dl = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024";
    Mod->setDataLayout(dl);
    Mod->setTargetTriple(target().triple);  //.getTriple()

    // Count it
    llvm_jit_stats_jitted(true);
  }


  namespace {
    TargetInfo make_target()
    {
      TargetInfo t;
      t.global_addrspace = 1;
      t.local_addrspace  = 3;

      //std::string str_triple("spir64-unknown-unknown");
      std::string str_triple("spir64");
      t.triple.setTriple(str_triple);

      QDPIO::cout << "LLVM initialization" << std::endl;
      QDPIO::cout << "  Target triple                       : " << str_triple << "\n";

      t.math["sin_f32"]="_Z3sinf";
      t.math["acos_f32"]="_Z4acosf";
      t.math["asin_f32"]="_Z4asinf";
      t.math["atan_f32"]="_Z4atanf";
      t.math["ceil_f32"]="_Z4ceilf";
      t.math["floor_f32"]="_Z4floorf";
      t.math["cos_f32"]="_Z3cosf";
      t.math["cosh_f32"]="_Z4coshf";
      t.math["exp_f32"]="_Z3expf";
      t.math["log_f32"]="_Z3logf";
      t.math["log10_f32"]="_Z5log10f";
      t.math["sinh_f32"]="_Z4sinhf";
      t.math["tan_f32"]="_Z3tanf";
      t.math["tanh_f32"]="_Z4tanhf";
      t.math["fabs_f32"]="_Z4fabsf";
      t.math["sqrt_f32"]="_Z4sqrtf";
      t.math["isfinite_f32"]="_Z8isfinitef";

      t.math["pow_f32"]="_Z3powff";
      t.math["atan2_f32"]="_Z5atan2ff";

      t.math["sin_f64"]="_Z3sind";
      t.math["acos_f64"]="_Z4acosd";
      t.math["asin_f64"]="_Z4asind";
      t.math["atan_f64"]="_Z4atand";
      t.math["ceil_f64"]="_Z4ceild";
      t.math["floor_f64"]="_Z4floord";
      t.math["cos_f64"]="_Z3cosd";
      t.math["cosh_f64"]="_Z4coshd";
      t.math["exp_f64"]="_Z3expd";
      t.math["log_f64"]="_Z3logd";
      t.math["log10_f64"]="_Z5log10d";
      t.math["sinh_f64"]="_Z4sinhd";
      t.math["tan_f64"]="_Z3tand";
      t.math["tanh_f64"]="_Z4tanhd";
      t.math["fabs_f64"]="_Z4fabsd";
      t.math["sqrt_f64"]="_Z4sqrtd";
      t.math["isfinite_f64"]="_Z8isfinited";

      t.math["pow_f64"]="_Z3powdd";
      t.math["atan2_f64"]="_Z5atan2dd";

      return t;
    }
  }


  const TargetInfo& CompilerL0::target_info()
  {
    static TargetInfo t = make_target();
    return t;
  }


  llvm::Value* CompilerL0::get_shared_ptr( llvm::Type *ty , int n )
  {
    llvm::GlobalVariable *gv = new llvm::GlobalVariable ( *Mod , 
							  llvm::ArrayType::get( ty , n ) ,
							  false , 
							  llvm::GlobalVariable::InternalLinkage, 
							  nullptr, 
							  "shared_buffer", 
							  0, //GlobalVariable *InsertBefore=0, 
							  llvm::GlobalVariable::NotThreadLocal, //ThreadLocalMode=NotThreadLocal
							  target().local_addrspace, // unsigned AddressSpace=0, 
							  false); //bool isExternallyInitialized=false)

    return builder->CreatePointerCast(gv, llvm::PointerType::get( Mod->getContext() , target().local_addrspace ) );
  }



  void CompilerL0::bar_sync()
  {
    special( "_Z7barrierj" , get_type<void>() , { get_type<int>() } , { create_value(3) } );
  }


  llvm::Value * CompilerL0::call_special_tidx()    { return cast( get_type<int>() , special( "_Z12get_local_idj"   ,get_type<size_t>() , { get_type<int>() } , { create_value(0) } )); }
  llvm::Value * CompilerL0::call_special_ntidx()   { return cast( get_type<int>() , special( "_Z14get_local_sizej" ,get_type<size_t>() , { get_type<int>() } , { create_value(0) } )); }
  llvm::Value * CompilerL0::call_special_ctaidx()  { return cast( get_type<int>() , special( "_Z12get_group_idj"   ,get_type<size_t>() , { get_type<int>() } , { create_value(0) } )); }
  llvm::Value * CompilerL0::call_special_nctaidx() { return cast( get_type<int>() , special( "_Z14get_num_groupsj" ,get_type<size_t>() , { get_type<int>() } , { create_value(0) } )); }
  llvm::Value * CompilerL0::call_special_ctaidy()  { return cast( get_type<int>() , special( "_Z12get_group_idj"   ,get_type<size_t>() , { get_type<int>() } , { create_value(1) } )); }



  void CompilerL0::build_function_levelzero_codegen(JitFunction& func, const std::string& spirv_path)
  {
    StopWatch swatch(false);
    func.time_math = 0.;
    swatch.reset();
    swatch.start();

    //QDPIO::cout << "setting module data layout\n";

    if ( options().opt_level != -1 )
      {
	opt();
      }

    
    swatch.stop();
    func.time_passes = swatch.getTimeInMicroseconds();

    
    if (options().verbose || options().keepfiles)
      {
	std::string module_name = "module_" + str_kernel_name + ".bc";
	QDPIO::cout << "write code to " << module_name << "\n";
	std::error_code EC;

	llvm::raw_fd_ostream OS(module_name, EC, llvm::sys::fs::OF_None);

	llvm::WriteBitcodeToFile(*Mod, OS);
	OS.flush();
      }
    
    swatch.reset();
    swatch.start();

    {
      std::string Err;

      SPIRV::TranslatorOpts::ExtensionsStatusMap ExtensionsStatus;

      // Match spirv-llvm default options as closely as possible.
      SPIRV::TranslatorOpts Opts(SPIRV::VersionNumber::MaximumVersion,
				 ExtensionsStatus);

      std::ofstream OutFile(spirv_path, std::ios::binary);
      if (!llvm::writeSpirv(Mod.get(), Opts, OutFile, Err)) {
	QDPIO::cout << "writeSpirv failed: " << Err << "\n";
	QDP_abort(1);
      }

      OutFile.flush();
      OutFile.close();
    }
    
    swatch.stop();
    func.time_codegen = swatch.getTimeInMicroseconds();
  }


  
  CodeObject CompilerL0::build_function_backend(JitFunction& func)
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

    std::string spirv_path =
      "module_" + str_kernel_name +
      "_node_" + std::to_string(nodeNumber) +
      "_pid_" + std::to_string(::getpid()) +
      ".spv";

    // call codegen
    build_function_levelzero_codegen( func , spirv_path );

    CodeObject obj;
    obj.image = spirv_path;
    if (! options().keepfiles )
      obj.remove_after_load = spirv_path;
    return obj;
  }



  Compiler* CompilerCreatorGPU::create( CompilerFactory* fac , string ftype , string pretty )
  {
    CompilerL0* c = new CompilerL0;
    c->setup( fac , ftype , pretty );
    return c;
  }

  
} // JIT
