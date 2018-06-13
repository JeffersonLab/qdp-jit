#ifndef QDP_LLVM
#define QDP_LLVM

#include "qdp_config.h"

//#define __STDC_LIMIT_MACROS
//#define __STDC_CONSTANT_MACROS

#include "llvm/IRReader/IRReader.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Bitcode/BitcodeReader.h"
//#include "llvm/Bitcode/Writer.h"
//#include "llvm/IR/PassManager.h"  // not ready yet in LLVM 3.8
#include "llvm/IR/LegacyPassManager.h"
//#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"


#include "llvm/CodeGen/TargetLowering.h"

// pre llvm6
#if 0
#include "llvm/Target/TargetLowering.h"
#endif

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Support/ToolOutputFile.h"
//#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/SourceMgr.h"
//#include "llvm/Assembly/Parser.h"
#include "llvm/IR/TypeBuilder.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Attributes.h"

#include "llvm/Support/raw_os_ostream.h"

//#include "llvm/Support/DataStream.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Program.h"
//#include "llvm/Support/system_error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Host.h"

#include <system_error>

//#include "llvm/ExecutionEngine/ObjectBuffer.h"
#include "llvm/IR/GlobalVariable.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"

namespace llvm {
ModulePass *createNVVMReflectPass(void);
}


namespace QDP {


  namespace llvm_debug {
    extern bool debug_func_build     ;
    extern bool debug_func_dump      ;
    extern bool debug_func_write     ;
    extern bool debug_loop_vectorizer;
  }

  namespace llvm_opt {
    extern int opt_level;   // opt -O level
    extern int nvptx_FTZ;   // NVPTX Flush subnormals to zero
  }

  namespace ptx_db {
    extern bool db_enabled;
    extern std::string dbname;
  }

  CUfunction llvm_ptx_db( const char * pretty );
  
  extern llvm::LLVMContext TheContext;

  typedef int ParamRef;

  // llvm::IRBuilder<> *builder;
  // llvm::BasicBlock  *entry;
  extern llvm::Function    *mainFunc;
  extern llvm::Function    *mainFunc_extern;
  //extern llvm::Module      *Mod;


  void llvm_set_debug( const char * str );
  void llvm_set_opt( const char * c_str );
  void llvm_set_ptxdb( const char * c_str );
  void llvm_debug_write_set_name( const char* pretty, const char* additional );

  llvm::Value * llvm_create_value( double v );
  llvm::Value * llvm_create_value( int v );
  llvm::Value * llvm_create_value( int64_t v );
  llvm::Value * llvm_create_value( size_t v );
  llvm::Value * llvm_create_value( bool v );


  template<class T> struct llvm_type;

  template<> struct llvm_type<float> { static llvm::Type* value; };
  template<> struct llvm_type<double> { static llvm::Type* value; };
  template<> struct llvm_type<int> { static llvm::Type* value; };
  template<> struct llvm_type<bool> { static llvm::Type* value; };
  template<> struct llvm_type<float*> { static llvm::Type* value; };
  template<> struct llvm_type<double*> { static llvm::Type* value; };
  template<> struct llvm_type<int*> { static llvm::Type* value; };
  template<> struct llvm_type<bool*> { static llvm::Type* value; };

  struct IndexRet {
    IndexRet(){}
    ParamRef p_multi_index;  // if neg. -> recv. buffer, otherwise it's the local index
    ParamRef p_recv_buf;
    // llvm::Value * r_newidx_local;
    // llvm::Value * r_newidx_buffer;
    // llvm::Value * r_pred_in_buf;
    // llvm::Value * r_rcvbuf;
  };

  std::string getPTXfromCUFunc(CUfunction f);
  
  void llvm_append_mattr( const char * attr );

  llvm::Value *llvm_get_arg_lo();
  llvm::Value *llvm_get_arg_hi();
  llvm::Value *llvm_get_arg_myId();
  llvm::Value *llvm_get_arg_ordered();
  llvm::Value *llvm_get_arg_start();

  void llvm_start_new_function();
  void llvm_wrapper_init();
  llvm::PHINode * llvm_phi( llvm::Type* type, unsigned num = 0 );
  llvm::Type* promote( llvm::Type* t0 , llvm::Type* t1 );
  llvm::Value* llvm_cast( llvm::Type *dest_type , llvm::Value *src );

  llvm::SwitchInst * llvm_switch( llvm::Value* val , llvm::BasicBlock* bb_default );

  //llvm::Type* llvm_normalize_values(llvm::Value* lhs , llvm::Value* rhs);

#if 0
  llvm::Value* llvm_b_op( llvm::Value *(*)(llvm::Value *, llvm::Value *) func_float,
			  llvm::Value *(*)(llvm::Value *, llvm::Value *) func_int,
			  llvm::Value* lhs , llvm::Value* rhs );

  llvm::Value* llvm_u_op( llvm::Value *(*)(llvm::Value *) func_float,
			  llvm::Value *(*)(llvm::Value *) func_int,
			  llvm::Value* lhs );
#endif

  llvm::Value* llvm_shl( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_shr( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_rem( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_xor( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_and( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_or( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_mul( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_add( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_sub( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_div( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_eq( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_ge( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_gt( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_le( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_lt( llvm::Value* lhs , llvm::Value* rhs );

  llvm::Value* llvm_neg( llvm::Value* lhs );

  llvm::Value* llvm_sin_f32( llvm::Value* lhs );

  //
  // Convenience function definitions
  //
  llvm::Value* llvm_not( llvm::Value* lhs );

  //  std::string param_next();

  template<class T> ParamRef llvm_add_param();

  template<> ParamRef llvm_add_param<bool>();
  template<> ParamRef llvm_add_param<bool*>();
  template<> ParamRef llvm_add_param<int64_t>();
  template<> ParamRef llvm_add_param<int>();
  template<> ParamRef llvm_add_param<int*>();
  template<> ParamRef llvm_add_param<float>();
  template<> ParamRef llvm_add_param<float*>();
  template<> ParamRef llvm_add_param<double>();
  template<> ParamRef llvm_add_param<double*>();

  llvm::Value * llvm_derefParam( ParamRef r );

  llvm::BasicBlock * llvm_get_insert_block();

  llvm::BasicBlock * llvm_new_basic_block();
  void llvm_cond_branch(llvm::Value * cond, llvm::BasicBlock * thenBB, llvm::BasicBlock * elseBB);
  void llvm_branch(llvm::BasicBlock * BB);
  void llvm_set_insert_point( llvm::BasicBlock * BB );
  llvm::BasicBlock * llvm_get_insert_point();
  void llvm_exit();
  llvm::BasicBlock * llvm_cond_exit( llvm::Value * cond );

  llvm::ConstantInt * llvm_create_const_int(int i);

  llvm::Value * llvm_create_value( double v );
  llvm::Value * llvm_create_value(int v );
  llvm::Value * llvm_create_value(size_t v);
  llvm::Value * llvm_create_value(bool v );


  llvm::Value * llvm_createGEP( llvm::Value * ptr , llvm::Value * idx );
  llvm::Value * llvm_load( llvm::Value * ptr );
  void          llvm_store( llvm::Value * val , llvm::Value * ptr );

  llvm::Value * llvm_load_ptr_idx( llvm::Value * ptr , llvm::Value * idx );
  void          llvm_store_ptr_idx( llvm::Value * val , llvm::Value * ptr , llvm::Value * idx );

  llvm::Value * llvm_array_type_indirection( ParamRef p , llvm::Value* idx );

  llvm::Value * llvm_special( const char * name );

  llvm::Value * llvm_call_special_tidx();
  llvm::Value * llvm_call_special_ntidx();
  llvm::Value * llvm_call_special_ctaidx();

  llvm::Value * llvm_alloca( llvm::Type* type , int elements );
  llvm::Value * llvm_get_shared_ptr( llvm::Type *ty );

  void llvm_bar_sync();

  llvm::Value * llvm_thread_idx();

  void addKernelMetadata(llvm::Function *F);

  CUfunction llvm_get_cufunction(const char* fname, const char* pretty);


  llvm::Value* llvm_sin_f32( llvm::Value* lhs );
  llvm::Value* llvm_acos_f32( llvm::Value* lhs );
  llvm::Value* llvm_asin_f32( llvm::Value* lhs );
  llvm::Value* llvm_atan_f32( llvm::Value* lhs );
  llvm::Value* llvm_ceil_f32( llvm::Value* lhs );
  llvm::Value* llvm_floor_f32( llvm::Value* lhs );
  llvm::Value* llvm_cos_f32( llvm::Value* lhs );
  llvm::Value* llvm_cosh_f32( llvm::Value* lhs );
  llvm::Value* llvm_exp_f32( llvm::Value* lhs );
  llvm::Value* llvm_log_f32( llvm::Value* lhs );
  llvm::Value* llvm_log10_f32( llvm::Value* lhs );
  llvm::Value* llvm_sinh_f32( llvm::Value* lhs );
  llvm::Value* llvm_tan_f32( llvm::Value* lhs ); 
  llvm::Value* llvm_tanh_f32( llvm::Value* lhs );
  llvm::Value* llvm_fabs_f32( llvm::Value* lhs ); 
  llvm::Value* llvm_sqrt_f32( llvm::Value* lhs );
  llvm::Value* llvm_isfinite_f32( llvm::Value* lhs );

  llvm::Value* llvm_pow_f32( llvm::Value* lhs, llvm::Value* rhs ); 
  llvm::Value* llvm_atan2_f32( llvm::Value* lhs, llvm::Value* rhs );

  llvm::Value* llvm_sin_f64( llvm::Value* lhs );
  llvm::Value* llvm_acos_f64( llvm::Value* lhs );
  llvm::Value* llvm_asin_f64( llvm::Value* lhs );
  llvm::Value* llvm_atan_f64( llvm::Value* lhs );
  llvm::Value* llvm_ceil_f64( llvm::Value* lhs );
  llvm::Value* llvm_floor_f64( llvm::Value* lhs );
  llvm::Value* llvm_cos_f64( llvm::Value* lhs ); 
  llvm::Value* llvm_cosh_f64( llvm::Value* lhs );
  llvm::Value* llvm_exp_f64( llvm::Value* lhs ); 
  llvm::Value* llvm_log_f64( llvm::Value* lhs ); 
  llvm::Value* llvm_log10_f64( llvm::Value* lhs );
  llvm::Value* llvm_sinh_f64( llvm::Value* lhs ); 
  llvm::Value* llvm_tan_f64( llvm::Value* lhs ); 
  llvm::Value* llvm_tanh_f64( llvm::Value* lhs );
  llvm::Value* llvm_fabs_f64( llvm::Value* lhs ); 
  llvm::Value* llvm_sqrt_f64( llvm::Value* lhs );
  llvm::Value* llvm_isfinite_f64( llvm::Value* lhs );

  llvm::Value* llvm_pow_f64( llvm::Value* lhs, llvm::Value* rhs );
  llvm::Value* llvm_atan2_f64( llvm::Value* lhs, llvm::Value* rhs );


} // namespace QDP


#endif
