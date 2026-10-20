#ifndef QDP_LLVM_H
#define QDP_LLVM_H

#include "qdp_config.h"

#include<string>
#include<vector>
#include<iostream>
#include<fstream>
#include<sstream>
#include<map>

using namespace std;


namespace QDP {
  class JitFunction;
  class DynKey;
  class ArrayBiDirectionalMap;
}


#include "qdp_shape.h"
#include "qdp_init.h"
#include "qdp_multi.h"
#include "qdp_stdio.h"
#include "qdp_jit_function.h"
#include "qdp_cache.h"
#include "qdp_gpu.h"
#include "qdp_jit_config.h"
#include "qdp_precision.h"
#include "qdp_stopwatch.h"



namespace llvm
{
  struct Value;
  struct BasicBlock;
  struct Type;
  struct Function;
  struct SwitchInst;
}

namespace JIT {
  class Compiler;
  class CompilerFactory;
}

namespace QDP {

  void llvm_set_libdevice_path(const char* path);
  void llvm_set_libdevice_name(const char* name);
  std::string llvm_get_libdevice_path();
  std::string llvm_get_libdevice_name();


  void llvm_factory_init( bool _isPrimary , int _nodeNumber);
  JIT::CompilerFactory& llvm_factory();

  typedef int ParamRef;

  void llvm_module_dump();
  void llvm_create_function();
  template<class T> llvm::Type* llvm_get_type();
  llvm::Type* llvm_val_type( llvm::Value* l );
  void llvm_start_new_function( const char* ftype , const char* pretty , bool host_jit = false );
  llvm::Value * llvm_phi( llvm::Type* type, unsigned num = 0 );
  llvm::Value* llvm_cast( llvm::Type *dest_type , llvm::Value *src );
  llvm::SwitchInst* llvm_switch_create( llvm::Value* val , llvm::BasicBlock* bb_default );
  void llvm_switch_add_case( llvm::SwitchInst * SI , int val , llvm::BasicBlock* bb );

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
  llvm::Value* llvm_ne( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_ge( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_gt( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_le( llvm::Value* lhs , llvm::Value* rhs );
  llvm::Value* llvm_lt( llvm::Value* lhs , llvm::Value* rhs );

  llvm::Value* llvm_neg( llvm::Value* lhs );
  llvm::Value* llvm_not( llvm::Value* lhs );

  template<class T> ParamRef llvm_add_param();
  template<> ParamRef llvm_add_param<bool>();
  template<> ParamRef llvm_add_param<bool*>();
  template<> ParamRef llvm_add_param<int64_t>();
  template<> ParamRef llvm_add_param<int>();
  template<> ParamRef llvm_add_param<int*>();
  template<> ParamRef llvm_add_param<jit_half_t>();
  template<> ParamRef llvm_add_param<jit_half_t*>();
  template<> ParamRef llvm_add_param<float>();
  template<> ParamRef llvm_add_param<float*>();
  template<> ParamRef llvm_add_param<double>();
  template<> ParamRef llvm_add_param<double*>();

  template<> ParamRef llvm_add_param<int**>();
  template<> ParamRef llvm_add_param<float**>();
  template<> ParamRef llvm_add_param<double**>();

  llvm::Value * llvm_derefParam( ParamRef r );
  llvm::BasicBlock * llvm_new_basic_block();

  void llvm_cond_branch(llvm::Value * cond, llvm::BasicBlock * thenBB, llvm::BasicBlock * elseBB);
  void llvm_branch(llvm::BasicBlock * BB);
  void llvm_loop_unroll_full_latch();
  int  llvm_const_trip_count( llvm::Value* start , llvm::Value* end );
  void llvm_set_insert_point( llvm::BasicBlock * BB );
  llvm::BasicBlock * llvm_get_insert_point();
  void llvm_exit();
  llvm::BasicBlock * llvm_cond_exit( llvm::Value * cond );

  llvm::Value * llvm_create_value( double v );
  llvm::Value * llvm_create_value(int v );
  llvm::Value * llvm_create_value(size_t v);
  llvm::Value * llvm_create_value(bool v );

  llvm::Value* llvm_builder_CreateLoad ( llvm::Type*  ty  , llvm::Value* ptr );
  void         llvm_builder_CreateStore( llvm::Type*  ty  , llvm::Value* val , llvm::Value* ptr );
  llvm::Value* llvm_builder_CreateGEP( llvm::Type*  ty  , llvm::Value* ptr , llvm::Value* idx );
  
  template<class T>
  llvm::Value * llvm_load( llvm::Value * ptr )
  {
    return llvm_builder_CreateLoad( llvm_get_type<T>() , ptr );
  }

  template<class T>
  void llvm_store( llvm::Value * val , llvm::Value * ptr )
  {
    llvm::Value * val_cast = llvm_cast( llvm_get_type<T>() , val );
    llvm_builder_CreateStore( llvm_get_type<T>() , val_cast , ptr );
  }
  

  template<class T>
  llvm::Value * llvm_load_ptr_idx ( llvm::Value * ptr , llvm::Value * idx )
  {
    return llvm_load<T>( llvm_builder_CreateGEP( llvm_get_type<T>() , ptr , idx ) );
  }

  
  template<class T>
  void llvm_store_ptr_idx( llvm::Value * val , llvm::Value * ptr , llvm::Value * idx )
  {
    llvm_store<T>( val , llvm_builder_CreateGEP( llvm_get_type<T>() , ptr , idx ) );
  }


  
  void          llvm_add_incoming( llvm::Value * phi , llvm::Value* val , llvm::BasicBlock* bb );

  
  template<class T>
  llvm::Value * llvm_array_type_indirection( ParamRef p        , llvm::Value* idx )
  {
    llvm::Value* base = llvm_derefParam( p );
    llvm::Value* gep = llvm_builder_CreateGEP( llvm_get_type<T>() , base , idx );
    return llvm_load<T>( gep );
  }

  template<class T>
  llvm::Value * llvm_array_type_indirection( llvm::Value* base , llvm::Value* idx )
  {
    llvm::Value* gep = llvm_builder_CreateGEP( llvm_get_type<T>() , base , idx );
    return llvm_load<T>( gep );
  }

  
  llvm::Value * llvm_call_special_tidx();
  llvm::Value * llvm_call_special_ntidx();
  llvm::Value * llvm_call_special_ctaidx();
  llvm::Value * llvm_call_special_nctaidx();
  llvm::Value * llvm_call_special_ctaidy();

  llvm::Value * llvm_alloca( llvm::Type* type , int elements );
  llvm::Value * llvm_get_shared_ptr( llvm::Type *ty , int n );

  void llvm_bar_sync();

  llvm::Value * llvm_thread_idx();


  void llvm_build_function(JitFunction&);

  void llvm_set_pointer_param_attributes( ParamRef written_param , bool noalias );


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


  void llvm_jit_stats_lattice2dev();
  void llvm_jit_stats_lattice2host();
  void llvm_jit_stats_jitted(bool gpu);

  long llvm_get_jit_stats_lattice2dev();
  long llvm_get_jit_stats_lattice2host();
  long llvm_get_jit_stats_jitted(bool gpu);


} // namespace QDP


#endif
