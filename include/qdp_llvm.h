#ifndef QDP_LLVM
#define QDP_LLVM

#include "qdp_config.h"

#include<string>
#include<vector>
#include<iostream>
#include<fstream>
#include<sstream>
#include<map>

using namespace std;

class JitFunction;
class DynKey;
class ArrayBiDirectionalMap;

#define QDP_CONST

#include "qdp_init.h"
#include "qdp_multi.h"
#include "qdp_stdio.h"
#include "qdp_jit_function.h"
#include "qdp_cache.h"
#include "qdp_gpu.h"
#include "qdp_jit_config.h"
#include "qdp_precision.h"
#include "qdp_layout.h"


//#define __STDC_LIMIT_MACROS
//#define __STDC_CONSTANT_MACROS

namespace llvm
{
  struct Value;
  struct BasicBlock;
  struct Type;
  struct PHINode;
  struct Function;
}

namespace QDP {


  namespace llvm_debug {
    extern bool debug_func_build     ;
    extern bool debug_func_dump      ;
    extern bool debug_func_write     ;
    extern bool debug_loop_vectorizer;
  }


  namespace ptx_db {
    extern bool db_enabled;
    extern std::string dbname;
  }

  void llvm_ptx_db( JitFunction& f, const char * pretty );

  
  typedef int ParamRef;


  void llvm_module_dump();

  void llvm_set_clang_codegen();
  void llvm_set_clang_opt(const char* opt);
  void llvm_set_codegen_optlevel( int i );
  
  void llvm_set_libdevice_path(const char* path);
  void llvm_set_libdevice_name(const char* name);


  void llvm_set_debug( const char * str );
  void llvm_set_ptxdb( const char * c_str );
  void llvm_debug_write_set_name( const char* pretty, const char* additional );

  std::string get_ptx_db_fname();
  bool        get_ptx_db_enabled();
  int         get_ptx_db_size();
    
  llvm::Value * llvm_create_value( double v );
  llvm::Value * llvm_create_value( int v );
  llvm::Value * llvm_create_value( int64_t v );
  llvm::Value * llvm_create_value( size_t v );
  llvm::Value * llvm_create_value( bool v );

  
  template<class T> llvm::Type* llvm_get_type();


  struct IndexRet {
    IndexRet(){}
    ParamRef p_multi_index;  // if neg. -> recv. buffer, otherwise it's the local index
    ParamRef p_recv_buf;
  };



  
  void llvm_append_mattr( const char * attr );

  llvm::Value *llvm_get_arg_lo();
  llvm::Value *llvm_get_arg_hi();
  llvm::Value *llvm_get_arg_myId();
  llvm::Value *llvm_get_arg_ordered();
  llvm::Value *llvm_get_arg_start();

  void llvm_start_new_function( const char* ftype , const char* pretty );


  void llvm_backend_init();
  llvm::PHINode * llvm_phi( llvm::Type* type, unsigned num = 0 );
  llvm::Type* promote( llvm::Type* t0 , llvm::Type* t1 );
  llvm::Value* llvm_cast( llvm::Type *dest_type , llvm::Value *src );


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


  int llvm_get_last_param_count();
  
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

  llvm::BasicBlock * llvm_get_insert_block();

  llvm::BasicBlock * llvm_new_basic_block();

  void llvm_cond_branch(llvm::Value * cond, llvm::BasicBlock * thenBB, llvm::BasicBlock * elseBB);
  void llvm_branch(llvm::BasicBlock * BB);
  void llvm_set_insert_point( llvm::BasicBlock * BB );
  llvm::BasicBlock * llvm_get_insert_point();
  void llvm_exit();
  llvm::BasicBlock * llvm_cond_exit( llvm::Value * cond );

  llvm::Value * llvm_create_value( double v );
  llvm::Value * llvm_create_value(int v );
  llvm::Value * llvm_create_value(size_t v);
  llvm::Value * llvm_create_value(bool v );


  llvm::Value * llvm_createGEP( llvm::Value * ptr , llvm::Value * idx );
  llvm::Value * llvm_load( llvm::Value * ptr );
  void          llvm_store( llvm::Value * val , llvm::Value * ptr );

  llvm::Value * llvm_load_ptr_idx( llvm::Value * ptr , llvm::Value * idx );
  void          llvm_store_ptr_idx( llvm::Value * val , llvm::Value * ptr , llvm::Value * idx );

  llvm::Value * llvm_array_type_indirection( ParamRef p        , llvm::Value* idx );
  llvm::Value * llvm_array_type_indirection( llvm::Value* base , llvm::Value* idx );

  llvm::Value * llvm_special( const char * name );

  llvm::Value * llvm_call_special_tidx();
  llvm::Value * llvm_call_special_ntidx();
  llvm::Value * llvm_call_special_ctaidx();
  llvm::Value * llvm_call_special_nctaidx();

  llvm::Value * llvm_alloca( llvm::Type* type , int elements );
  llvm::Value * llvm_get_shared_ptr( llvm::Type *ty );

  void llvm_bar_sync();

  llvm::Value * llvm_thread_idx();


  void llvm_build_function(JitFunction&);


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


  llvm::Value *jit_function_preamble_get_idx( const std::vector<ParamRef>& vec );


  class JitDefer
  {
  public:
    virtual llvm::Value* val() const = 0;
  };


  class JitDeferValue: public JitDefer
  {
    llvm::Value* r;
  public:
    JitDeferValue( llvm::Value* r ): r(r) {}
    virtual llvm::Value* val() const
    {
      return r;
    }
  };


  class JitDeferAdd: public JitDefer
  {
    llvm::Value* r;
    llvm::Value* l;
  public:
    JitDeferAdd( llvm::Value* l , llvm::Value* r ): l(l), r(r) {}
    virtual llvm::Value* val() const
    {
      return llvm_add( l , r );
    }
  };


  class JitDeferArrayTypeIndirection: public JitDefer
  {
    const ParamRef& p;
    llvm::Value* r;
  public:
    JitDeferArrayTypeIndirection( const ParamRef& p , llvm::Value* r ): p(p), r(r) {}
    virtual llvm::Value* val() const
    {
      return llvm_array_type_indirection( p , r );
    }
  };

  
  llvm::Value* jit_ternary( llvm::Value* cond , llvm::Value*    val_true , llvm::Value*    val_false );
  llvm::Value* jit_ternary( llvm::Value* cond , const JitDefer& val_true , llvm::Value*    val_false );
  llvm::Value* jit_ternary( llvm::Value* cond , llvm::Value*    val_true , const JitDefer& val_false );
  llvm::Value* jit_ternary( llvm::Value* cond , const JitDefer& val_true , const JitDefer& val_false );


  class JitForLoop
  {
  public:
    JitForLoop( int start          , int end ):           JitForLoop( llvm_create_value(start) , llvm_create_value(end) ) {}
    JitForLoop( int start          , llvm::Value*  end ): JitForLoop( llvm_create_value(start) , end ) {}
    JitForLoop( llvm::Value* start , int  end ):          JitForLoop( start , llvm_create_value(end) ) {}
    JitForLoop( llvm::Value* start , llvm::Value*  end );
    llvm::Value * index();
    void end();
  private:
    llvm::BasicBlock * block_outer;
    llvm::BasicBlock * block_loop_cond;
    llvm::BasicBlock * block_loop_body;
    llvm::BasicBlock * block_loop_exit;
    llvm::PHINode * r_i;
  };


  class JitForLoopPower
  {
  public:
    JitForLoopPower( llvm::Value* start );
    llvm::Value * index();
    void end();
  private:
    llvm::BasicBlock * block_outer;
    llvm::BasicBlock * block_loop_cond;
    llvm::BasicBlock * block_loop_body;
    llvm::BasicBlock * block_loop_exit;
    llvm::PHINode * r_i;
  };


  void jit_stats_lattice2dev();
  void jit_stats_lattice2host();
  void jit_stats_jitted();
  void jit_stats_special(int i);

  long get_jit_stats_lattice2dev();
  long get_jit_stats_lattice2host();
  long get_jit_stats_jitted();
  long get_jit_stats_special(int i);
  std::map<int,std::string>& get_jit_stats_special_names();

  std::string jit_util_get_static_dynamic_string( const std::string& pretty );

  

} // namespace QDP


#endif
