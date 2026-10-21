#ifndef QDP_JIT_H
#define QDP_JIT_H

#include "qdp_config.h"

#include<string>
#include<vector>
#include<iostream>
#include<fstream>
#include<sstream>
#include<map>
#include<functional>

using namespace std;



namespace QDP {
  class JitFunction;
  class DynKey;
  class ArrayBiDirectionalMap;

  //enum class JitDataOrder { SiteFast , SiteSlow };
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



// namespace llvm
// {
//   struct Value;
//   struct BasicBlock;
//   struct Type;
//   struct Function;
//   struct SwitchInst;
//   struct LLVMContext;
// }



using namespace llvm;
using namespace QDP;




namespace JIT {

  namespace globals {
    extern std::unique_ptr< llvm::orc::ThreadSafeContext > TSC;
    extern std::unique_ptr< llvm::orc::LLJIT >             TheJIT;
  }
  
  class CompilerCreatorGPU;
  class CompilerCreatorHost;
  class CompilerFactory;
  class Compiler;


  typedef int ParamRef;
  
  enum jitprec { i32 , f32 , f64 };



  // CompilerFactory initializes LLVM
  class CompilerFactory
  {
    bool isPrimary;
    int  nodeNumber;

    CompilerCreatorGPU*  creatorGPU;
    CompilerCreatorHost* creatorHost;
    
  public:
    CompilerFactory( bool _isPrimary , int _nodeNumber);
    ~CompilerFactory();

    bool getIsPrimary() const { return isPrimary; }
    int getNodeNumber() const { return nodeNumber; }
    
    void addGPU( CompilerCreatorGPU* cc )   { creatorGPU = cc; }
    void addHost( CompilerCreatorHost* cc ) { creatorHost = cc; }

    Compiler* createGPU( string ftype , string pretty );
    Compiler* createHost( string ftype , string pretty );
  };



  
    
  struct TargetInfo
  {
    llvm::Triple                          triple;
    std::unique_ptr<llvm::TargetMachine>  machine;
    std::string                           arch;
    std::map< std::string , std::string > math;
    unsigned                              global_addrspace = 0;
    unsigned                              local_addrspace  = 3;
  };


  struct CompileOptions
  {
    int                      opt_level = -1;
    bool                     verbose = false;
    bool                     keepfiles = false;
    std::string              prepend_path;
    std::vector<std::string> extra_libs;
    int                      cuda_ftz = 0;
    int                      workgroup_bound = 0;
    bool                     opt_prepass = false;
    bool                     is_primary = false;
    int                      node = 0;
  };


  struct CodeObject
  {
    std::string image;
    std::string remove_after_load;
  };


  class Compiler
  {
  protected:
    Compiler(): swatch_builder(false) {}

    const TargetInfo& target() const { return *target_; }
    const CompileOptions& options() const { return options_; }

    string str_pretty;
    string str_kernel_name;

    std::vector< llvm::Type* > vecParamType;
    std::vector< llvm::Value* > vecArgument;

    std::map< jitprec , std::map< std::string , int > > math_declarations;

    bool function_created;


    BasicBlock::iterator it_stack;
    
    llvm::Function    *mainFunc;
    llvm::BasicBlock  *bb_stack;
    llvm::BasicBlock  *bb_afterstack;

    std::unique_ptr< llvm::Module >      Mod;
    std::unique_ptr< llvm::Module >      module_libdevice;

    std::unique_ptr< llvm::IRBuilder<> > builder;


    int label_counter;

    StopWatch swatch_builder;

  private:
    const TargetInfo* target_ = nullptr;
    CompileOptions    options_;

    std::pair<llvm::Value*,llvm::Value*> normalize_values(llvm::Value* lhs , llvm::Value* rhs);
    llvm::Type*   promote( llvm::Type* t0 , llvm::Type* t1 );
    llvm::Value*  trunc_i1( llvm::Value* val );
    ParamRef      push_param( llvm::Type* type );
    llvm::Value*  call_f32( llvm::Function* func , llvm::Value* lhs );
    llvm::Value*  call_f32( llvm::Function* func , llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value*  call_f64( llvm::Function* func , llvm::Value* lhs );
    llvm::Value*  call_f64( llvm::Function* func , llvm::Value* lhs , llvm::Value* rhs );

    virtual void  addKernelMetadata(llvm::Function *F) {}
    virtual void  set_calling_conv( llvm::Function* F , bool kernel = true ) {}
    virtual CodeObject build_function_backend(JitFunction& func) = 0;
    virtual const TargetInfo& target_info() = 0;
    virtual void  create_llvm_kernel_function(llvm::FunctionType *funcType);

  protected:
    llvm::Value*  special( const char * name , llvm::Type* ret , std::vector<llvm::Type*> param_types , std::vector<llvm::Value*> param_values );
    void          opt( bool target_aware = false );
    void          run_passes( llvm::TargetMachine* tm , llvm::OptimizationLevel level ,
			      const std::function<void(llvm::ModulePassManager&)>& extra = nullptr );
    void          run_pipeline( llvm::TargetMachine* tm , const std::string& pipeline );
    void          internalize_and_dce();
    std::string   find_device_library( std::vector<std::string> paths ,
				       std::vector<std::string> appends ,
				       std::vector<std::string> names ,
				       const std::function<void(std::vector<std::string>&,std::vector<std::string>&)>& expand = nullptr ,
				       std::vector<std::string>* tried = nullptr );
    void          abort_device_library_not_found( const std::vector<std::string>& tried );

  private:
    void          load( JitFunction& func , const CodeObject& obj );

  protected:


  public:
    virtual ~Compiler() {}
    
    virtual llvm::Value* get_shared_ptr( llvm::Type *ty , int n ) = 0;
    virtual void         bar_sync() = 0;
    virtual llvm::Value * call_special_tidx() = 0;
    virtual llvm::Value * call_special_ntidx() = 0;
    virtual llvm::Value * call_special_ctaidx() = 0;
    virtual llvm::Value * call_special_nctaidx() = 0;
    virtual llvm::Value * call_special_ctaidy() = 0;
    
    void setup( CompilerFactory* fac , string ftype , string pretty );
 
    void          module_dump();

    llvm::LLVMContext* get_context();


    llvm::Function* get_func( std::string name , jitprec p_out , jitprec p_in , int num_args );

    void create_function();

    void set_pointer_param_attributes( int written_param , bool noalias );

  
    template<class T> llvm::Type* get_type();
  
    llvm::Type* val_type( llvm::Value* l );

    void start_new_function( const char* ftype , const char* pretty , bool host_jit = false );


    llvm::Value * phi( llvm::Type* type, unsigned num = 0 );
    llvm::Value* cast( llvm::Type *dest_type , llvm::Value *src );


    llvm::SwitchInst* switch_create( llvm::Value* val , llvm::BasicBlock* bb_default );
    void switch_add_case( llvm::SwitchInst * SI , int val , llvm::BasicBlock* bb );
  

    llvm::Value* func_shl( llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value* func_shr( llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value* func_rem( llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value* func_xor( llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value* func_and( llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value* func_or( llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value* func_mul( llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value* func_add( llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value* func_sub( llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value* func_div( llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value* func_eq( llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value* func_ne( llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value* func_ge( llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value* func_gt( llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value* func_le( llvm::Value* lhs , llvm::Value* rhs );
    llvm::Value* func_lt( llvm::Value* lhs , llvm::Value* rhs );

    llvm::Value* func_neg( llvm::Value* lhs );

    llvm::Value* func_not( llvm::Value* lhs );

    template<class T> ParamRef add_param();

    llvm::Value * derefParam( ParamRef r );

    llvm::BasicBlock * new_basic_block();

    void cond_branch(llvm::Value * cond, llvm::BasicBlock * thenBB, llvm::BasicBlock * elseBB);
    void branch(llvm::BasicBlock * BB);
    void set_insert_point( llvm::BasicBlock * BB );
    llvm::BasicBlock * get_insert_point();
    void func_exit();
    llvm::BasicBlock * cond_exit( llvm::Value * cond );
    
    llvm::Value * create_value(double v );
    llvm::Value * create_value(int v );
    llvm::Value * create_value(size_t v);
    llvm::Value * create_value(bool v );



    llvm::Value* builder_CreateLoad ( llvm::Type*  ty  , llvm::Value* ptr );
    void         builder_CreateStore( llvm::Type*  ty  , llvm::Value* val , llvm::Value* ptr );
    llvm::Value* builder_CreateGEP( llvm::Type*  ty  , llvm::Value* ptr , llvm::Value* idx );

  
    template<class T>
    llvm::Value * load( llvm::Value * ptr )
    {
      return builder_CreateLoad( get_type<T>() , ptr );
    }

    template<class T>
    void store( llvm::Value * val , llvm::Value * ptr )
    {
      //assert(ptr->getType()->isPointerTy() && "store: not a pointer type");
      llvm::Value * val_cast = cast( get_type<T>() , val );
      builder_CreateStore( get_type<T>() , val_cast , ptr );
    }
  

    template<class T>
    llvm::Value * load_ptr_idx ( llvm::Value * ptr , llvm::Value * idx )
    {
      return load<T>( builder_CreateGEP( get_type<T>() , ptr , idx ) );
    }

  
    template<class T>
    void store_ptr_idx( llvm::Value * val , llvm::Value * ptr , llvm::Value * idx )
    {
      store<T>( val , builder_CreateGEP( get_type<T>() , ptr , idx ) );
    }


  
    void add_incoming( llvm::Value * phi , llvm::Value* val , llvm::BasicBlock* bb );

  
    template<class T>
    llvm::Value * array_type_indirection( ParamRef p        , llvm::Value* idx )
    {
      llvm::Value* base = derefParam( p );
      //llvm::Value* gep = builder->CreateGEP( get_type<T>() , base , idx );
      llvm::Value* gep = builder_CreateGEP( get_type<T>() , base , idx );
      return load<T>( gep );
    }

    template<class T>
    llvm::Value * array_type_indirection( llvm::Value* base , llvm::Value* idx )
    {
      //llvm::Value* gep = builder->CreateGEP( get_type<T>() , base , idx );
      llvm::Value* gep = builder_CreateGEP( get_type<T>() , base , idx );
      return load<T>( gep );
    }




    
    llvm::Value * func_alloca( llvm::Type* type , int elements );
    //llvm::Value * func_alloca( llvm::Type* type , llvm::Value* elements );  // introduced PTX 7.3, as of LLVM 16: not yet implemented


    llvm::Value * thread_idx();


    void build_function(JitFunction&);


    llvm::Value* func_sin_f32( llvm::Value* lhs );
    llvm::Value* func_acos_f32( llvm::Value* lhs );
    llvm::Value* func_asin_f32( llvm::Value* lhs );
    llvm::Value* func_atan_f32( llvm::Value* lhs );
    llvm::Value* func_ceil_f32( llvm::Value* lhs );
    llvm::Value* func_floor_f32( llvm::Value* lhs );
    llvm::Value* func_cos_f32( llvm::Value* lhs );
    llvm::Value* func_cosh_f32( llvm::Value* lhs );
    llvm::Value* func_exp_f32( llvm::Value* lhs );
    llvm::Value* func_log_f32( llvm::Value* lhs );
    llvm::Value* func_log10_f32( llvm::Value* lhs );
    llvm::Value* func_sinh_f32( llvm::Value* lhs );
    llvm::Value* func_tan_f32( llvm::Value* lhs ); 
    llvm::Value* func_tanh_f32( llvm::Value* lhs );
    llvm::Value* func_fabs_f32( llvm::Value* lhs ); 
    llvm::Value* func_sqrt_f32( llvm::Value* lhs );
    llvm::Value* func_isfinite_f32( llvm::Value* lhs );

    llvm::Value* func_pow_f32( llvm::Value* lhs, llvm::Value* rhs ); 
    llvm::Value* func_atan2_f32( llvm::Value* lhs, llvm::Value* rhs );

    llvm::Value* func_sin_f64( llvm::Value* lhs );
    llvm::Value* func_acos_f64( llvm::Value* lhs );
    llvm::Value* func_asin_f64( llvm::Value* lhs );
    llvm::Value* func_atan_f64( llvm::Value* lhs );
    llvm::Value* func_ceil_f64( llvm::Value* lhs );
    llvm::Value* func_floor_f64( llvm::Value* lhs );
    llvm::Value* func_cos_f64( llvm::Value* lhs ); 
    llvm::Value* func_cosh_f64( llvm::Value* lhs );
    llvm::Value* func_exp_f64( llvm::Value* lhs ); 
    llvm::Value* func_log_f64( llvm::Value* lhs ); 
    llvm::Value* func_log10_f64( llvm::Value* lhs );
    llvm::Value* func_sinh_f64( llvm::Value* lhs ); 
    llvm::Value* func_tan_f64( llvm::Value* lhs ); 
    llvm::Value* func_tanh_f64( llvm::Value* lhs );
    llvm::Value* func_fabs_f64( llvm::Value* lhs ); 
    llvm::Value* func_sqrt_f64( llvm::Value* lhs );
    llvm::Value* func_isfinite_f64( llvm::Value* lhs );

    llvm::Value* func_pow_f64( llvm::Value* lhs, llvm::Value* rhs );
    llvm::Value* func_atan2_f64( llvm::Value* lhs, llvm::Value* rhs );


  };


  
  class CompilerCreatorGPU
  {
  public:
    Compiler* create( CompilerFactory* fac , string ftype , string pretty );
  };

  
  class CompilerCreatorHost
  {
  public:
    Compiler* create( CompilerFactory* fac , string ftype , string pretty );
  };

  

  

  class CompilerCUDA: public Compiler
  {
    std::string get_ptx();
    CodeObject build_function_backend(JitFunction& func);
    void init_libdevice();
    std::unique_ptr< llvm::Module > load_lazy( const llvm::MemoryBuffer& bc );
    const TargetInfo& target_info();
    void addKernelMetadata(llvm::Function *F);

  public:
    void setup(CompilerFactory* fac , string ftype , string pretty );
    llvm::Value* get_shared_ptr( llvm::Type *ty , int n );
    void         bar_sync();
    void set_calling_conv(llvm::Function* F , bool kernel = true )
    {
      if (kernel) F->setCallingConv( llvm::CallingConv::PTX_Kernel );
    }
    llvm::Value * call_special_tidx();
    llvm::Value * call_special_ntidx();
    llvm::Value * call_special_ctaidx();
    llvm::Value * call_special_nctaidx();
    llvm::Value * call_special_ctaidy();
  };



  class CompilerROCM: public Compiler
  {
    llvm::Value* load_dispatch_field( unsigned offset , llvm::Type* ty );

    void build_function_rocm_codegen( JitFunction& func , const std::string& shared_path , std::string& shared );
    CodeObject build_function_backend(JitFunction& func);
    const std::vector< std::pair< std::string , std::unique_ptr< llvm::MemoryBuffer > > >& device_libraries();
    void read_ocml( std::vector< std::pair< std::string , std::unique_ptr< llvm::MemoryBuffer > > >& buffers );
    std::unique_ptr< llvm::MemoryBuffer > read_libdevice();
    void link_device_libraries();
    const TargetInfo& target_info();

  public:
    void setup(CompilerFactory* fac , string ftype , string pretty );
    llvm::Value* get_shared_ptr( llvm::Type *ty , int n );
    void         bar_sync();
    void set_calling_conv(llvm::Function* F , bool kernel = true )
    {
      if (kernel) F->setCallingConv( llvm::CallingConv::AMDGPU_KERNEL );
    }
    llvm::Value * call_special_tidx();
    llvm::Value * call_special_ntidx();
    llvm::Value * call_special_ctaidx();
    llvm::Value * call_special_nctaidx();
    llvm::Value * call_special_ctaidy();
  };


  
  class CompilerL0: public Compiler
  {
    CodeObject build_function_backend(JitFunction& func);
    void build_function_levelzero_codegen(JitFunction& func, std::string& spirv);
    const TargetInfo& target_info();
  public:
    void setup(CompilerFactory* fac , string ftype , string pretty );
    void         bar_sync();
    llvm::Value* get_shared_ptr( llvm::Type *ty , int n );

    void set_calling_conv(llvm::Function* F , bool kernel = true )
    {
      if (kernel)
	F->setCallingConv( llvm::CallingConv::SPIR_KERNEL );
      else
	F->setCallingConv( llvm::CallingConv::SPIR_FUNC );
    }
    llvm::Value * call_special_tidx();
    llvm::Value * call_special_ntidx();
    llvm::Value * call_special_ctaidx();
    llvm::Value * call_special_nctaidx();
    llvm::Value * call_special_ctaidy();
      
  };


  class CompilerHost: public Compiler
  {
    CodeObject build_function_backend(JitFunction& func);
    const TargetInfo& target_info();
    void create_llvm_kernel_function(llvm::FunctionType *funcType);

  public:
    void setup(CompilerFactory* fac , string ftype , string pretty );
    void         bar_sync();
    llvm::Value* get_shared_ptr( llvm::Type *ty , int n );
    llvm::Value * call_special_tidx();
    llvm::Value * call_special_ntidx();
    llvm::Value * call_special_ctaidx();
    llvm::Value * call_special_nctaidx();
    llvm::Value * call_special_ctaidy();
  };


  
  
}

#endif
