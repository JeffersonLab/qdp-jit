#include "qdp_forward_jit.h"
#include "qdp_llvm.h"
#include "qdp_config.h"
#include "qdp_params.h"

#include "qdp_llvm_includes.h"

#include "qdp_jit.h"

#include <memory>
    
namespace {
  JIT::Compiler* jit = nullptr;

  JIT::Compiler& current()
  {
    assert( jit && "llvm_* code generation outside a kernel build" );
    return *jit;
  }
}


namespace {
  std::string user_libdevice_path;
  std::string user_libdevice_name;
}


namespace {
  JIT::CompilerFactory* _factory;
}


using namespace JIT;

namespace QDP
{
  namespace JITSTATS {
    long lattice2dev  = 0;
    long lattice2host = 0;
    long jitted_gpu   = 0;
    long jitted_host  = 0;
  }
  
  void llvm_jit_stats_lattice2dev()  { ++JITSTATS::lattice2dev; }
  void llvm_jit_stats_lattice2host() { ++JITSTATS::lattice2host; }
  void llvm_jit_stats_jitted(bool gpu) { if (gpu) ++JITSTATS::jitted_gpu; else ++JITSTATS::jitted_host; }
  
  long llvm_get_jit_stats_lattice2dev()  { return JITSTATS::lattice2dev; }
  long llvm_get_jit_stats_lattice2host() { return JITSTATS::lattice2host; }
  long llvm_get_jit_stats_jitted(bool gpu) { return gpu ? JITSTATS::jitted_gpu : JITSTATS::jitted_host; }


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

  std::string llvm_get_libdevice_path() { return user_libdevice_path; }
  std::string llvm_get_libdevice_name() { return user_libdevice_name; }


  
  JIT::CompilerFactory& llvm_factory()
  {
    assert(_factory);
    return *_factory;
  }

  
  void llvm_factory_init( bool _isPrimary , int _nodeNumber)
  {
    assert(!_factory);
    _factory = new JIT::CompilerFactory( _isPrimary , _nodeNumber );

    QDPIO::cout << "Adding JIT compiler (GPU)" << endl;
    JIT::CompilerCreatorGPU* creatorGPU = new JIT::CompilerCreatorGPU();
    llvm_factory().addGPU( creatorGPU );

#if defined (QDP_HOST_COMPILER)
    if (qdp_jit_config_get_hostjit())
      {
	QDPIO::cout << "Adding JIT compiler (host)" << endl;
	JIT::CompilerCreatorHost* creatorHost = new JIT::CompilerCreatorHost();
	llvm_factory().addHost( creatorHost );
      }
#endif
  }

  
  void llvm_start_new_function( const char* ftype , const char* pretty , bool host_jit )
  {
    assert( !jit && "nested kernel build" );

    if (!host_jit)
      jit = _factory->createGPU( ftype , pretty );
    else
      jit = _factory->createHost( ftype , pretty );
  }

  
  void llvm_create_function()
  {
    current().create_function();
  }


  void llvm_set_pointer_param_attributes( ParamRef written_param , bool noalias )
  {
    current().set_pointer_param_attributes( written_param , noalias );
  }


  void llvm_build_function(JitFunction& f)
  {
    current().build_function(f);

    delete jit;
    jit = nullptr;
  }


  void llvm_module_dump()
  {
    current().module_dump();
  }

  
  llvm::Value* llvm_builder_CreateLoad( llvm::Type* ty , llvm::Value* ptr )
  {
    return current().builder_CreateLoad( ty , ptr );
  }

  void llvm_builder_CreateStore( llvm::Type* ty , llvm::Value* val , llvm::Value* ptr )
  {
    current().builder_CreateStore( ty , val , ptr );
  }

  llvm::Value* llvm_builder_CreateGEP( llvm::Type*  ty  , llvm::Value* ptr , llvm::Value* idx )
  {
    return current().builder_CreateGEP( ty  , ptr , idx );
  }

  

  template<> llvm::Type* llvm_get_type<void>()       { return current().get_type<void>(); }
  template<> llvm::Type* llvm_get_type<jit_half_t>() { return current().get_type<jit_half_t>(); }
  template<> llvm::Type* llvm_get_type<float>()      { return current().get_type<float>(); }
  template<> llvm::Type* llvm_get_type<double>()     { return current().get_type<double>(); }
  template<> llvm::Type* llvm_get_type<int>()        { return current().get_type<int>(); }
  template<> llvm::Type* llvm_get_type<bool>()       { return current().get_type<bool>(); }
  template<> llvm::Type* llvm_get_type<size_t>()     { return current().get_type<size_t>(); }

  template<> llvm::Type* llvm_get_type<jit_half_t*>() { return current().get_type<jit_half_t*>(); }
  template<> llvm::Type* llvm_get_type<float*>()      { return current().get_type<float*>(); }
  template<> llvm::Type* llvm_get_type<double*>()     { return current().get_type<double*>(); }
  template<> llvm::Type* llvm_get_type<int*>()        { return current().get_type<int*>(); }
  template<> llvm::Type* llvm_get_type<bool*>()       { return current().get_type<bool*>(); }

  
  
  llvm::Value * llvm_derefParam( ParamRef r ) {
    return current().derefParam( r );
  }


  llvm::SwitchInst * llvm_switch_create( llvm::Value* val , llvm::BasicBlock* bb_default ) 
  {
    return current().switch_create( val , bb_default );
  }

  
  void llvm_switch_add_case( llvm::SwitchInst * SI , int val , llvm::BasicBlock* bb )
  {
    current().switch_add_case( SI , val , bb );
  }


  llvm::Value * llvm_phi( llvm::Type* type, unsigned num )
  {
    return current().phi(type,num);
  }


  llvm::Value* llvm_cast( llvm::Type *dest_type , llvm::Value *src )
  {
    return current().cast( dest_type , src );
  }


  llvm::Value* llvm_neg( llvm::Value* val ) {
    return current().func_neg(val);
  }


  llvm::Value* llvm_rem( llvm::Value* lhs , llvm::Value* rhs ) {
    return current().func_rem( lhs , rhs );
  }


  llvm::Value* llvm_shr( llvm::Value* lhs , llvm::Value* rhs ) {
    return current().func_shr( lhs , rhs );
  }


  llvm::Value* llvm_shl( llvm::Value* lhs , llvm::Value* rhs ) {  
    return current().func_shl( lhs , rhs );
  }


  llvm::Value* llvm_and( llvm::Value* lhs , llvm::Value* rhs ) {
    return current().func_and( lhs , rhs );
  }


  llvm::Value* llvm_or( llvm::Value* lhs , llvm::Value* rhs ) {  
    return current().func_or( lhs , rhs );
  }


  llvm::Value* llvm_xor( llvm::Value* lhs , llvm::Value* rhs ) {  
    return current().func_xor( lhs , rhs );
  }


  llvm::Value* llvm_mul( llvm::Value* lhs , llvm::Value* rhs ) {
    return current().func_mul( lhs , rhs );
  }


  llvm::Value* llvm_add( llvm::Value* lhs , llvm::Value* rhs ) {
    return current().func_add( lhs , rhs );
  }


  llvm::Value* llvm_sub( llvm::Value* lhs , llvm::Value* rhs ) {
    return current().func_sub( lhs , rhs );
  }


  llvm::Value* llvm_div( llvm::Value* lhs , llvm::Value* rhs ) {
    return current().func_div( lhs , rhs );
  }


  llvm::Value* llvm_eq( llvm::Value* lhs , llvm::Value* rhs ) {
    return current().func_eq( lhs , rhs );
  }

  
  llvm::Value* llvm_ne( llvm::Value* lhs , llvm::Value* rhs ) {
    return current().func_ne( lhs , rhs );
  }


  llvm::Value* llvm_ge( llvm::Value* lhs , llvm::Value* rhs ) {
    return current().func_ge( lhs , rhs );
  }


  llvm::Value* llvm_gt( llvm::Value* lhs , llvm::Value* rhs ) {
    return current().func_gt( lhs , rhs );
  }


  llvm::Value* llvm_le( llvm::Value* lhs , llvm::Value* rhs ) {
    return current().func_le( lhs , rhs );
  }


  llvm::Value* llvm_lt( llvm::Value* lhs , llvm::Value* rhs ) {
    return current().func_lt( lhs , rhs );
  }


  llvm::Value* llvm_not( llvm::Value* lhs ) {
    return current().func_not( lhs );
  }


  llvm::Value* llvm_get_shared_ptr( llvm::Type *ty , int n ) {
    return current().get_shared_ptr( ty , n );
  }


  llvm::Value * llvm_alloca( llvm::Type* type , int elements )
  {
    return current().func_alloca( type , elements );
  }


  template<> ParamRef llvm_add_param<bool>() {
    return current().add_param<bool>();
  }
  template<> ParamRef llvm_add_param<bool*>() { 
    return current().add_param<bool*>();
  }
  template<> ParamRef llvm_add_param<int64_t>() { 
    return current().add_param<int64_t>();
  }
  template<> ParamRef llvm_add_param<int>() { 
    return current().add_param<int>();
  }
  template<> ParamRef llvm_add_param<int*>() { 
    return current().add_param<int*>();
  }
  template<> ParamRef llvm_add_param<jit_half_t>() { 
    return current().add_param<jit_half_t>();
  }
  template<> ParamRef llvm_add_param<float>() { 
    return current().add_param<float>();
  }
  template<> ParamRef llvm_add_param<jit_half_t*>() { 
    return current().add_param<jit_half_t*>();
  }
  template<> ParamRef llvm_add_param<float*>() { 
    return current().add_param<float*>();
  }
  template<> ParamRef llvm_add_param<double>() { 
    return current().add_param<double>();
  }
  template<> ParamRef llvm_add_param<double*>() { 
    return current().add_param<double*>();
  }

  template<> ParamRef llvm_add_param<int**>() {
    return current().add_param<int**>();
  }
  template<> ParamRef llvm_add_param<float**>() {
    return current().add_param<float**>();
  }
  template<> ParamRef llvm_add_param<double**>() {
    return current().add_param<double**>();
  }


  llvm::BasicBlock * llvm_new_basic_block()
  {
    return current().new_basic_block();
  }

  void llvm_cond_branch(llvm::Value * cond, llvm::BasicBlock * thenBB, llvm::BasicBlock * elseBB)
  {
    current().cond_branch( cond , thenBB , elseBB );
  }

  
  void llvm_loop_unroll_full_latch()
  {
    llvm::Instruction* latch = llvm_get_insert_point()->getTerminator();
    assert( latch );
    llvm::LLVMContext& C = latch->getContext();
    llvm::Metadata* full = llvm::MDNode::get( C , llvm::MDString::get( C , "llvm.loop.unroll.full" ) );
    llvm::TempMDTuple self = llvm::MDNode::getTemporary( C , {} );
    llvm::MDNode* id = llvm::MDNode::get( C , { self.get() , full } );
    id->replaceOperandWith( 0 , id );
    latch->setMetadata( llvm::LLVMContext::MD_loop , id );
  }


  int llvm_const_trip_count( llvm::Value* start , llvm::Value* end )
  {
    auto* s = llvm::dyn_cast<llvm::ConstantInt>( start );
    auto* e = llvm::dyn_cast<llvm::ConstantInt>( end );
    if ( !s || !e )
      return -1;
    return std::max<int64_t>( 0 , e->getSExtValue() - s->getSExtValue() );
  }


  void llvm_branch(llvm::BasicBlock * BB)
  {
    current().branch(BB);
  }


  void llvm_set_insert_point( llvm::BasicBlock * BB )
  {
    current().set_insert_point( BB );
  }


  llvm::BasicBlock * llvm_get_insert_point()
  {
    return current().get_insert_point();
  }


  void llvm_exit()
  {
    current().func_exit();
  }


  llvm::Type* llvm_val_type( llvm::Value* l )
  {
    return current().val_type(l);
  }


  llvm::BasicBlock * llvm_cond_exit( llvm::Value * cond )
  {
    return current().cond_exit(cond);
  }

  
  llvm::Value * llvm_create_value(double v )  {return current().create_value( v ); }
  llvm::Value * llvm_create_value(int v )     {return current().create_value( v ); }
  llvm::Value * llvm_create_value(size_t v)   {return current().create_value( v ); }
  llvm::Value * llvm_create_value(bool v )    {return current().create_value( v ); }


  void llvm_add_incoming( llvm::Value* phi , llvm::Value* val , llvm::BasicBlock* bb )
  {
    current().add_incoming( phi , val , bb );
  }
  
  void llvm_bar_sync()
  {
    current().bar_sync();
  }

  
  llvm::Value * llvm_call_special_tidx()    { return current().call_special_tidx(); }
  llvm::Value * llvm_call_special_ntidx()   { return current().call_special_ntidx(); }
  llvm::Value * llvm_call_special_ctaidx()  { return current().call_special_ctaidx(); }
  llvm::Value * llvm_call_special_nctaidx() { return current().call_special_nctaidx(); }
  llvm::Value * llvm_call_special_ctaidy()  { return current().call_special_ctaidy(); }


  llvm::Value * llvm_thread_idx()
  {
    return current().thread_idx();
  }


  llvm::Value* llvm_sin_f32( llvm::Value* lhs )      { return current().func_sin_f32( lhs ); }
  llvm::Value* llvm_acos_f32( llvm::Value* lhs )     { return current().func_acos_f32( lhs ); }
  llvm::Value* llvm_asin_f32( llvm::Value* lhs )     { return current().func_asin_f32( lhs ); }
  llvm::Value* llvm_atan_f32( llvm::Value* lhs )     { return current().func_atan_f32( lhs ); }
  llvm::Value* llvm_ceil_f32( llvm::Value* lhs )     { return current().func_ceil_f32( lhs ); }
  llvm::Value* llvm_floor_f32( llvm::Value* lhs )    { return current().func_floor_f32( lhs ); }
  llvm::Value* llvm_cos_f32( llvm::Value* lhs )      { return current().func_cos_f32( lhs ); }
  llvm::Value* llvm_cosh_f32( llvm::Value* lhs )     { return current().func_cosh_f32( lhs ); }
  llvm::Value* llvm_exp_f32( llvm::Value* lhs )      { return current().func_exp_f32( lhs ); }
  llvm::Value* llvm_log_f32( llvm::Value* lhs )      { return current().func_log_f32( lhs ); }
  llvm::Value* llvm_log10_f32( llvm::Value* lhs )    { return current().func_log10_f32( lhs ); }
  llvm::Value* llvm_sinh_f32( llvm::Value* lhs )     { return current().func_sinh_f32( lhs ); }
  llvm::Value* llvm_tan_f32( llvm::Value* lhs )      { return current().func_tan_f32( lhs ); }
  llvm::Value* llvm_tanh_f32( llvm::Value* lhs )     { return current().func_tanh_f32( lhs ); }
  llvm::Value* llvm_fabs_f32( llvm::Value* lhs )     { return current().func_fabs_f32( lhs ); }
  llvm::Value* llvm_sqrt_f32( llvm::Value* lhs )     { return current().func_sqrt_f32( lhs ); }
  llvm::Value* llvm_isfinite_f32( llvm::Value* lhs ) { return current().func_isfinite_f32( lhs ); }

  llvm::Value* llvm_pow_f32( llvm::Value* lhs, llvm::Value* rhs )   { return current().func_pow_f32( lhs , rhs ); }
  llvm::Value* llvm_atan2_f32( llvm::Value* lhs, llvm::Value* rhs ) { return current().func_atan2_f32( lhs , rhs ); }

  llvm::Value* llvm_sin_f64( llvm::Value* lhs )      { return current().func_sin_f64( lhs ); }
  llvm::Value* llvm_acos_f64( llvm::Value* lhs )     { return current().func_acos_f64( lhs ); }
  llvm::Value* llvm_asin_f64( llvm::Value* lhs )     { return current().func_asin_f64( lhs ); }
  llvm::Value* llvm_atan_f64( llvm::Value* lhs )     { return current().func_atan_f64( lhs ); }
  llvm::Value* llvm_ceil_f64( llvm::Value* lhs )     { return current().func_ceil_f64( lhs ); }
  llvm::Value* llvm_floor_f64( llvm::Value* lhs )    { return current().func_floor_f64( lhs ); }
  llvm::Value* llvm_cos_f64( llvm::Value* lhs )      { return current().func_cos_f64( lhs ); }
  llvm::Value* llvm_cosh_f64( llvm::Value* lhs )     { return current().func_cosh_f64( lhs ); }
  llvm::Value* llvm_exp_f64( llvm::Value* lhs )      { return current().func_exp_f64( lhs ); }
  llvm::Value* llvm_log_f64( llvm::Value* lhs )      { return current().func_log_f64( lhs ); }
  llvm::Value* llvm_log10_f64( llvm::Value* lhs )    { return current().func_log10_f64( lhs ); }
  llvm::Value* llvm_sinh_f64( llvm::Value* lhs )     { return current().func_sinh_f64( lhs ); }
  llvm::Value* llvm_tan_f64( llvm::Value* lhs )      { return current().func_tan_f64( lhs ); }
  llvm::Value* llvm_tanh_f64( llvm::Value* lhs )     { return current().func_tanh_f64( lhs ); }
  llvm::Value* llvm_fabs_f64( llvm::Value* lhs )     { return current().func_fabs_f64( lhs ); }
  llvm::Value* llvm_sqrt_f64( llvm::Value* lhs )     { return current().func_sqrt_f64( lhs ); }
  llvm::Value* llvm_isfinite_f64( llvm::Value* lhs ) { return current().func_isfinite_f64( lhs ); }
 
  llvm::Value* llvm_pow_f64( llvm::Value* lhs, llvm::Value* rhs )   { return current().func_pow_f64( lhs , rhs ); }
  llvm::Value* llvm_atan2_f64( llvm::Value* lhs, llvm::Value* rhs ) { return current().func_atan2_f64( lhs , rhs ); }

  
} // namespace QDP

