#include "qdp_config.h"
#include "qdp_params.h"

#include "qdp_llvm_includes.h"

#include <system_error>
#include <memory>
#include <unistd.h>

#include <signal.h>

#include "qdp_forward_jit.h"
#include "qdp_llvm.h"
#include "qdp_jit.h"

using namespace QDP;
using namespace llvm;

namespace JIT {

  namespace globals {
    std::unique_ptr< llvm::orc::ThreadSafeContext > TSC;
    map<string,int> map_func_counter;
  }

  
  namespace {
    CompileOptions options_from_config( const CompilerFactory& fac )
    {
      CompileOptions o;
      o.opt_level = jit_config_get_codegen_opt();
      o.verbose   = jit_config_get_verbose_output();
#if defined (QDP_BACKEND_ROCM) || defined (QDP_BACKEND_L0)
      o.keepfiles = jit_config_get_keepfiles();
#endif
#if defined (QDP_BACKEND_ROCM)
      o.prepend_path = jit_config_get_prepend_path();
      o.extra_libs   = jit_config_get_extra_lib();
      o.workgroup_bound = jit_config_get_workgroup_bound() ? jit_config_get_threads_per_block() : 0;
      o.opt_prepass     = jit_config_get_opt_prepass();
#endif
#if defined (QDP_BACKEND_CUDA)
      o.cuda_ftz = jit_config_get_CUDA_FTZ();
#endif
      o.is_primary    = fac.getIsPrimary();
      o.node          = fac.getNodeNumber();
      return o;
    }
  }


  void Compiler::setup( CompilerFactory* fac , string ftype , string pretty )
  {
    options_ = options_from_config( *fac );
    str_pretty = pretty;
    str_kernel_name = ftype + to_string( globals::map_func_counter[ftype]++ );

    function_created = false;

    target_ = &target_info();
      
    swatch_builder.reset();
    swatch_builder.start();
    
    math_declarations.clear();
    
    Mod.reset();
    
    Mod = std::make_unique<llvm::Module>("module", *get_context());

    builder.reset( new llvm::IRBuilder<>( *get_context() ) );

    if (options().verbose)
      {
	QDPIO::cout << "setting module data layout\n";
      }

    vecParamType.clear();
    vecArgument.clear();
  }
  


  llvm::Value* Compiler::trunc_i1( llvm::Value* val )
  {
    llvm::Type* Dest = llvm::Type::getInt1Ty(*get_context());
    
    return builder->CreateTrunc( val , Dest );
  }


  llvm::Value * Compiler::special( const char * name , llvm::Type* ret , std::vector<llvm::Type*> param_types , std::vector<llvm::Value*> param_values )
  {
    llvm::FunctionType *FT = llvm::FunctionType::get( ret ,
    						      llvm::ArrayRef<llvm::Type*>( param_types.data() , param_types.size() ) , 
    						      false );

    llvm::AttrBuilder ABuilder(*get_context());
    
    ABuilder.addAttribute(llvm::Attribute::Convergent);
  
    auto F = Mod->getOrInsertFunction( name , 
					      FT , 
					      llvm::AttributeList::get(*get_context(), 
								       llvm::AttributeList::FunctionIndex, 
								       ABuilder)
					      );


    if (auto FF = dyn_cast<llvm::Function>(F.getCallee()))
      {
	set_calling_conv( FF , false );
      }
    
    return builder->CreateCall(F,param_values);
  }



  
  llvm::Value* Compiler::builder_CreateLoad( llvm::Type* ty , llvm::Value* ptr )
  {
    return builder->CreateLoad( ty , ptr );
  }


  void Compiler::builder_CreateStore( llvm::Type* ty , llvm::Value* val , llvm::Value* ptr )
  {
    builder->CreateStore( val , ptr );
  }


  llvm::Value* Compiler::builder_CreateGEP( llvm::Type*  ty  , llvm::Value* ptr , llvm::Value* idx )
  {
    return builder->CreateGEP( ty , ptr , idx );
  }


  llvm::Value * Compiler::derefParam( ParamRef r )
  {
    if (!function_created)
      create_function();
    if ( vecArgument.size() <= (unsigned)r )
      raise(SIGSEGV);
    assert( vecArgument.size() > (unsigned)r && "derefParam out of range");
    return vecArgument.at(r);
  }

  
  llvm::SwitchInst* Compiler::switch_create( llvm::Value* val , llvm::BasicBlock* bb_default )
  {
    return builder->CreateSwitch( val , bb_default );
  }


  void Compiler::switch_add_case( llvm::SwitchInst * SI , int val , llvm::BasicBlock* bb )
  {
    SI->addCase( builder->getInt32(val) , bb );
  }


  llvm::Value* Compiler::phi( llvm::Type* type, unsigned num )
  {
    return builder->CreatePHI( type , num );
  }



  llvm::Type* Compiler::promote( llvm::Type* t0 , llvm::Type* t1 )
  {
    if ( t0->isFloatingPointTy() || t1->isFloatingPointTy() ) {
      if ( t0->isDoubleTy() || t1->isDoubleTy() ) {
	return llvm::Type::getDoubleTy(*get_context());
      } else {
	return llvm::Type::getFloatTy(*get_context());
      }
    } else {
      unsigned upper = std::max( t0->getScalarSizeInBits() , t1->getScalarSizeInBits() );
      return llvm::Type::getIntNTy(*get_context() , upper );
    }
  }


  std::pair<llvm::Value*,llvm::Value*> Compiler::normalize_values(llvm::Value* lhs , llvm::Value* rhs)
  {
    llvm::Value* lhs_new = lhs;
    llvm::Value* rhs_new = rhs;
    llvm::Type* args_type = promote( lhs->getType() , rhs->getType() );
    if ( args_type != lhs->getType() ) {
      lhs_new = cast( args_type , lhs );
    }
    if ( args_type != rhs->getType() ) {
      rhs_new = cast( args_type , rhs );
    }
    return std::make_pair(lhs_new,rhs_new);
  }


  
  llvm::Value* Compiler::cast( llvm::Type *dest_type , llvm::Value *src )
  {
    assert( dest_type && "llvm_cast" );
    assert( src       && "llvm_cast" );
    
    if ( src->getType() == dest_type)
      return src;

    auto i1_type = llvm::Type::getIntNTy(*get_context() , 1 );
    auto i8_type = llvm::Type::getIntNTy(*get_context() , 8 );

    if ( src->getType() == i1_type && dest_type == i8_type )
      {
	return builder->CreateZExt( src , i8_type );
      }
    
    if ( dest_type->isArrayTy() )
      if ( dest_type->getArrayElementType() == src->getType() )
	return src;

    llvm::Value* ret = builder->CreateCast( llvm::CastInst::getCastOpcode( src , true , dest_type , true ) , 
					    src , dest_type , "" );
    return ret;
  }



  
  llvm::Value* Compiler::func_neg( llvm::Value* val )
  {
    if ( val->getType()->getScalarType()->isFloatingPointTy() )
      return builder->CreateFNeg( val );
    else
      return builder->CreateNeg( val );
  }



  llvm::Value* Compiler::func_rem( llvm::Value* lhs , llvm::Value* rhs )
  {
    auto vals = normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFRem( vals.first , vals.second );
    else
      return builder->CreateSRem( vals.first , vals.second );
  }


  llvm::Value* Compiler::func_shr( llvm::Value* lhs , llvm::Value* rhs )
  {
    auto vals = normalize_values(lhs,rhs);

    assert( ! ( vals.first->getType()->getScalarType()->isFloatingPointTy() ) );
    return builder->CreateAShr( vals.first , vals.second );
  }


  llvm::Value* Compiler::func_shl( llvm::Value* lhs , llvm::Value* rhs )
  {  
    auto vals = normalize_values(lhs,rhs);

    assert( ! ( vals.first->getType()->getScalarType()->isFloatingPointTy()  ) );
    return builder->CreateShl( vals.first , vals.second );
  }


  llvm::Value* Compiler::func_and( llvm::Value* lhs , llvm::Value* rhs )
  {
    auto vals = normalize_values(lhs,rhs);

    assert( ! ( vals.first->getType()->getScalarType()->isFloatingPointTy()  ) );
    return builder->CreateAnd( vals.first , vals.second );
  }


  llvm::Value* Compiler::func_or( llvm::Value* lhs , llvm::Value* rhs )
  {  
    auto vals = normalize_values(lhs,rhs);

    assert( ! ( vals.first->getType()->getScalarType()->isFloatingPointTy()  ) );

    return builder->CreateOr( vals.first , vals.second );
  }


  llvm::Value* Compiler::func_xor( llvm::Value* lhs , llvm::Value* rhs )
  {  
    auto vals = normalize_values(lhs,rhs);

    assert( ! ( vals.first->getType()->getScalarType()->isFloatingPointTy()  ) );

    return builder->CreateXor( vals.first , vals.second );
  }


  llvm::Value* Compiler::func_mul( llvm::Value* lhs , llvm::Value* rhs )
  {
    auto vals = normalize_values(lhs,rhs);

    if ( vals.first->getType()->getScalarType()->isFloatingPointTy() )
      return builder->CreateFMul( vals.first , vals.second );
    else
      return builder->CreateMul( vals.first , vals.second );
  }


  llvm::Value* Compiler::func_add( llvm::Value* lhs , llvm::Value* rhs )
  {
    auto vals = normalize_values(lhs,rhs);

    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFAdd( vals.first , vals.second );
    else
      return builder->CreateNSWAdd( vals.first , vals.second );
  }


  llvm::Value* Compiler::func_sub( llvm::Value* lhs , llvm::Value* rhs )
  {
    auto vals = normalize_values(lhs,rhs);

    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFSub( vals.first , vals.second );
    else
      return builder->CreateSub( vals.first , vals.second );
  }


  llvm::Value* Compiler::func_div( llvm::Value* lhs , llvm::Value* rhs )
  {
    auto vals = normalize_values(lhs,rhs);

    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFDiv( vals.first , vals.second );
    else 
      return builder->CreateSDiv( vals.first , vals.second );
  }


  llvm::Value* Compiler::func_eq( llvm::Value* lhs , llvm::Value* rhs )
  {
    auto vals = normalize_values(lhs,rhs);

    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFCmpOEQ( vals.first , vals.second );
    else
      return builder->CreateICmpEQ( vals.first , vals.second );
  }

  
  llvm::Value* Compiler::func_ne( llvm::Value* lhs , llvm::Value* rhs )
  {
    auto vals = normalize_values(lhs,rhs);

    llvm::Type* args_type = vals.first->getType();
    if ( args_type->isFloatingPointTy() )
      return builder->CreateFCmpONE( vals.first , vals.second );
    else
      return builder->CreateICmpNE( vals.first , vals.second );
  }


  llvm::Value* Compiler::func_ge( llvm::Value* lhs , llvm::Value* rhs )
  {
    auto vals = normalize_values(lhs,rhs);

    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFCmpOGE( vals.first , vals.second );
    else
      return builder->CreateICmpSGE( vals.first , vals.second );
  }


  llvm::Value* Compiler::func_gt( llvm::Value* lhs , llvm::Value* rhs )
  {
    auto vals = normalize_values(lhs,rhs);

    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFCmpOGT( vals.first , vals.second );
    else
      return builder->CreateICmpSGT( vals.first , vals.second );
  }


  llvm::Value* Compiler::func_le( llvm::Value* lhs , llvm::Value* rhs )
  {
    auto vals = normalize_values(lhs,rhs);

    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFCmpOLE( vals.first , vals.second );
    else
      return builder->CreateICmpSLE( vals.first , vals.second );
  }


  llvm::Value* Compiler::func_lt( llvm::Value* lhs , llvm::Value* rhs )
  {
    auto vals = normalize_values(lhs,rhs);
    llvm::Type* args_type = vals.first->getType();
    if ( args_type->getScalarType()->isFloatingPointTy() )
      return builder->CreateFCmpOLT( vals.first , vals.second );
    else 
      return builder->CreateICmpSLT( vals.first , vals.second );
  }


  llvm::Value* Compiler::func_not( llvm::Value* lhs )
  {
    llvm::Value* tr = llvm::ConstantInt::getTrue( llvm::Type::getInt1Ty(*get_context()) );

    return func_xor( trunc_i1( lhs ) , tr );
  }



  llvm::Value * Compiler::func_alloca( llvm::Type* type , int elements )
  {
    auto it_save = builder->GetInsertPoint();
    auto bb_save = builder->GetInsertBlock();

    builder->SetInsertPoint(bb_stack, it_stack);

    auto DL = Mod->getDataLayout();
    unsigned AddrSpace = DL.getAllocaAddrSpace();

    llvm::Value* ret = builder->CreateAlloca( type , AddrSpace , create_value(elements) );    // This can be a llvm::Value*

    it_stack = builder->GetInsertPoint();
    
    builder->SetInsertPoint(bb_save,it_save);

    return ret;
  }



  llvm::BasicBlock * Compiler::new_basic_block()
  {
    std::ostringstream oss;
    oss << "L" << label_counter++;
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(*get_context(), oss.str() );
    mainFunc->insert(mainFunc->end(), BB);
    return BB;
  }


  void Compiler::cond_branch(llvm::Value * cond, llvm::BasicBlock * thenBB, llvm::BasicBlock * elseBB)
  {
    builder->CreateCondBr( trunc_i1( cond ) , thenBB, elseBB);
  }


  void Compiler::branch(llvm::BasicBlock * BB)
  {
    builder->CreateBr( BB );
  }


  void Compiler::set_insert_point( llvm::BasicBlock * BB )
  {
    builder->SetInsertPoint(BB);
  }

  llvm::BasicBlock * Compiler::get_insert_point()
  {
    return builder->GetInsertBlock();
  }


  void Compiler::func_exit()
  {
    builder->CreateRetVoid();
  }


  llvm::Type* Compiler::val_type( llvm::Value* l )
  {
    return l->getType();
  }


  llvm::BasicBlock * Compiler::cond_exit( llvm::Value * cond )
  {
    llvm::BasicBlock * thenBB = Compiler::new_basic_block();
    llvm::BasicBlock * elseBB = Compiler::new_basic_block();
    Compiler::cond_branch( cond , thenBB , elseBB );
    Compiler::set_insert_point(thenBB);
    Compiler::func_exit();
    Compiler::set_insert_point(elseBB);
    return elseBB;
  }



  llvm::Value * Compiler::create_value( double v )
  {
    if (sizeof(REAL) == 4)
      return llvm::ConstantFP::get( llvm::Type::getFloatTy(*Compiler::get_context()) , v );
    else
      return llvm::ConstantFP::get( llvm::Type::getDoubleTy(*Compiler::get_context()) , v );
  }

  llvm::Value * Compiler::create_value(int v )     {return llvm::ConstantInt::get( llvm::Type::getInt32Ty(*Compiler::get_context()) , v );}
  llvm::Value * Compiler::create_value(size_t v)   {return llvm::ConstantInt::get( llvm::Type::getInt32Ty(*Compiler::get_context()) , v );}
  llvm::Value * Compiler::create_value(bool v )    {return llvm::ConstantInt::get( llvm::Type::getInt8Ty (*Compiler::get_context()) , v );}

  

  llvm::LLVMContext* Compiler::get_context()
  {
    return globals::TSC->withContextDo([](llvm::LLVMContext* C) {
      return C;
    });
  }
  

  template<> llvm::Type* Compiler::get_type<void>()       { return llvm::Type::getVoidTy( *get_context() ); }
  template<> llvm::Type* Compiler::get_type<jit_half_t>() { return llvm::Type::getHalfTy(*get_context()); }
  template<> llvm::Type* Compiler::get_type<float>()      { return llvm::Type::getFloatTy(*get_context()); }
  template<> llvm::Type* Compiler::get_type<double>()     { return llvm::Type::getDoubleTy(*get_context()); }
  template<> llvm::Type* Compiler::get_type<int>()        { return llvm::Type::getIntNTy(*get_context(),32); }
  template<> llvm::Type* Compiler::get_type<bool>()       { return llvm::Type::getIntNTy(*get_context(),8); }
  template<> llvm::Type* Compiler::get_type<size_t>()     { return llvm::Type::getIntNTy(*get_context(),64); }

  
  template<> llvm::Type* Compiler::get_type<jit_half_t*>() { return llvm::PointerType::get(*get_context() , target().global_addrspace); }
  template<> llvm::Type* Compiler::get_type<float*>()      { return llvm::PointerType::get(*get_context() , target().global_addrspace); }
  template<> llvm::Type* Compiler::get_type<double*>()     { return llvm::PointerType::get(*get_context() , target().global_addrspace); }
  template<> llvm::Type* Compiler::get_type<int*>()        { return llvm::PointerType::get(*get_context() , target().global_addrspace); }
  template<> llvm::Type* Compiler::get_type<bool*>()       { return llvm::PointerType::get(*get_context() , target().global_addrspace); }




  llvm::Function *Compiler::get_func( std::string name , jitprec p_out , jitprec p_in , int num_args )
  {
    if (math_declarations[p_in][name] == 1)
      {
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
	    type_in = llvm::Type::getInt32Ty(*get_context());
	    break;
	  case jitprec::f32:
	    type_in = llvm::Type::getFloatTy(*get_context());
	    break;
	  case jitprec::f64:
	    type_in = llvm::Type::getDoubleTy(*get_context());
	    break;
	  }

	llvm::Type* type_out;
	switch(p_out)
	  {
	  case jitprec::i32:
	    type_out = llvm::Type::getInt32Ty(*get_context());
	    break;
	  case jitprec::f32:
	    type_out = llvm::Type::getFloatTy(*get_context());
	    break;
	  case jitprec::f64:
	    type_out = llvm::Type::getDoubleTy(*get_context());
	    break;
	  }
	
	std::vector< llvm::Type* > Args( num_args , type_in );
	llvm::FunctionType *funcType = llvm::FunctionType::get( type_out , Args , false); // no vararg
	return llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, name.c_str() , Mod.get());
      }
  }



  ParamRef Compiler::push_param( llvm::Type* type )
  {
    assert( !function_created && "kernel parameter added after the function body was started" );
    vecParamType.push_back( type );
    return vecParamType.size()-1;
  }


  template<> ParamRef Compiler::add_param<bool>() { 
    return push_param( llvm::Type::getInt8Ty( *get_context() ) );
  }
  template<> ParamRef Compiler::add_param<bool*>() { 
    return push_param( llvm::PointerType::get( *get_context() , target().global_addrspace) );
  }
  template<> ParamRef Compiler::add_param<int64_t>() { 
    return push_param( llvm::Type::getInt64Ty( *get_context() ) );
  }
  template<> ParamRef Compiler::add_param<int>() { 
    return push_param( llvm::Type::getInt32Ty( *get_context() ) );
  }
  template<> ParamRef Compiler::add_param<int*>() { 
    return push_param( llvm::PointerType::get( *get_context() , target().global_addrspace) );
  }
  template<> ParamRef Compiler::add_param<jit_half_t>() { 
    return push_param( llvm::Type::getHalfTy( *get_context() ) );
  }
  template<> ParamRef Compiler::add_param<float>() { 
    return push_param( llvm::Type::getFloatTy( *get_context() ) );
  }
  template<> ParamRef Compiler::add_param<jit_half_t*>() { 
    return push_param( llvm::PointerType::get( *get_context() , target().global_addrspace) );
  }
  template<> ParamRef Compiler::add_param<float*>() { 
    return push_param( llvm::PointerType::get( *get_context() , target().global_addrspace) );
  }
  template<> ParamRef Compiler::add_param<double>() { 
    return push_param( llvm::Type::getDoubleTy( *get_context() ) );
  }
  template<> ParamRef Compiler::add_param<double*>() { 
    return push_param( llvm::PointerType::get( *get_context() , target().global_addrspace) );
  }

  template<> ParamRef Compiler::add_param<int**>() {
    return push_param( llvm::PointerType::get( *get_context() , target().global_addrspace) );
  }
  template<> ParamRef Compiler::add_param<float**>() {
    return push_param( llvm::PointerType::get( *get_context() , target().global_addrspace) );
  }
  template<> ParamRef Compiler::add_param<double**>() {
    return push_param( llvm::PointerType::get( *get_context() , target().global_addrspace) );
  }


  void Compiler::create_llvm_kernel_function( llvm::FunctionType *funcType )
  {
    mainFunc = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, str_kernel_name , Mod.get());
  }
  

  void Compiler::set_pointer_param_attributes( int written_param , bool noalias )
  {
    if (!function_created)
      create_function();

    for ( unsigned i = 0 ; i < mainFunc->arg_size() ; ++i )
      {
	llvm::Argument* a = mainFunc->getArg(i);
	if ( !a->getType()->isPointerTy() )
	  continue;
	if ( noalias )
	  a->addAttr( llvm::Attribute::NoAlias );
	if ( (int)i != written_param )
	  a->addAttr( llvm::Attribute::ReadOnly );
      }
  }


  void Compiler::create_function()
  {
    assert( !function_created );
    assert( vecParamType.size() > 0 );

    llvm::FunctionType *funcType = 
      llvm::FunctionType::get( builder->getVoidTy() , 
			       llvm::ArrayRef<llvm::Type*>( vecParamType.data() , vecParamType.size() ) , 
			       false); // no vararg


    // Initialize mainFunc
    create_llvm_kernel_function( funcType );



    // set the calling conventions, if any
    set_calling_conv( mainFunc , true );
    
    
    unsigned Idx = 0;
    for (llvm::Function::arg_iterator AI = mainFunc->arg_begin(), AE = mainFunc->arg_end() ; AI != AE ; ++AI, ++Idx) {
      AI->setName( std::string("arg")+std::to_string(Idx) );
      vecArgument.push_back( &*AI );
    }

    bb_stack = llvm::BasicBlock::Create(*get_context(), "stack", mainFunc);
    builder->SetInsertPoint(bb_stack);
    it_stack = builder->GetInsertPoint(); // probly bb_stack.begin()
    
    bb_afterstack = llvm::BasicBlock::Create(*get_context(), "afterstack" );
    mainFunc->insert(mainFunc->end(), bb_afterstack);

    builder->SetInsertPoint(bb_afterstack);

    label_counter = 0;
    function_created = true;
  }



  void Compiler::add_incoming( llvm::Value* phi , llvm::Value* val , llvm::BasicBlock* bb )
  {
    dyn_cast<llvm::PHINode>(phi)->addIncoming( val , bb );
  }



  llvm::Value * Compiler::thread_idx()
  {
    if (!function_created)
      create_function();

    llvm::Value * tidx    = call_special_tidx();
    llvm::Value * ntidx   = call_special_ntidx();
    llvm::Value * ctaidx  = call_special_ctaidx();
    llvm::Value * ctaidy  = call_special_ctaidy();
    llvm::Value * nctaidx = call_special_nctaidx();
    return func_add( func_mul( func_add( func_mul( ctaidy , nctaidx ) , ctaidx ) , ntidx ) , tidx );
  }


  void Compiler::module_dump()
  {
    QDPIO::cout << "--------------------------  Module dump...\n";
    Mod->print(llvm::errs(), nullptr);
    QDPIO::cout << "--------------------------\n";
  }


  
  void Compiler::run_passes( llvm::TargetMachine* tm , llvm::OptimizationLevel level ,
			     const std::function<void(llvm::ModulePassManager&)>& extra )
  {
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PassBuilder PB(tm);

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(level);

    if (extra)
      extra(MPM);

    MPM.run(*Mod, MAM);
  }


  void Compiler::run_pipeline( llvm::TargetMachine* tm , const std::string& pipeline )
  {
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PassBuilder PB(tm);

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    ModulePassManager MPM;
    if (llvm::Error err = PB.parsePassPipeline( MPM , pipeline ))
      {
	QDPIO::cerr << "invalid pass pipeline '" << pipeline << "': " << llvm::toString( std::move(err) ) << std::endl;
	QDP_abort(1);
      }

    MPM.run(*Mod, MAM);
  }


  void Compiler::opt( bool target_aware )
  {
    auto level = OptimizationLevel::O2;
    
    switch (options().opt_level) {
    case 0:
      level = OptimizationLevel::O0;
      break;
    case 1:
      level = OptimizationLevel::O1;
      break;
    case 2:
      level = OptimizationLevel::O2;
      break;
    case 3:
      level = OptimizationLevel::O3;
      break;
    default:
      QDPIO::cerr << "optimization level not valid: " <<  options().opt_level << std::endl;
    }

    run_passes( target_aware ? target().machine.get() : nullptr , level );
  }


  void Compiler::internalize_and_dce()
  {
    const std::string kernel = str_kernel_name;

    run_passes( target().machine.get() , OptimizationLevel::O0 , [&kernel]( ModulePassManager& MPM ) {
	MPM.addPass( InternalizePass( [kernel]( const llvm::GlobalValue& gv ) { return gv.getName().str() == kernel; } ) );
	MPM.addPass( GlobalDCEPass() );
      });
  }


  namespace {
    std::string append_slash( std::string tmp )
    {
      if (tmp.back() != '/')
	tmp.append("/");
      return tmp;
    }
  }


  std::string Compiler::find_device_library( std::vector<std::string> paths ,
					     std::vector<std::string> appends ,
					     std::vector<std::string> names ,
					     const std::function<void(std::vector<std::string>&,std::vector<std::string>&)>& expand ,
					     std::vector<std::string>* tried )
  {
    if (!llvm_get_libdevice_path().empty())
      {
	paths   = { llvm_get_libdevice_path() };
	appends = { "" };
      }

    if (!llvm_get_libdevice_name().empty())
      {
	names = { llvm_get_libdevice_name() };
      }

    std::string arch = target().arch;
    auto index = arch.find("sm_", 0);
    if (index != std::string::npos)
      arch.replace(index, 3, "" );

    for ( std::string& name : names )
      {
	index = name.find("ARCH", 0);
	if (index != std::string::npos)
	  name.replace(index, 4, arch );
      }

    if (expand)
      expand( paths , appends );

    for ( const std::string& path : paths )
      for ( const std::string& append : appends )
	for ( const std::string& name : names )
	  {
	    std::string fname = append_slash( path ) + append + name;

	    if (tried)
	      tried->push_back( fname );

	    if (options().verbose)
	      {
		QDPIO::cout << "trying: " << fname << std::endl;
	      }

	    std::ifstream ftmp(fname.c_str());
	    if (ftmp.good())
	      {
		if (options().verbose)
		  {
		    QDPIO::cout << "libdevice found.\n";
		  }
		return fname;
	      }
	  }

    return "";
  }


  void Compiler::abort_device_library_not_found( const std::vector<std::string>& tried )
  {
    QDPIO::cerr << "libdevice not found. Tried:\n";
    for ( const std::string& fname : tried )
      QDPIO::cerr << "  " << fname << "\n";
    QDPIO::cerr << "Set its location with -libdevice-path <dir> and -libdevice-name <file>.\n";
    QDP_abort(1);
  }


  void Compiler::load( JitFunction& func , const CodeObject& obj )
  {
    StopWatch swatch(false);
    swatch.start();

    if (!get_jitf( func , obj.image , str_kernel_name , str_pretty , target().arch ))
      {
	QDPIO::cout << "Module declarations:" << std::endl;
	for ( const llvm::Function& F : *Mod )
	  if (F.isDeclaration())
	    QDPIO::cout << F.getName().str() << std::endl;
	sleep(1);
      }

    swatch.stop();
    func.time_dynload = swatch.getTimeInMicroseconds();

    if (!obj.remove_after_load.empty())
      std::remove( obj.remove_after_load.c_str() );
  }

  
  void Compiler::build_function(JitFunction& func)
  {
    builder->SetInsertPoint(bb_stack,it_stack);
    builder->CreateBr( bb_afterstack );

    swatch_builder.stop();
    func.time_builder = swatch_builder.getTimeInMicroseconds();

    func.set_num_params( vecParamType.size() );

    //llvm_module_dump();

    CodeObject obj = build_function_backend(func);

    if (func.isGPU())
      load( func , obj );

    //llvm_module_dump();
  }
  

  llvm::Value* Compiler::call_f32( llvm::Function* func , llvm::Value* lhs )
  {
    llvm::Value* lhs_f32 = cast( get_type<float>() , lhs );
    return builder->CreateCall(func,lhs_f32);
  }

  llvm::Value* Compiler::call_f32( llvm::Function* func , llvm::Value* lhs , llvm::Value* rhs )
  {
    llvm::Value* lhs_f32 = cast( get_type<float>() , lhs );
    llvm::Value* rhs_f32 = cast( get_type<float>() , rhs );
    return builder->CreateCall(func,{lhs_f32,rhs_f32});
  }

  llvm::Value* Compiler::call_f64( llvm::Function* func , llvm::Value* lhs )
  {
    llvm::Value* lhs_f64 = cast( get_type<double>() , lhs );
    return builder->CreateCall(func,lhs_f64);
  }

  llvm::Value* Compiler::call_f64( llvm::Function* func , llvm::Value* lhs , llvm::Value* rhs )
  {
    llvm::Value* lhs_f64 = cast( get_type<double>() , lhs );
    llvm::Value* rhs_f64 = cast( get_type<double>() , rhs );
    return builder->CreateCall(func,{lhs_f64,rhs_f64});
  }



  llvm::Value* Compiler::func_sin_f32( llvm::Value* lhs )      { return call_f32( get_func( target().math.at("sin_f32")      , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_acos_f32( llvm::Value* lhs )     { return call_f32( get_func( target().math.at("acos_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_asin_f32( llvm::Value* lhs )     { return call_f32( get_func( target().math.at("asin_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_atan_f32( llvm::Value* lhs )     { return call_f32( get_func( target().math.at("atan_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_ceil_f32( llvm::Value* lhs )     { return call_f32( get_func( target().math.at("ceil_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_floor_f32( llvm::Value* lhs )    { return call_f32( get_func( target().math.at("floor_f32")    , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_cos_f32( llvm::Value* lhs )      { return call_f32( get_func( target().math.at("cos_f32")      , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_cosh_f32( llvm::Value* lhs )     { return call_f32( get_func( target().math.at("cosh_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_exp_f32( llvm::Value* lhs )      { return call_f32( get_func( target().math.at("exp_f32")      , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_log_f32( llvm::Value* lhs )      { return call_f32( get_func( target().math.at("log_f32")      , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_log10_f32( llvm::Value* lhs )    { return call_f32( get_func( target().math.at("log10_f32")    , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_sinh_f32( llvm::Value* lhs )     { return call_f32( get_func( target().math.at("sinh_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_tan_f32( llvm::Value* lhs )      { return call_f32( get_func( target().math.at("tan_f32")      , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_tanh_f32( llvm::Value* lhs )     { return call_f32( get_func( target().math.at("tanh_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_fabs_f32( llvm::Value* lhs )     { return call_f32( get_func( target().math.at("fabs_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_sqrt_f32( llvm::Value* lhs )     { return call_f32( get_func( target().math.at("sqrt_f32")     , jitprec::f32 , jitprec::f32 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_isfinite_f32( llvm::Value* lhs ) { return call_f32( get_func( target().math.at("isfinite_f32") , jitprec::i32 , jitprec::f32 , 1 )  , lhs ); }

  llvm::Value* Compiler::func_pow_f32( llvm::Value* lhs, llvm::Value* rhs )   { return call_f32( get_func( target().math.at("pow_f32")   , jitprec::f32 , jitprec::f32 , 2 )  , lhs , rhs ); }
  llvm::Value* Compiler::func_atan2_f32( llvm::Value* lhs, llvm::Value* rhs ) { return call_f32( get_func( target().math.at("atan2_f32") , jitprec::f32 , jitprec::f32 , 2 )  , lhs , rhs ); }

  
  llvm::Value* Compiler::func_sin_f64( llvm::Value* lhs )      { return call_f64( get_func( target().math.at("sin_f64")      , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_acos_f64( llvm::Value* lhs )     { return call_f64( get_func( target().math.at("acos_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_asin_f64( llvm::Value* lhs )     { return call_f64( get_func( target().math.at("asin_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_atan_f64( llvm::Value* lhs )     { return call_f64( get_func( target().math.at("atan_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_ceil_f64( llvm::Value* lhs )     { return call_f64( get_func( target().math.at("ceil_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_floor_f64( llvm::Value* lhs )    { return call_f64( get_func( target().math.at("floor_f64")    , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_cos_f64( llvm::Value* lhs )      { return call_f64( get_func( target().math.at("cos_f64")      , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_cosh_f64( llvm::Value* lhs )     { return call_f64( get_func( target().math.at("cosh_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_exp_f64( llvm::Value* lhs )      { return call_f64( get_func( target().math.at("exp_f64")      , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_log_f64( llvm::Value* lhs )      { return call_f64( get_func( target().math.at("log_f64")      , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_log10_f64( llvm::Value* lhs )    { return call_f64( get_func( target().math.at("log10_f64")    , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_sinh_f64( llvm::Value* lhs )     { return call_f64( get_func( target().math.at("sinh_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_tan_f64( llvm::Value* lhs )      { return call_f64( get_func( target().math.at("tan_f64")      , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_tanh_f64( llvm::Value* lhs )     { return call_f64( get_func( target().math.at("tanh_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_fabs_f64( llvm::Value* lhs )     { return call_f64( get_func( target().math.at("fabs_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_sqrt_f64( llvm::Value* lhs )     { return call_f64( get_func( target().math.at("sqrt_f64")     , jitprec::f64 , jitprec::f64 , 1 )  , lhs ); }
  llvm::Value* Compiler::func_isfinite_f64( llvm::Value* lhs ) { return call_f64( get_func( target().math.at("isfinite_f64") , jitprec::i32 , jitprec::f64 , 1 )  , lhs ); }

  llvm::Value* Compiler::func_pow_f64( llvm::Value* lhs, llvm::Value* rhs )   { return call_f64( get_func( target().math.at("pow_f64")   , jitprec::f64 , jitprec::f64 , 2 )  , lhs , rhs ); }
  llvm::Value* Compiler::func_atan2_f64( llvm::Value* lhs, llvm::Value* rhs ) { return call_f64( get_func( target().math.at("atan2_f64") , jitprec::f64 , jitprec::f64 , 2 )  , lhs , rhs ); }

  
} // JIT
