#include "qdp.h"

namespace QDP {

  int jit_label::count = 0;

  template<class T>
  std::shared_ptr<T> get(const jit_value_t & pA) {
    return std::dynamic_pointer_cast< T >( pA );
  }


  namespace PTX {
    // { "f32","f64","u16","u32","u64","s16","s32","s64", "u8","b16","b32","b64","pred" };
    // { "f"  ,"d"  ,"h"  ,"u"  ,"w"  ,"q"  ,"i"  ,"l"  ,"s"  ,"x"  ,"y"  ,"z"  ,"p" };
    // { ""   ,""   ,"lo.","lo.","lo.","lo.","lo.","lo.","lo.",""   ,""   ,""   ,"" };

    std::map< int , std::array<const char*,4> > ptx_type_matrix = {
      {jit_ptx_type::f32 ,{{"f32" ,"f" ,""   ,"rn."}}},
      {jit_ptx_type::f64 ,{{"f64" ,"d" ,""   ,"rn."}}},
      {jit_ptx_type::u16 ,{{"u16" ,"h" ,"lo.",""}}},
      {jit_ptx_type::u32 ,{{"u32" ,"u" ,"lo.",""}}},
      {jit_ptx_type::u64 ,{{"u64" ,"w" ,"lo.",""}}},
      {jit_ptx_type::s16 ,{{"s16" ,"q" ,"lo.",""}}},
      {jit_ptx_type::s32 ,{{"s32" ,"i" ,"lo.",""}}},
      {jit_ptx_type::s64 ,{{"s64" ,"l" ,"lo.",""}}},
      {jit_ptx_type::u8  ,{{"u8"  ,"s" ,"lo.",""}}},
      {jit_ptx_type::b16 ,{{"b16" ,"x" ,""   ,""}}},
      {jit_ptx_type::b32 ,{{"b32" ,"y" ,""   ,""}}},
      {jit_ptx_type::b64 ,{{"b64" ,"z" ,""   ,""}}},
      {jit_ptx_type::pred,{{"pred","p" ,""   ,""}}} };

    const std::array<const char *,3> jit_state_space_str = { "global","local","shared" };
    const char * jit_identifier_local_memory = "loc";

    std::map< int , std::map<int,int> > create_promote()
    {
      std::map< int , std::map<int,int> > map_promote;
      map_promote[jit_ptx_type::u32][jit_ptx_type::u16] = jit_ptx_type::u32;
      map_promote[jit_ptx_type::u16][jit_ptx_type::u32] = jit_ptx_type::u32;
      map_promote[jit_ptx_type::s32][jit_ptx_type::u32] = jit_ptx_type::s32;
      map_promote[jit_ptx_type::u32][jit_ptx_type::s32] = jit_ptx_type::s32;
      map_promote[jit_ptx_type::u64][jit_ptx_type::s32] = jit_ptx_type::s64;
      map_promote[jit_ptx_type::s32][jit_ptx_type::u64] = jit_ptx_type::s64;
      map_promote[jit_ptx_type::s32][jit_ptx_type::u16] = jit_ptx_type::s32;
      map_promote[jit_ptx_type::u16][jit_ptx_type::s32] = jit_ptx_type::s32;
      map_promote[jit_ptx_type::u32][jit_ptx_type::u64] = jit_ptx_type::u64;
      map_promote[jit_ptx_type::u16][jit_ptx_type::u64] = jit_ptx_type::u64;
      map_promote[jit_ptx_type::u64][jit_ptx_type::u16] = jit_ptx_type::u64;
      map_promote[jit_ptx_type::u64][jit_ptx_type::u32] = jit_ptx_type::u64;
      map_promote[jit_ptx_type::s64][jit_ptx_type::u64] = jit_ptx_type::s64;
      map_promote[jit_ptx_type::u64][jit_ptx_type::s64] = jit_ptx_type::s64;
      map_promote[jit_ptx_type::s64][jit_ptx_type::s32] = jit_ptx_type::s64;
      map_promote[jit_ptx_type::s32][jit_ptx_type::s64] = jit_ptx_type::s64;
      return map_promote;
    }
    std::map< int , int > create_wide_promote()
    {
      std::map< int , int > map_wide_promote;
      map_wide_promote[ jit_ptx_type::f32 ] = jit_ptx_type::f64;
      map_wide_promote[ jit_ptx_type::u16 ] = jit_ptx_type::u32;
      map_wide_promote[ jit_ptx_type::u32 ] = jit_ptx_type::u64;
      map_wide_promote[ jit_ptx_type::u64 ] = jit_ptx_type::u64;
      map_wide_promote[ jit_ptx_type::s32 ] = jit_ptx_type::s64;
      map_wide_promote[ jit_ptx_type::s64 ] = jit_ptx_type::s64;
      return map_wide_promote;
    }
    const std::map< int , std::map<int,int> > map_promote      = create_promote();
    const std::map< int , int >               map_wide_promote = create_wide_promote();
  }

  int jit_number_of_types() { return PTX::ptx_type_matrix.size(); }

  const char * jit_value_reg::get_state_space_str() const { 
    assert(mem_state >= 0 && mem_state < PTX::jit_state_space_str.size());
    return PTX::jit_state_space_str[mem_state]; 
  }

  const char * jit_get_identifier_local_memory() {
    return PTX::jit_identifier_local_memory;
  }

  int jit_type_promote(int t0,int t1) {
    if (t0==t1) return t0;
    //std::cout << "promote: " << jit_get_ptx_type(t0) << " " << jit_get_ptx_type(t1) << "\n";
    assert( PTX::map_promote.count(t0) > 0 );
    assert( PTX::map_promote.at(t0).count(t1) > 0 );
    int ret = PTX::map_promote.at(t0).at(t1);
    assert((ret >= 0) && (ret < jit_number_of_types()));
    //std::cout << "         ->  " << PTX::ptx_type_matrix.at( ret )[0] << "\n";
    return ret;
  }

  int jit_type_wide_promote(int t0) {
    //std::cout << "wide_promote: " << jit_get_ptx_type(t0) << "\n";
    assert( PTX::map_wide_promote.count(t0) > 0 );
    return PTX::map_wide_promote.at(t0);
  }

  const char * jit_get_ptx_type( int type ) {
    assert((type >= 0) && (type < PTX::ptx_type_matrix.size()));
    return PTX::ptx_type_matrix.at(type)[0];
  }

  const char * jit_get_ptx_letter( int type ) {
    assert((type >= 0) && (type < PTX::ptx_type_matrix.size()));
    return PTX::ptx_type_matrix.at(type)[1];
  }

  const char * jit_get_mul_specifier_lo_str( int type ) {
    assert((type >= 0) && (type < PTX::ptx_type_matrix.size()));
    return PTX::ptx_type_matrix.at(type)[2];
  }

  const char * jit_get_div_specifier( int type ) {
    assert((type >= 0) && (type < PTX::ptx_type_matrix.size()));
    return PTX::ptx_type_matrix.at(type)[3];
  }




  // FUNCTION

  jit_function::jit_function( const char * fname_): fname(fname_), 
						    reg_count( jit_number_of_types() ), 
						    param_count(0), 
						    local_count(0)
  {
    // std::cout << "Constructing function " << fname 
    // 	      << "reg_count vector size = " << reg_count.size() << "\n";
    std::fill ( reg_count.begin() , reg_count.end() , 0 );
    
  }



  int jit_function::reg_alloc( int type ) {
    assert((type >= 0) && (type < jit_number_of_types() ));
    return reg_count[type]++;
  }

  std::ostringstream& jit_function::get_prg() { return oss_prg; }
  std::ostringstream& jit_function::get_signature() { return oss_signature; }

  int jit_function::get_param_count() {
    return param_count;
  }

  void jit_function::inc_param_count() {
    param_count++;
  }




void jit_function::write_reg_defs()
{
  int i=0;
  for (auto& x: reg_count) {
    if (x>0) {
      oss_reg_defs << ".reg ." << jit_get_ptx_type(i) << " " << jit_get_ptx_letter(i) << "<" << x << ">;\n";
    }
    i++;
  }
  i=0;
  for (auto& x: vec_local_count) {
    oss_reg_defs << ".local ." 
		 << jit_get_ptx_type( x.first ) << " " 
		 << jit_get_identifier_local_memory() << i 
		 << "[" << x.second << "];\n";
    i++;
  }
}



  void jit_function::write() 
  {
    write_reg_defs();
    std::ofstream out(fname.c_str());

    int major = DeviceParams::Instance().getMajor();
    int minor = DeviceParams::Instance().getMinor();
    
    if (major >= 2) {
      out << ".version 2.3\n";
      out << ".target sm_20" << "\n";
      out << ".address_size 64\n";
    } else {
      out << ".version 1.4\n";
      out << ".target sm_" << major << minor << "\n";
    }
    out << ".entry function (" 
	<< get_signature().str() 
	<< ")\n" 
	<< "{\n" 
	<< oss_reg_defs.str() 
	<< oss_prg.str() 
	<< "}\n";
    out.close();
  }


  jit_function_t jit_get_valid_func( jit_function_t f0 ,jit_function_t f1 ) {
    if (f0) 
      return f0;
    else
      return f1;
  }



  jit_value_t jit_add_param( jit_function_t func , int type ) {
    assert(func);
    if (func->get_param_count() > 0)
      func->get_signature() << ",\n";
    func->get_signature() << ".param ." 
			  << jit_get_ptx_type(type) 
			  << " param" 
			  << func->get_param_count();
    jit_value_t ret = jit_val_create_new( func , type );
    func->get_prg() << "ld.param." 
		    << jit_get_ptx_type(type) 
		    << " " 
		    << jit_get_reg_name(ret) 
		    << ",[" 
		    << "param" 
		    << func->get_param_count() 
		    << "];\n";
    func->inc_param_count();
    return ret;
  }


  int jit_function::local_alloc( int type, int count ) {
    assert((type >= 0) && (type < jit_number_of_types() ));
    assert(count>0);
    int ret =  vec_local_count.size();
    vec_local_count.push_back( std::make_pair(type,count) );
    return ret;
  }
  
  
  jit_value_t jit_allocate_local( jit_function_t func , int type , int count ) {
    assert(func);
    int num = func->local_alloc(type,count);
    jit_value_reg_t ret = get<jit_value_reg>(jit_val_create_new( func , jit_ptx_type::u64 ));
    func->get_prg() << "mov.u64 " 
		    << jit_get_reg_name(ret) 
		    << "," 
		    << jit_get_identifier_local_memory() << num 
		    << ";\n";
    ret->set_local_state();
    return ret;
  }



  jit_function_t jit_create_function(const char * fname_) {
    //std::cout << "Creating jit function\n";
    jit_function_t func( new jit_function(fname_) );
    return func;
  }


  jit_label_t jit_label_create( jit_function_t func ) {
    return std::make_shared< jit_label >();
  }


  // VALUE REG

  jit_value_reg::jit_value_reg(jit_function_t func_, int type_): jit_value(type_),
								 func(func_),
								 mem_state(state_global) {
    assert(func);
    number = func->reg_alloc(type);
  }
  

  void jit_value_reg::set_local_state() { 
    mem_state = state_local; 
  }


  void jit_value_reg::set_state_space( StateSpace ss ) {
    assert(ss==state_global || ss==state_local || ss==state_shared);
    mem_state = ss;
  }


  jit_value_reg::StateSpace jit_value_reg::get_state_space() {
    return mem_state;
  }




  std::string jit_value_reg::get_name() const {
    std::ostringstream tmp;
    tmp << jit_get_ptx_letter( type ) << number;
    return tmp.str();
  }


  int jit_value_reg::get_number() const { return number; }
  jit_function_t jit_value_reg::get_func() const { return func; };



  jit_value_reg::StateSpace jit_propagate_state_space( jit_value_reg::StateSpace ss0 , 
						       jit_value_reg::StateSpace ss1 ) {
    if ( ss0 == jit_value_reg::state_local || ss1 == jit_value_reg::state_local ) 
      return jit_value_reg::state_local;
    else 
      return jit_value_reg::state_global;
  }



  std::string jit_get_reg_name( jit_value_t val ) {
    assert(val);
    if (auto val_reg = get<jit_value_reg>(val) )
      return val_reg->get_name();
    assert(!"Problem");
  }

    
  jit_value_reg_t jit_val_create_new( jit_function_t func , int type ) {
    assert(func);
    //std::cout << "Creating jit value, type = " << type << "\n";
    jit_value_reg_t val(new jit_value_reg(func,type));
    return val;
  }

  jit_value_reg_t jit_val_create_from_const( jit_function_t func , int type , int val_i , jit_value_t pred) {
    assert(func);
    //std::cout << "Creating const jit value, type = " << type << "\n";
    jit_value_const_t val_const(new jit_value_const_int(val_i));
    return jit_val_create_convert( func , type , val_const , pred );
  }




  jit_value_reg_t jit_val_create_convert_const( jit_function_t func , int type , jit_value_const_t val , jit_value_t pred ) {
    jit_value_reg_t ret = jit_val_create_new( func , type );
    func->get_prg() << jit_predicate(pred)
		    << "mov." 
		    << jit_get_ptx_type( type ) << " " 
		    << ret->get_name() << ","
		    << val->getAsString() << ";\n";
    return ret;
  }
  jit_value_reg_t jit_val_create_convert_reg( jit_function_t func , int type , jit_value_reg_t val , jit_value_t pred ) {
    jit_value_reg_t ret = jit_val_create_new( func , type );
    if (type == val->get_type()) {
      func->get_prg() << jit_predicate(pred)
		      << "mov." 
		      << jit_get_ptx_type( type ) << " " 
		      << ret->get_name() << ","
		      << val->get_name() << ";\n";
    } else {
      func->get_prg() << jit_predicate(pred)
		      << "cvt." 
		      << jit_get_ptx_type( type ) << "." 
		      << jit_get_ptx_type( val->get_type() ) << " " 
		      << ret->get_name() << ","
		      << val->get_name() << ";\n";
    }
    return ret;
  }
  jit_value_reg_t jit_val_create_convert( jit_function_t func , int type , jit_value_t val , jit_value_t pred ) {
    assert(func);
    assert(val);
    if (auto val_const = get< jit_value_const >(val))
      return jit_val_create_convert_const( func , type , val_const , pred );
    if (auto val_reg = get< jit_value_reg >(val))
      return jit_val_create_convert_reg( func , type , val_reg , pred );
    assert(!"Probs");
  }

  jit_value_t jit_val_create_copy( jit_value_t val , jit_value_t pred ) {
    assert(val);
    if (auto val_const = get< jit_value_const >(val)) 
      {
	if (jit_value_const_int_t val_const_int = get<jit_value_const_int>(val_const)) 
	  {
	    return jit_val_create_const_int( val_const_int->getValue()  );
	  } 
	if (jit_value_const_float_t val_const_float = get<jit_value_const_float>(val_const)) 
	  {
	    return jit_val_create_const_float( val_const_float->getValue()  );
	  }
	assert(!"Problem");
      }
    if (auto val_reg = get< jit_value_reg >(val))
      {
	// std::cout << "TYPE reg = " << val_reg->get_type() << "\n";
	// std::cout << "TYPE     = " << val->get_type() << "\n";

	jit_value_reg_t ret = jit_val_create_convert( val_reg->get_func() , val_reg->get_type() , val_reg , pred );
	return ret;
      }
    assert(!"Problem");
  }


  jit_value_const_t jit_val_create_const_int( int val ) {
    return std::make_shared< jit_value_const_int >(val);
  }

  jit_value_const_t jit_val_create_const_float( float val ) {
    return std::make_shared< jit_value_const_float >(val);
  }



  jit_function_t getFunc(jit_value_t val) {
    auto val_reg = get< jit_value_reg >(val);
    assert(val_reg);
    return val_reg->get_func();
  }

  // Thread Geometry

  jit_value_t jit_geom_get_tidx( jit_function_t func ) {
    assert(func);
    jit_value_t tidx = jit_val_create_new( func , jit_ptx_type::u16 );
    func->get_prg() << "mov.u16 " 
		    << jit_get_reg_name( tidx ) << ",%tid.x;\n";
    return tidx;
  }
  jit_value_t jit_geom_get_ntidx( jit_function_t func ) {
    assert(func);
    jit_value_t tidx = jit_val_create_new( func , jit_ptx_type::u16 );
    func->get_prg() << "mov.u16 " 
		    << jit_get_reg_name( tidx ) << ",%ntid.x;\n";
    return tidx;
  }
  jit_value_t jit_geom_get_ctaidx( jit_function_t func ) {
    assert(func);
    jit_value_t tidx = jit_val_create_new( func , jit_ptx_type::u16 );
    func->get_prg() << "mov.u16 " 
		    << jit_get_reg_name( tidx ) << ",%ctaid.x;\n";
    return tidx;
  }




  jit_value_t jit_op_const_const( jit_value_const_t lhs , jit_value_const_t rhs , const JitOp& op ) {
    if (lhs->isInt() && rhs->isInt())  {
      jit_value_const_int_t c1 = get<jit_value_const_int>(lhs);
      jit_value_const_int_t c2 = get<jit_value_const_int>(rhs);
      return std::make_shared<jit_value_const_int>( op(c1->getValue(),c2->getValue()) );
    } else {
      return std::make_shared<jit_value_const_float>( op( lhs->getAsFloat() , rhs->getAsFloat() ) );
    }
  }
  jit_value_t jit_op_reg_const( jit_function_t func, jit_value_reg_t lhs , jit_value_const_t rhs , const JitOp& op , jit_value_t pred ) {
    jit_value_reg_t ret = jit_val_create_new( func , op.getDestType() );
    jit_value_t lhs_new = jit_val_create_convert( func , op.getArgsType() , lhs , pred );
    func->get_prg() << jit_predicate(pred)
		    << op << " "
		    << jit_get_reg_name( ret ) << ","
		    << jit_get_reg_name( lhs_new ) << ","
		    << rhs->getAsString() << ";\n";
    ret->set_state_space( lhs->get_state_space() );
    return ret;
  }
  jit_value_t jit_op_reg_reg( jit_function_t func, jit_value_reg_t lhs , jit_value_reg_t rhs , const JitOp& op , jit_value_t pred ) {
    jit_value_reg_t ret     = jit_val_create_new( func , op.getDestType() );
    jit_value_t lhs_new = jit_val_create_convert( func , op.getArgsType() , lhs , pred );
    jit_value_t rhs_new = jit_val_create_convert( func , op.getArgsType() , rhs , pred );
    func->get_prg() << jit_predicate(pred)
		    << op << " "
		    << jit_get_reg_name( ret ) << ","
		    << jit_get_reg_name( lhs_new ) << ","
		    << jit_get_reg_name( rhs_new ) << ";\n";
    ret->set_state_space( jit_propagate_state_space( lhs->get_state_space() , rhs->get_state_space() ) );
    return ret;
  }
  jit_value_t jit_ins_op( jit_value_t lhs , jit_value_t rhs , const JitOp& op , jit_value_t pred ) {
    if (auto c1 = get< jit_value_const >(lhs))
      if (auto c2 = get< jit_value_const >(rhs))
	return jit_op_const_const(c1,c2,op);
    if (auto c1 = get< jit_value_const >(lhs))
      if (auto r2 = get< jit_value_reg >(rhs))
	return jit_op_reg_const(r2->get_func(),r2,c1,op,pred);
    if (auto r1 = get< jit_value_reg >(lhs))
      if (auto c2 = get< jit_value_const >(rhs))
	return jit_op_reg_const(r1->get_func(),r1,c2,op,pred);
    if (auto r1 = get< jit_value_reg >(lhs))
      if (auto r2 = get< jit_value_reg >(rhs))
	return jit_op_reg_reg(r1->get_func(),r1,r2,op,pred);
    assert(!"Should never be here");
  }



  jit_value_t jit_ins_add( jit_value_t lhs , jit_value_t rhs , jit_value_t pred ) {
    return jit_ins_op( lhs , rhs , JitOpAdd( lhs->get_type() , rhs->get_type() ) , pred );
  }
  jit_value_t jit_ins_sub( jit_value_t lhs , jit_value_t rhs , jit_value_t pred) {
    return jit_ins_op( lhs , rhs , JitOpSub( lhs->get_type() , rhs->get_type() ) , pred );
  }
  jit_value_t jit_ins_mul( jit_value_t lhs , jit_value_t rhs , jit_value_t pred) {
    return jit_ins_op( lhs , rhs , JitOpMul( lhs->get_type() , rhs->get_type() ) , pred );
  }
  jit_value_t jit_ins_div( jit_value_t lhs , jit_value_t rhs , jit_value_t pred) {
    return jit_ins_op( lhs , rhs , JitOpDiv( lhs->get_type() , rhs->get_type() ) , pred );
  }
  jit_value_t jit_ins_mul_wide( jit_value_t lhs , jit_value_t rhs , jit_value_t pred) {
    return jit_ins_op( lhs , rhs , JitOpMulWide( lhs->get_type() , rhs->get_type() ) , pred );
  }


  jit_value_t jit_ins_lt( jit_value_t lhs , jit_value_t rhs , jit_value_t pred ) {
    return jit_ins_op( lhs , rhs , JitOpLT( lhs->get_type() , rhs->get_type() ) , pred );
  }
  jit_value_t jit_ins_ne( jit_value_t lhs , jit_value_t rhs , jit_value_t pred ) {
    return jit_ins_op( lhs , rhs , JitOpNE( lhs->get_type() , rhs->get_type() ) , pred );
  }
  jit_value_t jit_ins_eq( jit_value_t lhs , jit_value_t rhs , jit_value_t pred ) {
    return jit_ins_op( lhs , rhs , JitOpEQ( lhs->get_type() , rhs->get_type() ) , pred );
  }
  jit_value_t jit_ins_ge( jit_value_t lhs , jit_value_t rhs , jit_value_t pred ) {
    return jit_ins_op( lhs , rhs , JitOpGE( lhs->get_type() , rhs->get_type() ) , pred );
  }


  jit_value_t jit_ins_or( jit_value_t lhs , jit_value_t rhs ) { assert(!"ni"); }
  jit_value_t jit_ins_and( jit_value_t lhs , jit_value_t rhs ) { assert(!"ni"); }
  jit_value_t jit_ins_shl( jit_value_t lhs , jit_value_t rhs ) { assert(!"ni"); }
  jit_value_t jit_ins_shr( jit_value_t lhs , jit_value_t rhs ) { assert(!"ni"); }
  jit_value_t jit_ins_xor( jit_value_t lhs , jit_value_t rhs ) { assert(!"ni"); }
  jit_value_t jit_ins_mod( jit_value_t lhs , jit_value_t rhs ) { assert(!"ni"); }





  jit_value_t jit_ins_unary_op( jit_value_t rhs , const JitUnaryOp& op , jit_value_t pred ) {
    if (auto reg = get< jit_value_reg >(rhs)) {
      jit_value_reg_t ret = jit_val_create_new( reg->get_func() , reg->get_type() );
      reg->get_func()->get_prg() << jit_predicate(pred)
				 << op << " "
				 << jit_get_reg_name( ret ) << ","
				 << jit_get_reg_name( reg ) << ";\n";
      ret->set_state_space( ret->get_state_space() );
      return ret;
    }
    assert(!"Should never be here");
  }

  jit_value_t jit_ins_neg( jit_value_t rhs , jit_value_t pred ) {
    return jit_ins_unary_op( rhs , JitUnaryOpNeg( rhs->get_type() ) , pred );
  }



  void jit_ins_mov_no_create( jit_value_t dest , jit_value_t src , jit_value_t pred ){
    assert(dest);
    assert(src);
    auto dest_reg = get< jit_value_reg >(dest);
    jit_value_reg_t src_conv = jit_val_create_convert( dest_reg->get_func() , dest_reg->get_type() , src , pred );
    //auto src_reg  = get< jit_value_reg >(src);
    dest_reg->get_func()->get_prg() << jit_predicate(pred)
				    << "mov."
				    << jit_get_ptx_type( dest->get_type() ) << " "
				    << jit_get_reg_name( dest_reg ) << ","
				    << jit_get_reg_name( src_conv ) << ";\n";
  }


  jit_value_t jit_ins_load( jit_value_t base , int offset , int type , jit_value_t pred ) {
    assert(base);
    auto base_reg = get< jit_value_reg >(base);
    if (!base_reg)
      assert(!"Problem");
    jit_value_reg_t ret = jit_val_create_new( base_reg->get_func() , type );
    base_reg->get_func()->get_prg() << jit_predicate(pred)
				    << "ld." << base_reg->get_state_space_str() << "."
				    << jit_get_ptx_type( type ) << " "
				    << jit_get_reg_name( ret ) << ",["
				    << jit_get_reg_name( base_reg ) << " + "
				    << offset << "];\n";
    return ret;
  }



  void jit_ins_store( jit_value_t base , int offset , int type , jit_value_t val , jit_value_t pred ) {
    assert(base);
    assert(val);
    auto base_reg = get< jit_value_reg >(base);
    if (!base_reg)
      assert(!"Problem");
    base_reg->get_func()->get_prg() << jit_predicate(pred)
				    << "st." << base_reg->get_state_space_str() << "."
				    << jit_get_ptx_type( type ) << " ["
				    << jit_get_reg_name( base_reg ) << " + "
				    << offset << "],"
				    << jit_get_reg_name( val ) << ";\n";
  }


  jit_value_t jit_geom_get_linear_th_idx( jit_function_t func ) {
    //    std::cout << "jit_geom_get_linear_th_idx, should use wide_mul\n";
    assert(func);
    
    jit_value_t tmp = jit_ins_mul_wide( jit_geom_get_ctaidx(func) , jit_geom_get_ntidx(func) );
    assert(tmp);
    jit_value_t tmp1 = jit_ins_add( tmp , jit_geom_get_tidx(func) );
    return tmp1;
  }


  std::string jit_predicate( jit_value_t pred ) {
    if (!pred)
      return "";
    if (pred->get_type() != jit_ptx_type::pred) {
      std::cout << "not a predicate!\n";
      exit(1);
    }
    auto pred_reg = get< jit_value_reg >(pred);
    std::ostringstream oss;
    oss << "@" << jit_get_reg_name(pred_reg) << " ";
    return oss.str();
  }


  void jit_ins_label( jit_function_t func , jit_label_t& label ) {
    assert(func);
    if (!label)
      label = jit_label_create( func );
    func->get_prg() << *label << ":\n";
  }

  void jit_ins_exit( jit_function_t func , jit_value_t pred ) {
    assert(func);
    func->get_prg() << jit_predicate( pred )
		    << "exit;\n";
  }

  void jit_ins_branch( jit_function_t func , jit_label_t& label , jit_value_t pred ) {
    assert(func);
    if (!label)
      label = jit_label_create( func );
    func->get_prg() << jit_predicate( pred )
		    << "bra "
		    << *label << ";\n";
  }

  void jit_ins_comment(  jit_function_t func , const char * comment ) {
    assert(func);
    func->get_prg() << "// " << comment << "\n";
  }

}



