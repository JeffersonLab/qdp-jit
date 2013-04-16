#include "qdp.h"

// Unary single precision
#include "../lib/func_sin_f32.inc"
#include "../lib/func_acos_f32.inc"
#include "../lib/func_asin_f32.inc"
#include "../lib/func_atan_f32.inc"
#include "../lib/func_cos_f32.inc"
#include "../lib/func_cosh_f32.inc"
#include "../lib/func_exp_f32.inc"
#include "../lib/func_log_f32.inc"
#include "../lib/func_log10_f32.inc"
#include "../lib/func_sinh_f32.inc"
#include "../lib/func_tan_f32.inc"
#include "../lib/func_tanh_f32.inc"

#include "../lib/func_sin_f64.inc"
#include "../lib/func_acos_f64.inc"
#include "../lib/func_asin_f64.inc"
#include "../lib/func_atan_f64.inc"
#include "../lib/func_cos_f64.inc"
#include "../lib/func_cosh_f64.inc"
#include "../lib/func_exp_f64.inc"
#include "../lib/func_log_f64.inc"
#include "../lib/func_log10_f64.inc"
#include "../lib/func_sinh_f64.inc"
#include "../lib/func_tan_f64.inc"
#include "../lib/func_tanh_f64.inc"

// Binary single precision
#include "../lib/func_pow_f32.inc"
#include "../lib/func_atan2_f32.inc"
#include "../lib/func_pow_f64.inc"
#include "../lib/func_atan2_f64.inc"


namespace QDP {

  int jit_label::count = 0;

  jit_function_t jit_internal_function;

  std::ostream& operator<< (std::ostream& stream, const jit_ptx_type& type ) {
    stream << jit_get_ptx_type(type);
    return stream;
  }

  std::ostream& operator<< (std::ostream& stream, const jit_state_space& space ) {
    stream << get_state_space_str(space);
    return stream;
  }

  namespace PTX {
    // { "f32","f64","u16","u32","u64","s16","s32","s64", "u8","b16","b32","b64","pred" };
    // { "f"  ,"d"  ,"h"  ,"u"  ,"w"  ,"q"  ,"i"  ,"l"  ,"s"  ,"x"  ,"y"  ,"z"  ,"p" };
    // { ""   ,""   ,"lo.","lo.","lo.","lo.","lo.","lo.","lo.",""   ,""   ,""   ,"" };

    std::map< int , std::pair<const char *,std::string> > create_ptx_math_functions_binary()
    {
      std::map< int , std::pair<const char *,std::string> > map_ptx_math_functions_binary;
      map_ptx_math_functions_binary[0] = 
	std::make_pair("func_pow_f32",std::string(  (const char *)func_pow_f32_ptx  , func_pow_f32_ptx_len ));
      map_ptx_math_functions_binary[1] = 
	std::make_pair("func_atan2_f32",std::string(  (const char *)func_atan2_f32_ptx  , func_atan2_f32_ptx_len ));
      map_ptx_math_functions_binary[2] = 
	std::make_pair("func_pow_f64",std::string(  (const char *)func_pow_f64_ptx  , func_pow_f64_ptx_len ));
      map_ptx_math_functions_binary[3] = 
	std::make_pair("func_atan2_f64",std::string(  (const char *)func_atan2_f64_ptx  , func_atan2_f64_ptx_len ));
      return map_ptx_math_functions_binary;
    }

    std::map< int , std::pair<const char *,std::string> > create_ptx_math_functions_unary()
    {
      std::map< int , std::pair<const char *,std::string> > map_ptx_math_functions_unary;
      map_ptx_math_functions_unary[0] = 
	std::make_pair("func_sin_f32",std::string(  (const char *)func_sin_f32_ptx  , func_sin_f32_ptx_len ));
      map_ptx_math_functions_unary[1] = 
	std::make_pair("func_acos_f32",std::string( (const char *)func_acos_f32_ptx , func_acos_f32_ptx_len ));
      map_ptx_math_functions_unary[2] = 
	std::make_pair("func_asin_f32",std::string( (const char *)func_asin_f32_ptx , func_asin_f32_ptx_len ));
      map_ptx_math_functions_unary[3] = 
	std::make_pair("func_atan_f32",std::string( (const char *)func_atan_f32_ptx , func_atan_f32_ptx_len ));
      map_ptx_math_functions_unary[4] = 
	std::make_pair("func_cos_f32",std::string( (const char *)func_cos_f32_ptx , func_cos_f32_ptx_len ));
      map_ptx_math_functions_unary[5] = 
	std::make_pair("func_cosh_f32",std::string( (const char *)func_cosh_f32_ptx , func_cosh_f32_ptx_len ));
      map_ptx_math_functions_unary[6] = 
	std::make_pair("func_exp_f32",std::string( (const char *)func_exp_f32_ptx , func_exp_f32_ptx_len ));
      map_ptx_math_functions_unary[7] = 
	std::make_pair("func_log_f32",std::string( (const char *)func_log_f32_ptx , func_log_f32_ptx_len ));
      map_ptx_math_functions_unary[8] = 
	std::make_pair("func_log10_f32",std::string( (const char *)func_log10_f32_ptx , func_log10_f32_ptx_len ));
      map_ptx_math_functions_unary[9] = 
	std::make_pair("func_sinh_f32",std::string( (const char *)func_sinh_f32_ptx , func_sinh_f32_ptx_len ));
      map_ptx_math_functions_unary[10] = 
	std::make_pair("func_tan_f32",std::string( (const char *)func_tan_f32_ptx , func_tan_f32_ptx_len ));
      map_ptx_math_functions_unary[11] = 
	std::make_pair("func_tanh_f32",std::string( (const char *)func_tanh_f32_ptx , func_tanh_f32_ptx_len ));

      map_ptx_math_functions_unary[12] = 
	std::make_pair("func_sin_f64",std::string(  (const char *)func_sin_f64_ptx  , func_sin_f64_ptx_len ));
      map_ptx_math_functions_unary[13] = 
	std::make_pair("func_acos_f64",std::string( (const char *)func_acos_f64_ptx , func_acos_f64_ptx_len ));
      map_ptx_math_functions_unary[14] = 
	std::make_pair("func_asin_f64",std::string( (const char *)func_asin_f64_ptx , func_asin_f64_ptx_len ));
      map_ptx_math_functions_unary[15] = 
	std::make_pair("func_atan_f64",std::string( (const char *)func_atan_f64_ptx , func_atan_f64_ptx_len ));
      map_ptx_math_functions_unary[16] = 
       	std::make_pair("func_cos_f64",std::string( (const char *)func_cos_f64_ptx , func_cos_f64_ptx_len ));
      map_ptx_math_functions_unary[17] = 
       	std::make_pair("func_cosh_f64",std::string( (const char *)func_cosh_f64_ptx , func_cosh_f64_ptx_len ));
      map_ptx_math_functions_unary[18] = 
       	std::make_pair("func_exp_f64",std::string( (const char *)func_exp_f64_ptx , func_exp_f64_ptx_len ));
      map_ptx_math_functions_unary[19] = 
       	std::make_pair("func_log_f64",std::string( (const char *)func_log_f64_ptx , func_log_f64_ptx_len ));
      map_ptx_math_functions_unary[20] = 
       	std::make_pair("func_log10_f64",std::string( (const char *)func_log10_f64_ptx , func_log10_f64_ptx_len ));
      map_ptx_math_functions_unary[21] = 
       	std::make_pair("func_sinh_f64",std::string( (const char *)func_sinh_f64_ptx , func_sinh_f64_ptx_len ));
      map_ptx_math_functions_unary[22] = 
       	std::make_pair("func_tan_f64",std::string( (const char *)func_tan_f64_ptx , func_tan_f64_ptx_len ));
      map_ptx_math_functions_unary[23] = 
       	std::make_pair("func_tanh_f64",std::string( (const char *)func_tanh_f64_ptx , func_tanh_f64_ptx_len ));

      return map_ptx_math_functions_unary;
    }

    // PTX_type , reg_name_prefix, mul_lo_specifier , div_lo_specifier
    std::map< jit_ptx_type , std::array<const char*,4> > create_ptx_type_matrix(int cc) {
      if (cc >= 20) {
	QDP_info_primary("Using ptx_type_matrix for sm_20 or higher");
	std::map< jit_ptx_type , std::array<const char*,4> > ptx_type_matrix {
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
	return ptx_type_matrix;
      } else {
	QDP_info_primary("Using ptx_type_matrix for sm_1x");
	std::map< jit_ptx_type , std::array<const char*,4> > ptx_type_matrix {
	  {jit_ptx_type::f32 ,{{"f32" ,"f" ,""   ,"full."}}},
	    {jit_ptx_type::f64 ,{{"f64" ,"d" ,""   ,"full."}}},
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
	return ptx_type_matrix;
      }
    }

    std::map< jit_ptx_type , std::array<const char*,4> > ptx_type_matrix;
	
    const char * jit_identifier_local_memory = "loc";

    std::map< jit_state_space , const char * > create_state_space_map()
    {
      std::map< jit_state_space , const char * > map_state_space_map;
      map_state_space_map[ jit_state_space::state_default] = "default";
      map_state_space_map[ jit_state_space::state_global]  = "global";
      map_state_space_map[ jit_state_space::state_local]   = "local";
      map_state_space_map[ jit_state_space::state_shared]  = "shared";
      return map_state_space_map;
    }

    std::map< jit_ptx_type , std::map<jit_ptx_type,const char *> > create_cvt_rnd_from_to()
    {
      std::map< jit_ptx_type , std::map<jit_ptx_type,const char *> > map_cvt_rnd_from_to;
      map_cvt_rnd_from_to[jit_ptx_type::s32][jit_ptx_type::f32] = "rn.";
      map_cvt_rnd_from_to[jit_ptx_type::u32][jit_ptx_type::f32] = "rn.";
      map_cvt_rnd_from_to[jit_ptx_type::s32][jit_ptx_type::f64] = "rn.";
      map_cvt_rnd_from_to[jit_ptx_type::u32][jit_ptx_type::f64] = "rn.";
      map_cvt_rnd_from_to[jit_ptx_type::s64][jit_ptx_type::f32] = "rn.";
      map_cvt_rnd_from_to[jit_ptx_type::u64][jit_ptx_type::f32] = "rn.";
      map_cvt_rnd_from_to[jit_ptx_type::s64][jit_ptx_type::f64] = "rn.";
      map_cvt_rnd_from_to[jit_ptx_type::u64][jit_ptx_type::f64] = "rn.";
      map_cvt_rnd_from_to[jit_ptx_type::f64][jit_ptx_type::f32] = "rn.";
      return map_cvt_rnd_from_to;
    }

    std::map< jit_ptx_type , std::map<jit_ptx_type,jit_ptx_type> > create_promote()
    {
      std::map< jit_ptx_type , std::map<jit_ptx_type,jit_ptx_type> > map_promote;
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
      map_promote[jit_ptx_type::s32][jit_ptx_type::u8] = jit_ptx_type::s32;
      map_promote[jit_ptx_type::u8][jit_ptx_type::s32] = jit_ptx_type::s32;
      map_promote[jit_ptx_type::f64][jit_ptx_type::f32] = jit_ptx_type::f64;
      map_promote[jit_ptx_type::f32][jit_ptx_type::f64] = jit_ptx_type::f64;
      map_promote[jit_ptx_type::f64][jit_ptx_type::s32] = jit_ptx_type::f64;
      map_promote[jit_ptx_type::s32][jit_ptx_type::f64] = jit_ptx_type::f64;
      map_promote[jit_ptx_type::f32][jit_ptx_type::s32] = jit_ptx_type::f32;
      map_promote[jit_ptx_type::s32][jit_ptx_type::f32] = jit_ptx_type::f32;
      return map_promote;
    }
    std::map< jit_ptx_type , jit_ptx_type > create_wide_promote()
    {
      std::map< jit_ptx_type , jit_ptx_type > map_wide_promote;
      map_wide_promote[ jit_ptx_type::f32 ] = jit_ptx_type::f64;
      map_wide_promote[ jit_ptx_type::u16 ] = jit_ptx_type::u32;
      map_wide_promote[ jit_ptx_type::u32 ] = jit_ptx_type::u64;
      map_wide_promote[ jit_ptx_type::u64 ] = jit_ptx_type::u64;
      map_wide_promote[ jit_ptx_type::s32 ] = jit_ptx_type::s64;
      map_wide_promote[ jit_ptx_type::s64 ] = jit_ptx_type::s64;
      return map_wide_promote;
    }
    std::map< jit_ptx_type , jit_ptx_type > create_bit_type()
    {
      std::map< jit_ptx_type , jit_ptx_type > map_bit_type;
      map_bit_type[ jit_ptx_type::u32 ] = jit_ptx_type::b32;
      map_bit_type[ jit_ptx_type::s32 ] = jit_ptx_type::b32;
      map_bit_type[ jit_ptx_type::f32 ] = jit_ptx_type::b32;
      map_bit_type[ jit_ptx_type::u64 ] = jit_ptx_type::b64;
      map_bit_type[ jit_ptx_type::s64 ] = jit_ptx_type::b64;
      map_bit_type[ jit_ptx_type::f64 ] = jit_ptx_type::b64;
      map_bit_type[ jit_ptx_type::u16 ] = jit_ptx_type::b16;
      map_bit_type[ jit_ptx_type::s16 ] = jit_ptx_type::b16;
      map_bit_type[ jit_ptx_type::pred ] = jit_ptx_type::pred;
      return map_bit_type;
    }
    std::map< jit_state_space , 
	      std::map< jit_state_space , 
			jit_state_space > >
    create_state_promote()
    {
      std::map< jit_state_space , 
		std::map< jit_state_space , 
			  jit_state_space > > map_state_promote;
      map_state_promote[ jit_state_space::state_default ][ jit_state_space::state_shared ] = jit_state_space::state_shared;
      map_state_promote[ jit_state_space::state_shared ][ jit_state_space::state_default ] = jit_state_space::state_shared;
      map_state_promote[ jit_state_space::state_default ][ jit_state_space::state_global ] = jit_state_space::state_global;
      map_state_promote[ jit_state_space::state_global ][ jit_state_space::state_default ] = jit_state_space::state_global;
      map_state_promote[ jit_state_space::state_default ][ jit_state_space::state_local ] = jit_state_space::state_local;
      map_state_promote[ jit_state_space::state_local ][ jit_state_space::state_default ] = jit_state_space::state_local;

      map_state_promote[ jit_state_space::state_shared ][ jit_state_space::state_shared ] = jit_state_space::state_shared;
      map_state_promote[ jit_state_space::state_shared ][ jit_state_space::state_global ] = jit_state_space::state_shared;
      map_state_promote[ jit_state_space::state_global ][ jit_state_space::state_shared ] = jit_state_space::state_shared;
      map_state_promote[ jit_state_space::state_local  ][ jit_state_space::state_local  ] = jit_state_space::state_local;
      map_state_promote[ jit_state_space::state_local  ][ jit_state_space::state_global ] = jit_state_space::state_local;
      map_state_promote[ jit_state_space::state_global ][ jit_state_space::state_local  ] = jit_state_space::state_local;
      map_state_promote[ jit_state_space::state_global ][ jit_state_space::state_global ] = jit_state_space::state_global;
      return map_state_promote;
    }
    const std::map< jit_state_space , const char * > map_state_space = create_state_space_map();
    const std::map< int , std::pair<const char *,
				    std::string> >     map_ptx_math_functions_unary = create_ptx_math_functions_unary();
    const std::map< int , std::pair<const char *,
				    std::string> >     map_ptx_math_functions_binary = create_ptx_math_functions_binary();
    const std::map< jit_ptx_type , std::map<jit_ptx_type,const char *> > map_cvt_rnd_from_to    = create_cvt_rnd_from_to();
    const std::map< jit_ptx_type , std::map<jit_ptx_type,jit_ptx_type> >          map_promote            = create_promote();
    const std::map< jit_ptx_type , jit_ptx_type >                        map_wide_promote       = create_wide_promote();
    const std::map< jit_ptx_type , jit_ptx_type >                        map_bit_type           = create_bit_type();
    const std::map< jit_state_space , 
		    std::map< jit_state_space , 
			      jit_state_space > > map_state_promote  = create_state_promote();
  }


  int jit_number_of_types() { return PTX::ptx_type_matrix.size(); }

  const char * jit_get_map_ptx_math_functions_funcname_unary(int i) {
    if (!PTX::map_ptx_math_functions_unary.count(i))
      QDP_error_exit("PTX math function_unary: Out of range %d",i);
    return PTX::map_ptx_math_functions_unary.at(i).first;
  }

  const std::string& jit_get_map_ptx_math_functions_prg_unary(int i) {
    if (!PTX::map_ptx_math_functions_unary.count(i))
      QDP_error_exit("PTX math function_unary: Out of range %d",i);
    return PTX::map_ptx_math_functions_unary.at(i).second;
  }

  const char * jit_get_map_ptx_math_functions_funcname_binary(int i) {
    if (!PTX::map_ptx_math_functions_binary.count(i))
      QDP_error_exit("PTX math function_binary: Out of range %d",i);
    return PTX::map_ptx_math_functions_binary.at(i).first;
  }

  const std::string& jit_get_map_ptx_math_functions_prg_binary(int i) {
    if (!PTX::map_ptx_math_functions_binary.count(i))
      QDP_error_exit("PTX math function_binary: Out of range %d",i);
    return PTX::map_ptx_math_functions_binary.at(i).second;
  }


  const char * get_state_space_str( jit_state_space mem_state ){ 
    assert( PTX::map_state_space.count(mem_state) > 0 );
    return PTX::map_state_space.at(mem_state); 
  }


  const char * jit_get_identifier_local_memory() {
    return PTX::jit_identifier_local_memory;
  }

  const char * jit_get_map_cvt_rnd_from_to(jit_ptx_type from,jit_ptx_type to) {
    static const char * nullstr = "";
    if (!PTX::map_cvt_rnd_from_to.count(from))
      return nullstr;
    if (!PTX::map_cvt_rnd_from_to.at(from).count(to))
      return nullstr;
    return PTX::map_cvt_rnd_from_to.at(from).at(to);
  }


  jit_ptx_type jit_bit_type(jit_ptx_type type) {
    assert( PTX::map_bit_type.count( type ) > 0 );
    return PTX::map_bit_type.at( type );
  }


  jit_ptx_type jit_type_promote(jit_ptx_type t0,jit_ptx_type t1) {
    //std::cout << "type promote: " << t0 << " " << t1 << "\n";
    if (t0==t1) return t0;
    if ( PTX::map_promote.count(t0) == 0 )
      std::cout << "promote: " << jit_get_ptx_type(t0) << " " << jit_get_ptx_type(t1) << "\n";
    assert( PTX::map_promote.count(t0) > 0 );
    if ( PTX::map_promote.at(t0).count(t1) == 0 )
      std::cout << "promote: " << jit_get_ptx_type(t0) << " " << jit_get_ptx_type(t1) << "\n";
    assert( PTX::map_promote.at(t0).count(t1) > 0 );
    jit_ptx_type ret = PTX::map_promote.at(t0).at(t1);
    //assert((ret >= 0) && (ret < jit_number_of_types()));
    //std::cout << "         ->  " << PTX::ptx_type_matrix.at( ret )[0] << "\n";
    return ret;
  }


  jit_value::jit_value( int val ): 
    ever_assigned(true), 
    mem_state(jit_state_space::state_default)
  {
    type = val < 0 ? jit_ptx_type::s32 : jit_ptx_type::u32;
    reg_alloc();
    std::ostringstream oss; oss << val;
    jit_ins_mov( *this , oss.str() );
  }


  jit_value::jit_value( size_t val ): 
    ever_assigned(true), 
    mem_state(jit_state_space::state_default)
  {
    type = val <= (size_t)std::numeric_limits<int32_t>::max() ? jit_ptx_type::u32 : jit_ptx_type::u64;
    reg_alloc();
    std::ostringstream oss; oss << val;
    jit_ins_mov( *this , oss.str() );
  }


  jit_value::jit_value( double val ): 
    ever_assigned(true), 
    mem_state(jit_state_space::state_default),
    type(jit_type<REAL>::value)
  {
    reg_alloc();
    std::ostringstream oss; 
    oss.setf(ios::scientific);
    oss.precision(std::numeric_limits<double>::digits10 + 1);
    oss << val;
    jit_ins_mov( *this , oss.str() );
  }

  jit_state_space jit_state_promote( jit_state_space ss0 , jit_state_space ss1 ) {
    //std::cout << "state_promote: " << ss0 << " " << ss1 << "\n";
    if ( ss0 == ss1 ) return ss0;
    assert( PTX::map_state_promote.count( ss0 ) > 0 );
    assert( PTX::map_state_promote.at( ss0 ).count( ss1 ) > 0 );
    jit_state_space ret = PTX::map_state_promote.at( ss0 ).at( ss1 );
    assert( ret == jit_state_space::state_global || ret == jit_state_space::state_shared || ret == jit_state_space::state_local );
    //std::cout << "         ->  " << PTX::ptx_type_matrix.at( ret )[0] << "\n";
    return ret;
  }


  jit_ptx_type jit_type_wide_promote(jit_ptx_type t0) {
    assert( PTX::map_wide_promote.count(t0) > 0 );
    return PTX::map_wide_promote.at(t0);
  }

  const char * jit_get_ptx_type( jit_ptx_type type ) {
    assert( PTX::ptx_type_matrix.count(type) > 0 );
    return PTX::ptx_type_matrix.at(type)[0];
  }

  const char * jit_get_ptx_letter( jit_ptx_type type ) {
    assert( PTX::ptx_type_matrix.count(type) > 0 );
    return  PTX::ptx_type_matrix.at(type)[1];
  }

  const char * jit_get_mul_specifier_lo_str( jit_ptx_type type ) {
    assert( PTX::ptx_type_matrix.count(type) > 0 );
    return PTX::ptx_type_matrix.at(type)[2];
  }

  const char * jit_get_div_specifier( jit_ptx_type type ) {
    assert( PTX::ptx_type_matrix.count(type) > 0 );
    return PTX::ptx_type_matrix.at(type)[3];
  }




  // FUNCTION

  jit_function::jit_function(): param_count(0), 
				local_count(0),
				m_shared(false),
				m_include_math_ptx_unary(PTX::map_ptx_math_functions_unary.size(),false),
				m_include_math_ptx_binary(PTX::map_ptx_math_functions_binary.size(),false)
  {}


  void jit_function::emitShared() {
    m_shared=true;
  }


  int jit_function::reg_alloc( jit_ptx_type type ) {
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
    for( RegCountMap::const_iterator it = reg_count.begin(); it != reg_count.end(); ++it )
      {
	jit_ptx_type type = it->first;
	int count = it->second;
	oss_reg_defs << ".reg ." 
		     << jit_get_ptx_type(type) 
		     << " " 
		     << jit_get_ptx_letter(type) 
		     << "<" 
		     << count
		     << ">;\n";
      }

    // for (auto& x: ) {
    //   if (x>0) {
    //     oss_reg_defs << ".reg ." << jit_get_ptx_type(x.first) << " " << jit_get_ptx_letter(x.first) << "<" << x << ">;\n";
    //     //oss_reg_defs << ".reg ." << jit_get_ptx_type(i) << " " << jit_get_ptx_letter(i) << "<" << x << ">;\n";
    //   }
    //   i++;
    // }

    for( int i = 0 ; i < vec_local_count.size() ; ++i ) 
      {
	jit_ptx_type type = vec_local_count.at(i).first;
	int count = vec_local_count.at(i).second;
	oss_reg_defs << ".local ." 
		     << jit_get_ptx_type( type ) << " " 
		     << jit_get_identifier_local_memory() << i 
		     << "[" << count << "];\n";
      }
  }


  std::string jit_function::get_kernel_as_string()
  {
    std::ostringstream final_ptx;
    write_reg_defs();

    int major = DeviceParams::Instance().getMajor();
    int minor = DeviceParams::Instance().getMinor();
    
    if (major >= 2) {
      final_ptx << ".version 3.1\n";
      final_ptx << ".target sm_" << major << minor << "\n";
      final_ptx << ".address_size 64\n";
    } else {
      final_ptx << ".version 1.4\n";
      final_ptx << ".target sm_" << major << minor << "\n";
    }

    if (m_shared)
      final_ptx << ".extern .shared .align 4 .b8 sdata[];\n";

    for( int i=0 ; i < PTX::map_ptx_math_functions_unary.size() ; i++ ) {
      if (m_include_math_ptx_unary.at(i)) {
	QDP_info_primary("including unary PTX math function %d",(int)i);
	final_ptx << jit_get_map_ptx_math_functions_prg_unary(i) << "\n";
      }
    }
    for( int i=0 ; i < PTX::map_ptx_math_functions_binary.size() ; i++ ) {
      if (m_include_math_ptx_binary.at(i)) {
	QDP_info_primary("including binary PTX math function %i",(int)i);
	final_ptx << jit_get_map_ptx_math_functions_prg_binary(i) << "\n";
      }
    }

    final_ptx << ".entry function (" 
	<< get_signature().str() 
	<< ")\n" 
	<< "{\n" 
	<< oss_reg_defs.str() 
	<< oss_prg.str() 
	<< "}\n";

    return final_ptx.str();
  }



  jit_value jit_add_param( jit_ptx_type type ) {
    assert( type != jit_ptx_type::u8 );
    jit_function_t func = jit_get_function();
    if (func->get_param_count() > 0)
      func->get_signature() << ",\n";

    if (type == jit_ptx_type::pred) {
      func->get_signature() << ".param ." 
			    << jit_get_ptx_type( jit_ptx_type::u8 )
			    << " param" 
			    << func->get_param_count();

      int num = jit_get_function()->reg_alloc( jit_ptx_type::u8 );
      jit_get_function()->get_prg() << "ld.param."
				    << jit_get_ptx_type( jit_ptx_type::u8 ) << " "
				    << jit_get_ptx_letter( jit_ptx_type::u8 ) << num 
				    << ",[" 
				    << "param" 
				    << func->get_param_count() 
				    << "];\n";

      jit_value s32( jit_ptx_type::s32 );
      jit_get_function()->get_prg() << "cvt."
				    << jit_get_ptx_type( s32.get_type() )
				    << ".u8"
				    << " "
				    << jit_get_reg_name( s32 )
				    << ","
				    << jit_get_ptx_letter( jit_ptx_type::u8 ) << num
				    << ";\n";
      func->inc_param_count();
      jit_value ret = jit_ins_ne( s32 , jit_value(0) );
      ret.set_state_space( jit_state_space::state_default );
      ret.set_ever_assigned();
      return ret;
    } else {
      func->get_signature() << ".param ." 
			    << jit_get_ptx_type(type) 
			    << " param" 
			    << func->get_param_count();
      jit_value ret( type );
      func->get_prg() << "ld.param." 
		      << jit_get_ptx_type(type) 
		      << " " 
		      << jit_get_reg_name(ret) 
		      << ",[" 
		      << "param" 
		      << func->get_param_count() 
		      << "];\n";
      ret.set_state_space( jit_state_space::state_global );
      ret.set_ever_assigned();
      func->inc_param_count();
      return ret;
    }
  }


  int jit_function::local_alloc( jit_ptx_type type, int count ) {
    assert(count>0);
    int ret =  vec_local_count.size();
    vec_local_count.push_back( std::make_pair(type,count) );
    return ret;
  }
  
  
  jit_value jit_allocate_local( jit_ptx_type type , int count ) {
    jit_function_t func = jit_get_function();
    int num = func->local_alloc(type,count);
    jit_value ret( jit_ptx_type::u64 );
    func->get_prg() << "mov.u64 " 
		    << jit_get_reg_name(ret) 
		    << "," 
		    << jit_get_identifier_local_memory() << num 
		    << ";\n";
    ret.set_ever_assigned();
    ret.set_state_space( jit_state_space::state_local );
    return ret;
  }


  jit_value jit_get_shared_mem_ptr( ) {
    jit_function_t func = jit_get_function();
    jit_value ret( jit_ptx_type::u64 );
    func->get_prg() << "mov.u64 " 
		    << jit_get_reg_name(ret) 
		    << ",sdata;\n";
    ret.set_ever_assigned();
    ret.set_state_space( jit_state_space::state_shared );
    func->emitShared();
    return ret;
  }

  std::string jit_get_kernel_as_string() {
    std::string ret = jit_get_function()->get_kernel_as_string();
    QDP_info_primary("Resetting jit function");
    jit_internal_function.reset();
    return ret;
  }


  void jit_start_new_function() {
    if (jit_internal_function) {
      QDP_error_exit("New jit function requested, but previous one not finished yet!");
      //QDP_info_primary("Resetting old jit function (use_count = %d) ...",(int)jit_internal_function.use_count());
      //jit_internal_function.reset();
    }
    QDP_info_primary("Starting new jit function");
    jit_internal_function = make_shared<jit_function>();
  }


  jit_function_t jit_get_function() {
    assert( jit_internal_function );
    return jit_internal_function;
  }


  CUfunction jit_get_cufunction(const char* fname)
  {
    CUfunction func;
    CUresult ret;
    CUmodule cuModule;

    std::string ptx_kernel = jit_get_kernel_as_string();

#if 0
    // Write kernel to file ?
    if (Layout::primaryNode()) {
      std::ofstream out(fname);
      out << ptx_kernel;
      out.close();
    }
#endif

    ret = cuModuleLoadDataEx( &cuModule , ptx_kernel.c_str() , 0 , 0 , 0 );
    if (ret) {
      if (Layout::primaryNode()) {
	QDP_info_primary("Error loading external data. Dumping kernel to %s.",fname);
	std::ofstream out(fname);
	out << ptx_kernel;
	out.close();
	QDP_error_exit("Abort.");
      }
    }

    ret = cuModuleGetFunction(&func, cuModule, "function");
    if (ret)
      QDP_error_exit("Error returned from cuModuleGetFunction. Abort.");

    return func;
  }




  jit_label_t jit_label_create() {
    return std::make_shared< jit_label >();
  }


  // VALUE REG


  jit_value::jit_value( jit_ptx_type type_ ):   
    type(type_),    
    mem_state(jit_state_space::state_default), 
    ever_assigned(false) 
  { 
    reg_alloc(); 
  }

  jit_value::jit_value( const jit_value& rhs ): 
    type(rhs.type), 
    mem_state(rhs.mem_state),
    ever_assigned(rhs.ever_assigned)
  {
    reg_alloc();
    assign( rhs );
  }
    
  void jit_value::reg_alloc() {
    assert( type != jit_ptx_type::u8 );
    number = jit_get_function()->reg_alloc(type);
  }
  

  void jit_value::assign( const jit_value& rhs ) {
    assert( rhs.get_ever_assigned() );
    assert( type != jit_ptx_type::u8 );
    if ( type != rhs.get_type() ) {
      jit_ins_mov( *this , jit_val_convert( type , rhs ) );
    } else {
      jit_ins_mov( *this , rhs );
    }
    ever_assigned = rhs.ever_assigned;
  }


  jit_value& jit_value::operator=( const jit_value& rhs ) {
    assign(rhs);
  }
    


  jit_ptx_type    jit_value::get_type() const {return type;}
  void            jit_value::set_state_space( jit_state_space ss ) { mem_state = ss; }
  jit_state_space jit_value::get_state_space() const { 
    return mem_state; 
  }



  void jit_ins_bar_sync( int a ) {
    jit_function_t func = jit_get_function();
    assert( a >= 0 && a <= 15 );
    func->get_prg() << "bar.sync " << a << ";\n";
  }

  
  std::string jit_value::get_name() const {
    std::ostringstream tmp;
    tmp << jit_get_ptx_letter( type ) << number;
    return tmp.str();
  }
  
  
  // int jit_value::get_number() const { return number; }
  
  // jit_function_t jit_value_reg::get_func() const { return func; };






  std::string jit_get_reg_name( const jit_value& val ) {
    return val.get_name();
  }

    
  // jit_value_reg_t jit_val_create_new( int type ) {
  //   jit_function_t func = jit_get_function();
  //   //std::cout << "Creating jit value, type = " << type << "\n";
  //   jit_value_reg_t val( new jit_value_reg( type ) );
  //   return val;
  // }

  // jit_value_reg_t jit_val_create_from_const( int type , int val_i , const jit_value& pred) {
  //   jit_function_t func = jit_get_function();
  //   //std::cout << "Creating const jit value, type = " << type << "\n";
  //   jit_value_const_t val_const(new jit_value_const_int(val_i));
  //   return jit_val_create_convert( type , val_const , pred );
  // }


  jit_value jit_val_convert( jit_ptx_type type , const jit_value& rhs , const jit_value& pred ) {
    jit_value ret(type);
    if (rhs.get_type() == type) {
      assert(type != jit_ptx_type::u8);
      jit_ins_mov( ret , rhs , pred );
    } else {
      if (rhs.get_type() == jit_ptx_type::pred) {
	jit_value ret_s32(jit_ptx_type::s32);
	ret_s32 = jit_ins_selp( jit_value(1), jit_value(0), rhs );
	return jit_val_convert( type , ret_s32 , pred );
      } else {
	jit_get_function()->get_prg() << jit_predicate(pred)
				      << "cvt."
				      << jit_get_map_cvt_rnd_from_to(rhs.get_type(),type) 
				      << jit_get_ptx_type( type ) << "." 
				      << jit_get_ptx_type( rhs.get_type() ) << " " 
				      << ret.get_name() << ","
				      << rhs.get_name() << ";\n";
      }
    }
    ret.set_state_space( rhs.get_state_space() );
    ret.set_ever_assigned();
    return ret;
  }


  // jit_value_reg_t jit_val_create_convert_const( int type , jit_value_const_t val , const jit_value& pred ) {
  //   jit_function_t func = jit_get_function();
  //   jit_value_reg_t ret = jit_val_create_new( type );
  //   assert( type != jit_ptx_type::u8 );
  //   func->get_prg() << jit_predicate(pred)
  // 		    << "mov." 
  // 		    << jit_get_ptx_type( type ) << " " 
  // 		    << ret->get_name() << ","
  // 		    << val->getAsString() << ";\n";
  //   return ret;
  // }
  // jit_value_reg_t jit_val_create_convert_reg( int type , jit_value_reg_t val , const jit_value& pred ) {
  //   jit_function_t func = jit_get_function();
  //   jit_value_reg_t ret = jit_val_create_new( type );
  //   if (type == val.get_type()) {
  //     assert( type != jit_ptx_type::u8 );
  //     func->get_prg() << jit_predicate(pred)
  // 		      << "mov." 
  // 		      << jit_get_ptx_type( type ) << " " 
  // 		      << ret->get_name() << ","
  // 		      << val->get_name() << ";\n";
  //   } else {
  //     if ( type == jit_ptx_type::pred ) {
  // 	ret = get<jit_value_reg>(jit_ins_ne( val , jit_value(0) , pred ));
  //     } else if ( val.get_type() == jit_ptx_type::pred ) {
  // 	jit_value ret_s32 = jit_ins_selp( jit_value(1) , jit_value(0) , val );
  // 	if (type != jit_ptx_type::s32)
  // 	  return jit_val_create_convert( jit_ptx_type::s32 , ret_s32 , pred );
  // 	else
  // 	  return get<jit_value_reg>(ret_s32);
  //     } else {
  // 	func->get_prg() << jit_predicate(pred)
  // 			<< "cvt."
  // 			<< jit_get_map_cvt_rnd_from_to(val.get_type(),type) 
  // 			<< jit_get_ptx_type( type ) << "." 
  // 			<< jit_get_ptx_type( val.get_type() ) << " " 
  // 			<< ret->get_name() << ","
  // 			<< val->get_name() << ";\n";
  //     }
  //   }
  //   return ret;
  // }
  // jit_value_reg_t jit_val_create_convert( int type , jit_value val , const jit_value& pred ) {
  //   assert(val);
  //   if (auto val_const = get< jit_value_const >(val))
  //     return jit_val_create_convert_const( type , val_const , pred );
  //   if (auto val_reg = get< jit_value_reg >(val))
  //     return jit_val_create_convert_reg( type , val_reg , pred );
  //   assert(!"Probs");
  // }

  // jit_value jit_val_create_copy( jit_value val , const jit_value& pred ) {
  //   assert(val);
  //   if (auto val_const = get< jit_value_const >(val)) 
  //     {
  // 	if (jit_value_const_int_t val_const_int = get<jit_value_const_int>(val_const)) 
  // 	  {
  // 	    return jit_value( val_const_int->getValue()  );
  // 	  } 
  // 	if (jit_value_const_float_t val_const_float = get<jit_value_const_float>(val_const)) 
  // 	  {
  // 	    return jit_val_create_const_float( val_const_float->getValue()  );
  // 	  }
  // 	assert(!"Problem");
  //     }
  //   if (auto val_reg = get< jit_value_reg >(val))
  //     {
  // 	// std::cout << "TYPE reg = " << val_reg.get_type() << "\n";
  // 	// std::cout << "TYPE     = " << val.get_type() << "\n";
  // 	assert( val_reg.get_type() != jit_ptx_type::u8 );
  // 	jit_value_reg_t ret = jit_val_create_convert( val_reg->get_func() , val_reg.get_type() , val_reg , pred );
  // 	return ret;
  //     }
  //   assert(!"Problem");
  // }


  // jit_value_const_t jit_value( int val ) {
  //   return std::make_shared< jit_value_const_int >(val);
  // }

  // jit_value_const_t jit_val_create_const_float( double val ) {
  //   return std::make_shared< jit_value_const_float >(val);
  // }




  // Thread Geometry

  jit_value jit_geom_get_tidx() {
    jit_ptx_type th_reg = DeviceParams::Instance().getMajor() >= 2 ? jit_ptx_type::u32 : jit_ptx_type::u16;
    jit_value tidx( th_reg );
    jit_get_function()->get_prg() << "mov."
				  << jit_get_ptx_type( th_reg )
				  << " "
				  << jit_get_reg_name( tidx ) 
				  << ",%tid.x;\n";
    tidx.set_state_space( jit_state_space::state_default );
    tidx.set_ever_assigned();
    return tidx;
  }
  jit_value jit_geom_get_ntidx() {
    jit_ptx_type th_reg = DeviceParams::Instance().getMajor() >= 2 ? jit_ptx_type::u32 : jit_ptx_type::u16;
    jit_value tidx( th_reg );
    jit_get_function()->get_prg() << "mov."
				  << jit_get_ptx_type( th_reg )
				  << " "
				  << jit_get_reg_name( tidx ) 
				  << ",%ntid.x;\n";
    tidx.set_state_space( jit_state_space::state_default );
    tidx.set_ever_assigned();
    return tidx;
  }
  jit_value jit_geom_get_ctaidx() {
    jit_ptx_type th_reg = DeviceParams::Instance().getMajor() >= 2 ? jit_ptx_type::u32 : jit_ptx_type::u16;
    jit_value tidx( th_reg );
    jit_get_function()->get_prg() << "mov."
				  << jit_get_ptx_type( th_reg )
				  << " "
				  << jit_get_reg_name( tidx ) 
				  << ",%ctaid.x;\n";
    tidx.set_state_space( jit_state_space::state_default );
    tidx.set_ever_assigned();
    return tidx;
  }




  jit_value jit_ins_selp( const jit_value& lhs ,  const jit_value& rhs , const jit_value& p ) {
    jit_ptx_type typebase = jit_type_promote( lhs.get_type() , rhs.get_type() );
    jit_value ret( typebase );
    std::ostringstream instr;
    
    if (typebase == jit_ptx_type::pred) {
      assert( lhs.get_type() == jit_ptx_type::pred );
      assert( rhs.get_type() == jit_ptx_type::pred );
      typebase = jit_ptx_type::s32;
      jit_value lhs_s32(typebase);
      jit_value rhs_s32(typebase);
      jit_value ret_s32(typebase);
      lhs_s32 = jit_ins_selp( jit_value(1) , jit_value(0) , lhs );
      rhs_s32 = jit_ins_selp( jit_value(1) , jit_value(0) , rhs );
      instr << "selp." 
	    << jit_get_ptx_type( typebase ) 
	    << " "
	    << jit_get_reg_name( ret_s32 ) 
	    << ","
	    << jit_get_reg_name( lhs_s32 ) 
	    << ","
	    << jit_get_reg_name( rhs_s32 ) 
	    << ","
	    << jit_get_reg_name( p ) 
	    << ";\n";
      jit_get_function()->get_prg() << instr.str();
      ret = jit_ins_ne( ret_s32 , jit_value(0) );
      ret.set_state_space( jit_state_promote( lhs.get_state_space() , rhs.get_state_space() ) );
      ret.set_ever_assigned();
      return ret;
    }

    jit_value lhs_tb(typebase);
    jit_value rhs_tb(typebase);

    lhs_tb = lhs.get_type() != typebase ? jit_val_convert( typebase , lhs ) : lhs;
    rhs_tb = rhs.get_type() != typebase ? jit_val_convert( typebase , rhs ) : rhs;

    instr << "selp." 
	  << jit_get_ptx_type( typebase ) 
	  << " "
	  << jit_get_reg_name( ret ) 
	  << ","
	  << jit_get_reg_name( lhs_tb ) 
	  << ","
	  << jit_get_reg_name( rhs_tb ) 
	  << ","
	  << jit_get_reg_name( p ) 
	  << ";\n";

    jit_get_function()->get_prg() << instr.str();
    ret.set_state_space( jit_state_promote( lhs.get_state_space() , rhs.get_state_space() ) );
    ret.set_ever_assigned();
    return ret;
  }





  jit_value jit_ins_op( const jit_value& lhs , const jit_value& rhs , const JitOp& op , const jit_value& pred ) {
    jit_ptx_type dest_type = op.getDestType();
    jit_ptx_type args_type = op.getArgsType();
    jit_value ret(dest_type);
    jit_value lhs_new = jit_val_convert( args_type , lhs , pred );
    jit_value rhs_new = jit_val_convert( args_type , rhs , pred );
    jit_get_function()->get_prg() << jit_predicate(pred)
				  << op << " "
				  << jit_get_reg_name( ret ) << ","
				  << jit_get_reg_name( lhs_new ) << ","
				  << jit_get_reg_name( rhs_new ) << ";\n";
    ret.set_ever_assigned();
    ret.set_state_space( jit_state_promote( lhs.get_state_space() , rhs.get_state_space() ) );
    return ret;
  }



  jit_value jit_ins_add( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_op( lhs , rhs , JitOpAdd( lhs , rhs ) , pred );
  }
  jit_value jit_ins_sub( const jit_value& lhs , const jit_value& rhs , const jit_value& pred) {
    return jit_ins_op( lhs , rhs , JitOpSub( lhs , rhs ) , pred );
  }
  jit_value jit_ins_mul( const jit_value& lhs , const jit_value& rhs , const jit_value& pred) {
    return jit_ins_op( lhs , rhs , JitOpMul( lhs , rhs ) , pred );
  }
  jit_value jit_ins_div( const jit_value& lhs , const jit_value& rhs , const jit_value& pred) {
    return jit_ins_op( lhs , rhs , JitOpDiv( lhs , rhs ) , pred );
  }
  jit_value jit_ins_shl( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_op( lhs , rhs , JitOpSHL( lhs , rhs ) , pred );
  }
  jit_value jit_ins_shr( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_op( lhs , rhs , JitOpSHR( lhs , rhs ) , pred );
  }
  jit_value jit_ins_mul_wide( const jit_value& lhs , const jit_value& rhs , const jit_value& pred) {
    return jit_ins_op( lhs , rhs , JitOpMulWide( lhs , rhs ) , pred );
  }
  jit_value jit_ins_and( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_op( lhs , rhs , JitOpAnd( lhs , rhs ) , pred );
  }
  jit_value jit_ins_or( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_op( lhs , rhs , JitOpOr( lhs , rhs ) , pred );
  }
  jit_value jit_ins_xor( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_op( lhs , rhs , JitOpXOr( lhs , rhs ) , pred );
  }
  jit_value jit_ins_rem( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_op( lhs , rhs , JitOpRem( lhs , rhs ) , pred );
  }

  jit_value jit_ins_lt( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_op( lhs , rhs , JitOpLT( lhs , rhs ) , pred );
  }
  jit_value jit_ins_ne( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_op( lhs , rhs , JitOpNE( lhs , rhs ) , pred );
  }
  jit_value jit_ins_eq( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_op( lhs , rhs , JitOpEQ( lhs , rhs ) , pred );
  }
  jit_value jit_ins_ge( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_op( lhs , rhs , JitOpGE( lhs , rhs ) , pred );
  }
  jit_value jit_ins_le( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_op( lhs , rhs , JitOpLE( lhs , rhs ) , pred );
  }
  jit_value jit_ins_gt( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_op( lhs , rhs , JitOpGT( lhs , rhs ) , pred );
  }


  jit_value jit_ins_or( const jit_value& lhs , const jit_value& rhs ) { assert(!"ni"); }
  jit_value jit_ins_and( const jit_value& lhs , const jit_value& rhs ) { assert(!"ni"); }
  jit_value jit_ins_xor( const jit_value& lhs , const jit_value& rhs ) { assert(!"ni"); }
  jit_value jit_ins_mod( const jit_value& lhs , const jit_value& rhs ) { assert(!"ni"); }




  jit_value jit_ins_unary_op( const jit_value& reg , const JitUnaryOp& op , const jit_value& pred ) {
    jit_ptx_type type = reg.get_type();
    jit_value ret( type );
    jit_get_function()->get_prg() << jit_predicate(pred)
				  << op << " "
				  << jit_get_reg_name( ret ) << ","
				  << jit_get_reg_name( reg ) << ";\n";
    ret.set_ever_assigned();
    ret.set_state_space( ret.get_state_space() );
    return ret;
  }


  jit_value jit_ins_neg( const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_unary_op( rhs , JitUnaryOpNeg( rhs.get_type() ) , pred );
  }
  jit_value jit_ins_not( const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_unary_op( rhs , JitUnaryOpNot( rhs.get_type() ) , pred );
  }
  jit_value jit_ins_fabs( const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_unary_op( rhs , JitUnaryOpAbs( rhs.get_type() ) , pred );
  }
  jit_value jit_ins_floor( const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_unary_op( rhs , JitUnaryOpFloor( rhs.get_type() ) , pred );
  }
  jit_value jit_ins_ceil( const jit_value& rhs , const jit_value& pred ) {
    return jit_ins_unary_op( rhs , JitUnaryOpCeil( rhs.get_type() ) , pred );
  }
  jit_value jit_ins_sqrt( const jit_value& rhs , const jit_value& pred ) { 
    return jit_ins_unary_op( rhs , JitUnaryOpSqrt( rhs.get_type() ) , pred );
  }


  jit_value jit_ins_math_unary( int num , 
				jit_ptx_type arg_type , 
				const jit_value& lhs , 
				const jit_value& pred ) {
    assert( arg_type == jit_ptx_type::f32 || arg_type == jit_ptx_type::f64);
    assert( num >= 0 && num < PTX::map_ptx_math_functions_unary.size() );
    jit_value lhs_new(arg_type);
    if (lhs.get_type() != arg_type )
      lhs_new = jit_val_convert( arg_type , lhs );
    else
      lhs_new = lhs;
    jit_value ret( arg_type );
    jit_get_function()->get_prg() << jit_predicate(pred)
				  << "call (" 
				  << jit_get_reg_name( ret ) 
				  << ")," 
				  << jit_get_map_ptx_math_functions_funcname_unary(num)
				  << ",(" 
				  << jit_get_reg_name( lhs ) 
				  << ");\n";
    ret.set_ever_assigned();
    jit_get_function()->set_include_math_ptx_unary(num);
    return ret;
  }
  jit_value jit_ins_math_binary( int num , 
				 jit_ptx_type arg_type , 
				 const jit_value& lhs , 
				 const jit_value& rhs , 
				 const jit_value& pred ) {
    assert( arg_type == jit_ptx_type::f32 || arg_type == jit_ptx_type::f64);
    assert( num >= 0 && num < PTX::map_ptx_math_functions_binary.size() );

    jit_value lhs_new(arg_type);
    if (lhs.get_type() != arg_type )
      lhs_new = jit_val_convert( arg_type , lhs );
    else
      lhs_new = lhs;

    jit_value rhs_new(arg_type);
    if (rhs.get_type() != arg_type )
      rhs_new = jit_val_convert( arg_type , rhs );
    else
      rhs_new = lhs;

    jit_value ret( arg_type );

    jit_get_function()->get_prg() << jit_predicate(pred)
				  << "call (" 
				  << jit_get_reg_name( ret ) 
				  << ")," 
				  << jit_get_map_ptx_math_functions_funcname_binary(num)
				  << ",(" 
				  << jit_get_reg_name( lhs ) 
				  << "," 
				  << jit_get_reg_name( rhs ) 
				  << ");\n";

    ret.set_ever_assigned();
    jit_get_function()->set_include_math_ptx_binary(num);
    return ret;
  }

  // Imported PTX Unary operations single precicion
  jit_value jit_ins_sin_f32(  const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 0 , jit_ptx_type::f32 , lhs , pred ); }
  jit_value jit_ins_acos_f32( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 1 , jit_ptx_type::f32 , lhs , pred ); }
  jit_value jit_ins_asin_f32( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 2 , jit_ptx_type::f32 , lhs , pred ); }
  jit_value jit_ins_atan_f32( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 3 , jit_ptx_type::f32 , lhs , pred ); }
  jit_value jit_ins_cos_f32( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 4 , jit_ptx_type::f32 , lhs , pred ); }
  jit_value jit_ins_cosh_f32( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 5 , jit_ptx_type::f32 , lhs , pred ); }
  jit_value jit_ins_exp_f32( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 6 , jit_ptx_type::f32 , lhs , pred ); }
  jit_value jit_ins_log_f32( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 7 , jit_ptx_type::f32 , lhs , pred ); }
  jit_value jit_ins_log10_f32( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 8 , jit_ptx_type::f32 , lhs , pred ); }
  jit_value jit_ins_sinh_f32( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 9 , jit_ptx_type::f32 , lhs , pred ); }
  jit_value jit_ins_tan_f32( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 10 , jit_ptx_type::f32 , lhs , pred ); }
  jit_value jit_ins_tanh_f32( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 11 , jit_ptx_type::f32 , lhs , pred ); }

  // Imported PTX Binary operations single precicion
  jit_value jit_ins_pow_f32( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) { 
    return jit_ins_math_binary( 0 , jit_ptx_type::f32 , lhs , rhs , pred ); }
  jit_value jit_ins_atan2_f32( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) { 
    return jit_ins_math_binary( 1 , jit_ptx_type::f32 , lhs , rhs , pred ); }


  // Imported PTX Unary operations double precicion
  jit_value jit_ins_sin_f64(  const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 12 , jit_ptx_type::f64 , lhs , pred ); }
  jit_value jit_ins_acos_f64( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 13 , jit_ptx_type::f64 , lhs , pred ); }
  jit_value jit_ins_asin_f64( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 14 , jit_ptx_type::f64 , lhs , pred ); }
  jit_value jit_ins_atan_f64( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 15 , jit_ptx_type::f64 , lhs , pred ); }
  jit_value jit_ins_cos_f64( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 16 , jit_ptx_type::f64 , lhs , pred ); }
  jit_value jit_ins_cosh_f64( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 17 , jit_ptx_type::f64 , lhs , pred ); }
  jit_value jit_ins_exp_f64( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 18 , jit_ptx_type::f64 , lhs , pred ); }
  jit_value jit_ins_log_f64( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 19 , jit_ptx_type::f64 , lhs , pred ); }
  jit_value jit_ins_log10_f64( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 20 , jit_ptx_type::f64 , lhs , pred ); }
  jit_value jit_ins_sinh_f64( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 21 , jit_ptx_type::f64 , lhs , pred ); }
  jit_value jit_ins_tan_f64( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 22 , jit_ptx_type::f64 , lhs , pred ); }
  jit_value jit_ins_tanh_f64( const jit_value& lhs , const jit_value& pred ) { 
    return jit_ins_math_unary( 23 , jit_ptx_type::f64 , lhs , pred ); }

  // Imported PTX Binary operations single precicion
  jit_value jit_ins_pow_f64( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) { 
    return jit_ins_math_binary( 2 , jit_ptx_type::f64 , lhs , rhs , pred ); }
  jit_value jit_ins_atan2_f64( const jit_value& lhs , const jit_value& rhs , const jit_value& pred ) { 
    return jit_ins_math_binary( 3 , jit_ptx_type::f64 , lhs , rhs , pred ); }


  void jit_ins_mov( jit_value& dest , const std::string& src , const jit_value& pred ) {
    assert( dest.get_type() != jit_ptx_type::u8 );
    jit_get_function()->get_prg() << jit_predicate(pred)
				  << "mov."
				  << jit_get_ptx_type( dest.get_type() ) << " "
				  << jit_get_reg_name( dest ) << ","
				  << src << ";\n";
    dest.set_state_space( jit_state_space::state_default );
    dest.set_ever_assigned();
  }


  void jit_ins_mov( jit_value& dest , const jit_value& src , const jit_value& pred ) {
    if ( dest.get_type() != src.get_type() ) {
      jit_value src_new(dest.get_type());
      jit_ins_mov( src_new , jit_val_convert( dest.get_type() , src , pred ), pred );
      jit_ins_mov( dest , src_new , pred );
      return;
    }
    assert( dest.get_type() != jit_ptx_type::u8 );
    jit_get_function()->get_prg() << jit_predicate(pred)
				  << "mov."
				  << jit_get_ptx_type( dest.get_type() ) << " "
				  << jit_get_reg_name( dest ) << ","
				  << jit_get_reg_name( src ) << ";\n";
    dest.set_state_space( src.get_state_space() );
    dest.set_ever_assigned();
  }


  jit_value jit_ins_load( const jit_value& base , int offset , jit_ptx_type type , const jit_value& pred ) {
    if ( type == jit_ptx_type::pred ) {
      int num = jit_get_function()->reg_alloc( jit_ptx_type::u8 );
      jit_get_function()->get_prg() << jit_predicate(pred)
				    << "ld." << get_state_space_str(base.get_state_space()) << "."
				    << jit_get_ptx_type( jit_ptx_type::u8 ) << " "
				    << jit_get_ptx_letter( jit_ptx_type::u8 ) << num << ",["
				    << jit_get_reg_name( base ) << " + "
				    << offset << "];\n";

      jit_value s32( jit_ptx_type::s32 );
      jit_get_function()->get_prg() << "cvt."
				    << jit_get_ptx_type( s32.get_type() )
				    << ".u8"
				    << " "
				    << jit_get_reg_name( s32 )
				    << ","
				    << jit_get_ptx_letter( jit_ptx_type::u8 ) << num
				    << ";\n";
      return jit_ins_ne( s32 , jit_value(0) );
    }
    jit_value loaded( type );
    jit_get_function()->get_prg() << jit_predicate(pred)
				  << "ld." << get_state_space_str(base.get_state_space()) << "."
				  << jit_get_ptx_type( type ) << " "
				  << jit_get_reg_name( loaded ) << ",["
				  << jit_get_reg_name( base ) << " + "
				  << offset << "];\n";
    
    loaded.set_state_space( jit_state_space::state_default );
    loaded.set_ever_assigned();
    return loaded;
  }

  void jit_ins_store(const jit_value& base, int offset , jit_ptx_type type , const jit_value& reg , const jit_value& pred ) {
    if (type == jit_ptx_type::pred ) {
      if ( reg.get_type() != jit_ptx_type::pred ) {
	// I need to convert reg to an 'u8' and the store it
	int num = jit_get_function()->reg_alloc( jit_ptx_type::u8 );
	jit_get_function()->get_prg() << "cvt.u8."
				      << jit_get_ptx_type( reg.get_type() )
				      << " "
				      << jit_get_ptx_letter( jit_ptx_type::u8 ) << num
				      << ","
				      << jit_get_reg_name( reg )
				      << ";\n";
	jit_get_function()->get_prg() << jit_predicate(pred)
				      << "st." << get_state_space_str(base.get_state_space()) << "."
				      << jit_get_ptx_type( jit_ptx_type::u8 ) << " ["
				      << jit_get_reg_name( base ) << " + "
				      << offset << "],"
				      << jit_get_ptx_letter( jit_ptx_type::u8 ) << num << ";\n";
      } else {
	jit_value reg_s32 = jit_ins_selp( jit_value(1) , jit_value(0) , reg );

	// I need to convert reg_s32 to an 'u8' and the store it
	int num = jit_get_function()->reg_alloc( jit_ptx_type::u8 );
	jit_get_function()->get_prg() << "cvt.u8."
				      << jit_get_ptx_type( reg_s32.get_type() )
				      << " "
				      << jit_get_ptx_letter( jit_ptx_type::u8 ) << num
				      << ","
				      << jit_get_reg_name( reg_s32 )
				      << ";\n";
	jit_get_function()->get_prg() << jit_predicate(pred)
				      << "st." << get_state_space_str(base.get_state_space()) << "."
				      << jit_get_ptx_type( jit_ptx_type::u8 ) << " ["
				      << jit_get_reg_name( base ) << " + "
				      << offset << "],"
				      << jit_get_ptx_letter( jit_ptx_type::u8 ) << num << ";\n";
      }
    } else {
      if ( reg.get_type() != type ) {
	jit_value reg_type = jit_val_convert( type , reg );
	jit_ins_store( base , offset , type , reg_type , pred );
      } else {
	jit_get_function()->get_prg() << jit_predicate(pred)
				      << "st." << get_state_space_str(base.get_state_space()) << "."
				      << jit_get_ptx_type( type ) << " ["
				      << jit_get_reg_name( base ) << " + "
				      << offset << "],"
				      << jit_get_reg_name( reg ) << ";\n";
      }
    }
  }




  jit_value jit_geom_get_linear_th_idx() {
    jit_value ctaidx = jit_geom_get_ctaidx();
    jit_value ntidx  = jit_geom_get_ntidx();

    if ( ctaidx.get_type() == jit_ptx_type::u16 )
      return jit_ins_add( jit_ins_mul_wide( ctaidx , ntidx ) , jit_geom_get_tidx() );
    else 
      return jit_ins_add( jit_ins_mul( ctaidx , ntidx ) , jit_geom_get_tidx() );
  }


  std::string jit_predicate( const jit_value& pred ) {
    if (!pred.get_ever_assigned())
      return "";
    assert( pred.get_type() == jit_ptx_type::pred );
    std::ostringstream oss;
    oss << "@" << jit_get_reg_name(pred) << " ";
    return oss.str();
  }


  void jit_ins_label( jit_label_t& label ) {
    if (!label)
      label = jit_label_create();
    jit_get_function()->get_prg() << *label << ":\n";
  }

  void jit_ins_exit( const jit_value& pred ) {
    jit_get_function()->get_prg() << jit_predicate( pred )
				  << "exit;\n";
  }

  void jit_ins_branch( jit_label_t& label , const jit_value& pred ) {
    if (!label)
      label = jit_label_create();
    jit_get_function()->get_prg() << jit_predicate( pred )
				  << "bra "
				  << *label << ";\n";
  }

  void jit_ins_comment( const char * comment ) {
    jit_get_function()->get_prg() << "// " << comment << "\n";
  }

}



