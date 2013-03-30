#include "qdp.h"

namespace QDP 
{


  template<>
  WordREG<double>::WordREG(int i) {
    val = jit_val_create_const_float( i );
    setup_m=true;
  }

  template<>
  WordREG<double>::WordREG(double f) {
    val = jit_val_create_const_float( f );
    setup_m=true;
  }

  template<>
  WordREG<float>::WordREG(int i) {
    val = jit_val_create_const_float( i );
    setup_m=true;
  }

  template<>
  WordREG<float>::WordREG(double f) {
    val = jit_val_create_const_float( f );
    setup_m=true;
  }

  template<>
  WordREG<int>::WordREG(int i) {
    val = jit_val_create_const_int( i );
    setup_m=true;
  }

  template<>
  WordREG<int>::WordREG(double f) {
    val = jit_val_create_const_int( f );
    setup_m=true;
  }



  template<>
  void WordREG<int>::setup(jit_value_t v) {
    assert(v);
    if (v->get_type() == jit_type<int>::value) {
      val=v;
    } else {
      if (auto v_const = get<jit_value_const>(v)) {
      	if (v_const->isInt()) {
      	  val = v;
      	} else {
      	  val = jit_val_create_const_int( v_const->getAsFloat() );
      	}
      } else {
      	val = jit_val_create_convert( getFunc(v) , jit_type<int>::value , v );
      }
    }
    setup_m=true;
  }


  template<>
  void WordREG<float>::setup(jit_value_t v) {
    assert(v);
    if (v->get_type() == jit_type<float>::value) {
      val=v;
    } else {
      if (auto v_const = get<jit_value_const>(v)) {
      	if (v_const->isInt()) {
	  if (auto v_const_int = get<jit_value_const_int>(v_const)) {
	    val = jit_val_create_const_int( v_const_int->getValue() );
	  } else {
	    assert(!"oops");
	  }
      	} else {
      	  val = jit_val_create_const_float( v_const->getAsFloat() );
      	}
      } else {
      	val = jit_val_create_convert( getFunc(v) , jit_type<float>::value , v );
      }
    }
    setup_m=true;
  }


  template<>
  void WordREG<double>::setup(jit_value_t v) {
    assert(v);
    if (v->get_type() == jit_type<double>::value) {
      val=v;
    } else {
      if (auto v_const = get<jit_value_const>(v)) {
      	if (v_const->isInt()) {
	  if (auto v_const_int = get<jit_value_const_int>(v_const)) {
	    val = jit_val_create_const_int( v_const_int->getValue() );
	  } else {
	    assert(!"oops");
	  }
      	} else {
      	  val = jit_val_create_const_float( v_const->getAsFloat() );
      	}
      } else {
      	val = jit_val_create_convert( getFunc(v) , jit_type<double>::value , v );
      }
    }
    setup_m=true;
  }


  template<>
  void WordREG<bool>::setup(jit_value_t v) {
    assert(v);
    if (v->get_type() == jit_type<bool>::value) {
      val=v;
    } else {
      //std::cout << "trying... " << __PRETTY_FUNCTION__ << v->get_type() << "\n";
      if (auto v_const = get<jit_value_const>(v)) {
      	if (v_const->isInt()) {
	  auto v_const_int = get<jit_value_const_int>(v_const);
	  assert(v_const_int);
	  val = jit_ins_ne( v_const_int , jit_val_create_const_int(0) );
	  //std::cout << "it was a const_int!n";
	} else {
	  auto v_const_float = get<jit_value_const_float>(v_const);
	  assert(v_const_float);
	  val = jit_ins_ne( v_const_float , jit_val_create_const_float(0.0) );
	  //std::cout << "it was a const_float!n";
	}
      } else {
	//std::cout << "its a reg!n";
	auto v_reg = get<jit_value_reg>(v);
	assert(v_reg);
	jit_function_t func = getFunc(v_reg);
	assert( func );
	val = jit_val_create_convert( func , jit_type<bool>::value , v );
	//std::cout << "converted from reg!n";
      }
    }
    setup_m=true;
  }




  typename UnaryReturn<WordREG<float>, FnArcCos>::Type_t
  acos(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnArcCos>::Type_t ret;
    ret.setup( jit_ins_acos_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnArcSin>::Type_t 
  asin(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnArcSin>::Type_t ret;
    ret.setup( jit_ins_asin_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnArcTan>::Type_t
  atan(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnArcTan>::Type_t ret;
    ret.setup( jit_ins_atan_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnCos>::Type_t
  cos(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnCos>::Type_t ret;
    ret.setup( jit_ins_cos_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnHypCos>::Type_t
  cosh(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnHypCos>::Type_t ret;
    ret.setup( jit_ins_cosh_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnExp>::Type_t
  exp(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnExp>::Type_t ret;
    ret.setup( jit_ins_exp_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnLog>::Type_t
  log(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnLog>::Type_t ret;
    ret.setup( jit_ins_log_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnLog10>::Type_t
  log10(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnLog10>::Type_t ret;
    ret.setup( jit_ins_log10_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnSin>::Type_t
  sin(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnSin>::Type_t ret;
    ret.setup( jit_ins_sin_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnHypSin>::Type_t
  sinh(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnHypSin>::Type_t ret;
    ret.setup( jit_ins_sinh_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnTan>::Type_t
  tan(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnTan>::Type_t ret;
    ret.setup( jit_ins_tan_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnHypTan>::Type_t
  tanh(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnHypTan>::Type_t ret;
    ret.setup( jit_ins_tanh_f32( s1.get_val() ) );
    return ret;
  }


  typename BinaryReturn<WordREG<float>, WordREG<float>, FnPow>::Type_t
  pow(const WordREG<float>& s1, const WordREG<float>& s2)
  {
    typename UnaryReturn<WordREG<float>, FnHypTan>::Type_t ret;
    ret.setup( jit_ins_pow_f32( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  typename BinaryReturn<WordREG<float>, WordREG<float>, FnArcTan2>::Type_t
  atan2(const WordREG<float>& s1, const WordREG<float>& s2)
  {
    typename UnaryReturn<WordREG<float>, FnArcTan2>::Type_t ret;
    ret.setup( jit_ins_atan2_f32( s1.get_val() , s2.get_val() ) );
    return ret;
  }




  typename UnaryReturn<WordREG<double>, FnSin>::Type_t
  sin(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnSin>::Type_t ret;
    ret.setup( jit_ins_sin_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnArcCos>::Type_t
  acos(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnArcCos>::Type_t ret;
    ret.setup( jit_ins_acos_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnArcSin>::Type_t
  asin(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnArcSin>::Type_t ret;
    ret.setup( jit_ins_asin_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnArcTan>::Type_t
  atan(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnArcTan>::Type_t ret;
    ret.setup( jit_ins_atan_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnCos>::Type_t
  cos(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnCos>::Type_t ret;
    ret.setup( jit_ins_cos_f64( s1.get_val() ) );
    return ret;
  }
  
  typename UnaryReturn<WordREG<double>, FnHypCos>::Type_t
  cosh(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnHypCos>::Type_t ret;
    ret.setup( jit_ins_cosh_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnExp>::Type_t
  exp(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnExp>::Type_t ret;
    ret.setup( jit_ins_exp_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnLog>::Type_t
  log(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnLog>::Type_t ret;
    ret.setup( jit_ins_log_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnLog10>::Type_t
  log10(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnLog10>::Type_t ret;
    ret.setup( jit_ins_log10_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnHypSin>::Type_t
  sinh(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnHypSin>::Type_t ret;
    ret.setup( jit_ins_sinh_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnTan>::Type_t
  tan(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnTan>::Type_t ret;
    ret.setup( jit_ins_tan_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnHypTan>::Type_t
  tanh(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnHypTan>::Type_t ret;
    ret.setup( jit_ins_tanh_f64( s1.get_val() ) );
    return ret;
  }


  typename BinaryReturn<WordREG<double>, WordREG<double>, FnPow>::Type_t
  pow(const WordREG<double>& s1, const WordREG<double>& s2)
  {
    typename UnaryReturn<WordREG<double>, FnHypTan>::Type_t ret;
    ret.setup( jit_ins_pow_f64( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  typename BinaryReturn<WordREG<double>, WordREG<double>, FnArcTan2>::Type_t
  atan2(const WordREG<double>& s1, const WordREG<double>& s2)
  {
    typename UnaryReturn<WordREG<double>, FnArcTan2>::Type_t ret;
    ret.setup( jit_ins_atan2_f64( s1.get_val() , s2.get_val() ) );
    return ret;
  }




}
