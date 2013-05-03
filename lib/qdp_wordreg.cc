#include "qdp.h"

namespace QDP 
{

#if 0
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
#endif



}
