#include "qdp.h"

namespace QDP 
{
  typename UnaryReturn<WordJIT<float>, FnCeil>::Type_t 
  ceil(WordJIT<float> s1)
  {
    auto d = unary_return<FnCeil>( s1 );
    d.store( llvm_ceil_f32( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<float>, FnFloor>::Type_t 
  floor(WordJIT<float> s1)
  {
    auto d = unary_return<FnFloor>( s1 );
    d.store( llvm_floor_f32( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<float>, FnFabs>::Type_t 
  fabs(WordJIT<float> s1)
  {
    auto d = unary_return<FnFabs>( s1 );
    d.store( llvm_fabs_f32( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<float>, FnSqrt>::Type_t 
  sqrt(WordJIT<float> s1)
  {
    auto d = unary_return<FnSqrt>( s1 );
    d.store( llvm_sqrt_f32( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<float>, FnIsFinite>::Type_t 
  isfinite(WordJIT<float> s1)
  {
    auto d = unary_return<FnIsFinite>( s1 );
    d.store( llvm_isfinite_f32( s1.get_val() ) );
    return d;
  }


  typename UnaryReturn<WordJIT<float>, FnArcCos>::Type_t
  acos(WordJIT<float> s1)
  {
    auto d = unary_return<FnArcCos>( s1 );
    d.store( llvm_acos_f32( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<float>, FnArcSin>::Type_t 
  asin(WordJIT<float> s1)
  {
    auto d = unary_return<FnArcSin>( s1 );
    d.store( llvm_asin_f32( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<float>, FnArcTan>::Type_t
  atan(WordJIT<float> s1)
  {
    auto d = unary_return<FnArcTan>( s1 );
    d.store( llvm_atan_f32( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<float>, FnCos>::Type_t
  cos(WordJIT<float> s1)
  {
    auto d = unary_return<FnCos>( s1 );
    d.store( llvm_cos_f32( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<float>, FnHypCos>::Type_t
  cosh(WordJIT<float> s1)
  {
    auto d = unary_return<FnHypCos>( s1 );
    d.store( llvm_cosh_f32( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<float>, FnExp>::Type_t
  exp(WordJIT<float> s1)
  {
    auto d = unary_return<FnExp>( s1 );
    d.store( llvm_exp_f32( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<float>, FnLog>::Type_t
  log(WordJIT<float> s1)
  {
    auto d = unary_return<FnLog>( s1 );
    d.store( llvm_log_f32( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<float>, FnLog10>::Type_t
  log10(WordJIT<float> s1)
  {
    auto d = unary_return<FnLog10>( s1 );
    d.store( llvm_log10_f32( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<float>, FnSin>::Type_t
  sin(WordJIT<float> s1)
  {
    auto d = unary_return<FnSin>( s1 );
    d.store( llvm_sin_f32( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<float>, FnHypSin>::Type_t
  sinh(WordJIT<float> s1)
  {
    auto d = unary_return<FnHypSin>( s1 );
    d.store( llvm_sinh_f32( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<float>, FnTan>::Type_t
  tan(WordJIT<float> s1)
  {
    auto d = unary_return<FnTan>( s1 );
    d.store( llvm_tan_f32( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<float>, FnHypTan>::Type_t
  tanh(WordJIT<float> s1)
  {
    auto d = unary_return<FnHypTan>( s1 );
    d.store( llvm_tanh_f32( s1.get_val() ) );
    return d;
  }



    typename UnaryReturn<WordJIT<double>, FnCeil>::Type_t 
  ceil(WordJIT<double> s1)
  {
    auto d = unary_return<FnCeil>( s1 );
    d.store( llvm_ceil_f64( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<double>, FnFloor>::Type_t 
  floor(WordJIT<double> s1)
  {
    auto d = unary_return<FnFloor>( s1 );
    d.store( llvm_floor_f64( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<double>, FnFabs>::Type_t 
  fabs(WordJIT<double> s1)
  {
    auto d = unary_return<FnFabs>( s1 );
    d.store( llvm_fabs_f64( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<double>, FnSqrt>::Type_t 
  sqrt(WordJIT<double> s1)
  {
    auto d = unary_return<FnSqrt>( s1 );
    d.store( llvm_sqrt_f64( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<double>, FnIsFinite>::Type_t 
  isfinite(WordJIT<double> s1)
  {
    auto d = unary_return<FnIsFinite>( s1 );
    d.store( llvm_isfinite_f64( s1.get_val() ) );
    return d;
  }


  typename UnaryReturn<WordJIT<double>, FnArcCos>::Type_t
  acos(WordJIT<double> s1)
  {
    auto d = unary_return<FnArcCos>( s1 );
    d.store( llvm_acos_f64( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<double>, FnArcSin>::Type_t 
  asin(WordJIT<double> s1)
  {
    auto d = unary_return<FnArcSin>( s1 );
    d.store( llvm_asin_f64( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<double>, FnArcTan>::Type_t
  atan(WordJIT<double> s1)
  {
    auto d = unary_return<FnArcTan>( s1 );
    d.store( llvm_atan_f64( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<double>, FnCos>::Type_t
  cos(WordJIT<double> s1)
  {
    auto d = unary_return<FnCos>( s1 );
    d.store( llvm_cos_f64( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<double>, FnHypCos>::Type_t
  cosh(WordJIT<double> s1)
  {
    auto d = unary_return<FnHypCos>( s1 );
    d.store( llvm_cosh_f64( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<double>, FnExp>::Type_t
  exp(WordJIT<double> s1)
  {
    auto d = unary_return<FnExp>( s1 );
    d.store( llvm_exp_f64( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<double>, FnLog>::Type_t
  log(WordJIT<double> s1)
  {
    auto d = unary_return<FnLog>( s1 );
    d.store( llvm_log_f64( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<double>, FnLog10>::Type_t
  log10(WordJIT<double> s1)
  {
    auto d = unary_return<FnLog10>( s1 );
    d.store( llvm_log10_f64( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<double>, FnSin>::Type_t
  sin(WordJIT<double> s1)
  {
    auto d = unary_return<FnSin>( s1 );
    d.store( llvm_sin_f64( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<double>, FnHypSin>::Type_t
  sinh(WordJIT<double> s1)
  {
    auto d = unary_return<FnHypSin>( s1 );
    d.store( llvm_sinh_f64( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<double>, FnTan>::Type_t
  tan(WordJIT<double> s1)
  {
    auto d = unary_return<FnTan>( s1 );
    d.store( llvm_tan_f64( s1.get_val() ) );
    return d;
  }

  typename UnaryReturn<WordJIT<double>, FnHypTan>::Type_t
  tanh(WordJIT<double> s1)
  {
    auto d = unary_return<FnHypTan>( s1 );
    d.store( llvm_tanh_f64( s1.get_val() ) );
    return d;
  }




  typename BinaryReturn<WordJIT<float>, WordJIT<float>, FnPow>::Type_t
  pow(WordJIT<float> s1, WordJIT<float> s2)
  {
    auto ret = binary_return<FnPow>( s1 , s2 );
    ret.store( llvm_pow_f32( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  typename BinaryReturn<WordJIT<float>, WordJIT<float>, FnArcTan2>::Type_t
  atan2(WordJIT<float> s1, WordJIT<float> s2)
  {
    auto ret = binary_return<FnArcTan2>( s1 , s2 );
    ret.store( llvm_atan2_f32( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  
  typename BinaryReturn<WordJIT<double>, WordJIT<double>, FnPow>::Type_t
  pow(WordJIT<double> s1, WordJIT<double> s2)
  {
    auto ret = binary_return<FnPow>( s1 , s2 );
    ret.store( llvm_pow_f64( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  typename BinaryReturn<WordJIT<double>, WordJIT<double>, FnArcTan2>::Type_t
  atan2(WordJIT<double> s1, WordJIT<double> s2)
  {
    auto ret = binary_return<FnArcTan2>( s1 , s2 );
    ret.store( llvm_atan2_f64( s1.get_val() , s2.get_val() ) );
    return ret;
  }


  typename BinaryReturn<WordJIT<double>, WordJIT<float>, FnPow>::Type_t
  pow(WordJIT<double> s1, WordJIT<float> s2)
  {
    auto ret = binary_return<FnPow>( s1 , s2 );
    ret.store( llvm_pow_f64( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  typename BinaryReturn<WordJIT<double>, WordJIT<float>, FnArcTan2>::Type_t
  atan2(WordJIT<double> s1, WordJIT<float> s2)
  {
    auto ret = binary_return<FnArcTan2>( s1 , s2 );
    ret.store( llvm_atan2_f64( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  
  typename BinaryReturn<WordJIT<float>, WordJIT<double>, FnPow>::Type_t
  pow(WordJIT<float> s1, WordJIT<double> s2)
  {
    auto ret = binary_return<FnPow>( s1 , s2 );
    ret.store( llvm_pow_f32( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  typename BinaryReturn<WordJIT<float>, WordJIT<double>, FnArcTan2>::Type_t
  atan2(WordJIT<float> s1, WordJIT<double> s2)
  {
    auto ret = binary_return<FnArcTan2>( s1 , s2 );
    ret.store( llvm_atan2_f32( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  

}

