#include "qdp.h"

namespace QDP 
{

  void zero_rep(WordREG<double>& dest) 
  {
    dest.setup(llvm_create_value(0.0));
  }

  void zero_rep(WordREG<jit_half_t>& dest) 
  {
    dest.setup(llvm_create_value(0.0));
  }

  void zero_rep(WordREG<float>& dest) 
  {
    dest.setup(llvm_create_value(0.0));
  }

  void zero_rep(WordREG<int>& dest)
  {
    dest.setup(llvm_create_value(0));
  }



#if defined (QDP_BACKEND_AVX)
  void zero_rep(WordVecREG<double>& dest) 
  {
    dest.setup(llvm_fill_vector(llvm_create_value(0.0)));
  }

  void zero_rep(WordVecREG<jit_half_t>& dest) 
  {
    dest.setup(llvm_fill_vector(llvm_create_value(0.0)));
  }

  void zero_rep(WordVecREG<float>& dest) 
  {
    dest.setup(llvm_fill_vector(llvm_create_value(0.0)));
  }

  void zero_rep(WordVecREG<int>& dest)
  {
    dest.setup(llvm_fill_vector(llvm_create_value(0)));
  }
#endif
  
  

  
  // scalar f32
  //
  
  typename UnaryReturn<WordREG<float>, FnCeil>::Type_t 
  ceil(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnSin>::Type_t ret;
    ret.setup( llvm_ceil_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnFloor>::Type_t 
  floor(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnSin>::Type_t ret;
    ret.setup( llvm_floor_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnFabs>::Type_t 
  fabs(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnSin>::Type_t ret;
    ret.setup( llvm_fabs_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnSqrt>::Type_t 
  sqrt(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnSin>::Type_t ret;
    ret.setup( llvm_sqrt_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnIsFinite>::Type_t 
  isfinite(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnIsFinite>::Type_t ret;
    ret.setup( llvm_isfinite_f32( s1.get_val() ) );
    return ret;
  }


  typename UnaryReturn<WordREG<float>, FnArcCos>::Type_t
  acos(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnArcCos>::Type_t ret;
    ret.setup( llvm_acos_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnArcSin>::Type_t 
  asin(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnArcSin>::Type_t ret;
    ret.setup( llvm_asin_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnArcTan>::Type_t
  atan(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnArcTan>::Type_t ret;
    ret.setup( llvm_atan_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnCos>::Type_t
  cos(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnCos>::Type_t ret;
    ret.setup( llvm_cos_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnHypCos>::Type_t
  cosh(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnHypCos>::Type_t ret;
    ret.setup( llvm_cosh_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnExp>::Type_t
  exp(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnExp>::Type_t ret;
    ret.setup( llvm_exp_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnLog>::Type_t
  log(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnLog>::Type_t ret;
    ret.setup( llvm_log_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnLog10>::Type_t
  log10(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnLog10>::Type_t ret;
    ret.setup( llvm_log10_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnSin>::Type_t
  sin(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnSin>::Type_t ret;
    ret.setup( llvm_sin_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnHypSin>::Type_t
  sinh(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnHypSin>::Type_t ret;
    ret.setup( llvm_sinh_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnTan>::Type_t
  tan(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnTan>::Type_t ret;
    ret.setup( llvm_tan_f32( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<float>, FnHypTan>::Type_t
  tanh(const WordREG<float>& s1)
  {
    typename UnaryReturn<WordREG<float>, FnHypTan>::Type_t ret;
    ret.setup( llvm_tanh_f32( s1.get_val() ) );
    return ret;
  }



  // scalar f64
  //

  typename UnaryReturn<WordREG<double>, FnCeil>::Type_t 
  ceil(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnSin>::Type_t ret;
    ret.setup( llvm_ceil_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnFloor>::Type_t 
  floor(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnSin>::Type_t ret;
    ret.setup( llvm_floor_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnFabs>::Type_t 
  fabs(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnSin>::Type_t ret;
    ret.setup( llvm_fabs_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnSqrt>::Type_t 
  sqrt(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnSin>::Type_t ret;
    ret.setup( llvm_sqrt_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnIsFinite>::Type_t 
  isfinite(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnSin>::Type_t ret;
    ret.setup( llvm_isfinite_f64( s1.get_val() ) );
    return ret;
  }


  typename UnaryReturn<WordREG<double>, FnSin>::Type_t
  sin(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnSin>::Type_t ret;
    ret.setup( llvm_sin_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnArcCos>::Type_t
  acos(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnArcCos>::Type_t ret;
    ret.setup( llvm_acos_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnArcSin>::Type_t
  asin(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnArcSin>::Type_t ret;
    ret.setup( llvm_asin_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnArcTan>::Type_t
  atan(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnArcTan>::Type_t ret;
    ret.setup( llvm_atan_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnCos>::Type_t
  cos(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnCos>::Type_t ret;
    ret.setup( llvm_cos_f64( s1.get_val() ) );
    return ret;
  }
  
  typename UnaryReturn<WordREG<double>, FnHypCos>::Type_t
  cosh(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnHypCos>::Type_t ret;
    ret.setup( llvm_cosh_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnExp>::Type_t
  exp(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnExp>::Type_t ret;
    ret.setup( llvm_exp_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnLog>::Type_t
  log(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnLog>::Type_t ret;
    ret.setup( llvm_log_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnLog10>::Type_t
  log10(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnLog10>::Type_t ret;
    ret.setup( llvm_log10_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnHypSin>::Type_t
  sinh(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnHypSin>::Type_t ret;
    ret.setup( llvm_sinh_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnTan>::Type_t
  tan(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnTan>::Type_t ret;
    ret.setup( llvm_tan_f64( s1.get_val() ) );
    return ret;
  }

  typename UnaryReturn<WordREG<double>, FnHypTan>::Type_t
  tanh(const WordREG<double>& s1)
  {
    typename UnaryReturn<WordREG<double>, FnHypTan>::Type_t ret;
    ret.setup( llvm_tanh_f64( s1.get_val() ) );
    return ret;
  }

  // scalar 2 argument
  //
  // f32 f32

  typename BinaryReturn<WordREG<float>, WordREG<float>, FnPow>::Type_t
  pow(const WordREG<float>& s1, const WordREG<float>& s2)
  {
    typename BinaryReturn<WordREG<float>, WordREG<float>, FnPow>::Type_t ret;
    ret.setup( llvm_pow_f32( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  typename BinaryReturn<WordREG<float>, WordREG<float>, FnArcTan2>::Type_t
  atan2(const WordREG<float>& s1, const WordREG<float>& s2)
  {
    typename BinaryReturn<WordREG<float>, WordREG<float>, FnArcTan2>::Type_t ret;
    ret.setup( llvm_atan2_f32( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  // f64 f64
  
  typename BinaryReturn<WordREG<double>, WordREG<double>, FnPow>::Type_t
  pow(const WordREG<double>& s1, const WordREG<double>& s2)
  {
    typename BinaryReturn<WordREG<double>, WordREG<double>, FnPow>::Type_t ret;
    ret.setup( llvm_pow_f64( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  typename BinaryReturn<WordREG<double>, WordREG<double>, FnArcTan2>::Type_t
  atan2(const WordREG<double>& s1, const WordREG<double>& s2)
  {
    typename BinaryReturn<WordREG<double>, WordREG<double>, FnArcTan2>::Type_t ret;
    ret.setup( llvm_atan2_f64( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  // f64 f32

  typename BinaryReturn<WordREG<double>, WordREG<float>, FnPow>::Type_t
  pow(const WordREG<double>& s1, const WordREG<float>& s2)
  {
    typename BinaryReturn<WordREG<double>, WordREG<float>, FnPow>::Type_t ret;
    ret.setup( llvm_pow_f64( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  typename BinaryReturn<WordREG<double>, WordREG<float>, FnArcTan2>::Type_t
  atan2(const WordREG<double>& s1, const WordREG<float>& s2)
  {
    typename BinaryReturn<WordREG<double>, WordREG<float>, FnArcTan2>::Type_t ret;
    ret.setup( llvm_atan2_f64( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  // f32 f64
  
  typename BinaryReturn<WordREG<float>, WordREG<double>, FnPow>::Type_t
  pow(const WordREG<float>& s1, const WordREG<double>& s2)
  {
    typename BinaryReturn<WordREG<float>, WordREG<double>, FnPow>::Type_t ret;
    ret.setup( llvm_pow_f32( s1.get_val() , s2.get_val() ) );
    return ret;
  }

  typename BinaryReturn<WordREG<float>, WordREG<double>, FnArcTan2>::Type_t
  atan2(const WordREG<float>& s1, const WordREG<double>& s2)
  {
    typename BinaryReturn<WordREG<float>, WordREG<double>, FnArcTan2>::Type_t ret;
    ret.setup( llvm_atan2_f32( s1.get_val() , s2.get_val() ) );
    return ret;
  }



#if defined (QDP_BACKEND_AVX)
  // vector f32
  //
  typename UnaryReturn<WordVecREG<float>, FnCeil>::Type_t 
  ceil(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnSin>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_ceil_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<float>, FnFloor>::Type_t 
  floor(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnSin>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_floor_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<float>, FnFabs>::Type_t 
  fabs(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnSin>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_fabs_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<float>, FnSqrt>::Type_t 
  sqrt(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnSin>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_sqrt_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<float>, FnIsFinite>::Type_t 
  isfinite(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnIsFinite>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_isfinite_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }


  typename UnaryReturn<WordVecREG<float>, FnArcCos>::Type_t
  acos(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnArcCos>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_acos_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<float>, FnArcSin>::Type_t 
  asin(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnArcSin>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_asin_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<float>, FnArcTan>::Type_t
  atan(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnArcTan>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_atan_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<float>, FnCos>::Type_t
  cos(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnCos>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_cos_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<float>, FnHypCos>::Type_t
  cosh(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnHypCos>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_cosh_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<float>, FnExp>::Type_t
  exp(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnExp>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_exp_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<float>, FnLog>::Type_t
  log(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnLog>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_log_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<float>, FnLog10>::Type_t
  log10(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnLog10>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_log10_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<float>, FnSin>::Type_t
  sin(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnSin>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_sin_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<float>, FnHypSin>::Type_t
  sinh(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnHypSin>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_sinh_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<float>, FnTan>::Type_t
  tan(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnTan>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_tan_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<float>, FnHypTan>::Type_t
  tanh(const WordVecREG<float>& s1)
  {
    typename UnaryReturn<WordVecREG<float>, FnHypTan>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_tanh_f32( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  // vector f64
  //
  
    typename UnaryReturn<WordVecREG<double>, FnCeil>::Type_t 
  ceil(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnSin>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_ceil_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<double>, FnFloor>::Type_t 
  floor(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnSin>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_floor_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<double>, FnFabs>::Type_t 
  fabs(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnSin>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_fabs_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<double>, FnSqrt>::Type_t 
  sqrt(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnSin>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_sqrt_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<double>, FnIsFinite>::Type_t 
  isfinite(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnIsFinite>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_isfinite_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }


  typename UnaryReturn<WordVecREG<double>, FnArcCos>::Type_t
  acos(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnArcCos>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_acos_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<double>, FnArcSin>::Type_t 
  asin(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnArcSin>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_asin_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<double>, FnArcTan>::Type_t
  atan(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnArcTan>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_atan_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<double>, FnCos>::Type_t
  cos(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnCos>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_cos_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<double>, FnHypCos>::Type_t
  cosh(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnHypCos>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_cosh_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<double>, FnExp>::Type_t
  exp(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnExp>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_exp_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<double>, FnLog>::Type_t
  log(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnLog>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_log_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<double>, FnLog10>::Type_t
  log10(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnLog10>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_log10_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<double>, FnSin>::Type_t
  sin(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnSin>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_sin_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<double>, FnHypSin>::Type_t
  sinh(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnHypSin>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_sinh_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<double>, FnTan>::Type_t
  tan(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnTan>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_tan_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename UnaryReturn<WordVecREG<double>, FnHypTan>::Type_t
  tanh(const WordVecREG<double>& s1)
  {
    typename UnaryReturn<WordVecREG<double>, FnHypTan>::Type_t ret;
    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_tanh_f64( llvm_extract_element( s1.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }



  typename BinaryReturn<WordVecREG<float>, WordREG<float>, FnPow>::Type_t
  pow(const WordVecREG<float>& s1, const WordREG<float>& s2)
  {
    typename BinaryReturn<WordVecREG<float>, WordREG<float>, FnPow>::Type_t ret;

    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_pow_f32( llvm_extract_element( s1.get_val() , i ) , s2.get_val() ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename BinaryReturn<WordVecREG<double>, WordREG<double>, FnPow>::Type_t
  pow(const WordVecREG<double>& s1, const WordREG<double>& s2)
  {
    typename BinaryReturn<WordVecREG<double>, WordREG<double>, FnPow>::Type_t ret;

    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_pow_f64( llvm_extract_element( s1.get_val() , i ) , s2.get_val() ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename BinaryReturn<WordVecREG<double>, WordREG<float>, FnPow>::Type_t
  pow(const WordVecREG<double>& s1, const WordREG<float>& s2)
  {
    typename BinaryReturn<WordVecREG<double>, WordREG<float>, FnPow>::Type_t ret;

    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_pow_f64( llvm_extract_element( s1.get_val() , i ) , s2.get_val() ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }

  typename BinaryReturn<WordVecREG<float>, WordREG<double>, FnPow>::Type_t
  pow(const WordVecREG<float>& s1, const WordREG<double>& s2)
  {
    typename BinaryReturn<WordVecREG<float>, WordREG<double>, FnPow>::Type_t ret;

    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_pow_f32( llvm_extract_element( s1.get_val() , i ) , s2.get_val() ) , i );
      }
    ret.setup( ret_val );
    return ret;
  }


  typename BinaryReturn<WordVecREG<float>, WordVecREG<float>, FnArcTan2>::Type_t
  atan2(const WordVecREG<float>& s1, const WordVecREG<float>& s2)
  {
    typename BinaryReturn<WordVecREG<float>, WordVecREG<float>, FnArcTan2>::Type_t ret;

    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_atan2_f32( llvm_extract_element( s1.get_val() , i ) ,
								 llvm_extract_element( s2.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );

    return ret;
  }


  typename BinaryReturn<WordVecREG<double>, WordVecREG<double>, FnArcTan2>::Type_t
  atan2(const WordVecREG<double>& s1, const WordVecREG<double>& s2)
  {
    typename BinaryReturn<WordVecREG<double>, WordVecREG<double>, FnArcTan2>::Type_t ret;

    llvm::Value* ret_val = llvm_get_zero_vector( s1.get_val() );
    for( int i = 0 ; i < Layout::virtualNodeNumber() ; ++i )
      {
	ret_val = llvm_insert_element( ret_val , llvm_atan2_f64( llvm_extract_element( s1.get_val() , i ) ,
								 llvm_extract_element( s2.get_val() , i ) ) , i );
      }
    ret.setup( ret_val );

    return ret;
  }

  
  
#endif

  

}
