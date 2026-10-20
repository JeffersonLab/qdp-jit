#ifndef QDP_PRECISION_H
#define QDP_PRECISION_H

#include <qdp_config.h>

#if ! defined(BASE_PRECISION)
#define BASE_PRECISION 32
#endif

struct jit_half_t
{
  unsigned char a[2];
};

#include <cmath>
#include <cstdint>
#include <cstring>

inline double qdp_half_bits_to_double( uint16_t h )
{
  const int sign = h >> 15;
  const int exp  = ( h >> 10 ) & 0x1f;
  const int man  = h & 0x3ff;
  double v;
  if ( exp == 0 )
    v = std::ldexp( (double)man , -24 );
  else if ( exp == 31 )
    v = man ? NAN : INFINITY;
  else
    v = std::ldexp( (double)( man | 0x400 ) , exp - 25 );
  return sign ? -v : v;
}

inline uint16_t qdp_double_to_half_bits( double x )
{
  const uint16_t sign = std::signbit( x ) ? 0x8000 : 0;
  if ( std::isnan( x ) )
    return 0x7e00;
  if ( std::isinf( x ) )
    return sign | 0x7c00;
  double a = std::fabs( x );
  if ( a == 0.0 )
    return sign;
  int e;
  std::frexp( a , &e );
  int ex = e - 1;
  if ( ex < -14 )
    {
      double m = std::nearbyint( std::ldexp( a , 24 ) );
      return sign | (uint16_t)m;
    }
  double m = std::nearbyint( std::ldexp( a , 10 - ex ) );
  if ( m == 2048.0 )
    {
      m = 1024.0;
      ++ex;
    }
  if ( ex > 15 )
    return sign | 0x7c00;
  return sign | (uint16_t)( ( ex + 15 ) << 10 ) | (uint16_t)( (int)m - 1024 );
}

inline jit_half_t qdp_to_half( double x )
{
  uint16_t b = qdp_double_to_half_bits( x );
  jit_half_t h;
  std::memcpy( &h , &b , 2 );
  return h;
}

inline double qdp_from_half( jit_half_t h )
{
  uint16_t b;
  std::memcpy( &b , &h , 2 );
  return qdp_half_bits_to_double( b );
}

typedef int       INTEGER32;
typedef float     REAL32;
typedef double    REAL64;
typedef bool      LOGICAL;
typedef jit_half_t REAL16;

#if BASE_PRECISION == 32
typedef REAL32    REAL;
typedef REAL64    DOUBLE;


#elif BASE_PRECISION == 64
typedef REAL64    REAL;
typedef REAL64    DOUBLE;


#else
#error "Unknown BASE_PRECISION"
#endif

#endif
