#ifndef QDP_TRAITS_H
#define QDP_TRAITS_H


namespace QDP {

template<class T>       struct JITType;
template<class T>       struct BASEType;
template<class T,int N> struct GetLimit;

template<> struct JITType<int>    { typedef int    Type_t; };
template<> struct JITType<jit_half_t>  { typedef jit_half_t  Type_t; };
template<> struct JITType<float>  { typedef float  Type_t; };
template<> struct JITType<double> { typedef double Type_t; };
template<> struct JITType<bool>   { typedef bool   Type_t; };

template<> struct BASEType<int>    { typedef int    Type_t; };
template<> struct BASEType<jit_half_t>  { typedef jit_half_t  Type_t; };
template<> struct BASEType<float>  { typedef float  Type_t; };
template<> struct BASEType<double> { typedef double Type_t; };
template<> struct BASEType<bool>   { typedef bool   Type_t; };

template<class T>
struct QDPSubTypeTrait {};

  template<class T> struct WordType;

  template<> struct WordType<jit_half_t>  { typedef jit_half_t  Type_t; };
  template<> struct WordType<float>  { typedef float  Type_t; };
  template<> struct WordType<double> { typedef double Type_t; };
  template<> struct WordType<int>    { typedef int    Type_t; };
  template<> struct WordType<bool>   { typedef bool   Type_t; };



template<class T>
struct SinglePrecType
{
  typedef T Type_t; // This should never be instantiated as such
};


template<class T>
struct DoublePrecType
{
  typedef T Type_t; // This should never be instantiated as such
};


template<>
struct SinglePrecType<REAL32>
{
  typedef REAL32 Type_t;
};


template<>
struct SinglePrecType<REAL64>
{
  typedef REAL32 Type_t;
};


template<>
struct DoublePrecType<REAL32>
{
  typedef REAL64 Type_t;
};


template<>
struct DoublePrecType<REAL64>
{
  typedef REAL64 Type_t;
};


  template<typename T> 
  struct DoublePrecType< multi1d< T > >
  {
    typedef multi1d< typename DoublePrecType< T >::Type_t > Type_t;
  };



template<class T>
struct SimpleScalar {};



template<class T>
struct InternalScalar {};



template<class T>
struct PrimitiveScalar {};



template<class T>
struct LatticeScalar {};



template<class T>
struct RealScalar {};



template<class T>
struct NoComplex {};



struct Zero {};


extern Zero zero;







} // namespace QDP

#endif

