// -*- C++ -*-

/*! @file
 * @brief Traits classes
 * 
 * Traits classes needed internally
 */

#ifndef QDP_TRAITS_H
#define QDP_TRAITS_H

namespace QDP {

  //template<class T>       struct SignedType;
template<class T>       struct JITContainerType;
template<class T>       struct WordSize;
template<class T,int N> struct GetLimit;

  // template<> struct SignedType<float> { typedef float Type_t; }
  // template<> struct SignedType<double> { typedef double Type_t; }
  // template<> struct SignedType<unsigned int> { typedef int Type_t; }
  // template<> struct SignedType<bool> { typedef bool Type_t; }

template<> struct WordSize< float > { enum { Size = sizeof(float) }; };
template<> struct WordSize< double > { enum { Size = sizeof(double) }; };
template<> struct WordSize< int > { enum { Size = sizeof(int) }; };
  //template<> struct WordSize< unsigned int > { enum { Size = sizeof(unsigned int) }; };
template<> struct WordSize< bool > { enum { Size = sizeof(bool) }; };

  // GetLimit extracts the size of the specified QDP type level

  template<class T> 
  struct GetLimit<T,0>
  {
    enum { Limit_v = T::ThisSize };
  };
  
  template<class T,int N> 
  struct GetLimit
  {
    enum { Limit_v = GetLimit<typename T::Sub_t,N-1>::Limit_v };
  };




template<> struct JITContainerType<int>  { typedef int  Type_t; };
template<> struct JITContainerType<float>  { typedef float  Type_t; };
template<> struct JITContainerType<double> { typedef double  Type_t; };
template<> struct JITContainerType<bool> { typedef bool  Type_t; };


// template< template<class> class T, class T2> 
// struct WordSize< T<T2> >
// {
//   enum { Size = WordSize<T2>::Size };
// };

// template<class T>
// struct WordSize {
//   enum { Size=sizeof(T) };
// };


//-----------------------------------------------------------------------------
// Traits class for returning the subset-ted class name of a outer grid class
//-----------------------------------------------------------------------------

template<class T>
struct QDPSubTypeTrait {};

//-----------------------------------------------------------------------------
// Traits classes to support operations of simple scalars (floating constants, 
// etc.) on QDPTypes
//-----------------------------------------------------------------------------

//! Find the underlying word type of a field
template<class T>
struct WordType
{
  typedef T  Type_t;
};

  // template<>
  // template<class T>
  // struct WordType<Reference<T> >
  // {
  //   typedef typename WordType<T>::Type_t  Type_t;
  // };




//-----------------------------------------------------------------------------
// Traits Classes to support getting fixed precision versions of floating 
// precision classes
// ----------------------------------------------------------------------------
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

// Now we need to specialise to the bit whose precisions float
// The single prec types for both REAL32 and REAL64 are REAL32
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

// The Double prec types for both REAL32 and REAL64 are REAL64
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


//-----------------------------------------------------------------------------
// Constructors for simple word types
//-----------------------------------------------------------------------------

//! Construct simple word type. Default behavior is empty
template<class T>
struct SimpleScalar {};


//! Construct simple word type used at some level within primitives
template<class T>
struct InternalScalar {};


//! Makes a primitive scalar leaving grid alone
template<class T>
struct PrimitiveScalar {};


//! Makes a lattice scalar leaving primitive indices alone
template<class T>
struct LatticeScalar {};


//! Construct simple word type used at some level within primitives
template<class T>
struct RealScalar {};


//! Construct primitive type of input but always RScalar complex type
template<class T>
struct NoComplex {};


//! Simple zero tag
struct Zero {};

//! Put zero in some unnamed space
namespace {
 Zero zero;
}

} // namespace QDP

#endif

