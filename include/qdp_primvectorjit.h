// -*- C++ -*-

/*! \file
 * \brief Primitive Vector
 */


#ifndef QDP_PRIMVECTORJIT_H
#define QDP_PRIMVECTORJIT_H

namespace QDP {


//-------------------------------------------------------------------------------------
/*! \addtogroup primvector Vector primitive
 * \ingroup fiber
 *
 * Primitive type that transforms like a vector
 *
 * @{
 */

//! Primitive Vector class
/*!
 * All vector classes inherit this class
 * NOTE: For efficiency, there can be no virtual methods, so the data
 * portion is a part of the generic class, hence it is called a domain
 * and not a category
 */
template <class T, int N, template<class,int> class C> class PVectorJIT: public BaseJIT<T,N>
{
public:
  typedef C<T,N>  CC;

  //! PVectorJIT = PVectorJIT
  /*! Set equal to another PVectorJIT */
  //template<class T1>
  // inline
  // CC& assign(const typename REGType< C<T,N> >::Type_t& rhs) 
  //   {
  //     for(int i=0; i < N; ++i)
  // 	elem(i) = rhs.elem(i);

  //     return static_cast<CC&>(*this);
  //   }

  //! PVectorJIT = PVectorJIT
  /*! Set equal to another PVectorJIT */
  // template<class T1>
  // inline
  // CC& operator=(const typename REGType< C<T1,N> >::Type_t& rhs) 
  //   {
  //     return assign(rhs);
  //   }

  //! PVectorJIT += PVectorJIT
  // template<class T1>
  // inline
  // CC& operator+=(const typename REGType< C<T1,N> >::Type_t& rhs) 
  //   {
  //     for(int i=0; i < N; ++i)
  // 	elem(i) += rhs.elem(i);

  //     return static_cast<CC&>(*this);
  //   }

  // //! PVectorJIT -= PVectorJIT
  // template<class T1>
  // inline
  // CC& operator-=(const typename REGType< C<T1,N> >::Type_t& rhs) 
  //   {
  //     for(int i=0; i < N; ++i)
  // 	elem(i) -= rhs.elem(i);

  //     return static_cast<CC&>(*this);
  //   }

  //! PVectorJIT *= PScalarJIT
  template<class T1>
  inline
  CC& operator*=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) *= rhs.elem();

      return static_cast<CC&>(*this);
    }

  //! PVectorJIT /= PScalarJIT
  template<class T1>
  inline
  CC& operator/=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) /= rhs.elem();

      return static_cast<CC&>(*this);
    }


#if 0
  // NOTE: intentially avoid defining a copy constructor - let the compiler
  // generate one via the bit copy mechanism. This effectively achieves
  // the first form of the if below (QDP_USE_ARRAY_INITIALIZER) without having
  // to use that syntax which is not strictly legal in C++.

  //! Deep copy constructor
#if defined(QDP_USE_ARRAY_INITIALIZER)
  PVectorJIT(const PVectorJIT& a) : F(a.F) {}
#else
  /*! This is a copy form - legal but not necessarily efficient */
  PVectorJIT(const PVectorJIT& a)
    {
     
      for(int i=0; i < N; ++i)
	F[i] = a.F[i];
    }
#endif
#endif


public:
        T& elem(int i)       {return this->arrayF(i);}
  const T& elem(int i) const {return this->arrayF(i);}

  // T& elem(int i) {return JV<T,N>::getF()[i];}
  // const T& elem(int i) const {return JV<T,N>::getF()[i];}
};




//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

// Underlying word type
template<class T1, int N, template<class,int> class C>
struct WordType<PVectorJIT<T1,N,C> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

template<class T1, int N, template<class, int> class C> 
struct SinglePrecType< PVectorJIT<T1,N,C> >
{
  typedef PVectorJIT< typename SinglePrecType<T1>::Type_t, N, C> Type_t;
};

template<class T1, int N, template<class, int> class C> 
struct DoublePrecType< PVectorJIT<T1,N,C> >
{
  typedef PVectorJIT< typename DoublePrecType<T1>::Type_t, N, C> Type_t;
};

// Internally used scalars
template<class T, int N, template<class,int> class C>
struct InternalScalar<PVectorJIT<T,N,C> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive scalar leaving grid alone
template<class T, int N, template<class,int> class C>
struct PrimitiveScalar<PVectorJIT<T,N,C> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T, int N, template<class,int> class C>
struct LatticeScalar<PVectorJIT<T,N,C> > {
  typedef C<typename LatticeScalar<T>::Type_t, N>  Type_t;
};

//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PVectorJIT) -> PVectorJIT
template<class T1, int N, template<class,int> class C, class Op>
struct UnaryReturn<PVectorJIT<T1,N,C>, Op> {
  typedef C<typename UnaryReturn<T1, Op>::Type_t, N>  Type_t;
};
// Default binary(PScalarJIT,PVectorJIT) -> PVectorJIT
template<class T1, class T2, int N, template<class,int> class C, class Op>
struct BinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,N,C>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PMatrixJIT,PVectorJIT) -> PVectorJIT
template<class T1, class T2, int N, template<class,int> class C1, 
  template<class,int> class C2, class Op>
struct BinaryReturn<PMatrixJIT<T1,N,C1>, PVectorJIT<T2,N,C2>, Op> {
  typedef C2<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PVectorJIT,PScalarJIT) -> PVectorJIT
template<class T1, class T2, int N, template<class,int> class C, class Op>
struct BinaryReturn<PVectorJIT<T1,N,C>, PScalarJIT<T2>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PVectorJIT,PVectorJIT) -> PVectorJIT
template<class T1, class T2, int N, template<class,int> class C, class Op>
struct BinaryReturn<PVectorJIT<T1,N,C>, PVectorJIT<T2,N,C>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};


#if 0
template<class T1, class T2>
struct UnaryReturn<PScalarJIT<T2>, OpCast<T1> > {
  typedef PScalarJIT<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif


// Assignment is different
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorJIT<T1,N,C>, PVectorJIT<T2,N,C>, OpAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorJIT<T1,N,C>, PVectorJIT<T2,N,C>, OpAddAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorJIT<T1,N,C>, PVectorJIT<T2,N,C>, OpSubtractAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorJIT<T1,N,C>, PScalarJIT<T2>, OpMultiplyAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorJIT<T1,N,C>, PScalarJIT<T2>, OpDivideAssign > {
  typedef C<T1,N> &Type_t;
};
 



//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

/*! \addtogroup primvector */
/*! @{ */

// Primitive Vectors

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorJIT<T1,N,C>, OpUnaryPlus>::Type_t
operator+(const PVectorJIT<T1,N,C>& l)
{
  typename UnaryReturn<PVectorJIT<T1,N,C>, OpUnaryPlus>::Type_t  d(l.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = +l.elem(i);
  return d;
}


template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorJIT<T1,N,C>, OpUnaryMinus>::Type_t
operator-(const PVectorJIT<T1,N,C>& l)
{
  typename UnaryReturn<PVectorJIT<T1,N,C>, OpUnaryMinus>::Type_t  d(l.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = -l.elem(i);
  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVectorJIT<T1,N,C>, PVectorJIT<T2,N,C>, OpAdd>::Type_t
operator+(const PVectorJIT<T1,N,C>& l, const PVectorJIT<T2,N,C>& r)
{
  typename BinaryReturn<PVectorJIT<T1,N,C>, PVectorJIT<T2,N,C>, OpAdd>::Type_t  d(l.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) + r.elem(i);
  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVectorJIT<T1,N,C>, PVectorJIT<T2,N,C>, OpSubtract>::Type_t
operator-(const PVectorJIT<T1,N,C>& l, const PVectorJIT<T2,N,C>& r)
{
  typename BinaryReturn<PVectorJIT<T1,N,C>, PVectorJIT<T2,N,C>, OpSubtract>::Type_t  d(l.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) - r.elem(i);
  return d;
}


// PVectorJIT * PScalarJIT
template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVectorJIT<T1,N,C>, PScalarJIT<T2>, OpMultiply>::Type_t
operator*(const PVectorJIT<T1,N,C>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PVectorJIT<T1,N,C>, PScalarJIT<T2>, OpMultiply>::Type_t  d(l.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) * r.elem();
  return d;
}

// Optimized  PVectorJIT * adj(PScalarJIT)
template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVectorJIT<T1,N,C>, PScalarJIT<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(const PVectorJIT<T1,N,C>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PVectorJIT<T1,N,C>, PScalarJIT<T2>, OpMultiplyAdj>::Type_t  d(l.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = multiplyAdj(l.elem(i), r.elem());
  return d;
}


// PScalarJIT * PVectorJIT
template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,N,C>, OpMultiply>::Type_t
operator*(const PScalarJIT<T1>& l, const PVectorJIT<T2,N,C>& r)
{
  typename BinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,N,C>, OpMultiply>::Type_t  d(l.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem() * r.elem(i);
  return d;
}

// Optimized  adj(PScalarJIT) * PVectorJIT
template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,N,C>, OpAdjMultiply>::Type_t
adjMultiply(const PScalarJIT<T1>& l, const PVectorJIT<T2,N,C>& r)
{
  typename BinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,N,C>, OpAdjMultiply>::Type_t  d(l.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = adjMultiply(l.elem(), r.elem(i));
  return d;
}


// PMatrixJIT * PVectorJIT
template<class T1, class T2, int N, template<class,int> class C1, template<class,int> class C2>
inline typename BinaryReturn<PMatrixJIT<T1,N,C1>, PVectorJIT<T2,N,C2>, OpMultiply>::Type_t
operator*(const PMatrixJIT<T1,N,C1>& l, const PVectorJIT<T2,N,C2>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C1>, PVectorJIT<T2,N,C2>, OpMultiply>::Type_t  d(l.func());

  for(int i=0; i < N; ++i)
  {
    d.elem(i) = l.elem(i,0) * r.elem(0);
    for(int j=1; j < N; ++j)
      d.elem(i) += l.elem(i,j) * r.elem(j);
  }

  return d;
}

// Optimized  adj(PMatrixJIT)*PVectorJIT
template<class T1, class T2, int N, template<class,int> class C1, template<class,int> class C2>
inline typename BinaryReturn<PMatrixJIT<T1,N,C1>, PVectorJIT<T2,N,C2>, OpAdjMultiply>::Type_t
adjMultiply(const PMatrixJIT<T1,N,C1>& l, const PVectorJIT<T2,N,C2>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C1>, PVectorJIT<T2,N,C2>, OpAdjMultiply>::Type_t  d(l.func());

  for(int i=0; i < N; ++i)
  {
    d.elem(i) = adjMultiply(l.elem(0,i), r.elem(0));
    for(int j=1; j < N; ++j)
      d.elem(i) += adjMultiply(l.elem(j,i), r.elem(j));
  }

  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVectorJIT<T1,N,C>, PScalarJIT<T2>, OpDivide>::Type_t
operator/(const PVectorJIT<T1,N,C>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PVectorJIT<T1,N,C>, PScalarJIT<T2>, OpDivide>::Type_t  d(l.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) / r.elem();
  return d;
}



//! PVectorJIT = Re(PVectorJIT)
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorJIT<T,N,C>, FnReal>::Type_t
real(const PVectorJIT<T,N,C>& s1)
{
  typename UnaryReturn<PVectorJIT<T,N,C>, FnReal>::Type_t  d(s1.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = real(s1.elem(i));

  return d;
}


//! PVectorJIT = Im(PVectorJIT)
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorJIT<T,N,C>, FnImag>::Type_t
imag(const PVectorJIT<T,N,C>& s1)
{
  typename UnaryReturn<PVectorJIT<T,N,C>, FnImag>::Type_t  d(s1.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = imag(s1.elem(i));

  return d;
}


//! PVectorJIT<T> = (PVectorJIT<T> , PVectorJIT<T>)
template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVectorJIT<T1,N,C>, PVectorJIT<T2,N,C>, FnCmplx>::Type_t
cmplx(const PVectorJIT<T1,N,C>& s1, const PVectorJIT<T2,N,C>& s2)
{
  typename BinaryReturn<PVectorJIT<T1,N,C>, PVectorJIT<T2,N,C>, FnCmplx>::Type_t  d(s1.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = cmplx(s1.elem(i), s2.elem(i));

  return d;
}


//-----------------------------------------------------------------------------
// Functions
// Conjugate
template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorJIT<T1,N,C>, FnConjugate>::Type_t
conj(const PVectorJIT<T1,N,C>& l)
{
  typename UnaryReturn<PVectorJIT<T1,N,C>, FnConjugate>::Type_t  d(l.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = conj(l.elem(i));

  return d;
}

//! PVectorJIT = i * PVectorJIT
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorJIT<T,N,C>, FnTimesI>::Type_t
timesI(const PVectorJIT<T,N,C>& s1)
{
  typename UnaryReturn<PVectorJIT<T,N,C>, FnTimesI>::Type_t  d(s1.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = timesI(s1.elem(i));

  return d;
}

//! PVectorJIT = -i * PVectorJIT
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorJIT<T,N,C>, FnTimesMinusI>::Type_t
timesMinusI(const PVectorJIT<T,N,C>& s1)
{
  typename UnaryReturn<PVectorJIT<T,N,C>, FnTimesMinusI>::Type_t  d(s1.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = timesMinusI(s1.elem(i));

  return d;
}


//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorJIT<T,N,C>, FnGetSite>::Type_t
getSite(const PVectorJIT<T,N,C>& s1, int innersite)
{ 
  typename UnaryReturn<PVectorJIT<T,N,C>, FnGetSite>::Type_t  d(s1.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = getSite(s1.elem(i), innersite);

  return d;
}

// #if 0
// //! Extract color vector components 
// /*! Generically, this is an identity operation. Defined differently under color */
// template<class T, int N, template<class,int> class C>
// inline typename UnaryReturn<PVectorJIT<T,N,C>, FnPeekColorVectorJIT>::Type_t
// peekColor(const PVectorJIT<T,N,C>& l, int row)
// {
//   typename UnaryReturn<PVectorJIT<T,N,C>, FnPeekColorVectorJIT>::Type_t  d(l.func());

//   for(int i=0; i < N; ++i)
//     d.elem(i) = peekColor(l.elem(i),row);
//   return d;
// }

// //! Extract color matrix components 
// /*! Generically, this is an identity operation. Defined differently under color */
// template<class T, int N, template<class,int> class C>
// inline typename UnaryReturn<PVectorJIT<T,N,C>, FnPeekColorMatrixJIT>::Type_t
// peekColor(const PVectorJIT<T,N,C>& l, int row, int col)
// {
//   typename UnaryReturn<PVectorJIT<T,N,C>, FnPeekColorMatrixJIT>::Type_t  d(l.func());

//   for(int i=0; i < N; ++i)
//     d.elem(i) = peekColor(l.elem(i),row,col);
//   return d;
// }

// //! Extract spin vector components 
// /*! Generically, this is an identity operation. Defined differently under spin */
// template<class T, int N, template<class,int> class C>
// inline typename UnaryReturn<PVectorJIT<T,N,C>, FnPeekSpinVectorJIT>::Type_t
// peekSpin(const PVectorJIT<T,N,C>& l, int row)
// {
//   typename UnaryReturn<PVectorJIT<T,N,C>, FnPeekSpinVectorJIT>::Type_t  d(l.func());

//   for(int i=0; i < N; ++i)
//     d.elem(i) = peekSpin(l.elem(i),row);
//   return d;
// }

// //! Extract spin matrix components 
// /*! Generically, this is an identity operation. Defined differently under spin */
// template<class T, int N, template<class,int> class C>
// inline typename UnaryReturn<PVectorJIT<T,N,C>, FnPeekSpinMatrixJIT>::Type_t
// peekSpin(const PVectorJIT<T,N,C>& l, int row, int col)
// {
//   typename UnaryReturn<PVectorJIT<T,N,C>, FnPeekSpinMatrixJIT>::Type_t  d(l.func());

//   for(int i=0; i < N; ++i)
//     d.elem(i) = peekSpin(l.elem(i),row,col);
//   return d;
// }

// //! Insert color vector components 
// /*! Generically, this is an identity operation. Defined differently under color */
// template<class T1, class T2, int N, template<class,int> class C>
// inline typename UnaryReturn<PVectorJIT<T1,N,C>, FnPokeColorVectorJIT>::Type_t&
// pokeColor(PVectorJIT<T1,N,C>& l, const PVectorJIT<T2,N,C>& r, int row)
// {
//   typedef typename UnaryReturn<PVectorJIT<T1,N,C>, FnPokeColorVectorJIT>::Type_t  Return_t;

//   for(int i=0; i < N; ++i)
//     pokeColor(l.elem(i),r.elem(i),row);
//   return static_cast<Return_t&>(l);
// }

// //! Insert color matrix components 
// /*! Generically, this is an identity operation. Defined differently under color */
// template<class T1, class T2, int N, template<class,int> class C>
// inline typename UnaryReturn<PVectorJIT<T1,N,C>, FnPokeColorVectorJIT>::Type_t&
// pokeColor(PVectorJIT<T1,N,C>& l, const PVectorJIT<T2,N,C>& r, int row, int col)
// {
//   typedef typename UnaryReturn<PVectorJIT<T1,N,C>, FnPokeColorVectorJIT>::Type_t  Return_t;

//   for(int i=0; i < N; ++i)
//     pokeColor(l.elem(i),r.elem(i),row,col);
//   return static_cast<Return_t&>(l);
// }

// //! Insert spin vector components 
// /*! Generically, this is an identity operation. Defined differently under spin */
// template<class T1, class T2, int N, template<class,int> class C>
// inline typename UnaryReturn<PVectorJIT<T1,N,C>, FnPokeSpinVectorJIT>::Type_t&
// pokeSpin(PVectorJIT<T1,N,C>& l, const PVectorJIT<T2,N,C>& r, int row)
// {
//   typedef typename UnaryReturn<PVectorJIT<T1,N,C>, FnPokeSpinVectorJIT>::Type_t  Return_t;

//   for(int i=0; i < N; ++i)
//     pokeSpin(l.elem(i),r.elem(i),row);
//   return static_cast<Return_t&>(l);
// }

// //! Insert spin matrix components 
// /*! Generically, this is an identity operation. Defined differently under spin */
// template<class T1, class T2, int N, template<class,int> class C>
// inline typename UnaryReturn<PVectorJIT<T1,N,C>, FnPokeSpinVectorJIT>::Type_t&
// pokeSpin(PVectorJIT<T1,N,C>& l, const PVectorJIT<T2,N,C>& r, int row, int col)
// {
//   typedef typename UnaryReturn<PVectorJIT<T1,N,C>, FnPokeSpinVectorJIT>::Type_t  Return_t;

//   for(int i=0; i < N; ++i)
//     pokeSpin(l.elem(i),r.elem(i),row,col);
//   return static_cast<Return_t&>(l);
// }
// #endif


//! dest = 0
template<class T, int N, template<class,int> class C> 
inline void 
zero_rep(PVectorJIT<T,N,C>& dest) 
{
  for(int i=0; i < N; ++i)
    zero_rep(dest.elem(i));
}

//! dest = (mask) ? s1 : dest
template<class T, class T1, int N, template<class,int> class C> 
inline void 
copymask(PVectorJIT<T,N,C>& d, const PScalarJIT<T1>& mask, const PVectorJIT<T,N,C>& s1) 
{
  for(int i=0; i < N; ++i)
    copymask(d.elem(i),mask.elem(),s1.elem(i));
}


//! dest [some type] = source [some type]
template<class T, class T1, int N, template<class,int> class C>
inline void 
copy_site(PVectorJIT<T,N,C>& d, int isite, const PVectorJIT<T1,N,C>& s1)
{
  for(int i=0; i < N; ++i)
    copy_site(d.elem(i), isite, s1.elem(i));
}

//! dest [some type] = source [some type]
template<class T, class T1, int N, template<class,int> class C>
inline void 
copy_site(PVectorJIT<T,N,C>& d, int isite, const PScalarJIT<T1>& s1)
{
  for(int i=0; i < N; ++i)
    copy_site(d.elem(i), isite, s1.elem());
}


//! gather several inner sites together
template<class T, class T1, int N, template<class,int> class C>
inline void 
gather_sites(PVectorJIT<T,N,C>& d, 
	     const PVectorJIT<T1,N,C>& s0, int i0, 
	     const PVectorJIT<T1,N,C>& s1, int i1,
	     const PVectorJIT<T1,N,C>& s2, int i2,
	     const PVectorJIT<T1,N,C>& s3, int i3)
{
  for(int i=0; i < N; ++i)
    gather_sites(d.elem(i), 
		 s0.elem(i), i0, 
		 s1.elem(i), i1, 
		 s2.elem(i), i2, 
		 s3.elem(i), i3);
}


//! dest  = random  
  template<class T, int N, template<class,int> class C, class T1, class T2, class T3>
inline void
fill_random(PVectorJIT<T,N,C>& d, T1& seed, T2& skewed_seed, const T3& seed_mult)
{
  // Loop over rows the slowest
  for(int i=0; i < N; ++i)
    fill_random(d.elem(i), seed, skewed_seed, seed_mult);
}


//! dest  = gaussian
template<class T, int N, template<class,int> class C>
inline void
fill_gaussian(PVectorJIT<T,N,C>& d, PVectorJIT<T,N,C>& r1, PVectorJIT<T,N,C>& r2)
{
  for(int i=0; i < N; ++i)
    fill_gaussian(d.elem(i), r1.elem(i), r2.elem(i));
}


#if 0
// Global sum over site indices only
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PVectorJIT<T,N,C>, FnSum > {
  typedef C<typename UnaryReturn<T, FnSum>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorJIT<T,N,C>, FnSum>::Type_t
sum(const PVectorJIT<T,N,C>& s1)
{
  typename UnaryReturn<PVectorJIT<T,N,C>, FnSum>::Type_t  d(s1.func());

  for(int i=0; i < N; ++i)
    d.elem(i) = sum(s1.elem(i));

  return d;
}
#endif


// InnerProduct (norm-seq) global sum = sum(tr(adj(s1)*s1))
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PVectorJIT<T,N,C>, FnNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N, template<class,int> class C>
struct UnaryReturn<PVectorJIT<T,N,C>, FnLocalNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorJIT<T,N,C>, FnLocalNorm2>::Type_t
localNorm2(const PVectorJIT<T,N,C>& s1)
{
  typename UnaryReturn<PVectorJIT<T,N,C>, FnLocalNorm2>::Type_t  d(s1.func());

  d.elem() = localNorm2(s1.elem(0));
  for(int i=1; i < N; ++i)
    d.elem() += localNorm2(s1.elem(i));

  return d;
}


//! PScalarJIT<T> = InnerProduct(adj(PVectorJIT<T1>)*PVectorJIT<T1>)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorJIT<T1,N,C>, PVectorJIT<T2,N,C>, FnInnerProduct > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorJIT<T1,N,C>, PVectorJIT<T2,N,C>, FnLocalInnerProduct > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>
localInnerProduct(const PVectorJIT<T1,N,C>& s1, const PVectorJIT<T2,N,C>& s2)
{
  PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  d(s1.func());

  d.elem() = localInnerProduct(s1.elem(0), s2.elem(0));
  for(int i=1; i < N; ++i)
    d.elem() += localInnerProduct(s1.elem(i), s2.elem(i));

  return d;
}


//! PScalarJIT<T> = InnerProductReal(adj(PVectorJIT<T1>)*PVectorJIT<T1>)
/*!
 * return  realpart of InnerProduct(adj(s1)*s2)
 */
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorJIT<T1,N,C>, PVectorJIT<T2,N,C>, FnInnerProductReal > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorJIT<T1,N,C>, PVectorJIT<T2,N,C>, FnLocalInnerProductReal > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>
localInnerProductReal(const PVectorJIT<T1,N,C>& s1, const PVectorJIT<T2,N,C>& s2)
{
  PScalarJIT<typename BinaryReturn<T1,T2, FnLocalInnerProductReal>::Type_t>  d(s1.func());

  d.elem() = localInnerProductReal(s1.elem(0), s2.elem(0));
  for(int i=1; i < N; ++i)
    d.elem() += localInnerProductReal(s1.elem(i), s2.elem(i));

  return d;
}


// This PVectorJIT<T1,N,C> stuff versus PSpinVector<T1,N> is causing problems. 
// When searching for type matching functions, the language does not allow
// for varying template arguments to match a function. We should just move
// away from PVectorJIT to use PSpinVector and PColorVector. However, have to
// replicate all the functions. Uggh - another day...

//
////! PVectorJIT<T> = localInnerProduct(adj(PScalarJIT<T1>)*PVectorJIT<T1>)
//template<class T1, class T2, int N, template<class,int> class C>
//struct BinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,N,C>, FnLocalInnerProduct> {
//  typedef PVectorJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t, N, C>  Type_t;
//};
//
//template<class T1, class T2, int N, template<class,int> class C>
//inline PVectorJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t,N,C>
//localInnerProduct(const PScalarJIT<T1>& s1, const PVectorJIT<T2,N,C>& s2)
//{
//  typename BinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,N,C>, FnLocalInnerProduct>::Type_t  d;
//
//  for(int i=0; i < N; ++i)
//    d.elem(i) = localInnerProduct(s1.elem(0), s2.elem(i));
//
//  return d;
//}
//
//
////! PScalarJIT<T> = InnerProductReal(adj(PScalarJIT<T1>)*PVectorJIT<T1>)
///*!
// * return  realpart of InnerProduct(adj(s1)*s2)
// */
//template<class T1, class T2, int N, template<class,int> class C>
//struct BinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,N,C>, FnLocalInnerProductReal > {
//  typedef PVectorJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t, N,C>  Type_t;
//};
//
//template<class T1, class T2, int N, template<class,int> class C>
//inline PVectorJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t,N,C>
//localInnerProductReal(const PScalarJIT<T1>& s1, const PVectorJIT<T2,N,C>& s2)
//{
//  typename BinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,N,C>, FnLocalInnerProductReal>::Type_t  d;
//
//  for(int i=0; i < N; ++i)
//    d.elem(i) = localInnerProductReal(s1.elem(), s2.elem(i));
//
//  return d;
//}


//! PVectorJIT<T> = where(PScalarJIT, PVectorJIT, PVectorJIT)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3, int N, template<class,int> class C>
struct TrinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,N,C>, PVectorJIT<T3,N,C>, FnWhere> {
  typedef C<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t, N>  Type_t;
};

template<class T1, class T2, class T3, int N, template<class,int> class C>
inline typename TrinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,N,C>, PVectorJIT<T3,N,C>, FnWhere>::Type_t
where(const PScalarJIT<T1>& a, const PVectorJIT<T2,N,C>& b, const PVectorJIT<T3,N,C>& c)
{
  typename TrinaryReturn<PScalarJIT<T1>, PVectorJIT<T2,N,C>, PVectorJIT<T3,N,C>, FnWhere>::Type_t  d(a.func());

  // Not optimal - want to have where outside assignment
  for(int i=0; i < N; ++i)
    d.elem(i) = where(a.elem(), b.elem(i), c.elem(i));

  return d;
}

/*! @} */  // end of group primvector

} // namespace QDP

#endif
