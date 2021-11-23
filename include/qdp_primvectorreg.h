// -*- C++ -*-

/*! \file
 * \brief Primitive Vector
 */


#ifndef QDP_PRIMVECTORREG_H
#define QDP_PRIMVECTORREG_H

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
  template <class T, int N, template<class,int> class C> class PVectorREG
{
  T F[N];
public:
  typedef C<T,N>  CC;


  //! PVectorREG = PVectorREG
  /*! Set equal to another PVectorREG */
  template<class T1>
  inline
  CC& assign(const C<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs.elem(i);

      return static_cast<CC&>(*this);
    }

  //! PVectorREG = PVectorREG
  /*! Set equal to another PVectorREG */
  template<class T1>
  inline
  CC& operator=(const C<T1,N>& rhs) 
    {
      return assign(rhs);
    }

  //! PVectorREG += PVectorREG
  template<class T1>
  inline
  CC& operator+=(const C<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) += rhs.elem(i);

      return static_cast<CC&>(*this);
    }

  //! PVectorREG -= PVectorREG
  template<class T1>
  inline
  CC& operator-=(const C<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) -= rhs.elem(i);

      return static_cast<CC&>(*this);
    }

  //! PVectorREG *= PScalarREG
  template<class T1>
  inline
  CC& operator*=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) *= rhs.elem();

      return static_cast<CC&>(*this);
    }

  //! PVectorREG /= PScalarREG
  template<class T1>
  inline
  CC& operator/=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) /= rhs.elem();

      return static_cast<CC&>(*this);
    }


public:
        T& elem(int i)       {return F[i];}
  const T& elem(int i) const {return F[i];}
};





//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------


  template <class T, int N, template<class,int> class C >
  struct IsWordVec< PVectorREG<T,N,C > >
  {
    constexpr static bool value = IsWordVec<T>::value;
  };

  
// Underlying word type
template<class T1, int N, template<class,int> class C>
struct WordType<PVectorREG<T1,N,C> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

template<class T1, int N, template<class, int> class C> 
struct SinglePrecType< PVectorREG<T1,N,C> >
{
  typedef PVectorREG< typename SinglePrecType<T1>::Type_t, N, C> Type_t;
};

template<class T1, int N, template<class, int> class C> 
struct DoublePrecType< PVectorREG<T1,N,C> >
{
  typedef PVectorREG< typename DoublePrecType<T1>::Type_t, N, C> Type_t;
};

// Internally used scalars
template<class T, int N, template<class,int> class C>
struct InternalScalar<PVectorREG<T,N,C> > {
  typedef PScalarREG<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive scalar leaving grid alone
template<class T, int N, template<class,int> class C>
struct PrimitiveScalar<PVectorREG<T,N,C> > {
  typedef PScalarREG<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T, int N, template<class,int> class C>
struct LatticeScalar<PVectorREG<T,N,C> > {
  typedef C<typename LatticeScalar<T>::Type_t, N>  Type_t;
};

//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PVectorREG) -> PVectorREG
template<class T1, int N, template<class,int> class C, class Op>
struct UnaryReturn<PVectorREG<T1,N,C>, Op> {
  typedef C<typename UnaryReturn<T1, Op>::Type_t, N>  Type_t;
};
// Default binary(PScalarREG,PVectorREG) -> PVectorREG
template<class T1, class T2, int N, template<class,int> class C, class Op>
struct BinaryReturn<PScalarREG<T1>, PVectorREG<T2,N,C>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PMatrixREG,PVectorREG) -> PVectorREG
template<class T1, class T2, int N, template<class,int> class C1, 
  template<class,int> class C2, class Op>
struct BinaryReturn<PMatrixREG<T1,N,C1>, PVectorREG<T2,N,C2>, Op> {
  typedef C2<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PVectorREG,PScalarREG) -> PVectorREG
template<class T1, class T2, int N, template<class,int> class C, class Op>
struct BinaryReturn<PVectorREG<T1,N,C>, PScalarREG<T2>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PVectorREG,PVectorREG) -> PVectorREG
template<class T1, class T2, int N, template<class,int> class C, class Op>
struct BinaryReturn<PVectorREG<T1,N,C>, PVectorREG<T2,N,C>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};


#if 0
template<class T1, class T2>
struct UnaryReturn<PScalarREG<T2>, OpCast<T1> > {
  typedef PScalarREG<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif


// Assignment is different
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorREG<T1,N,C>, PVectorREG<T2,N,C>, OpAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorREG<T1,N,C>, PVectorREG<T2,N,C>, OpAddAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorREG<T1,N,C>, PVectorREG<T2,N,C>, OpSubtractAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorREG<T1,N,C>, PScalarREG<T2>, OpMultiplyAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorREG<T1,N,C>, PScalarREG<T2>, OpDivideAssign > {
  typedef C<T1,N> &Type_t;
};
 



//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

/*! \addtogroup primvector */
/*! @{ */

// Primitive Vectors

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorREG<T1,N,C>, OpUnaryPlus>::Type_t
operator+(const PVectorREG<T1,N,C>& l)
{
  typename UnaryReturn<PVectorREG<T1,N,C>, OpUnaryPlus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = +l.elem(i);
  return d;
}


template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorREG<T1,N,C>, OpUnaryMinus>::Type_t
operator-(const PVectorREG<T1,N,C>& l)
{
  typename UnaryReturn<PVectorREG<T1,N,C>, OpUnaryMinus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = -l.elem(i);
  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVectorREG<T1,N,C>, PVectorREG<T2,N,C>, OpAdd>::Type_t
operator+(const PVectorREG<T1,N,C>& l, const PVectorREG<T2,N,C>& r)
{
  typename BinaryReturn<PVectorREG<T1,N,C>, PVectorREG<T2,N,C>, OpAdd>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) + r.elem(i);
  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVectorREG<T1,N,C>, PVectorREG<T2,N,C>, OpSubtract>::Type_t
operator-(const PVectorREG<T1,N,C>& l, const PVectorREG<T2,N,C>& r)
{
  typename BinaryReturn<PVectorREG<T1,N,C>, PVectorREG<T2,N,C>, OpSubtract>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) - r.elem(i);
  return d;
}


// PVectorREG * PScalarREG
template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVectorREG<T1,N,C>, PScalarREG<T2>, OpMultiply>::Type_t
operator*(const PVectorREG<T1,N,C>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PVectorREG<T1,N,C>, PScalarREG<T2>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) * r.elem();
  return d;
}

// Optimized  PVectorREG * adj(PScalarREG)
template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVectorREG<T1,N,C>, PScalarREG<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(const PVectorREG<T1,N,C>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PVectorREG<T1,N,C>, PScalarREG<T2>, OpMultiplyAdj>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = multiplyAdj(l.elem(i), r.elem());
  return d;
}


// PScalarREG * PVectorREG
template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarREG<T1>, PVectorREG<T2,N,C>, OpMultiply>::Type_t
operator*(const PScalarREG<T1>& l, const PVectorREG<T2,N,C>& r)
{
  typename BinaryReturn<PScalarREG<T1>, PVectorREG<T2,N,C>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem() * r.elem(i);
  return d;
}

// Optimized  adj(PScalarREG) * PVectorREG
template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarREG<T1>, PVectorREG<T2,N,C>, OpAdjMultiply>::Type_t
adjMultiply(const PScalarREG<T1>& l, const PVectorREG<T2,N,C>& r)
{
  typename BinaryReturn<PScalarREG<T1>, PVectorREG<T2,N,C>, OpAdjMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = adjMultiply(l.elem(), r.elem(i));
  return d;
}


// PMatrixREG * PVectorREG
template<class T1, class T2, int N, template<class,int> class C1, template<class,int> class C2>
inline typename BinaryReturn<PMatrixREG<T1,N,C1>, PVectorREG<T2,N,C2>, OpMultiply>::Type_t
operator*(const PMatrixREG<T1,N,C1>& l, const PVectorREG<T2,N,C2>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C1>, PVectorREG<T2,N,C2>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
  {
    d.elem(i) = l.elem(i,0) * r.elem(0);
    for(int j=1; j < N; ++j)
      d.elem(i) += l.elem(i,j) * r.elem(j);
  }

  return d;
}

// Optimized  adj(PMatrixREG)*PVectorREG
template<class T1, class T2, int N, template<class,int> class C1, template<class,int> class C2>
inline typename BinaryReturn<PMatrixREG<T1,N,C1>, PVectorREG<T2,N,C2>, OpAdjMultiply>::Type_t
adjMultiply(const PMatrixREG<T1,N,C1>& l, const PVectorREG<T2,N,C2>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C1>, PVectorREG<T2,N,C2>, OpAdjMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
  {
    d.elem(i) = adjMultiply(l.elem(0,i), r.elem(0));
    for(int j=1; j < N; ++j)
      d.elem(i) += adjMultiply(l.elem(j,i), r.elem(j));
  }

  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVectorREG<T1,N,C>, PScalarREG<T2>, OpDivide>::Type_t
operator/(const PVectorREG<T1,N,C>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PVectorREG<T1,N,C>, PScalarREG<T2>, OpDivide>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) / r.elem();
  return d;
}



//! PVectorREG = Re(PVectorREG)
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorREG<T,N,C>, FnReal>::Type_t
real(const PVectorREG<T,N,C>& s1)
{
  typename UnaryReturn<PVectorREG<T,N,C>, FnReal>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = real(s1.elem(i));

  return d;
}


//! PVectorREG = Im(PVectorREG)
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorREG<T,N,C>, FnImag>::Type_t
imag(const PVectorREG<T,N,C>& s1)
{
  typename UnaryReturn<PVectorREG<T,N,C>, FnImag>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = imag(s1.elem(i));

  return d;
}


//! PVectorREG<T> = (PVectorREG<T> , PVectorREG<T>)
template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVectorREG<T1,N,C>, PVectorREG<T2,N,C>, FnCmplx>::Type_t
cmplx(const PVectorREG<T1,N,C>& s1, const PVectorREG<T2,N,C>& s2)
{
  typename BinaryReturn<PVectorREG<T1,N,C>, PVectorREG<T2,N,C>, FnCmplx>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = cmplx(s1.elem(i), s2.elem(i));

  return d;
}




template<class T1, int N, template<class,int> class C>
struct UnaryReturn<PVectorREG<T1,N,C>, FnIsFinite> {
  typedef PScalarREG< typename UnaryReturn<T1, FnIsFinite >::Type_t > Type_t;
};

  
template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorREG<T1,N,C>, FnIsFinite>::Type_t
isfinite(const PVectorREG<T1,N,C>& l)
{
  typename UnaryReturn<PVectorREG<T1,N,C>, FnIsFinite>::Type_t d(true);

  for(int i=0; i < N; ++i)
    d.elem() &= isfinite(l.elem(i));

  return d;
}


  

//-----------------------------------------------------------------------------
// Functions
// Conjugate
template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorREG<T1,N,C>, FnConjugate>::Type_t
conj(const PVectorREG<T1,N,C>& l)
{
  typename UnaryReturn<PVectorREG<T1,N,C>, FnConjugate>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = conj(l.elem(i));

  return d;
}

//! PVectorREG = i * PVectorREG
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorREG<T,N,C>, FnTimesI>::Type_t
timesI(const PVectorREG<T,N,C>& s1)
{
  typename UnaryReturn<PVectorREG<T,N,C>, FnTimesI>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = timesI(s1.elem(i));

  return d;
}

//! PVectorREG = -i * PVectorREG
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorREG<T,N,C>, FnTimesMinusI>::Type_t
timesMinusI(const PVectorREG<T,N,C>& s1)
{
  typename UnaryReturn<PVectorREG<T,N,C>, FnTimesMinusI>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = timesMinusI(s1.elem(i));

  return d;
}


//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorREG<T,N,C>, FnGetSite>::Type_t
getSite(const PVectorREG<T,N,C>& s1, int innersite)
{ 
  typename UnaryReturn<PVectorREG<T,N,C>, FnGetSite>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = getSite(s1.elem(i), innersite);

  return d;
}



//! dest = 0
template<class T, int N, template<class,int> class C> 
inline void 
zero_rep(PVectorREG<T,N,C>& dest) 
{
  for(int i=0; i < N; ++i)
    zero_rep(dest.elem(i));
}


//! dest [some type] = source [some type]
template<class T, class T1, int N, template<class,int> class C>
inline void 
copy_site(PVectorREG<T,N,C>& d, int isite, const PVectorREG<T1,N,C>& s1)
{
  for(int i=0; i < N; ++i)
    copy_site(d.elem(i), isite, s1.elem(i));
}

//! dest [some type] = source [some type]
template<class T, class T1, int N, template<class,int> class C>
inline void 
copy_site(PVectorREG<T,N,C>& d, int isite, const PScalarREG<T1>& s1)
{
  for(int i=0; i < N; ++i)
    copy_site(d.elem(i), isite, s1.elem());
}


//! gather several inner sites together
template<class T, class T1, int N, template<class,int> class C>
inline void 
gather_sites(PVectorREG<T,N,C>& d, 
	     const PVectorREG<T1,N,C>& s0, int i0, 
	     const PVectorREG<T1,N,C>& s1, int i1,
	     const PVectorREG<T1,N,C>& s2, int i2,
	     const PVectorREG<T1,N,C>& s3, int i3)
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
fill_random(PVectorREG<T,N,C>& d, T1& seed, T2& skewed_seed, const T3& seed_mult)
{
  // Loop over rows the slowest
  for(int i=0; i < N; ++i)
    fill_random(d.elem(i), seed, skewed_seed, seed_mult);
}


//! dest  = gaussian
template<class T, int N, template<class,int> class C>
inline void
fill_gaussian(PVectorREG<T,N,C>& d, PVectorREG<T,N,C>& r1, PVectorREG<T,N,C>& r2)
{
  for(int i=0; i < N; ++i)
    fill_gaussian(d.elem(i), r1.elem(i), r2.elem(i));
}




// InnerProduct (norm-seq) global sum = sum(tr(adj(s1)*s1))
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PVectorREG<T,N,C>, FnNorm2 > {
  typedef PScalarREG<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N, template<class,int> class C>
struct UnaryReturn<PVectorREG<T,N,C>, FnLocalNorm2 > {
  typedef PScalarREG<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVectorREG<T,N,C>, FnLocalNorm2>::Type_t
localNorm2(const PVectorREG<T,N,C>& s1)
{
  typename UnaryReturn<PVectorREG<T,N,C>, FnLocalNorm2>::Type_t  d;

  d.elem() = localNorm2(s1.elem(0));
  for(int i=1; i < N; ++i)
    d.elem() += localNorm2(s1.elem(i));

  return d;
}


//! PScalarREG<T> = InnerProduct(adj(PVectorREG<T1>)*PVectorREG<T1>)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorREG<T1,N,C>, PVectorREG<T2,N,C>, FnInnerProduct > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorREG<T1,N,C>, PVectorREG<T2,N,C>, FnLocalInnerProduct > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>
localInnerProduct(const PVectorREG<T1,N,C>& s1, const PVectorREG<T2,N,C>& s2)
{
  PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  d;

  d.elem() = localInnerProduct(s1.elem(0), s2.elem(0));
  for(int i=1; i < N; ++i)
    d.elem() += localInnerProduct(s1.elem(i), s2.elem(i));

  return d;
}


//! PScalarREG<T> = InnerProductReal(adj(PVectorREG<T1>)*PVectorREG<T1>)
/*!
 * return  realpart of InnerProduct(adj(s1)*s2)
 */
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorREG<T1,N,C>, PVectorREG<T2,N,C>, FnInnerProductReal > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVectorREG<T1,N,C>, PVectorREG<T2,N,C>, FnLocalInnerProductReal > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>
localInnerProductReal(const PVectorREG<T1,N,C>& s1, const PVectorREG<T2,N,C>& s2)
{
  PScalarREG<typename BinaryReturn<T1,T2, FnLocalInnerProductReal>::Type_t>  d;

  d.elem() = localInnerProductReal(s1.elem(0), s2.elem(0));
  for(int i=1; i < N; ++i)
    d.elem() += localInnerProductReal(s1.elem(i), s2.elem(i));

  return d;
}




//! PVectorREG<T> = where(PScalarREG, PVectorREG, PVectorREG)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3, int N, template<class,int> class C>
struct TrinaryReturn<PScalarREG<T1>, PVectorREG<T2,N,C>, PVectorREG<T3,N,C>, FnWhere> {
  typedef C<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t, N>  Type_t;
};

template<class T1, class T2, class T3, int N, template<class,int> class C>
inline typename TrinaryReturn<PScalarREG<T1>, PVectorREG<T2,N,C>, PVectorREG<T3,N,C>, FnWhere>::Type_t
where(const PScalarREG<T1>& a, const PVectorREG<T2,N,C>& b, const PVectorREG<T3,N,C>& c)
{
  typename TrinaryReturn<PScalarREG<T1>, PVectorREG<T2,N,C>, PVectorREG<T3,N,C>, FnWhere>::Type_t  d;

  // Not optimal - want to have where outside assignment
  for(int i=0; i < N; ++i)
    d.elem(i) = where(a.elem(), b.elem(i), c.elem(i));

  return d;
}


template<class T, int N, template<class,int> class C>
inline void 
qdpPHI(PVectorREG<T,N,C>& d, 
       const PVectorREG<T,N,C>& phi0, llvm::BasicBlock* bb0 ,
       const PVectorREG<T,N,C>& phi1, llvm::BasicBlock* bb1 )
{
  for(int i=0; i < N; ++i)
    qdpPHI(d.elem(i),
	   phi0.elem(i),bb0,
	   phi1.elem(i),bb1);
}



/*! @} */  // end of group primvector

} // namespace QDP

#endif
