// -*- C++ -*-

/*! \file
 * \brief Primitive Spin Vector
 */


#ifndef QDP_PRIMSPINVECREG_H
#define QDP_PRIMSPINVECREG_H

namespace QDP {


//-------------------------------------------------------------------------------------
/*! \addtogroup primspinvector Spin vector primitive
 * \ingroup primvector
 *
 * Primitive type that transforms like a Spin vector
 *
 * @{
 */

//! Primitive spin Vector class
/*! 
 * Spin vector class supports gamma matrix algebra 
 *
 * NOTE: the class is mostly empty - it is the specialized versions below
 * that know for a fixed size how gamma matrices (constants) should act
 * on the spin vectors.
 */

template <class T, int N> class PSpinVectorREG //: public BaseREG<T,N,PSpinVectorREG<T,N> >
{
  T F[N];
public:
  PSpinVectorREG(){}


  void setup( const typename JITType< PSpinVectorREG >::Type_t& j ) {
    for (int i = 0 ; i < N ; i++ )
      this->elem(i).setup( j.elem(i) );
  }


  template<class T1>
  PSpinVectorREG(const PSpinVectorREG<T1,N>& a)
  {
    for(int i=0; i < N; i++) 
      elem(i) = a.elem(i);
  }



  template<class T1>
  inline
  PSpinVectorREG& assign(const PSpinVectorREG<T1, N>& rhs)
  {
    for(int i=0; i < N; i++) 
      elem(i) = rhs.elem(i);

    return *this;
  }

  template<class T1> 
  inline 
  PSpinVectorREG& operator=(const PSpinVectorREG<T1,N>& rhs)
  {
    return assign(rhs);
  }


  PSpinVectorREG& operator=(const PSpinVectorREG& rhs)
  {
    return assign(rhs);
  }

  //! PSpinVectorREG += PSpinVectorREG
  template<class T1>
  inline
  PSpinVectorREG& operator+=(const PSpinVectorREG<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) += rhs.elem(i);

      return *this;
    }


  template<class T1>
  inline
  PSpinVectorREG& operator*=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) *= rhs.elem();

      return *this;
    }

  template<class T1>
  inline
  PSpinVectorREG& operator-=(const PSpinVectorREG<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) -= rhs.elem(i);

      return *this;
    }

  //! PSpinVectorREG /= PScalarREG
  template<class T1>
  inline
  PSpinVectorREG& operator/=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) /= rhs.elem();

      return *this;
    }

public:
        T& elem(int i)       {return F[i];}
  const T& elem(int i) const {return F[i];}

  // T& elem(int i) {return JV<T,N>::getF()[i];}
  // const T& elem(int i) const {return JV<T,N>::getF()[i];}
};




// Primitive Vectors

template<class T1, int N>
inline typename UnaryReturn<PSpinVectorREG<T1,N>, OpUnaryPlus>::Type_t
operator+(const PSpinVectorREG<T1,N>& l)
{
  typename UnaryReturn<PSpinVectorREG<T1,N>, OpUnaryPlus>::Type_t  d;
  
  for(int i=0; i < N; ++i)
    d.elem(i) = +l.elem(i);
  return d;
}


template<class T1, int N>
inline typename UnaryReturn<PSpinVectorREG<T1,N>, OpUnaryMinus>::Type_t
operator-(const PSpinVectorREG<T1,N>& l)
{
  typename UnaryReturn<PSpinVectorREG<T1,N>, OpUnaryMinus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = -l.elem(i);
  return d;
}


template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinVectorREG<T1,N>, PSpinVectorREG<T2,N>, OpAdd>::Type_t
operator+(const PSpinVectorREG<T1,N>& l, const PSpinVectorREG<T2,N>& r)
{
  typename BinaryReturn<PSpinVectorREG<T1,N>, PSpinVectorREG<T2,N>, OpAdd>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) + r.elem(i);
  return d;
}


template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinVectorREG<T1,N>, PSpinVectorREG<T2,N>, OpSubtract>::Type_t
operator-(const PSpinVectorREG<T1,N>& l, const PSpinVectorREG<T2,N>& r)
{
  typename BinaryReturn<PSpinVectorREG<T1,N>, PSpinVectorREG<T2,N>, OpSubtract>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) - r.elem(i);
  return d;
}


// PSpinVectorREG * PScalarREG
template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinVectorREG<T1,N>, PScalarREG<T2>, OpMultiply>::Type_t
operator*(const PSpinVectorREG<T1,N>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PSpinVectorREG<T1,N>, PScalarREG<T2>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) * r.elem();
  return d;
}

// Optimized  PSpinVectorREG * adj(PScalarREG)
template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinVectorREG<T1,N>, PScalarREG<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(const PSpinVectorREG<T1,N>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PSpinVectorREG<T1,N>, PScalarREG<T2>, OpMultiplyAdj>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = multiplyAdj(l.elem(i), r.elem());
  return d;
}


// PScalarREG * PSpinVectorREG
template<class T1, class T2, int N>
inline typename BinaryReturn<PScalarREG<T1>, PSpinVectorREG<T2,N>, OpMultiply>::Type_t
operator*(const PScalarREG<T1>& l, const PSpinVectorREG<T2,N>& r)
{
  typename BinaryReturn<PScalarREG<T1>, PSpinVectorREG<T2,N>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem() * r.elem(i);
  return d;
}

// Optimized  adj(PScalarREG) * PSpinVectorREG
template<class T1, class T2, int N>
inline typename BinaryReturn<PScalarREG<T1>, PSpinVectorREG<T2,N>, OpAdjMultiply>::Type_t
adjMultiply(const PScalarREG<T1>& l, const PSpinVectorREG<T2,N>& r)
{
  typename BinaryReturn<PScalarREG<T1>, PSpinVectorREG<T2,N>, OpAdjMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = adjMultiply(l.elem(), r.elem(i));
  return d;
}


// PMatrix * PSpinVectorREG
template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinMatrixREG<T1,N>, PSpinVectorREG<T2,N>, OpMultiply>::Type_t
operator*(const PSpinMatrixREG<T1,N>& l, const PSpinVectorREG<T2,N>& r)
{
  typename BinaryReturn<PSpinMatrixREG<T1,N>, PSpinVectorREG<T2,N>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
  {
    d.elem(i) = l.elem(i,0) * r.elem(0);
    for(int j=1; j < N; ++j)
      d.elem(i) += l.elem(i,j) * r.elem(j);
  }

  return d;
}


// PMatrix * PSpinVectorREG
template<class T1, class T2,  template<class,int> class C, int N>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PSpinVectorREG<T2,N>, OpMultiply>::Type_t
operator*(const PSpinMatrixREG<T1,N>& l, const PSpinVectorREG<T2,N>& r)
{
  typename BinaryReturn<PSpinMatrixREG<T1,N>, PSpinVectorREG<T2,N>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
  {
    d.elem(i) = l.elem(i,0) * r.elem(0);
    for(int j=1; j < N; ++j)
      d.elem(i) += l.elem(i,j) * r.elem(j);
  }

  return d;
}

// Optimized  adj(PMatrixREG)*PSpinVectorREG
template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinMatrixREG<T1,N>, PSpinVectorREG<T2,N>, OpAdjMultiply>::Type_t
adjMultiply(const PSpinMatrixREG<T1,N>& l, const PSpinVectorREG<T2,N>& r)
{
  typename BinaryReturn<PSpinMatrixREG<T1,N>, PSpinVectorREG<T2,N>, OpAdjMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
  {
    d.elem(i) = adjMultiply(l.elem(0,i), r.elem(0));
    for(int j=1; j < N; ++j)
      d.elem(i) += adjMultiply(l.elem(j,i), r.elem(j));
  }

  return d;
}

// Optimized  adj(PMatrixREG)*PVector
template<class T1, class T2, int N, template<class,int> class C1>
inline typename BinaryReturn<PMatrixREG<T1,N,C1>, PSpinVectorREG<T2,N>, OpAdjMultiply>::Type_t
adjMultiply(const PMatrixREG<T1,N,C1>& l, const PSpinVectorREG<T2,N>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C1>, PSpinVectorREG<T2,N>, OpAdjMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
  {
    d.elem(i) = adjMultiply(l.elem(0,i), r.elem(0));
    for(int j=1; j < N; ++j)
      d.elem(i) += adjMultiply(l.elem(j,i), r.elem(j));
  }

  return d;
}

template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinVectorREG<T1,N>, PScalarREG<T2>, OpDivide>::Type_t
operator/(const PSpinVectorREG<T1,N>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PSpinVectorREG<T1,N>, PScalarREG<T2>, OpDivide>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) / r.elem();
  return d;
}



//! PSpinVectorREG = Re(PSpinVectorREG)
template<class T, int N>
inline typename UnaryReturn<PSpinVectorREG<T,N>, FnReal>::Type_t
real(const PSpinVectorREG<T,N>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,N>, FnReal>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = real(s1.elem(i));

  return d;
}


//! PSpinVectorREG = Im(PSpinVectorREG)
template<class T, int N>
inline typename UnaryReturn<PSpinVectorREG<T,N>, FnImag>::Type_t
imag(const PSpinVectorREG<T,N>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,N>, FnImag>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = imag(s1.elem(i));

  return d;
}


//! PSpinVectorREG<T> = (PSpinVectorREG<T> , PSpinVectorREG<T>)
template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinVectorREG<T1,N>, PSpinVectorREG<T2,N>, FnCmplx>::Type_t
cmplx(const PSpinVectorREG<T1,N>& s1, const PSpinVectorREG<T2,N>& s2)
{
  typename BinaryReturn<PSpinVectorREG<T1,N>, PSpinVectorREG<T2,N>, FnCmplx>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = cmplx(s1.elem(i), s2.elem(i));

  return d;
}


//-----------------------------------------------------------------------------
// Functions
// Conjugate
template<class T1, int N>
inline typename UnaryReturn<PSpinVectorREG<T1,N>, FnConjugate>::Type_t
conj(const PSpinVectorREG<T1,N>& l)
{
  typename UnaryReturn<PSpinVectorREG<T1,N>, FnConjugate>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = conj(l.elem(i));

  return d;
}

//! PSpinVectorREG = i * PSpinVectorREG
template<class T, int N>
inline typename UnaryReturn<PSpinVectorREG<T,N>, FnTimesI>::Type_t
timesI(const PSpinVectorREG<T,N>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,N>, FnTimesI>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = timesI(s1.elem(i));

  return d;
}

//! PSpinVectorREG = -i * PSpinVectorREG
template<class T, int N>
inline typename UnaryReturn<PSpinVectorREG<T,N>, FnTimesMinusI>::Type_t
timesMinusI(const PSpinVectorREG<T,N>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,N>, FnTimesMinusI>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = timesMinusI(s1.elem(i));

  return d;
}


//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T, int N>
inline typename UnaryReturn<PSpinVectorREG<T,N>, FnGetSite>::Type_t
getSite(const PSpinVectorREG<T,N>& s1, int innersite)
{ 
  typename UnaryReturn<PSpinVectorREG<T,N>, FnGetSite>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = getSite(s1.elem(i), innersite);

  return d;
}

//! Extract color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T, int N>
inline typename UnaryReturn<PSpinVectorREG<T,N>, FnPeekColorVectorREG >::Type_t
peekColor(const PSpinVectorREG<T,N>& l, jit_value row)
{
  typename UnaryReturn<PSpinVectorREG<T,N>, FnPeekColorVectorREG >::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = peekColor(l.elem(i),row);
  return d;
}

//! Extract color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T, int N>
inline typename UnaryReturn<PSpinVectorREG<T,N>, FnPeekColorMatrixREG >::Type_t
peekColor(const PSpinVectorREG<T,N>& l, jit_value row, jit_value col)
{
  typename UnaryReturn<PSpinVectorREG<T,N>, FnPeekColorMatrixREG >::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = peekColor(l.elem(i),row,col);
  return d;
}


//! Extract spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
#if 0
template<class T, int N>
inline typename UnaryReturn<PSpinVectorREG<T,N>, FnPeekSpinMatrixREG>::Type_t
peekSpin(const PSpinVectorREG<T,N>& l, jit_value row, jit_value col)
{
  typename UnaryReturn<PSpinVectorREG<T,N>, FnPeekSpinMatrixREG>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = peekSpin(l.elem(i),row,col);
  return d;
}
#endif

template<class T, int N>
inline typename UnaryReturn<PSpinVectorREG<T,N>, FnPeekSpinVectorREG>::Type_t
peekSpin(const PSpinVectorREG<T,N>& l, jit_value row)
{
  typename UnaryReturn<PSpinVectorREG<T,N>, FnPeekSpinVectorREG>::Type_t  d;

  typedef typename JITType< PSpinVectorREG<T,N> >::Type_t TTjit;

  jit_value ptr_local = jit_allocate_local( jit_type<typename WordType<T>::Type_t>::value , TTjit::Size_t );

  TTjit dj;
  dj.setup( ptr_local, jit_value(1) , jit_value(0) );
  dj=l;

  d.elem() = dj.getRegElem(row);
  return d;
}

//! Insert color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2, int N>
inline typename UnaryReturn<PSpinVectorREG<T1,N>, FnPokeColorVectorREG >::Type_t&
pokeColor(PSpinVectorREG<T1,N>& l, const PSpinVectorREG<T2,N>& r, jit_value row)
{
  typedef typename UnaryReturn<PSpinVectorREG<T1,N>, FnPokeColorVectorREG >::Type_t  Return_t;

  for(int i=0; i < N; ++i)
    pokeColor(l.elem(i),r.elem(i),row);
  return static_cast<Return_t&>(l);
}

//! Insert color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2, int N>
inline typename UnaryReturn<PSpinVectorREG<T1,N>, FnPokeColorVectorREG>::Type_t&
pokeColor(PSpinVectorREG<T1,N>& l, const PSpinVectorREG<T2,N>& r, jit_value row, jit_value col)
{
  typedef typename UnaryReturn<PSpinVectorREG<T1,N>, FnPokeColorVectorREG>::Type_t  Return_t;

  for(int i=0; i < N; ++i)
    pokeColor(l.elem(i),r.elem(i),row,col);
  return static_cast<Return_t&>(l);
}

//! Insert spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2, int N>
inline typename UnaryReturn<PSpinVectorREG<T1,N>, FnPokeSpinVectorREG>::Type_t&
pokeSpin(PSpinVectorREG<T1,N>& l, const PSpinVectorREG<T2,N>& r, jit_value row)
{
  typedef typename UnaryReturn<PSpinVectorREG<T1,N>, FnPokeSpinVectorREG>::Type_t  Return_t;

  for(int i=0; i < N; ++i)
    pokeSpin(l.elem(i),r.elem(i),row);
  return static_cast<Return_t&>(l);
}

//! Insert spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2, int N>
inline typename UnaryReturn<PSpinVectorREG<T1,N>, FnPokeSpinVectorREG>::Type_t&
pokeSpin(PSpinVectorREG<T1,N>& l, const PSpinVectorREG<T2,N>& r, jit_value row, jit_value col)
{
  typedef typename UnaryReturn<PSpinVectorREG<T1,N>, FnPokeSpinVectorREG>::Type_t  Return_t;

  for(int i=0; i < N; ++i)
    pokeSpin(l.elem(i),r.elem(i),row,col);
  return static_cast<Return_t&>(l);
}


//! dest = 0
template<class T, int N> 
inline void 
zero_rep(PSpinVectorREG<T,N>& dest) 
{
  for(int i=0; i < N; ++i)
    zero_rep(dest.elem(i));
}

//! dest = (mask) ? s1 : dest
template<class T, class T1, int N> 
inline void 
copymask(PSpinVectorREG<T,N>& d, const PScalarREG<T1>& mask, const PSpinVectorREG<T,N>& s1) 
{
  for(int i=0; i < N; ++i)
    copymask(d.elem(i),mask.elem(),s1.elem(i));
}


//! dest [some type] = source [some type]
template<class T, class T1, int N>
inline void 
copy_site(PSpinVectorREG<T,N>& d, int isite, const PSpinVectorREG<T1,N>& s1)
{
  for(int i=0; i < N; ++i)
    copy_site(d.elem(i), isite, s1.elem(i));
}

//! dest [some type] = source [some type]
template<class T, class T1, int N>
inline void 
copy_site(PSpinVectorREG<T,N>& d, int isite, const PScalarREG<T1>& s1)
{
  for(int i=0; i < N; ++i)
    copy_site(d.elem(i), isite, s1.elem());
}


//! gather several inner sites together
template<class T, class T1, int N>
inline void 
gather_sites(PSpinVectorREG<T,N>& d, 
	     const PSpinVectorREG<T1,N>& s0, int i0, 
	     const PSpinVectorREG<T1,N>& s1, int i1,
	     const PSpinVectorREG<T1,N>& s2, int i2,
	     const PSpinVectorREG<T1,N>& s3, int i3)
{
  for(int i=0; i < N; ++i)
    gather_sites(d.elem(i), 
		 s0.elem(i), i0, 
		 s1.elem(i), i1, 
		 s2.elem(i), i2, 
		 s3.elem(i), i3);
}



#if 0
// Global sum over site indices only
template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSum > {
  typedef PSpinVectorREGtypename UnaryReturn<T, FnSum>::Type_t, N>  Type_t;
};

template<class T, int N>
inline typename UnaryReturn<PSpinVectorREG<T,N>, FnSum>::Type_t
sum(const PSpinVectorREG<T,N>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,N>, FnSum>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = sum(s1.elem(i));

  return d;
}
#endif




template<class T, int N>
inline typename UnaryReturn<PSpinVectorREG<T,N>, FnLocalNorm2>::Type_t
localNorm2(const PSpinVectorREG<T,N>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,N>, FnLocalNorm2>::Type_t  d;

  d.elem() = localNorm2(s1.elem(0));
  for(int i=1; i < N; ++i)
    d.elem() += localNorm2(s1.elem(i));

  return d;
}




//! PSpinVectorREG<T> = where(PScalarREG, PSpinVectorREG, PSpinVectorREG)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3, int N>
struct TrinaryReturn<PScalarREG<T1>, PSpinVectorREG<T2,N>, PSpinVectorREG<T3,N>, FnWhere> {
  typedef PSpinVectorREG<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t, N>  Type_t;
};

template<class T1, class T2, class T3, int N>
inline typename TrinaryReturn<PScalarREG<T1>, PSpinVectorREG<T2,N>, PSpinVectorREG<T3,N>, FnWhere>::Type_t
where(const PScalarREG<T1>& a, const PSpinVectorREG<T2,N>& b, const PSpinVectorREG<T3,N>& c)
{
  typename TrinaryReturn<PScalarREG<T1>, PSpinVectorREG<T2,N>, PSpinVectorREG<T3,N>, FnWhere>::Type_t  d;

  // Not optimal - want to have where outside assignment
  for(int i=0; i < N; ++i)
    d.elem(i) = where(a.elem(), b.elem(i), c.elem(i));

  return d;
}


//! Specialization of primitive spin Vector class for 4 spin components
/*! 
 * Spin vector class supports gamma matrix algebra for 4 spin components
 */


//! Specialization of primitive spin Vector class for 2 spin components
/*! 
 * Spin vector class supports gamma matrix algebra for 2 spin components
 * NOTE: this can be used for spin projection tricks of a 4 component spinor
 * to 2 spin components, or a 2 spin component Dirac fermion in 2 dimensions
 */


/*! @} */   // end of group primspinvec

//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

template<class T1, int N>
struct JITType<PSpinVectorREG<T1,N> > 
{
  typedef PSpinVectorJIT<typename JITType<T1>::Type_t,N>  Type_t;
};


// Underlying word type
template<class T1, int N>
struct WordType<PSpinVectorREG<T1,N> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

// Fixed Precision
template<class T1, int N>
struct SinglePrecType< PSpinVectorREG<T1, N> > 
{
  typedef PSpinVectorREG< typename SinglePrecType<T1>::Type_t, N> Type_t;
};

template<class T1, int N>
struct DoublePrecType< PSpinVectorREG<T1, N> > 
{
  typedef PSpinVectorREG< typename DoublePrecType<T1>::Type_t, N> Type_t;
};

// Internally used scalars
template<class T, int N>
struct InternalScalar<PSpinVectorREG<T,N> > {
  typedef PScalarREG<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive into a scalar leaving grid alone
template<class T, int N>
struct PrimitiveScalar<PSpinVectorREG<T,N> > {
  typedef PScalarREG<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T, int N>
struct LatticeScalar<PSpinVectorREG<T,N> > {
  typedef PSpinVectorREG<typename LatticeScalar<T>::Type_t, N>  Type_t;
};

//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PSpinVectorREG) -> PSpinVectorREG
template<class T1, int N, class Op>
struct UnaryReturn<PSpinVectorREG<T1,N>, Op> {
  typedef PSpinVectorREG<typename UnaryReturn<T1, Op>::Type_t, N>  Type_t;
};

// Default binary(PScalarREG,PSpinVectorREG) -> PSpinVectorREG
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PScalarREG<T1>, PSpinVectorREG<T2,N>, Op> {
  typedef PSpinVectorREG<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PSpinMatrixREG,PSpinVectorREG) -> PSpinVectorREG
template<class T1, class T2, int N, class Op>
struct BinaryReturn< PSpinMatrixREG<T1,N>, PSpinVectorREG<T2,N>, Op> {
  typedef PSpinVectorREG<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};


// Default binary(PMatrixREG,PSpinVectorREG) -> PSpinVectorREG
template<class T1, class T2, int N, template <class,int> class C1, class Op>
struct BinaryReturn< PMatrixREG<T1,N,C1>, PSpinVectorREG<T2,N>, Op> {
  typedef PSpinVectorREG<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};


// Default binary(PSpinVectorREG,PScalarREG) -> PSpinVectorREG
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PSpinVectorREG<T1,N>, PScalarREG<T2>, Op> {
  typedef PSpinVectorREG<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PSpinVectorREG,PSpinVectorREG) -> PSpinVectorREG
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PSpinVectorREG<T1,N>, PSpinVectorREG<T2,N>, Op> {
  typedef PSpinVectorREG<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};


#if 0
template<class T1, class T2>
struct UnaryReturn<PScalarREG<T2>, OpCast<T1> > {
  typedef PScalarREG<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif


// Assignment is different
template<class T1, class T2, int N>
struct BinaryReturn<PSpinVectorREG<T1,N>, PSpinVectorREG<T2,N>, OpAssign > {
  typedef PSpinVectorREG<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinVectorREG<T1,N>, PSpinVectorREG<T2,N>, OpAddAssign > {
  typedef PSpinVectorREG<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PSpinVectorREG<T1,N>, PSpinVectorREG<T2,N>, OpSubtractAssign > {
  typedef PSpinVectorREG<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PSpinVectorREG<T1,N>, PScalarREG<T2>, OpMultiplyAssign > {
  typedef PSpinVectorREG<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PSpinVectorREG<T1,N>, PScalarREG<T2>, OpDivideAssign > {
  typedef PSpinVectorREG<T1,N> &Type_t;
};
 


// SpinVector
template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnNorm2 > {
  typedef PScalarREG<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnLocalNorm2 > {
  typedef PScalarREG<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinVectorREG<T1,N>, PSpinVectorREG<T2,N>, FnInnerProduct> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinVectorREG<T1,N>, PSpinVectorREG<T2,N>, FnLocalInnerProduct> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinVectorREG<T1,N>, PSpinVectorREG<T2,N>, FnInnerProductReal> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinVectorREG<T1,N>, PSpinVectorREG<T2,N>, FnLocalInnerProductReal> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};


template<class T1, class T2, int N>
inline PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>
localInnerProduct(const PSpinVectorREG<T1,N>& s1, const PSpinVectorREG<T2,N>& s2)
{
  PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  d;

  d.elem() = localInnerProduct(s1.elem(0), s2.elem(0));
  for(int i=1; i < N; ++i)
    d.elem() += localInnerProduct(s1.elem(i), s2.elem(i));

  return d;
}

template<class T1, class T2, int N>
inline PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>
localInnerProductReal(const PSpinVectorREG<T1,N>& s1, const PSpinVectorREG<T2,N>& s2)
{
  PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  d;

  d.elem() = localInnerProductReal(s1.elem(0), s2.elem(0));
  for(int i=1; i < N; ++i)
    d.elem() += localInnerProductReal(s1.elem(i), s2.elem(i));

  return d;
}

// Gamma algebra
template<int m, class T2, int N>
struct BinaryReturn<GammaConst<N,m>, PSpinVectorREG<T2,N>, OpGammaConstMultiply> {
  typedef PSpinVectorREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};

template<class T2, int N>
struct BinaryReturn<GammaType<N>, PSpinVectorREG<T2,N>, OpGammaTypeMultiply> {
  typedef PSpinVectorREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};

// Gamma algebra
template<int m, class T2, int N>
struct BinaryReturn<GammaConstDP<N,m>, PSpinVectorREG<T2,N>, OpGammaConstDPMultiply> {
  typedef PSpinVectorREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};

template<class T2, int N>
struct BinaryReturn<GammaTypeDP<N>, PSpinVectorREG<T2,N>, OpGammaTypeDPMultiply> {
  typedef PSpinVectorREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};

// Generic Spin projection
template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinProject > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinProject>::Type_t, (N>>1) >  Type_t;
};

// spin projection for each direction
template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinProjectDir0Plus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinProjectDir0Plus>::Type_t, (N>>1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinProjectDir1Plus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinProjectDir1Plus>::Type_t, (N>>1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinProjectDir2Plus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinProjectDir2Plus>::Type_t, (N>>1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinProjectDir3Plus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinProjectDir3Plus>::Type_t, (N>>1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinProjectDir0Minus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinProjectDir0Minus>::Type_t, (N>>1) > Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinProjectDir1Minus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinProjectDir1Minus>::Type_t, (N>>1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinProjectDir2Minus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinProjectDir2Minus>::Type_t, (N>>1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinProjectDir3Minus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinProjectDir3Minus>::Type_t, (N>>1) >  Type_t;
};


// Generic Spin reconstruction
template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinReconstruct > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinReconstruct>::Type_t, (N<<1) >  Type_t;
};

// spin reconstruction for each direction
template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinReconstructDir0Plus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinReconstructDir0Plus>::Type_t, (N<<1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinReconstructDir1Plus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinReconstructDir1Plus>::Type_t, (N<<1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinReconstructDir2Plus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinReconstructDir2Plus>::Type_t, (N<<1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinReconstructDir3Plus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinReconstructDir3Plus>::Type_t, (N<<1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinReconstructDir0Minus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinReconstructDir0Minus>::Type_t, (N<<1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinReconstructDir1Minus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinReconstructDir1Minus>::Type_t, (N<<1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinReconstructDir2Minus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinReconstructDir2Minus>::Type_t, (N<<1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnSpinReconstructDir3Minus > {
  typedef PSpinVectorREG<typename UnaryReturn<T, FnSpinReconstructDir3Minus>::Type_t, (N<<1) >  Type_t;
};




//! dest  = random  
template<class T, int N,  class T1, class T2, class T3>
inline void
fill_random(PSpinVectorREG<T,N>& d, T1& seed, T2& skewed_seed, const T3& seed_mult)
{
  // Loop over rows the slowest
  for(int i=0; i < N; ++i)
    fill_random(d.elem(i), seed, skewed_seed, seed_mult);
}


//! dest  = gaussian
template<class T, int N>
inline void
fill_gaussian(PSpinVectorREG<T,N>& d, PSpinVectorREG<T,N>& r1, PSpinVectorREG<T,N>& r2)
{
  for(int i=0; i < N; ++i)
    fill_gaussian(d.elem(i), r1.elem(i), r2.elem(i));
}




//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

/*! \addtogroup primspinvector */
/*! @{ */

// Peeking and poking
//! Extract spin vector components 
template<class T, int N>
struct UnaryReturn<PSpinVectorREG<T,N>, FnPeekSpinVectorREG > {
  typedef PScalarREG<typename UnaryReturn<T, FnPeekSpinVectorREG>::Type_t>  Type_t;
};

template<class T, int N>
inline typename UnaryReturn<PSpinVectorREG<T,N>, FnPeekSpinVectorREG>::Type_t
peekSpin(const PSpinVectorREG<T,N>& l, int row)
{
  typename UnaryReturn<PSpinVectorREG<T,N>, FnPeekSpinVectorREG>::Type_t  d;

  // Note, do not need to propagate down since the function is eaten at this level
  d.elem() = l.getRegElem(row);
  return d;
}

//! Insert spin vector components
template<class T1, class T2, int N>
inline PSpinVectorREG<T1,N>&
pokeSpin(PSpinVectorREG<T1,N>& l, const PScalarREG<T2>& r, int row)
{
  // Note, do not need to propagate down since the function is eaten at this level
  l.getRegElem(row) = r.elem();
  return l;
}



// SpinVector<4> = Gamma<4,m> * SpinVector<4>
// There are 16 cases here for Nd=4
template<class T2>
inline typename BinaryReturn<GammaConst<4,0>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,0>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,0>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;
  
  d.elem(0) =  r.elem(0);
  d.elem(1) =  r.elem(1);
  d.elem(2) =  r.elem(2);
  d.elem(3) =  r.elem(3);

  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,1>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,1>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,1>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = timesI(r.elem(3));
  d.elem(1) = timesI(r.elem(2));
  d.elem(2) = timesMinusI(r.elem(1));
  d.elem(3) = timesMinusI(r.elem(0));

  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,2>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,2>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,2>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = -r.elem(3);
  d.elem(1) =  r.elem(2);
  d.elem(2) =  r.elem(1);
  d.elem(3) = -r.elem(0);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,3>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,3>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,3>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = timesMinusI(r.elem(0));
  d.elem(1) = timesI(r.elem(1));
  d.elem(2) = timesMinusI(r.elem(2));
  d.elem(3) = timesI(r.elem(3));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,4>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,4>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,4>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = timesI(r.elem(2));
  d.elem(1) = timesMinusI(r.elem(3));
  d.elem(2) = timesMinusI(r.elem(0));
  d.elem(3) = timesI(r.elem(1));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,5>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,5>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,5>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = -r.elem(1);
  d.elem(1) =  r.elem(0);
  d.elem(2) = -r.elem(3);
  d.elem(3) =  r.elem(2);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,6>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,6>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,6>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = timesMinusI(r.elem(1));
  d.elem(1) = timesMinusI(r.elem(0));
  d.elem(2) = timesMinusI(r.elem(3));
  d.elem(3) = timesMinusI(r.elem(2));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,7>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,7>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,7>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) =  r.elem(2);
  d.elem(1) =  r.elem(3);
  d.elem(2) = -r.elem(0);
  d.elem(3) = -r.elem(1);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,8>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,8>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,8>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) =  r.elem(2);
  d.elem(1) =  r.elem(3);
  d.elem(2) =  r.elem(0);
  d.elem(3) =  r.elem(1);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,9>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,9>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,9>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = timesI(r.elem(1));
  d.elem(1) = timesI(r.elem(0));
  d.elem(2) = timesMinusI(r.elem(3));
  d.elem(3) = timesMinusI(r.elem(2));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,10>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,10>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,10>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = -r.elem(1);
  d.elem(1) =  r.elem(0);
  d.elem(2) =  r.elem(3);
  d.elem(3) = -r.elem(2);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,11>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,11>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,11>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = timesMinusI(r.elem(2));
  d.elem(1) = timesI(r.elem(3));
  d.elem(2) = timesMinusI(r.elem(0));
  d.elem(3) = timesI(r.elem(1));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,12>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,12>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,12>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = timesI(r.elem(0));
  d.elem(1) = timesMinusI(r.elem(1));
  d.elem(2) = timesMinusI(r.elem(2));
  d.elem(3) = timesI(r.elem(3));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,13>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,13>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,13>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = -r.elem(3);
  d.elem(1) =  r.elem(2);
  d.elem(2) = -r.elem(1);
  d.elem(3) =  r.elem(0);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,14>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,14>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,14>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = timesMinusI(r.elem(3));
  d.elem(1) = timesMinusI(r.elem(2));
  d.elem(2) = timesMinusI(r.elem(1));
  d.elem(3) = timesMinusI(r.elem(0));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,15>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,15>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,15>, PSpinVectorREG<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) =  r.elem(0);
  d.elem(1) =  r.elem(1);
  d.elem(2) = -r.elem(2);
  d.elem(3) = -r.elem(3);
  
  return d;
}


// SpinVector<2> = SpinProject(SpinVector<4>)
// There are 4 cases here for Nd=4 for each forward/backward direction
template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir0Minus>::Type_t
spinProjectDir0Minus(const PSpinVectorREG<T,4>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir0Minus>::Type_t  d;

  /*                              ( 1  0  0 -i)  ( a0 )    ( a0 - i a3 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -i  0)  ( a1 )  = ( a1 - i a2 )
   *                    0         ( 0  i  1  0)  ( a2 )    ( a2 + i a1 )
   *                              ( i  0  0  1)  ( a3 )    ( a3 + i a0 )

   * Therefore the top components are

   *      ( b0r + i b0i )  =  ( {a0r + a3i} + i{a0i - a3r} )
   *      ( b1r + i b1i )     ( {a1r + a2i} + i{a1i - a2r} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r - a1i} + i{a2i + a1r} )  =  ( - b1i + i b1r )
   *      ( b3r + i b3i )     ( {a3r - a0i} + i{a3i + a0r} )     ( - b0i + i b0r ) 
   */
  d.elem(0) = s1.elem(0) - timesI(s1.elem(3));
  d.elem(1) = s1.elem(1) - timesI(s1.elem(2));

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir1Minus>::Type_t
spinProjectDir1Minus(const PSpinVectorREG<T,4>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir1Minus>::Type_t  d;

  /*                              ( 1  0  0  1)  ( a0 )    ( a0 + a3 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -1  0)  ( a1 )  = ( a1 - a2 )
   *                    1         ( 0 -1  1  0)  ( a2 )    ( a2 - a1 )
   *                              ( 1  0  0  1)  ( a3 )    ( a3 + a0 )
	 
   * Therefore the top components are
      
   *      ( b0r + i b0i )  =  ( {a0r + a3r} + i{a0i + a3i} )
   *      ( b1r + i b1i )     ( {a1r - a2r} + i{a1i - a2i} )
      
   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r - a1r} + i{a2i - a1i} )  =  ( - b1r - i b1i )
   *      ( b3r + i b3i )     ( {a3r + a0r} + i{a3i + a0i} )     (   b0r + i b0i ) 
   */
  d.elem(0) = s1.elem(0) + s1.elem(3);
  d.elem(1) = s1.elem(1) - s1.elem(2);

  return d;
}
    
template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir2Minus>::Type_t
spinProjectDir2Minus(const PSpinVectorREG<T,4>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir2Minus>::Type_t  d;

  /*                              ( 1  0 -i  0)  ( a0 )    ( a0 - i a2 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1  0  i)  ( a1 )  = ( a1 + i a3 )
   *                    2         ( i  0  1  0)  ( a2 )    ( a2 + i a0 )
   *                              ( 0 -i  0  1)  ( a3 )    ( a3 - i a1 )

   * Therefore the top components are
      
   *      ( b0r + i b0i )  =  ( {a0r + a2i} + i{a0i - a2r} )
   *      ( b1r + i b1i )     ( {a1r - a3i} + i{a1i + a3r} )
      
   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r - a0i} + i{a2i + a0r} )  =  ( - b0i + i b0r )
   *      ( b3r + i b3i )     ( {a3r + a1i} + i{a3i - a1r} )     (   b1i - i b1r )
   */
  d.elem(0) = s1.elem(0) - timesI(s1.elem(2));
  d.elem(1) = s1.elem(1) + timesI(s1.elem(3));

  return d;
}
    
template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir3Minus>::Type_t
spinProjectDir3Minus(const PSpinVectorREG<T,4>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir3Minus>::Type_t  d;

  /*                              ( 1  0 -1  0)  ( a0 )    ( a0 - a2 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1  0 -1)  ( a1 )  = ( a1 - a3 )
   *                    3         (-1  0  1  0)  ( a2 )    ( a2 - a0 )
   *                              ( 0 -1  0  1)  ( a3 )    ( a3 - a1 )
      
   * Therefore the top components are
      
   *      ( b0r + i b0i )  =  ( {a0r - a2r} + i{a0i - a2i} )
   *      ( b1r + i b1i )     ( {a1r - a3r} + i{a1i - a3i} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r - a0r} + i{a2i - a0i} )  =  ( - b0r - i b0i )
   *      ( b3r + i b3i )     ( {a3r - a1r} + i{a3i - a1i} )     ( - b1r - i b1i ) 
   */
  d.elem(0) = s1.elem(0) - s1.elem(2);
  d.elem(1) = s1.elem(1) - s1.elem(3);

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir0Plus>::Type_t
spinProjectDir0Plus(const PSpinVectorREG<T,4>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir0Plus>::Type_t  d;

  /*                              ( 1  0  0 +i)  ( a0 )    ( a0 + i a3 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1 +i  0)  ( a1 )  = ( a1 + i a2 )
   *                    0         ( 0 -i  1  0)  ( a2 )    ( a2 - i a1 )
   *                              (-i  0  0  1)  ( a3 )    ( a3 - i a0 )

   * Therefore the top components are

   *      ( b0r + i b0i )  =  ( {a0r - a3i} + i{a0i + a3r} )
   *      ( b1r + i b1i )     ( {a1r - a2i} + i{a1i + a2r} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r + a1i} + i{a2i - a1r} )  =  ( b1i - i b1r )
   *      ( b3r + i b3i )     ( {a3r + a0i} + i{a3i - a0r} )     ( b0i - i b0r ) 
   */
  d.elem(0) = s1.elem(0) + timesI(s1.elem(3));
  d.elem(1) = s1.elem(1) + timesI(s1.elem(2));

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir1Plus>::Type_t
spinProjectDir1Plus(const PSpinVectorREG<T,4>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir1Plus>::Type_t  d;

  /*                              ( 1  0  0 -1)  ( a0 )    ( a0 - a3 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  1  0)  ( a1 )  = ( a1 + a2 )
   *                    1         ( 0  1  1  0)  ( a2 )    ( a2 + a1 )
   *                              (-1  0  0  1)  ( a3 )    ( a3 - a0 )

   * Therefore the top components are

   *      ( b0r + i b0i )  =  ( {a0r - a3r} + i{a0i - a3i} )
   *      ( b1r + i b1i )     ( {a1r + a2r} + i{a1i + a2i} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r + a1r} + i{a2i + a1i} )  =  (   b1r + i b1i )
   *      ( b3r + i b3i )     ( {a3r - a0r} + i{a3i - a0i} )     ( - b0r - i b0i ) 
   */
  d.elem(0) = s1.elem(0) - s1.elem(3);
  d.elem(1) = s1.elem(1) + s1.elem(2);

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir2Plus>::Type_t
spinProjectDir2Plus(const PSpinVectorREG<T,4>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir2Plus>::Type_t  d;

  /*                              ( 1  0  i  0)  ( a0 )    ( a0 + i a2 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0 -i)  ( a1 )  = ( a1 - i a3 )
   *                    2         (-i  0  1  0)  ( a2 )    ( a2 - i a0 )
   *                              ( 0  i  0  1)  ( a3 )    ( a3 + i a1 )

   * Therefore the top components are

   *      ( b0r + i b0i )  =  ( {a0r - a2i} + i{a0i + a2r} )
   *      ( b1r + i b1i )     ( {a1r + a3i} + i{a1i - a3r} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r + a0i} + i{a2i - a0r} )  =  (   b0i - i b0r )
   *      ( b3r + i b3i )     ( {a3r - a1i} + i{a3i + a1r} )     ( - b1i + i b1r ) 
   */
  d.elem(0) = s1.elem(0) + timesI(s1.elem(2));
  d.elem(1) = s1.elem(1) - timesI(s1.elem(3));

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir3Plus>::Type_t
spinProjectDir3Plus(const PSpinVectorREG<T,4>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,4>, FnSpinProjectDir3Plus>::Type_t  d;

  /*                              ( 1  0  1  0)  ( a0 )    ( a0 + a2 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0  1)  ( a1 )  = ( a1 + a3 )
   *                    3         ( 1  0  1  0)  ( a2 )    ( a2 + a0 )
   *                              ( 0  1  0  1)  ( a3 )    ( a3 + a1 )

   * Therefore the top components are

   *      ( b0r + i b0i )  =  ( {a0r + a2r} + i{a0i + a2i} )
   *      ( b1r + i b1i )     ( {a1r + a3r} + i{a1i + a3i} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r + a0r} + i{a2i + a0i} )  =  ( b0r + i b0i )
   *      ( b3r + i b3i )     ( {a3r + a1r} + i{a3i + a1i} )     ( b1r + i b1i ) 
   */
  d.elem(0) = s1.elem(0) + s1.elem(2);
  d.elem(1) = s1.elem(1) + s1.elem(3);

  return d;
}


// SpinVector<4> = SpinReconstruct(SpinVector<2>)
// There are 4 cases here for Nd=4 for each forward/backward direction
template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir0Minus>::Type_t
spinReconstructDir0Minus(const PSpinVectorREG<T,2>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir0Minus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = timesI(s1.elem(1));
  d.elem(3) = timesI(s1.elem(0));

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir1Minus>::Type_t
spinReconstructDir1Minus(const PSpinVectorREG<T,2>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir1Minus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = -s1.elem(1);
  d.elem(3) = s1.elem(0);

  return d;
}


template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir2Minus>::Type_t
spinReconstructDir2Minus(const PSpinVectorREG<T,2>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir2Minus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = timesI(s1.elem(0));
  d.elem(3) = timesMinusI(s1.elem(1));

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir3Minus>::Type_t
spinReconstructDir3Minus(const PSpinVectorREG<T,2>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir3Minus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = -s1.elem(0);
  d.elem(3) = -s1.elem(1);

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir0Plus>::Type_t
spinReconstructDir0Plus(const PSpinVectorREG<T,2>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir0Plus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = timesMinusI(s1.elem(1));
  d.elem(3) = timesMinusI(s1.elem(0));

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir1Plus>::Type_t
spinReconstructDir1Plus(const PSpinVectorREG<T,2>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir1Plus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = s1.elem(1);
  d.elem(3) = -s1.elem(0);

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir2Plus>::Type_t
spinReconstructDir2Plus(const PSpinVectorREG<T,2>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir2Plus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = timesMinusI(s1.elem(0));
  d.elem(3) = timesI(s1.elem(1));

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir3Plus>::Type_t
spinReconstructDir3Plus(const PSpinVectorREG<T,2>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,2>, FnSpinReconstructDir3Plus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = s1.elem(0);
  d.elem(3) = s1.elem(1);

  return d;
}

//-----------------------------------------------

// SpinVector<4> = GammaDP<4,m> * SpinVector<4>
// There are 16 cases here for Nd=4
template<class T2>
inline typename BinaryReturn<GammaConstDP<4,0>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,0>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,0>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;
  
  d.elem(0) =  r.elem(0);
  d.elem(1) =  r.elem(1);
  d.elem(2) =  r.elem(2);
  d.elem(3) =  r.elem(3);

  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,1>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,1>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,1>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;
  
  d.elem(0) = timesMinusI(r.elem(3));
  d.elem(1) = timesMinusI(r.elem(2));
  d.elem(2) = timesI(r.elem(1));
  d.elem(3) = timesI(r.elem(0));

  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,2>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,2>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,2>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;

  d.elem(0) = -r.elem(3);
  d.elem(1) =  r.elem(2);
  d.elem(2) =  r.elem(1);
  d.elem(3) = -r.elem(0);

  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,3>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,3>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,3>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;

  d.elem(0) = timesI(r.elem(0));
  d.elem(1) = timesMinusI(r.elem(1));
  d.elem(2) = timesI(r.elem(2));
  d.elem(3) = timesMinusI(r.elem(3));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,4>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,4>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,4>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;

  d.elem(0) = timesMinusI(r.elem(2));
  d.elem(1) = timesI(r.elem(3));
  d.elem(2) = timesI(r.elem(0));
  d.elem(3) = timesMinusI(r.elem(1));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,5>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,5>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,5>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;

  d.elem(0) = -r.elem(1);
  d.elem(1) =  r.elem(0);
  d.elem(2) = -r.elem(3);
  d.elem(3) =  r.elem(2);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,6>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,6>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,6>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;

  d.elem(0) = timesI(r.elem(1));
  d.elem(1) = timesI(r.elem(0));
  d.elem(2) = timesI(r.elem(3));
  d.elem(3) = timesI(r.elem(2));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,7>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,7>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,7>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;

  d.elem(0) =  r.elem(2);
  d.elem(1) =  r.elem(3);
  d.elem(2) = -r.elem(0);
  d.elem(3) = -r.elem(1);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,8>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,8>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,8>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;

  d.elem(0) =  r.elem(0);
  d.elem(1) =  r.elem(1);
  d.elem(2) = -r.elem(2);
  d.elem(3) = -r.elem(3);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,9>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,9>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,9>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;

  d.elem(0) = timesI(r.elem(3));
  d.elem(1) = timesI(r.elem(2));
  d.elem(2) = timesI(r.elem(1));
  d.elem(3) = timesI(r.elem(0));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,10>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,10>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,10>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;

  d.elem(0) =  r.elem(3);
  d.elem(1) = -r.elem(2);
  d.elem(2) = -r.elem(1);
  d.elem(3) =  r.elem(0);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,11>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,11>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,11>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;

  d.elem(0) = timesI(r.elem(0));
  d.elem(1) = timesMinusI(r.elem(1));
  d.elem(2) = timesMinusI(r.elem(2));
  d.elem(3) = timesI(r.elem(3));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,12>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,12>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,12>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;

  d.elem(0) = timesI(r.elem(2));
  d.elem(1) = timesMinusI(r.elem(3));
  d.elem(2) = timesI(r.elem(0));
  d.elem(3) = timesMinusI(r.elem(1));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,13>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,13>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,13>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;

  d.elem(0) = -r.elem(1);
  d.elem(1) =  r.elem(0);
  d.elem(2) =  r.elem(3);
  d.elem(3) = -r.elem(2);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,14>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,14>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,14>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;

  d.elem(0) = timesI(r.elem(1));
  d.elem(1) = timesI(r.elem(0));
  d.elem(2) = timesMinusI(r.elem(3));
  d.elem(3) = timesMinusI(r.elem(2));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,15>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,15>&, const PSpinVectorREG<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,15>, PSpinVectorREG<T2,4>, OpGammaConstDPMultiply>::Type_t  d;

  d.elem(0) = -r.elem(2);
  d.elem(1) = -r.elem(3);
  d.elem(2) = -r.elem(0);
  d.elem(3) = -r.elem(1);
  
  return d;
}


//-----------------------------------------------------------------------------
//! PSpinVectorREG<T,4> = P_+ * PSpinVectorREG<T,4>
template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,4>, FnChiralProjectPlus>::Type_t
chiralProjectPlus(const PSpinVectorREG<T,4>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,4>, FnChiralProjectPlus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  zero_rep(d.elem(2));
  zero_rep(d.elem(3));

  return d;
}

//! PSpinVectorREG<T,4> = P_- * PSpinVectorREG<T,4>
template<class T>
inline typename UnaryReturn<PSpinVectorREG<T,4>, FnChiralProjectMinus>::Type_t
chiralProjectMinus(const PSpinVectorREG<T,4>& s1)
{
  typename UnaryReturn<PSpinVectorREG<T,4>, FnChiralProjectMinus>::Type_t  d;

  zero_rep(d.elem(0));
  zero_rep(d.elem(1));
  d.elem(2) = s1.elem(2);
  d.elem(3) = s1.elem(3);

  return d;
}


/*! @} */   // end of group primspinvector

} // namespace QDP

#endif
