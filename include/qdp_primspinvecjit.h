// -*- C++ -*-

/*! \file
 * \brief Primitive Spin Vector
 */


#ifndef QDP_PRIMSPINVECJIT_H
#define QDP_PRIMSPINVECJIT_H

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
template <class T, int N> class PSpinVectorJIT: public BaseJIT<T,N>
{
public:

  // Default constructing should be possible
  PSpinVectorJIT(){}

  template<class T1>
  PSpinVectorJIT(const PSpinVectorREG<T1,N>& a)
  {
    for(int i=0; i < N; i++) 
      this->elem(i) = a.elem(i);
  }



  template<class T1> 
  inline 
  PSpinVectorJIT& operator=(const PSpinVectorREG<T1,N>& rhs)
  {
    for(int i=0; i < N; ++i)
      this->elem(i) = rhs.elem(i);
    return *this;
  }


  //! PSpinVectorJIT += PSpinVectorJIT
  template<class T1>
  inline
  PSpinVectorJIT& operator+=(const PSpinVectorREG<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	this->elem(i) += rhs.elem(i);

      return *this;
    }


  template<class T1>
  inline
  PSpinVectorJIT& operator*=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) *= rhs.elem();

      return *this;
    }

  template<class T1>
  inline
  PSpinVectorJIT& operator-=(const PSpinVectorREG<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	this->elem(i) -= rhs.elem(i);

      return *this;
    }

  //! PSpinVectorJIT /= PScalarJIT
  template<class T1>
  inline
  PSpinVectorJIT& operator/=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	this->elem(i) /= rhs.elem();

      return *this;
    }


public:
        T& elem(int i)       {return this->arrayF(i);}
  const T& elem(int i) const {return this->arrayF(i);}
};






// Primitive Vectors

template<class T1, int N>
inline typename UnaryReturn<PSpinVectorJIT<T1,N>, OpUnaryPlus>::Type_t
operator+(const PSpinVectorJIT<T1,N>& l)
{
  typename UnaryReturn<PSpinVectorJIT<T1,N>, OpUnaryPlus>::Type_t  d;
  
  for(int i=0; i < N; ++i)
    d.elem(i) = +l.elem(i);
  return d;
}


template<class T1, int N>
inline typename UnaryReturn<PSpinVectorJIT<T1,N>, OpUnaryMinus>::Type_t
operator-(const PSpinVectorJIT<T1,N>& l)
{
  typename UnaryReturn<PSpinVectorJIT<T1,N>, OpUnaryMinus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = -l.elem(i);
  return d;
}


template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinVectorJIT<T1,N>, PSpinVectorJIT<T2,N>, OpAdd>::Type_t
operator+(const PSpinVectorJIT<T1,N>& l, const PSpinVectorJIT<T2,N>& r)
{
  typename BinaryReturn<PSpinVectorJIT<T1,N>, PSpinVectorJIT<T2,N>, OpAdd>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) + r.elem(i);
  return d;
}


template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinVectorJIT<T1,N>, PSpinVectorJIT<T2,N>, OpSubtract>::Type_t
operator-(const PSpinVectorJIT<T1,N>& l, const PSpinVectorJIT<T2,N>& r)
{
  typename BinaryReturn<PSpinVectorJIT<T1,N>, PSpinVectorJIT<T2,N>, OpSubtract>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) - r.elem(i);
  return d;
}


// PSpinVectorJIT * PScalarJIT
template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinVectorJIT<T1,N>, PScalarJIT<T2>, OpMultiply>::Type_t
operator*(const PSpinVectorJIT<T1,N>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PSpinVectorJIT<T1,N>, PScalarJIT<T2>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) * r.elem();
  return d;
}

// Optimized  PSpinVectorJIT * adj(PScalarJIT)
template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinVectorJIT<T1,N>, PScalarJIT<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(const PSpinVectorJIT<T1,N>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PSpinVectorJIT<T1,N>, PScalarJIT<T2>, OpMultiplyAdj>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = multiplyAdj(l.elem(i), r.elem());
  return d;
}


// PScalarJIT * PSpinVectorJIT
template<class T1, class T2, int N>
inline typename BinaryReturn<PScalarJIT<T1>, PSpinVectorJIT<T2,N>, OpMultiply>::Type_t
operator*(const PScalarJIT<T1>& l, const PSpinVectorJIT<T2,N>& r)
{
  typename BinaryReturn<PScalarJIT<T1>, PSpinVectorJIT<T2,N>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem() * r.elem(i);
  return d;
}

// Optimized  adj(PScalarJIT) * PSpinVectorJIT
template<class T1, class T2, int N>
inline typename BinaryReturn<PScalarJIT<T1>, PSpinVectorJIT<T2,N>, OpAdjMultiply>::Type_t
adjMultiply(const PScalarJIT<T1>& l, const PSpinVectorJIT<T2,N>& r)
{
  typename BinaryReturn<PScalarJIT<T1>, PSpinVectorJIT<T2,N>, OpAdjMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = adjMultiply(l.elem(), r.elem(i));
  return d;
}


// PMatrix * PSpinVectorJIT
template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinVectorJIT<T2,N>, OpMultiply>::Type_t
operator*(const PSpinMatrixJIT<T1,N>& l, const PSpinVectorJIT<T2,N>& r)
{
  typename BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinVectorJIT<T2,N>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
  {
    d.elem(i) = l.elem(i,0) * r.elem(0);
    for(int j=1; j < N; ++j)
      d.elem(i) += l.elem(i,j) * r.elem(j);
  }

  return d;
}


// PMatrix * PSpinVectorJIT
template<class T1, class T2,  template<class,int> class C, int N>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PSpinVectorJIT<T2,N>, OpMultiply>::Type_t
operator*(const PSpinMatrixJIT<T1,N>& l, const PSpinVectorJIT<T2,N>& r)
{
  typename BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinVectorJIT<T2,N>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
  {
    d.elem(i) = l.elem(i,0) * r.elem(0);
    for(int j=1; j < N; ++j)
      d.elem(i) += l.elem(i,j) * r.elem(j);
  }

  return d;
}

// Optimized  adj(PMatrixJIT)*PSpinVectorJIT
template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinVectorJIT<T2,N>, OpAdjMultiply>::Type_t
adjMultiply(const PSpinMatrixJIT<T1,N>& l, const PSpinVectorJIT<T2,N>& r)
{
  typename BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinVectorJIT<T2,N>, OpAdjMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
  {
    d.elem(i) = adjMultiply(l.elem(0,i), r.elem(0));
    for(int j=1; j < N; ++j)
      d.elem(i) += adjMultiply(l.elem(j,i), r.elem(j));
  }

  return d;
}

// Optimized  adj(PMatrixJIT)*PVector
template<class T1, class T2, int N, template<class,int> class C1>
inline typename BinaryReturn<PMatrixJIT<T1,N,C1>, PSpinVectorJIT<T2,N>, OpAdjMultiply>::Type_t
adjMultiply(const PMatrixJIT<T1,N,C1>& l, const PSpinVectorJIT<T2,N>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C1>, PSpinVectorJIT<T2,N>, OpAdjMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
  {
    d.elem(i) = adjMultiply(l.elem(0,i), r.elem(0));
    for(int j=1; j < N; ++j)
      d.elem(i) += adjMultiply(l.elem(j,i), r.elem(j));
  }

  return d;
}

template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinVectorJIT<T1,N>, PScalarJIT<T2>, OpDivide>::Type_t
operator/(const PSpinVectorJIT<T1,N>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PSpinVectorJIT<T1,N>, PScalarJIT<T2>, OpDivide>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) / r.elem();
  return d;
}



//! PSpinVectorJIT = Re(PSpinVectorJIT)
template<class T, int N>
inline typename UnaryReturn<PSpinVectorJIT<T,N>, FnReal>::Type_t
real(const PSpinVectorJIT<T,N>& s1)
{
  typename UnaryReturn<PSpinVectorJIT<T,N>, FnReal>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = real(s1.elem(i));

  return d;
}


//! PSpinVectorJIT = Im(PSpinVectorJIT)
template<class T, int N>
inline typename UnaryReturn<PSpinVectorJIT<T,N>, FnImag>::Type_t
imag(const PSpinVectorJIT<T,N>& s1)
{
  typename UnaryReturn<PSpinVectorJIT<T,N>, FnImag>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = imag(s1.elem(i));

  return d;
}


//! PSpinVectorJIT<T> = (PSpinVectorJIT<T> , PSpinVectorJIT<T>)
template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinVectorJIT<T1,N>, PSpinVectorJIT<T2,N>, FnCmplx>::Type_t
cmplx(const PSpinVectorJIT<T1,N>& s1, const PSpinVectorJIT<T2,N>& s2)
{
  typename BinaryReturn<PSpinVectorJIT<T1,N>, PSpinVectorJIT<T2,N>, FnCmplx>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = cmplx(s1.elem(i), s2.elem(i));

  return d;
}


//-----------------------------------------------------------------------------
// Functions
// Conjugate
template<class T1, int N>
inline typename UnaryReturn<PSpinVectorJIT<T1,N>, FnConjugate>::Type_t
conj(const PSpinVectorJIT<T1,N>& l)
{
  typename UnaryReturn<PSpinVectorJIT<T1,N>, FnConjugate>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = conj(l.elem(i));

  return d;
}

//! PSpinVectorJIT = i * PSpinVectorJIT
template<class T, int N>
inline typename UnaryReturn<PSpinVectorJIT<T,N>, FnTimesI>::Type_t
timesI(const PSpinVectorJIT<T,N>& s1)
{
  typename UnaryReturn<PSpinVectorJIT<T,N>, FnTimesI>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = timesI(s1.elem(i));

  return d;
}

//! PSpinVectorJIT = -i * PSpinVectorJIT
template<class T, int N>
inline typename UnaryReturn<PSpinVectorJIT<T,N>, FnTimesMinusI>::Type_t
timesMinusI(const PSpinVectorJIT<T,N>& s1)
{
  typename UnaryReturn<PSpinVectorJIT<T,N>, FnTimesMinusI>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = timesMinusI(s1.elem(i));

  return d;
}


//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T, int N>
inline typename UnaryReturn<PSpinVectorJIT<T,N>, FnGetSite>::Type_t
getSite(const PSpinVectorJIT<T,N>& s1, int innersite)
{ 
  typename UnaryReturn<PSpinVectorJIT<T,N>, FnGetSite>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = getSite(s1.elem(i), innersite);

  return d;
}


//! Insert color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2, int N>
inline typename UnaryReturn<PSpinVectorJIT<T1,N>, FnPokeColorVectorREG >::Type_t&
pokeColor(PSpinVectorJIT<T1,N>& l, const PSpinVectorREG<T2,N>& r, llvm::Value* row)
{
  typedef typename UnaryReturn<PSpinVectorJIT<T1,N>, FnPokeColorVectorREG >::Type_t  Return_t;

  for(int i=0; i < N; ++i)
    pokeColor(l.elem(i),r.elem(i),row);
  return static_cast<Return_t&>(l);
}



//! dest = 0
template<class T, int N> 
inline void 
zero_rep(PSpinVectorJIT<T,N> dest) 
{
  for(int i=0; i < N; ++i)
    zero_rep(dest.elem(i));
}


//! dest [some type] = source [some type]
template<class T, class T1, int N>
inline void 
copy_site(PSpinVectorJIT<T,N> d, int isite, const PSpinVectorJIT<T1,N>& s1)
{
  for(int i=0; i < N; ++i)
    copy_site(d.elem(i), isite, s1.elem(i));
}

//! dest [some type] = source [some type]
template<class T, class T1, int N>
inline void 
copy_site(PSpinVectorJIT<T,N> d, int isite, const PScalarJIT<T1>& s1)
{
  for(int i=0; i < N; ++i)
    copy_site(d.elem(i), isite, s1.elem());
}


//! gather several inner sites together
template<class T, class T1, int N>
inline void 
gather_sites(PSpinVectorJIT<T,N> d, 
	     const PSpinVectorJIT<T1,N>& s0, int i0, 
	     const PSpinVectorJIT<T1,N>& s1, int i1,
	     const PSpinVectorJIT<T1,N>& s2, int i2,
	     const PSpinVectorJIT<T1,N>& s3, int i3)
{
  for(int i=0; i < N; ++i)
    gather_sites(d.elem(i), 
		 s0.elem(i), i0, 
		 s1.elem(i), i1, 
		 s2.elem(i), i2, 
		 s3.elem(i), i3);
}







template<class T, int N>
inline typename UnaryReturn<PSpinVectorJIT<T,N>, FnLocalNorm2>::Type_t
localNorm2(const PSpinVectorJIT<T,N>& s1)
{
  typename UnaryReturn<PSpinVectorJIT<T,N>, FnLocalNorm2>::Type_t  d;

  d.elem() = localNorm2(s1.elem(0));
  for(int i=1; i < N; ++i)
    d.elem() += localNorm2(s1.elem(i));

  return d;
}




//! PSpinVectorJIT<T> = where(PScalarJIT, PSpinVectorJIT, PSpinVectorJIT)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3, int N>
struct TrinaryReturn<PScalarJIT<T1>, PSpinVectorJIT<T2,N>, PSpinVectorJIT<T3,N>, FnWhere> {
  typedef PSpinVectorJIT<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t, N>  Type_t;
};

template<class T1, class T2, class T3, int N>
inline typename TrinaryReturn<PScalarJIT<T1>, PSpinVectorJIT<T2,N>, PSpinVectorJIT<T3,N>, FnWhere>::Type_t
where(const PScalarJIT<T1>& a, const PSpinVectorJIT<T2,N>& b, const PSpinVectorJIT<T3,N>& c)
{
  typename TrinaryReturn<PScalarJIT<T1>, PSpinVectorJIT<T2,N>, PSpinVectorJIT<T3,N>, FnWhere>::Type_t  d;

  // Not optimal - want to have where outside assignment
  for(int i=0; i < N; ++i)
    d.elem(i) = where(a.elem(), b.elem(i), c.elem(i));

  return d;
}


//! Specialization of primitive spin Vector class for 4 spin components


/*! @} */   // end of group primspinvec

//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------


template<class T1, int N>
struct ScalarType< PSpinVectorJIT<T1,N> >
{
  typedef PSpinVectorJIT< typename ScalarType<T1>::Type_t,N > Type_t;
};


template<class T1, int N>
struct REGType<PSpinVectorJIT<T1,N> > 
{
  typedef PSpinVectorREG<typename REGType<T1>::Type_t,N>  Type_t;
};

template<class T1, int N>
struct BASEType<PSpinVectorJIT<T1,N> > 
{
  typedef PSpinVector<typename BASEType<T1>::Type_t,N>  Type_t;
};


// Underlying word type
template<class T1, int N>
struct WordType<PSpinVectorJIT<T1,N> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

// Fixed Precision
template<class T1, int N>
struct SinglePrecType< PSpinVectorJIT<T1, N> > 
{
  typedef PSpinVectorJIT< typename SinglePrecType<T1>::Type_t, N> Type_t;
};

template<class T1, int N>
struct DoublePrecType< PSpinVectorJIT<T1, N> > 
{
  typedef PSpinVectorJIT< typename DoublePrecType<T1>::Type_t, N> Type_t;
};

// Internally used scalars
template<class T, int N>
struct InternalScalar<PSpinVectorJIT<T,N> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive into a scalar leaving grid alone
template<class T, int N>
struct PrimitiveScalar<PSpinVectorJIT<T,N> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T, int N>
struct LatticeScalar<PSpinVectorJIT<T,N> > {
  typedef PSpinVectorJIT<typename LatticeScalar<T>::Type_t, N>  Type_t;
};


//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------


template<class T1, class T2, int N>
inline PSpinVectorJIT<T1,N>&
pokeSpin(PSpinVectorJIT<T1,N>& l, const PScalarREG<T2>& r, llvm::Value* row)
{
  l.getJitElem(row) = r.elem();
  return l;
}


//! dest  = random  
template<class T, int N,  class T1, class T2, class T3>
inline void
fill_random_jit(PSpinVectorJIT<T,N> d, T1 seed, T2 skewed_seed, const T3& seed_mult)
{
  // Loop over rows the slowest
  for(int i=0; i < N; ++i)
    fill_random_jit(d.elem(i), seed, skewed_seed, seed_mult);
}


//! dest  = gaussian
template<class T,class T2, int N>
inline void
fill_gaussian(PSpinVectorJIT<T,N> d, PSpinVectorREG<T2,N>& r1, PSpinVectorREG<T2,N>& r2)
{
  for(int i=0; i < N; ++i)
    fill_gaussian(d.elem(i), r1.elem(i), r2.elem(i));
}


/*! @} */   // end of group primspinvector

} // namespace QDP

#endif
