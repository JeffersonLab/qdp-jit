// -*- C++ -*-

/*! \file
 * \brief Primitive Matrix
 */

#ifndef QDP_PRIMMATRIXJIT_H
#define QDP_PRIMMATRIXJIT_H

namespace QDP {


//-------------------------------------------------------------------------------------
/*! \addtogroup primmatrix Matrix primitive
 * \ingroup fiber
 *
 * Primitive type that transforms like a matrix
 *
 * @{
 */



//! Primitive Matrix class
/*!
 * All Matrix classes inherit this class
 * NOTE: For efficiency, there can be no virtual methods, so the data
 * portion is a part of the generic class, hence it is called a domain
 * and not a category
 */
template <class T, int N, template<class,int> class C> class PMatrixJIT : public JV<T,N*N>
{
public:
  typedef C<T,N>  CC;

  PMatrixJIT(Jit& j,int r , int of , int ol): JV<T,N*N>(j,r,of,ol) {}
  PMatrixJIT(Jit& j): JV<T,N*N>(j) {}


  //! PMatrixJIT = PScalarJIT
  /*! Fill with primitive scalar */
  template<class T1>
  inline
  CC& assign(const PScalarJIT<T1>& rhs)
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  if (i == j)
	    elem(i,j) = rhs.elem();
	  else
	    zero_rep(elem(i,j));

      return static_cast<CC&>(*this);
    }

  //! PMatrixJIT = PMatrixJIT
  /*! Set equal to another PMatrixJIT */
  template<class T1>
  inline
  CC& assign(const C<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) = rhs.elem(i,j);

      return static_cast<CC&>(*this);
    }

#if 0
  PMatrixJIT& assign(const PMatrixJIT& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) = rhs.elem(i,j);

      return static_cast<PMatrixJIT&>(*this);
    }
#endif

  //! PMatrixJIT += PMatrixJIT
  template<class T1>
  inline
  CC& operator+=(const C<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) += rhs.elem(i,j);

      return static_cast<CC&>(*this);
    }

  //! PMatrixJIT -= PMatrixJIT
  template<class T1>
  inline
  CC& operator-=(const C<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) -= rhs.elem(i,j);

      return static_cast<CC&>(*this);
    }

  //! PMatrixJIT += PScalarJIT
  template<class T1>
  inline
  CC& operator+=(const PScalarJIT<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i,i) += rhs.elem();

      return static_cast<CC&>(*this);
    }

  //! PMatrixJIT -= PScalarJIT
  template<class T1>
  inline
  CC& operator-=(const PScalarJIT<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i,i) -= rhs.elem();

      return static_cast<CC&>(*this);
    }

  //! PMatrixJIT *= PScalarJIT
  template<class T1>
  inline
  CC& operator*=(const PScalarJIT<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) *= rhs.elem();

      return static_cast<CC&>(*this);
    }

  //! PMatrixJIT /= PScalarJIT
  template<class T1>
  inline
  CC& operator/=(const PScalarJIT<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) /= rhs.elem();

      return static_cast<CC&>(*this);
    }


#if 0
  PMatrixJIT(const PMatrixJIT& a) : JV<T,N*N>::JV(a) {
    std::cout << "PMatrixJIT copy c-tor " << (void*)this << "\n";
  }
#endif

public:
  T& elem(int i, int j) {return JV<T,N*N>::getF()[j+N*i];}
  const T& elem(int i, int j) const {return JV<T,N*N>::getF()[j+N*i];}

};


//! Text input
template<class T, int N, template<class,int> class C>  
inline
TextReader& operator>>(TextReader& txt, PMatrixJIT<T,N,C>& d)
{
  for(int j=0; j < N; ++j)
    for(int i=0; i < N; ++i)
      txt >> d.elem(i,j);

  return txt;
}

//! Text output
template<class T, int N, template<class,int> class C>  
inline
TextWriter& operator<<(TextWriter& txt, const PMatrixJIT<T,N,C>& d)
{
  for(int j=0; j < N; ++j)
    for(int i=0; i < N; ++i)
      txt << d.elem(i,j);

  return txt;
}

#ifndef QDP_NO_LIBXML2
//! XML output
template<class T, int N, template<class,int> class C>  
inline
XMLWriter& operator<<(XMLWriter& xml, const PMatrixJIT<T,N,C>& d)
{
  xml.openTag("Matrix");

  XMLWriterAPI::AttributeList alist;

  for(int i=0; i < N; ++i)
  {
    for(int j=0; j < N; ++j)
    {
      alist.clear();
      alist.push_back(XMLWriterAPI::Attribute("row", i));
      alist.push_back(XMLWriterAPI::Attribute("col", j));

      xml.openTag("elem", alist);
      xml << d.elem(i,j);
      xml.closeTag();
    }
  }

  xml.closeTag(); // Matrix
  return xml;
}
#endif
/*! @} */  // end of group primmatrix

//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

// Underlying word type
template<class T1, int N, template<class,int> class C>
struct WordType<PMatrixJIT<T1,N,C> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

// Fixed Precision
template<class T1, int N, template<class,int> class C>
struct SinglePrecType< PMatrixJIT<T1, N, C> >
{
  typedef PMatrixJIT< typename SinglePrecType<T1>::Type_t, N, C > Type_t;
};


// Fixed Precision
template<class T1, int N, template<class,int> class C>
struct DoublePrecType< PMatrixJIT<T1, N, C> >
{
  typedef PMatrixJIT< typename DoublePrecType<T1>::Type_t, N, C > Type_t;
};

// Internally used scalars
template<class T, int N, template<class,int> class C>
struct InternalScalar<PMatrixJIT<T,N,C> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive scalar leaving grid alone
template<class T, int N, template<class,int> class C>
struct PrimitiveScalar<PMatrixJIT<T,N,C> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T, int N, template<class,int> class C>
struct LatticeScalar<PMatrixJIT<T,N,C> > {
  typedef C<typename LatticeScalar<T>::Type_t, N>  Type_t;
};


//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

/*
 * NOTE***: no Op defaults - they cause conflicts with specialized versions.
 * Avoid them.
 */


#if 0
template<class T1, class T2>
struct UnaryReturn<PScalarJIT<T2>, OpCast<T1> > {
  typedef PScalarJIT<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif

template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, OpIdentity> {
  typedef C<typename UnaryReturn<T, OpIdentity>::Type_t, N>  Type_t;
};


// Assignment is different
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpAddAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpSubtractAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpAddAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpSubtractAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpMultiplyAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpDivideAssign > {
  typedef C<T1,N> &Type_t;
};
 


//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------
/*! \addtogroup primmatrix */
/*! @{ */

// Primitive Matrices

// PMatrixJIT = + PMatrixJIT
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, OpUnaryPlus> {
  typedef C<typename UnaryReturn<T, OpUnaryPlus>::Type_t, N>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T1,N,C>, OpUnaryPlus>::Type_t
operator+(const PMatrixJIT<T1,N,C>& l)
{
  typename UnaryReturn<PMatrixJIT<T1,N,C>, OpUnaryPlus>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = +l.elem(i,j);

  return d;
}


// PMatrixJIT = - PMatrixJIT
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, OpUnaryMinus> {
  typedef C<typename UnaryReturn<T, OpUnaryMinus>::Type_t, N>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T1,N,C>, OpUnaryMinus>::Type_t
operator-(const PMatrixJIT<T1,N,C>& l)
{
  typename UnaryReturn<PMatrixJIT<T1,N,C>, OpUnaryMinus>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = -l.elem(i,j);

  return d;
}


// PMatrixJIT = PMatrixJIT + PMatrixJIT
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpAdd> {
  typedef C<typename BinaryReturn<T1, T2, OpAdd>::Type_t, N>  Type_t;
};


// template<class T1, class T2, int N, template<class,int> class C>
// inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpAdd>::Type_t
// operator+(const PMatrixJIT<T1,N,C>& l, const PMatrixJIT<T2,N,C>& r)
// {
//   typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpAdd>::Type_t  d;

//   for(int i=0; i < N; ++i)
//     for(int j=0; j < N; ++j)
//       d.elem(i,j) = l.elem(i,j) + r.elem(i,j);

//   return d;
// }

template<class T1, class T2, int N, template<class,int> class C>
inline void
addRep(const typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpAdd>::Type_t& dest, const PMatrixJIT<T1,N,C>& l, const PMatrixJIT<T2,N,C>& r)
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      addRep( dest.elem(i,j) , l.elem(i,j) , r.elem(i,j) );
}

// PMatrixJIT = PMatrixJIT + PScalarJIT
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpAdd> {
  typedef C<typename BinaryReturn<T1, T2, OpAdd>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpAdd>::Type_t
operator+(const PMatrixJIT<T1,N,C>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpAdd>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = (i == j) ? l.elem(i,i) + r.elem() : l.elem(i,j);

  return d;
}

// PMatrixJIT = PScalarJIT + PMatrixJIT
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpAdd> {
  typedef C<typename BinaryReturn<T1, T2, OpAdd>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpAdd>::Type_t
operator+(const PScalarJIT<T1>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpAdd>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = (i == j) ? l.elem() + r.elem(i,i) : r.elem(i,j);

  return d;
}


// PMatrixJIT = PMatrixJIT - PMatrixJIT
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpSubtract> {
  typedef C<typename BinaryReturn<T1, T2, OpSubtract>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpSubtract>::Type_t
operator-(const PMatrixJIT<T1,N,C>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpSubtract>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = l.elem(i,j) - r.elem(i,j);

  return d;
}

// PMatrixJIT = PMatrixJIT - PScalarJIT
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpSubtract> {
  typedef C<typename BinaryReturn<T1, T2, OpSubtract>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpSubtract>::Type_t
operator-(const PMatrixJIT<T1,N,C>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpSubtract>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = (i == j) ? l.elem(i,i) - r.elem() : l.elem(i,j);

  return d;
}

// PMatrixJIT = PScalarJIT - PMatrixJIT
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpSubtract> {
  typedef C<typename BinaryReturn<T1, T2, OpSubtract>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpSubtract>::Type_t
operator-(const PScalarJIT<T1>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpSubtract>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = (i == j) ? l.elem() - r.elem(i,i) : -r.elem(i,j);

  return d;
}


// PMatrixJIT = PMatrixJIT * PScalarJIT
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpMultiply>::Type_t
operator*(const PMatrixJIT<T1,N,C>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = l.elem(i,j) * r.elem();
  return d;
}

// Optimized  PMatrixJIT = adj(PMatrixJIT)*PScalarJIT
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpAdjMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpAdjMultiply>::Type_t
adjMultiply(const PMatrixJIT<T1,N,C>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpAdjMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = adjMultiply(l.elem(j,i), r.elem());
  return d;
}

// Optimized  PMatrixJIT = PMatrixJIT*adj(PScalarJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiplyAdj>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(const PMatrixJIT<T1,N,C>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpMultiplyAdj>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = multiplyAdj(l.elem(i,j), r.elem());
  return d;
}

// Optimized  PMatrixJIT = adj(PMatrixJIT)*adj(PScalarJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpAdjMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiplyAdj>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const PMatrixJIT<T1,N,C>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpAdjMultiplyAdj>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = adjMultiplyAdj(l.elem(j,i), r.elem());
  return d;
}



// PMatrixJIT = PScalarJIT * PMatrixJIT
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpMultiply>::Type_t
operator*(const PScalarJIT<T1>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = l.elem() * r.elem(i,j);
  return d;
}

// Optimized  PMatrixJIT = adj(PScalarJIT) * PMatrixJIT
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpAdjMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpAdjMultiply>::Type_t
adjMultiply(const PScalarJIT<T1>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpAdjMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = adjMultiply(l.elem(), r.elem(i,j));
  return d;
}

// Optimized  PMatrixJIT = PScalarJIT * adj(PMatrixJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiplyAdj>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpMultiplyAdj>::Type_t
multiplyAdj(const PScalarJIT<T1>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpMultiplyAdj>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = multiplyAdj(l.elem(), r.elem(j,i));
  return d;
}

// Optimized  PMatrixJIT = adj(PScalarJIT) * adj(PMatrixJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpAdjMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiplyAdj>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const PScalarJIT<T1>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, OpAdjMultiplyAdj>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = adjMultiplyAdj(l.elem(), r.elem(j,i));
  return d;
}


// PMatrixJIT = PMatrixJIT * PMatrixJIT
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiply>::Type_t, N>  Type_t;
};


// template<class T1, class T2, int N, template<class,int> class C>
// inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpMultiply>::Type_t
// operator*(const PMatrixJIT<T1,N,C>& l, const PMatrixJIT<T2,N,C>& r)
// {
//   typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpMultiply>::Type_t  d;

//   for(int i=0; i < N; ++i)
//     for(int j=0; j < N; ++j)
//     {
//       d.elem(i,j) = l.elem(i,0) * r.elem(0,j);
//       for(int k=1; k < N; ++k)
// 	d.elem(i,j) += l.elem(i,k) * r.elem(k,j);
//     }

//   return d;
// }


template<class T1, class T2, int N, template<class,int> class C>
inline void
mulRep(const typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpMultiply>::Type_t& dest,
       const PMatrixJIT<T1,N,C>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpMultiply>::Type_t& dd = const_cast<typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpMultiply>::Type_t&>(dest);
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
    {
      mulRep( dest.elem(i,j) , l.elem(i,0) , r.elem(0,j) );
      for(int k=1; k < N; ++k) {
	typename BinaryReturn<T1,T2,OpMultiply>::Type_t tmp(dest.func());
	mulRep( tmp , l.elem(i,k) , r.elem(k,j) ); 
	dd.elem(i,j) += tmp;
      }
    }
}

// Optimized  PMatrixJIT = adj(PMatrixJIT)*PMatrixJIT
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpAdjMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpAdjMultiply>::Type_t
adjMultiply(const PMatrixJIT<T1,N,C>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpAdjMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
    {
      d.elem(i,j) = adjMultiply(l.elem(0,i), r.elem(0,j));
      for(int k=1; k < N; ++k)
	d.elem(i,j) += adjMultiply(l.elem(k,i), r.elem(k,j));
    }

  return d;
}

// Optimized  PMatrixJIT = PMatrixJIT*adj(PMatrixJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiplyAdj>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpMultiplyAdj>::Type_t
multiplyAdj(const PMatrixJIT<T1,N,C>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpMultiplyAdj>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
    {
      d.elem(i,j) = multiplyAdj(l.elem(i,0), r.elem(j,0));
      for(int k=1; k < N; ++k)
	d.elem(i,j) += multiplyAdj(l.elem(i,k), r.elem(j,k));
    }

  return d;
}

// Optimized  PMatrixJIT = adj(PMatrixJIT)*adj(PMatrixJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpAdjMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiplyAdj>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const PMatrixJIT<T1,N,C>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, OpAdjMultiplyAdj>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
    {
      d.elem(i,j) = adjMultiplyAdj(l.elem(0,i), r.elem(j,0));
      for(int k=1; k < N; ++k)
	d.elem(i,j) += adjMultiplyAdj(l.elem(k,i), r.elem(j,k));
    }

  return d;
}


// PMatrixJIT = PMatrixJIT / PScalarJIT
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpDivide> {
  typedef C<typename BinaryReturn<T1, T2, OpDivide>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpDivide>::Type_t
operator/(const PMatrixJIT<T1,N,C>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, OpDivide>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = l.elem(i,j) / r.elem();
  return d;
}



//-----------------------------------------------------------------------------
// Functions

// Adjoint
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnAdjoint> {
  typedef C<typename UnaryReturn<T, FnAdjoint>::Type_t, N>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T1,N,C>, FnAdjoint>::Type_t
adj(const PMatrixJIT<T1,N,C>& l)
{
  typename UnaryReturn<PMatrixJIT<T1,N,C>, FnAdjoint>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = adj(l.elem(j,i));

  return d;
}


// Conjugate
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnConjugate> {
  typedef C<typename UnaryReturn<T, FnConjugate>::Type_t, N>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T1,N,C>, FnConjugate>::Type_t
conj(const PMatrixJIT<T1,N,C>& l)
{
  typename UnaryReturn<PMatrixJIT<T1,N,C>, FnConjugate>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = conj(l.elem(i,j));

  return d;
}


// Transpose
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnTranspose> {
  typedef C<typename UnaryReturn<T, FnTranspose>::Type_t, N>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T1,N,C>, FnTranspose>::Type_t
transpose(const PMatrixJIT<T1,N,C>& l)
{
  typename UnaryReturn<PMatrixJIT<T1,N,C>, FnTranspose>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = transpose(l.elem(j,i));

  return d;
}


// TRACE
// PScalarJIT = Trace(PMatrixJIT)
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnTrace> {
  typedef PScalarJIT<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnTrace>::Type_t
trace(const PMatrixJIT<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnTrace>::Type_t  d;

  d.elem() = trace(s1.elem(0,0));
  for(int i=1; i < N; ++i)
    d.elem() += trace(s1.elem(i,i));

  return d;
}


// PScalarJIT = Re(Trace(PMatrixJIT))
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnRealTrace> {
  typedef PScalarJIT<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T1,N,C>, FnRealTrace>::Type_t
realTrace(const PMatrixJIT<T1,N,C>& s1)
{
  typename UnaryReturn<PMatrixJIT<T1,N,C>, FnRealTrace>::Type_t  d;

  d.elem() = realTrace(s1.elem(0,0));
  for(int i=1; i < N; ++i)
    d.elem() += realTrace(s1.elem(i,i));

  return d;
}


//! PScalarJIT = Im(Trace(PMatrixJIT))
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnImagTrace> {
  typedef PScalarJIT<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T1,N,C>, FnImagTrace>::Type_t
imagTrace(const PMatrixJIT<T1,N,C>& s1)
{
  typename UnaryReturn<PMatrixJIT<T1,N,C>, FnImagTrace>::Type_t  d;

  d.elem() = imagTrace(s1.elem(0,0));
  for(int i=1; i < N; ++i)
    d.elem() += imagTrace(s1.elem(i,i));

  return d;
}


//! PMatrixJIT = traceColor(PMatrixJIT)   [this is an identity in general]
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnTraceColor> {
  typedef C<typename UnaryReturn<T, FnTraceColor>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnTraceColor>::Type_t
traceColor(const PMatrixJIT<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnTraceColor>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = traceColor(s1.elem(i,j));

  return d;
}


//! PMatrixJIT = traceSpin(PMatrixJIT)   [this is an identity in general]
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnTraceSpin> {
  typedef C<typename UnaryReturn<T, FnTraceSpin>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnTraceSpin>::Type_t
traceSpin(const PMatrixJIT<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnTraceSpin>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = traceSpin(s1.elem(i,j));

  return d;
}


//! PMatrixJIT = transposeColor(PMatrixJIT) [ this is an identity in general]
/*! define the return type */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnTransposeColor> {
  typedef C<typename UnaryReturn<T, FnTransposeColor>::Type_t, N> Type_t;
};

/*! define the function itself.Recurse down elements of the primmatrix
 *  and call transposeColor on each one */
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnTransposeColor>::Type_t
transposeColor(const PMatrixJIT<T,N,C>& s1)
{ 
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnTransposeColor>::Type_t d;
  for(int i=0; i < N; ++i) {
    for(int j=0; j < N; ++j) {
      d.elem(i,j) = transposeColor(s1.elem(i,j));
    }
  }

  return d;
}


//! PMatrixJIT = transposeSpin(PMatrixJIT) [ this is an identity in general]
/*! define the return type */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnTransposeSpin> {
  typedef C<typename UnaryReturn<T, FnTransposeSpin>::Type_t, N> Type_t;
};

/*! define the function itself.Recurse down elements of the primmatrix
 *  and call transposeSpin on each one */
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnTransposeSpin>::Type_t
transposeSpin(const PMatrixJIT<T,N,C>& s1)
{ 
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnTransposeSpin>::Type_t d;
  for(int i=0; i < N; ++i) {
    for(int j=0; j < N; ++j) {
      d.elem(i,j) = transposeSpin(s1.elem(i,j));
    }
  }

  return d;
}


// PScalarJIT = traceMultiply(PMatrixJIT,PMatrixJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnTraceMultiply>::Type_t
traceMultiply(const PMatrixJIT<T1,N,C>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnTraceMultiply>::Type_t  d;

  d.elem() = traceMultiply(l.elem(0,0), r.elem(0,0));
  for(int k=1; k < N; ++k)
    d.elem() += traceMultiply(l.elem(0,k), r.elem(k,0));

  for(int j=1; j < N; ++j)
    for(int k=0; k < N; ++k)
      d.elem() += traceMultiply(l.elem(j,k), r.elem(k,j));

  return d;
}

// PScalarJIT = traceMultiply(PMatrixJIT,PScalarJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, FnTraceMultiply>::Type_t
traceMultiply(const PMatrixJIT<T1,N,C>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, FnTraceMultiply>::Type_t  d;

  d.elem() = traceMultiply(l.elem(0,0), r.elem());
  for(int k=1; k < N; ++k)
    d.elem() += traceMultiply(l.elem(k,k), r.elem());

  return d;
}

// PScalarJIT = traceMultiply(PScalarJIT,PMatrixJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, FnTraceMultiply>::Type_t
traceMultiply(const PScalarJIT<T1>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, FnTraceMultiply>::Type_t  d;

  d.elem() = traceMultiply(l.elem(), r.elem(0,0));
  for(int k=1; k < N; ++k)
    d.elem() += traceMultiply(l.elem(), r.elem(k,k));

  return d;
}



//! PMatrixJIT = traceColorMultiply(PMatrixJIT,PMatrixJIT)   [the trace is an identity in general]
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnTraceColorMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnTraceColorMultiply>::Type_t
traceColorMultiply(const PMatrixJIT<T1,N,C>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnTraceColorMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
    {
      d.elem(i,j) = traceColorMultiply(l.elem(i,0), r.elem(0,j));
      for(int k=1; k < N; ++k)
	d.elem(i,j) += traceColorMultiply(l.elem(i,k), r.elem(k,j));
    }

  return d;
}

// PMatrixJIT = traceColorMultiply(PMatrixJIT,PScalarJIT)   [the trace is an identity in general]
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, FnTraceColorMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, FnTraceColorMultiply>::Type_t
traceColorMultiply(const PMatrixJIT<T1,N,C>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, FnTraceColorMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = traceColorMultiply(l.elem(i,j), r.elem());

  return d;
}

// PMatrixJIT = traceColorMultiply(PScalarJIT,PMatrixJIT)   [the trace is an identity in general]
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, FnTraceColorMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, FnTraceColorMultiply>::Type_t
traceColorMultiply(const PScalarJIT<T1>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, FnTraceColorMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = traceColorMultiply(l.elem(), r.elem(i,j));

  return d;
}


//! PMatrixJIT = traceSpinMultiply(PMatrixJIT,PMatrixJIT)   [the trace is an identity in general]
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnTraceSpinMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(const PMatrixJIT<T1,N,C>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnTraceSpinMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
    {
      d.elem(i,j) = traceSpinMultiply(l.elem(i,0), r.elem(0,j));
      for(int k=1; k < N; ++k)
	d.elem(i,j) += traceSpinMultiply(l.elem(i,k), r.elem(k,j));
    }

  return d;
}

// PScalarJIT = traceSpinMultiply(PMatrixJIT,PScalarJIT)   [the trace is an identity in general]
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, FnTraceSpinMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(const PMatrixJIT<T1,N,C>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, FnTraceSpinMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = traceSpinMultiply(l.elem(i,j), r.elem());

  return d;
}

// PScalarJIT = traceSpinMultiply(PScalarJIT,PMatrixJIT)   [the trace is an identity in general]
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, FnTraceSpinMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(const PScalarJIT<T1>& l, const PMatrixJIT<T2,N,C>& r)
{
  typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, FnTraceSpinMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = traceSpinMultiply(l.elem(), r.elem(i,j));

  return d;
}


//! PMatrixJIT = Re(PMatrixJIT)
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnReal> {
  typedef C<typename UnaryReturn<T, FnReal>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnReal>::Type_t
real(const PMatrixJIT<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnReal>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = real(s1.elem(i,j));

  return d;
}


//! PMatrixJIT = Im(PMatrixJIT)
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnImag> {
  typedef C<typename UnaryReturn<T, FnImag>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnImag>::Type_t
imag(const PMatrixJIT<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnImag>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = imag(s1.elem(i,j));

  return d;
}


//! PMatrixJIT<T> = (PMatrixJIT<T> , PMatrixJIT<T>)
template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnCmplx>::Type_t
cmplx(const PMatrixJIT<T1,N,C>& s1, const PMatrixJIT<T2,N,C>& s2)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnCmplx>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = cmplx(s1.elem(i,j), s2.elem(i,j));

  return d;
}




// Functions
//! PMatrixJIT = i * PMatrixJIT
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnTimesI> {
  typedef C<typename UnaryReturn<T, FnTimesI>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnTimesI>::Type_t
timesI(const PMatrixJIT<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnTimesI>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = timesI(s1.elem(i,j));

  return d;
}

//! PMatrixJIT = -i * PMatrixJIT
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnTimesMinusI> {
  typedef C<typename UnaryReturn<T, FnTimesMinusI>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnTimesMinusI>::Type_t
timesMinusI(const PMatrixJIT<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnTimesMinusI>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = timesMinusI(s1.elem(i,j));

  return d;
}

//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnGetSite> {
  typedef C<typename UnaryReturn<T, FnGetSite>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnGetSite>::Type_t
getSite(const PMatrixJIT<T,N,C>& s1, int innersite)
{ 
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnGetSite>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = getSite(s1.elem(i,j), innersite);

  return d;
}

//! Extract color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnPeekColorVector> {
  typedef C<typename UnaryReturn<T, FnPeekColorVector>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnPeekColorVector>::Type_t
peekColor(const PMatrixJIT<T,N,C>& l, int row)
{
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnPeekColorVector>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = peekColor(l.elem(i,j),row);
  return d;
}

//! Extract color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnPeekColorMatrix> {
  typedef C<typename UnaryReturn<T, FnPeekColorMatrix>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnPeekColorMatrix>::Type_t
peekColor(const PMatrixJIT<T,N,C>& l, int row, int col)
{
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnPeekColorMatrix>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = peekColor(l.elem(i,j),row,col);
  return d;
}

//! Extract spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnPeekSpinVector> {
  typedef C<typename UnaryReturn<T, FnPeekSpinVector>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnPeekSpinVector>::Type_t
peekSpin(const PMatrixJIT<T,N,C>& l, int row)
{
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnPeekSpinVector>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = peekSpin(l.elem(i,j),row);
  return d;
}

//! Extract spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnPeekSpinMatrix> {
  typedef C<typename UnaryReturn<T, FnPeekSpinMatrix>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnPeekSpinMatrix>::Type_t
peekSpin(const PMatrixJIT<T,N,C>& l, int row, int col)
{
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnPeekSpinMatrix>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = peekSpin(l.elem(i,j),row,col);
  return d;
}

//! Insert color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnPokeColorMatrix> {
  typedef C<typename UnaryReturn<T, FnPokeColorMatrix>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T1,N,C>, FnPokeColorMatrix>::Type_t&
pokeColor(PMatrixJIT<T1,N,C>& l, const PMatrixJIT<T2,N,C>& r, int row)
{
  typedef typename UnaryReturn<PMatrixJIT<T1,N,C>, FnPokeColorMatrix>::Type_t  Return_t;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      pokeColor(l.elem(i,j),r.elem(i,j),row);
  return static_cast<Return_t&>(l);
}

//! Insert color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T1,N,C>, FnPokeColorMatrix>::Type_t&
pokeColor(PMatrixJIT<T1,N,C>& l, const PMatrixJIT<T2,N,C>& r, int row, int col)
{
  typedef typename UnaryReturn<PMatrixJIT<T1,N,C>, FnPokeColorMatrix>::Type_t  Return_t;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      pokeColor(l.elem(i,j),r.elem(i,j),row,col);
  return static_cast<Return_t&>(l);
}

//! Insert spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnPokeSpinMatrix> {
  typedef C<typename UnaryReturn<T, FnPokeSpinMatrix>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T1,N,C>, FnPokeSpinMatrix>::Type_t&
pokeSpin(PMatrixJIT<T1,N,C>& l, const PMatrixJIT<T2,N,C>& r, int row)
{
  typedef typename UnaryReturn<PMatrixJIT<T1,N,C>, FnPokeSpinMatrix>::Type_t  Return_t;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      pokeSpin(l.elem(i,j),r.elem(i,j),row);
  return static_cast<Return_t&>(l);
}

//! Insert spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T1,N,C>, FnPokeSpinMatrix>::Type_t&
pokeSpin(PMatrixJIT<T1,N,C>& l, const PMatrixJIT<T2,N,C>& r, int row, int col)
{
  typedef typename UnaryReturn<PMatrixJIT<T1,N,C>, FnPokeSpinMatrix>::Type_t  Return_t;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      pokeSpin(l.elem(i,j),r.elem(i,j),row,col);
  return static_cast<Return_t&>(l);
}



//! dest = 0
template<class T, int N, template<class,int> class C> 
inline void 
zero_rep(PMatrixJIT<T,N,C>& dest) 
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      zero_rep(dest.elem(i,j));
}


//! dest = (mask) ? s1 : dest
template<class T, class T1, int N, template<class,int> class C> 
inline void 
copymask(PMatrixJIT<T,N,C>& d, const PScalarJIT<T1>& mask, const PMatrixJIT<T,N,C>& s1) 
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      copymask(d.elem(i,j),mask.elem(),s1.elem(i,j));
}


//! dest [some type] = source [some type]
template<class T, class T1, int N, template<class,int> class C>
inline void 
copy_site(PMatrixJIT<T,N,C>& d, int isite, const PMatrixJIT<T1,N,C>& s1)
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      copy_site(d.elem(i,j), isite, s1.elem(i,j));
}

//! dest [some type] = source [some type]
template<class T, class T1, int N, template<class,int> class C>
inline void 
copy_site(PMatrixJIT<T,N,C>& d, int isite, const PScalarJIT<T1>& s1)
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      copy_site(d.elem(i,j), isite, s1.elem());
}


//! gather several inner sites together
template<class T, class T1, int N, template<class,int> class C>
inline void 
gather_sites(PMatrixJIT<T,N,C>& d, 
	     const PMatrixJIT<T1,N,C>& s0, int i0, 
	     const PMatrixJIT<T1,N,C>& s1, int i1,
	     const PMatrixJIT<T1,N,C>& s2, int i2,
	     const PMatrixJIT<T1,N,C>& s3, int i3)
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      gather_sites(d.elem(i,j), 
		   s0.elem(i,j), i0, 
		   s1.elem(i,j), i1, 
		   s2.elem(i,j), i2, 
		   s3.elem(i,j), i3);
}


//! dest  = random  
template<class T, int N, template<class,int> class C, class T1, class T2>
inline void
fill_random(PMatrixJIT<T,N,C>& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  // The skewed_seed is the starting seed to use
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      fill_random(d.elem(i,j), seed, skewed_seed, seed_mult);
}

//! dest  = gaussian
template<class T, int N, template<class,int> class C>
inline void
fill_gaussian(PMatrixJIT<T,N,C>& d, PMatrixJIT<T,N,C>& r1, PMatrixJIT<T,N,C>& r2)
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      fill_gaussian(d.elem(i,j), r1.elem(i,j), r2.elem(i,j));
}



#if 0
// Global sum over site indices only
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnSum> {
  typedef C<typename UnaryReturn<T, FnSum>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnSum>::Type_t
sum(const PMatrixJIT<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnSum>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = sum(s1.elem(i,j));

  return d;
}
#endif


// InnerProduct (norm-seq) global sum = sum(tr(adj(s1)*s1))
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnNorm2> {
  typedef PScalarJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixJIT<T,N,C>, FnLocalNorm2> {
  typedef PScalarJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixJIT<T,N,C>, FnLocalNorm2>::Type_t
localNorm2(const PMatrixJIT<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixJIT<T,N,C>, FnLocalNorm2>::Type_t  d;

  d.elem() = localNorm2(s1.elem(0,0));
  for(int j=1; j < N; ++j)
    d.elem() += localNorm2(s1.elem(0,j));

  for(int i=1; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem() += localNorm2(s1.elem(i,j));

  return d;
}


//! PScalarJIT = innerProduct(PMatrixJIT,PMatrixJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

//! PScalarJIT = localInnerProduct(PMatrixJIT,PMatrixJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnLocalInnerProduct>::Type_t
localInnerProduct(const PMatrixJIT<T1,N,C>& s1, const PMatrixJIT<T2,N,C>& s2)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnLocalInnerProduct>::Type_t  d;

  d.elem() = localInnerProduct(s1.elem(0,0), s2.elem(0,0));
  for(int k=1; k < N; ++k)
    d.elem() += localInnerProduct(s1.elem(k,0), s2.elem(k,0));

  for(int j=1; j < N; ++j)
    for(int k=0; k < N; ++k)
      d.elem() += localInnerProduct(s1.elem(k,j), s2.elem(k,j));

  return d;
}

//! PScalarJIT = localInnerProduct(PMatrixJIT,PScalarJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(const PMatrixJIT<T1,N,C>& s1, const PScalarJIT<T2>& s2)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, FnLocalInnerProduct>::Type_t  d;

  d.elem() = localInnerProduct(s1.elem(0,0), s2.elem());
  for(int k=1; k < N; ++k)
    d.elem() += localInnerProduct(s1.elem(k,k), s2.elem());

  return d;
}

//! PScalarJIT = localInnerProduct(PScalarJIT,PMatrixJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, FnLocalInnerProduct>::Type_t
localInnerProduct(const PScalarJIT<T1>& s1, const PMatrixJIT<T2,N,C>& s2)
{
  typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, FnLocalInnerProduct>::Type_t  d;

  d.elem() = localInnerProduct(s1.elem(), s2.elem(0,0));
  for(int k=1; k < N; ++k)
    d.elem() += localInnerProduct(s1.elem(), s2.elem(k,k));

  return d;
}


//! PScalarJIT = innerProductReal(PMatrixJIT,PMatrixJIT)
/*!
 * return  realpart of InnerProduct(adj(s1)*s2)
 */
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnInnerProductReal > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

//! PScalarJIT = innerProductReal(PMatrixJIT,PMatrixJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnLocalInnerProductReal > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const PMatrixJIT<T1,N,C>& s1, const PMatrixJIT<T2,N,C>& s2)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PMatrixJIT<T2,N,C>, FnLocalInnerProductReal>::Type_t  d;

  d.elem() = localInnerProductReal(s1.elem(0,0), s2.elem(0,0));
  for(int k=1; k < N; ++k)
    d.elem() += localInnerProductReal(s1.elem(k,0), s2.elem(k,0));

  for(int j=1; j < N; ++j)
    for(int k=0; k < N; ++k)
      d.elem() += localInnerProductReal(s1.elem(k,j), s2.elem(k,j));

  return d;
}

//! PScalarJIT = localInnerProductReal(PMatrixJIT,PScalarJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, FnLocalInnerProductReal > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const PMatrixJIT<T1,N,C>& s1, const PScalarJIT<T2>& s2)
{
  typename BinaryReturn<PMatrixJIT<T1,N,C>, PScalarJIT<T2>, FnLocalInnerProductReal>::Type_t  d;

  d.elem() = localInnerProductReal(s1.elem(0,0), s2.elem());
  for(int k=1; k < N; ++k)
    d.elem() += localInnerProductReal(s1.elem(k,0), s2.elem(k,k));

  return d;
}

//! PScalarJIT = localInnerProductReal(PScalarJIT,PMatrixJIT)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, FnLocalInnerProductReal > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const PScalarJIT<T1>& s1, const PMatrixJIT<T2,N,C>& s2)
{
  typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, FnLocalInnerProductReal>::Type_t  d;

  d.elem() = localInnerProductReal(s1.elem(), s2.elem(0,0));
  for(int k=1; k < N; ++k)
    d.elem() += localInnerProductReal(s1.elem(), s2.elem(k,k));

  return d;
}


//! PMatrixJIT<T> = where(PScalarJIT, PMatrixJIT, PMatrixJIT)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3, int N, template<class,int> class C>
struct TrinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, PMatrixJIT<T3,N,C>, FnWhere> {
  typedef C<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t, N>  Type_t;
};

template<class T1, class T2, class T3, int N, template<class,int> class C>
inline typename TrinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, PMatrixJIT<T3,N,C>, FnWhere>::Type_t
where(const PScalarJIT<T1>& a, const PMatrixJIT<T2,N,C>& b, const PMatrixJIT<T3,N,C>& c)
{
  typename TrinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,N,C>, PMatrixJIT<T3,N,C>, FnWhere>::Type_t  d;

  // Not optimal - want to have where outside assignment
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = where(a.elem(), b.elem(i,j), c.elem(i,j));

  return d;
}

/*! @} */  // end of group primmatrix

} // namespace QDP

#endif
