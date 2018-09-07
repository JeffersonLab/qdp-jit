// -*- C++ -*-

/*! \file
 * \brief Primitive Matrix
 */

#ifndef QDP_PRIMMATRIXREG_H
#define QDP_PRIMMATRIXREG_H

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
  template <class T, int N, template<class,int> class C> class PMatrixREG //: public BaseREG<T,N*N,PMatrixREG<T,N,C> >
{
  T F[N*N];
public:
  typedef C<T,N>  CC;


  //! PMatrixREG = PScalarREG
  /*! Fill with primitive scalar */
  template<class T1>
  inline
  CC& assign(const PScalarREG<T1>& rhs)
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  if (i == j)
	    elem(i,j) = rhs.elem();
	  else
	    zero_rep(elem(i,j));

      return static_cast<CC&>(*this);
    }

  //! PMatrixREG = PMatrixREG
  /*! Set equal to another PMatrixREG */
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
  PMatrixREG& assign(const PMatrixREG& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) = rhs.elem(i,j);

      return static_cast<PMatrixREG&>(*this);
    }
#endif

  //! PMatrixREG += PMatrixREG
  template<class T1>
  inline
  CC& operator+=(const C<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) += rhs.elem(i,j);

      return static_cast<CC&>(*this);
    }

  //! PMatrixREG -= PMatrixREG
  template<class T1>
  inline
  CC& operator-=(const C<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) -= rhs.elem(i,j);

      return static_cast<CC&>(*this);
    }

  //! PMatrixREG += PScalarREG
  template<class T1>
  inline
  CC& operator+=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i,i) += rhs.elem();

      return static_cast<CC&>(*this);
    }

  //! PMatrixREG -= PScalarREG
  template<class T1>
  inline
  CC& operator-=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i,i) -= rhs.elem();

      return static_cast<CC&>(*this);
    }

  //! PMatrixREG *= PScalarREG
  template<class T1>
  inline
  CC& operator*=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) *= rhs.elem();

      return static_cast<CC&>(*this);
    }

  //! PMatrixREG /= PScalarREG
  template<class T1>
  inline
  CC& operator/=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) /= rhs.elem();

      return static_cast<CC&>(*this);
    }



public:
  T getRegElem(int row,int col) const {
    assert(!"ni");
#if 0
    int r_matidx = this->func().getRegs( Jit::s32 , 1 );
    int r_N = this->func().getRegs( Jit::s32 , 1 );
    this->func().asm_mov_literal( r_N , (int)N );
    this->func().asm_mul( r_matidx , col , r_N );
    this->func().asm_add( r_matidx , r_matidx , row );
    return JV<T,N*N>::getRegElem( r_matidx );
#endif
  }




        T& elem(int i, int j)       {return F[j+N*i];}
  const T& elem(int i, int j) const {return F[j+N*i];}

  //       T& elem(int i, int j)       {return this->arrayF(j+N*i);}
  // const T& elem(int i, int j) const {return this->arrayF(j+N*i);}


  // T& elem(int i, int j) {return JV<T,N*N>::getF()[j+N*i];}
  // const T& elem(int i, int j) const {return JV<T,N*N>::getF()[j+N*i];}

};



  template <class T, int N, template<class,int> class PColorMatrixREG >
  struct JITType< PMatrixREG<T,N, PColorMatrixREG > >
  {
    typedef PMatrixJIT<typename JITType<T>::Type_t,N,PColorMatrixREG >  Type_t;
  };




//! Text input
template<class T, int N, template<class,int> class C>  
inline
TextReader& operator>>(TextReader& txt, PMatrixREG<T,N,C>& d)
{
  for(int j=0; j < N; ++j)
    for(int i=0; i < N; ++i)
      txt >> d.elem(i,j);

  return txt;
}

//! Text output
template<class T, int N, template<class,int> class C>  
inline
TextWriter& operator<<(TextWriter& txt, const PMatrixREG<T,N,C>& d)
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
XMLWriter& operator<<(XMLWriter& xml, const PMatrixREG<T,N,C>& d)
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
struct WordType<PMatrixREG<T1,N,C> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

// Fixed Precision
template<class T1, int N, template<class,int> class C>
struct SinglePrecType< PMatrixREG<T1, N, C> >
{
  typedef PMatrixREG< typename SinglePrecType<T1>::Type_t, N, C > Type_t;
};


// Fixed Precision
template<class T1, int N, template<class,int> class C>
struct DoublePrecType< PMatrixREG<T1, N, C> >
{
  typedef PMatrixREG< typename DoublePrecType<T1>::Type_t, N, C > Type_t;
};

// Internally used scalars
template<class T, int N, template<class,int> class C>
struct InternalScalar<PMatrixREG<T,N,C> > {
  typedef PScalarREG<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive scalar leaving grid alone
template<class T, int N, template<class,int> class C>
struct PrimitiveScalar<PMatrixREG<T,N,C> > {
  typedef PScalarREG<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T, int N, template<class,int> class C>
struct LatticeScalar<PMatrixREG<T,N,C> > {
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
struct UnaryReturn<PScalarREG<T2>, OpCast<T1> > {
  typedef PScalarREG<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif

template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, OpIdentity> {
  typedef C<typename UnaryReturn<T, OpIdentity>::Type_t, N>  Type_t;
};


// Assignment is different
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpAddAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpSubtractAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpAddAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpSubtractAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpMultiplyAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpDivideAssign > {
  typedef C<T1,N> &Type_t;
};
 


//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------
/*! \addtogroup primmatrix */
/*! @{ */

// Primitive Matrices

// PMatrixREG = + PMatrixREG
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, OpUnaryPlus> {
  typedef C<typename UnaryReturn<T, OpUnaryPlus>::Type_t, N>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T1,N,C>, OpUnaryPlus>::Type_t
operator+(const PMatrixREG<T1,N,C>& l)
{
  typename UnaryReturn<PMatrixREG<T1,N,C>, OpUnaryPlus>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = +l.elem(i,j);

  return d;
}


// PMatrixREG = - PMatrixREG
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, OpUnaryMinus> {
  typedef C<typename UnaryReturn<T, OpUnaryMinus>::Type_t, N>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T1,N,C>, OpUnaryMinus>::Type_t
operator-(const PMatrixREG<T1,N,C>& l)
{
  typename UnaryReturn<PMatrixREG<T1,N,C>, OpUnaryMinus>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = -l.elem(i,j);

  return d;
}


// PMatrixREG = PMatrixREG + PMatrixREG
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpAdd> {
  typedef C<typename BinaryReturn<T1, T2, OpAdd>::Type_t, N>  Type_t;
};


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpAdd>::Type_t
operator+(const PMatrixREG<T1,N,C>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpAdd>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = l.elem(i,j) + r.elem(i,j);

  return d;
}


// PMatrixREG = PMatrixREG + PScalarREG
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpAdd> {
  typedef C<typename BinaryReturn<T1, T2, OpAdd>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpAdd>::Type_t
operator+(const PMatrixREG<T1,N,C>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpAdd>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = (i == j) ? l.elem(i,i) + r.elem() : l.elem(i,j);

  return d;
}

// PMatrixREG = PScalarREG + PMatrixREG
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpAdd> {
  typedef C<typename BinaryReturn<T1, T2, OpAdd>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpAdd>::Type_t
operator+(const PScalarREG<T1>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpAdd>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = (i == j) ? l.elem() + r.elem(i,i) : r.elem(i,j);

  return d;
}


// PMatrixREG = PMatrixREG - PMatrixREG
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpSubtract> {
  typedef C<typename BinaryReturn<T1, T2, OpSubtract>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpSubtract>::Type_t
operator-(const PMatrixREG<T1,N,C>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpSubtract>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = l.elem(i,j) - r.elem(i,j);

  return d;
}

// PMatrixREG = PMatrixREG - PScalarREG
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpSubtract> {
  typedef C<typename BinaryReturn<T1, T2, OpSubtract>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpSubtract>::Type_t
operator-(const PMatrixREG<T1,N,C>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpSubtract>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = (i == j) ? l.elem(i,i) - r.elem() : l.elem(i,j);

  return d;
}

// PMatrixREG = PScalarREG - PMatrixREG
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpSubtract> {
  typedef C<typename BinaryReturn<T1, T2, OpSubtract>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpSubtract>::Type_t
operator-(const PScalarREG<T1>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpSubtract>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = (i == j) ? l.elem() - r.elem(i,i) : -r.elem(i,j);

  return d;
}


// PMatrixREG = PMatrixREG * PScalarREG
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpMultiply>::Type_t
operator*(const PMatrixREG<T1,N,C>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = l.elem(i,j) * r.elem();
  return d;
}

// Optimized  PMatrixREG = adj(PMatrixREG)*PScalarREG
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpAdjMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpAdjMultiply>::Type_t
adjMultiply(const PMatrixREG<T1,N,C>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpAdjMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = adjMultiply(l.elem(j,i), r.elem());
  return d;
}

// Optimized  PMatrixREG = PMatrixREG*adj(PScalarREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiplyAdj>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(const PMatrixREG<T1,N,C>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpMultiplyAdj>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = multiplyAdj(l.elem(i,j), r.elem());
  return d;
}

// Optimized  PMatrixREG = adj(PMatrixREG)*adj(PScalarREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpAdjMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiplyAdj>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const PMatrixREG<T1,N,C>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpAdjMultiplyAdj>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = adjMultiplyAdj(l.elem(j,i), r.elem());
  return d;
}



// PMatrixREG = PScalarREG * PMatrixREG
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpMultiply>::Type_t
operator*(const PScalarREG<T1>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = l.elem() * r.elem(i,j);
  return d;
}

// Optimized  PMatrixREG = adj(PScalarREG) * PMatrixREG
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpAdjMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpAdjMultiply>::Type_t
adjMultiply(const PScalarREG<T1>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpAdjMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = adjMultiply(l.elem(), r.elem(i,j));
  return d;
}

// Optimized  PMatrixREG = PScalarREG * adj(PMatrixREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiplyAdj>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpMultiplyAdj>::Type_t
multiplyAdj(const PScalarREG<T1>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpMultiplyAdj>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = multiplyAdj(l.elem(), r.elem(j,i));
  return d;
}

// Optimized  PMatrixREG = adj(PScalarREG) * adj(PMatrixREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpAdjMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiplyAdj>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const PScalarREG<T1>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, OpAdjMultiplyAdj>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = adjMultiplyAdj(l.elem(), r.elem(j,i));
  return d;
}


// PMatrixREG = PMatrixREG * PMatrixREG
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiply>::Type_t, N>  Type_t;
};


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpMultiply>::Type_t
operator*(const PMatrixREG<T1,N,C>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
    {
      d.elem(i,j) = l.elem(i,0) * r.elem(0,j);
      for(int k=1; k < N; ++k)
	d.elem(i,j) += l.elem(i,k) * r.elem(k,j);
    }

  return d;
}



// Optimized  PMatrixREG = adj(PMatrixREG)*PMatrixREG
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpAdjMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpAdjMultiply>::Type_t
adjMultiply(const PMatrixREG<T1,N,C>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpAdjMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
    {
      d.elem(i,j) = adjMultiply(l.elem(0,i), r.elem(0,j));
      for(int k=1; k < N; ++k)
	d.elem(i,j) += adjMultiply(l.elem(k,i), r.elem(k,j));
    }

  return d;
}

// Optimized  PMatrixREG = PMatrixREG*adj(PMatrixREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiplyAdj>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpMultiplyAdj>::Type_t
multiplyAdj(const PMatrixREG<T1,N,C>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpMultiplyAdj>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
    {
      d.elem(i,j) = multiplyAdj(l.elem(i,0), r.elem(j,0));
      for(int k=1; k < N; ++k)
	d.elem(i,j) += multiplyAdj(l.elem(i,k), r.elem(j,k));
    }

  return d;
}


// Optimized  PMatrixREG = adj(PMatrixREG)*adj(PMatrixREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpAdjMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiplyAdj>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const PMatrixREG<T1,N,C>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, OpAdjMultiplyAdj>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
    {
      d.elem(i,j) = adjMultiplyAdj(l.elem(0,i), r.elem(j,0));
      for(int k=1; k < N; ++k)
	d.elem(i,j) += adjMultiplyAdj(l.elem(k,i), r.elem(j,k));
    }

  return d;
}


// PMatrixREG = PMatrixREG / PScalarREG
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpDivide> {
  typedef C<typename BinaryReturn<T1, T2, OpDivide>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpDivide>::Type_t
operator/(const PMatrixREG<T1,N,C>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, OpDivide>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = l.elem(i,j) / r.elem();
  return d;
}



//-----------------------------------------------------------------------------
// Functions

// Adjoint
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnAdjoint> {
  typedef C<typename UnaryReturn<T, FnAdjoint>::Type_t, N>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T1,N,C>, FnAdjoint>::Type_t
adj(const PMatrixREG<T1,N,C>& l)
{
  typename UnaryReturn<PMatrixREG<T1,N,C>, FnAdjoint>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = adj(l.elem(j,i));

  return d;
}


// Conjugate
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnConjugate> {
  typedef C<typename UnaryReturn<T, FnConjugate>::Type_t, N>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T1,N,C>, FnConjugate>::Type_t
conj(const PMatrixREG<T1,N,C>& l)
{
  typename UnaryReturn<PMatrixREG<T1,N,C>, FnConjugate>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = conj(l.elem(i,j));

  return d;
}


// Transpose
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnTranspose> {
  typedef C<typename UnaryReturn<T, FnTranspose>::Type_t, N>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T1,N,C>, FnTranspose>::Type_t
transpose(const PMatrixREG<T1,N,C>& l)
{
  typename UnaryReturn<PMatrixREG<T1,N,C>, FnTranspose>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = transpose(l.elem(j,i));

  return d;
}


// TRACE
// PScalarREG = Trace(PMatrixREG)
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnTrace> {
  typedef PScalarREG<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnTrace>::Type_t
trace(const PMatrixREG<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixREG<T,N,C>, FnTrace>::Type_t  d;

  d.elem() = trace(s1.elem(0,0));
  for(int i=1; i < N; ++i)
    d.elem() += trace(s1.elem(i,i));

  return d;
}


// PScalarREG = Re(Trace(PMatrixREG))
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnRealTrace> {
  typedef PScalarREG<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T1,N,C>, FnRealTrace>::Type_t
realTrace(const PMatrixREG<T1,N,C>& s1)
{
  typename UnaryReturn<PMatrixREG<T1,N,C>, FnRealTrace>::Type_t  d;

  d.elem() = realTrace(s1.elem(0,0));
  for(int i=1; i < N; ++i)
    d.elem() += realTrace(s1.elem(i,i));

  return d;
}


//! PScalarREG = Im(Trace(PMatrixREG))
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnImagTrace> {
  typedef PScalarREG<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T1,N,C>, FnImagTrace>::Type_t
imagTrace(const PMatrixREG<T1,N,C>& s1)
{
  typename UnaryReturn<PMatrixREG<T1,N,C>, FnImagTrace>::Type_t  d;

  d.elem() = imagTrace(s1.elem(0,0));
  for(int i=1; i < N; ++i)
    d.elem() += imagTrace(s1.elem(i,i));

  return d;
}


//! PMatrixREG = traceColor(PMatrixREG)   [this is an identity in general]
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnTraceColor> {
  typedef C<typename UnaryReturn<T, FnTraceColor>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnTraceColor>::Type_t
traceColor(const PMatrixREG<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixREG<T,N,C>, FnTraceColor>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = traceColor(s1.elem(i,j));

  return d;
}


//! PMatrixREG = traceSpin(PMatrixREG)   [this is an identity in general]
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnTraceSpin> {
  typedef C<typename UnaryReturn<T, FnTraceSpin>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnTraceSpin>::Type_t
traceSpin(const PMatrixREG<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixREG<T,N,C>, FnTraceSpin>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = traceSpin(s1.elem(i,j));

  return d;
}


//! PMatrixREG = transposeColor(PMatrixREG) [ this is an identity in general]
/*! define the return type */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnTransposeColor> {
  typedef C<typename UnaryReturn<T, FnTransposeColor>::Type_t, N> Type_t;
};

/*! define the function itself.Recurse down elements of the primmatrix
 *  and call transposeColor on each one */
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnTransposeColor>::Type_t
transposeColor(const PMatrixREG<T,N,C>& s1)
{ 
  typename UnaryReturn<PMatrixREG<T,N,C>, FnTransposeColor>::Type_t d;
  for(int i=0; i < N; ++i) {
    for(int j=0; j < N; ++j) {
      d.elem(i,j) = transposeColor(s1.elem(i,j));
    }
  }

  return d;
}


//! PMatrixREG = transposeSpin(PMatrixREG) [ this is an identity in general]
/*! define the return type */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnTransposeSpin> {
  typedef C<typename UnaryReturn<T, FnTransposeSpin>::Type_t, N> Type_t;
};

/*! define the function itself.Recurse down elements of the primmatrix
 *  and call transposeSpin on each one */
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnTransposeSpin>::Type_t
transposeSpin(const PMatrixREG<T,N,C>& s1)
{ 
  typename UnaryReturn<PMatrixREG<T,N,C>, FnTransposeSpin>::Type_t d;
  for(int i=0; i < N; ++i) {
    for(int j=0; j < N; ++j) {
      d.elem(i,j) = transposeSpin(s1.elem(i,j));
    }
  }

  return d;
}


// PScalarREG = traceMultiply(PMatrixREG,PMatrixREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnTraceMultiply> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnTraceMultiply>::Type_t
traceMultiply(const PMatrixREG<T1,N,C>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnTraceMultiply>::Type_t  d;

  d.elem() = traceMultiply(l.elem(0,0), r.elem(0,0));
  for(int k=1; k < N; ++k)
    d.elem() += traceMultiply(l.elem(0,k), r.elem(k,0));

  for(int j=1; j < N; ++j)
    for(int k=0; k < N; ++k)
      d.elem() += traceMultiply(l.elem(j,k), r.elem(k,j));

  return d;
}

// PScalarREG = traceMultiply(PMatrixREG,PScalarREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, FnTraceMultiply> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, FnTraceMultiply>::Type_t
traceMultiply(const PMatrixREG<T1,N,C>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, FnTraceMultiply>::Type_t  d;

  d.elem() = traceMultiply(l.elem(0,0), r.elem());
  for(int k=1; k < N; ++k)
    d.elem() += traceMultiply(l.elem(k,k), r.elem());

  return d;
}

// PScalarREG = traceMultiply(PScalarREG,PMatrixREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, FnTraceMultiply> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, FnTraceMultiply>::Type_t
traceMultiply(const PScalarREG<T1>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, FnTraceMultiply>::Type_t  d;

  d.elem() = traceMultiply(l.elem(), r.elem(0,0));
  for(int k=1; k < N; ++k)
    d.elem() += traceMultiply(l.elem(), r.elem(k,k));

  return d;
}



//! PMatrixREG = traceColorMultiply(PMatrixREG,PMatrixREG)   [the trace is an identity in general]
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnTraceColorMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnTraceColorMultiply>::Type_t
traceColorMultiply(const PMatrixREG<T1,N,C>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnTraceColorMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
    {
      d.elem(i,j) = traceColorMultiply(l.elem(i,0), r.elem(0,j));
      for(int k=1; k < N; ++k)
	d.elem(i,j) += traceColorMultiply(l.elem(i,k), r.elem(k,j));
    }

  return d;
}

// PMatrixREG = traceColorMultiply(PMatrixREG,PScalarREG)   [the trace is an identity in general]
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, FnTraceColorMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, FnTraceColorMultiply>::Type_t
traceColorMultiply(const PMatrixREG<T1,N,C>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, FnTraceColorMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = traceColorMultiply(l.elem(i,j), r.elem());

  return d;
}

// PMatrixREG = traceColorMultiply(PScalarREG,PMatrixREG)   [the trace is an identity in general]
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, FnTraceColorMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, FnTraceColorMultiply>::Type_t
traceColorMultiply(const PScalarREG<T1>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, FnTraceColorMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = traceColorMultiply(l.elem(), r.elem(i,j));

  return d;
}


//! PMatrixREG = traceSpinMultiply(PMatrixREG,PMatrixREG)   [the trace is an identity in general]
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnTraceSpinMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(const PMatrixREG<T1,N,C>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnTraceSpinMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
    {
      d.elem(i,j) = traceSpinMultiply(l.elem(i,0), r.elem(0,j));
      for(int k=1; k < N; ++k)
	d.elem(i,j) += traceSpinMultiply(l.elem(i,k), r.elem(k,j));
    }

  return d;
}

// PScalarREG = traceSpinMultiply(PMatrixREG,PScalarREG)   [the trace is an identity in general]
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, FnTraceSpinMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(const PMatrixREG<T1,N,C>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, FnTraceSpinMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = traceSpinMultiply(l.elem(i,j), r.elem());

  return d;
}

// PScalarREG = traceSpinMultiply(PScalarREG,PMatrixREG)   [the trace is an identity in general]
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, FnTraceSpinMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(const PScalarREG<T1>& l, const PMatrixREG<T2,N,C>& r)
{
  typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, FnTraceSpinMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = traceSpinMultiply(l.elem(), r.elem(i,j));

  return d;
}


//! PMatrixREG = Re(PMatrixREG)
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnReal> {
  typedef C<typename UnaryReturn<T, FnReal>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnReal>::Type_t
real(const PMatrixREG<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixREG<T,N,C>, FnReal>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = real(s1.elem(i,j));

  return d;
}


//! PMatrixREG = Im(PMatrixREG)
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnImag> {
  typedef C<typename UnaryReturn<T, FnImag>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnImag>::Type_t
imag(const PMatrixREG<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixREG<T,N,C>, FnImag>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = imag(s1.elem(i,j));

  return d;
}


//! PMatrixREG<T> = (PMatrixREG<T> , PMatrixREG<T>)
template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnCmplx>::Type_t
cmplx(const PMatrixREG<T1,N,C>& s1, const PMatrixREG<T2,N,C>& s2)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnCmplx>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = cmplx(s1.elem(i,j), s2.elem(i,j));

  return d;
}




  //! isfinite
template<class T1, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T1,N,C>, FnIsFinite> {
  typedef PScalarREG< typename UnaryReturn<T1, FnIsFinite >::Type_t > Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T1,N,C>, FnIsFinite>::Type_t
isfinite(const PMatrixREG<T1,N,C>& l)
{
  typename UnaryReturn<PMatrixREG<T1,N,C>, FnIsFinite>::Type_t d(true);

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem() &= isfinite(l.elem(i,j));

  return d;
}



// Functions
//! PMatrixREG = i * PMatrixREG
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnTimesI> {
  typedef C<typename UnaryReturn<T, FnTimesI>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnTimesI>::Type_t
timesI(const PMatrixREG<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixREG<T,N,C>, FnTimesI>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = timesI(s1.elem(i,j));

  return d;
}

//! PMatrixREG = -i * PMatrixREG
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnTimesMinusI> {
  typedef C<typename UnaryReturn<T, FnTimesMinusI>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnTimesMinusI>::Type_t
timesMinusI(const PMatrixREG<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixREG<T,N,C>, FnTimesMinusI>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = timesMinusI(s1.elem(i,j));

  return d;
}

//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnGetSite> {
  typedef C<typename UnaryReturn<T, FnGetSite>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnGetSite>::Type_t
getSite(const PMatrixREG<T,N,C>& s1, int innersite)
{ 
  typename UnaryReturn<PMatrixREG<T,N,C>, FnGetSite>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = getSite(s1.elem(i,j), innersite);

  return d;
}

//! Extract color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnPeekColorVectorREG> {
  typedef C<typename UnaryReturn<T, FnPeekColorVectorREG>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnPeekColorVectorREG>::Type_t
peekColor(const PMatrixREG<T,N,C>& l, llvm::Value* row)
{
  typename UnaryReturn<PMatrixREG<T,N,C>, FnPeekColorVectorREG>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = peekColor(l.elem(i,j),row);
  return d;
}

//! Extract color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnPeekColorMatrixREG> {
  typedef C<typename UnaryReturn<T, FnPeekColorMatrixREG>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnPeekColorMatrixREG>::Type_t
peekColor(const PMatrixREG<T,N,C>& l, llvm::Value* row, llvm::Value* col)
{
  typename UnaryReturn<PMatrixREG<T,N,C>, FnPeekColorMatrixREG>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = peekColor(l.elem(i,j),row,col);
  return d;
}

//! Extract spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnPeekSpinVector> {
  typedef C<typename UnaryReturn<T, FnPeekSpinVector>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnPeekSpinVector>::Type_t
peekSpin(const PMatrixREG<T,N,C>& l, int row)
{
  typename UnaryReturn<PMatrixREG<T,N,C>, FnPeekSpinVector>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = peekSpin(l.elem(i,j),row);
  return d;
}

//! Extract spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnPeekSpinMatrix> {
  typedef C<typename UnaryReturn<T, FnPeekSpinMatrix>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnPeekSpinMatrix>::Type_t
peekSpin(const PMatrixREG<T,N,C>& l, int row, int col)
{
  typename UnaryReturn<PMatrixREG<T,N,C>, FnPeekSpinMatrix>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = peekSpin(l.elem(i,j),row,col);
  return d;
}

//! Insert color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnPokeColorMatrix> {
  typedef C<typename UnaryReturn<T, FnPokeColorMatrix>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T1,N,C>, FnPokeColorMatrix>::Type_t&
pokeColor(PMatrixREG<T1,N,C>& l, const PMatrixREG<T2,N,C>& r, int row)
{
  typedef typename UnaryReturn<PMatrixREG<T1,N,C>, FnPokeColorMatrix>::Type_t  Return_t;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      pokeColor(l.elem(i,j),r.elem(i,j),row);
  return static_cast<Return_t&>(l);
}

//! Insert color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T1,N,C>, FnPokeColorMatrix>::Type_t&
pokeColor(PMatrixREG<T1,N,C>& l, const PMatrixREG<T2,N,C>& r, int row, int col)
{
  typedef typename UnaryReturn<PMatrixREG<T1,N,C>, FnPokeColorMatrix>::Type_t  Return_t;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      pokeColor(l.elem(i,j),r.elem(i,j),row,col);
  return static_cast<Return_t&>(l);
}

//! Insert spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnPokeSpinMatrix> {
  typedef C<typename UnaryReturn<T, FnPokeSpinMatrix>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T1,N,C>, FnPokeSpinMatrix>::Type_t&
pokeSpin(PMatrixREG<T1,N,C>& l, const PMatrixREG<T2,N,C>& r, int row)
{
  typedef typename UnaryReturn<PMatrixREG<T1,N,C>, FnPokeSpinMatrix>::Type_t  Return_t;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      pokeSpin(l.elem(i,j),r.elem(i,j),row);
  return static_cast<Return_t&>(l);
}

//! Insert spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T1,N,C>, FnPokeSpinMatrix>::Type_t&
pokeSpin(PMatrixREG<T1,N,C>& l, const PMatrixREG<T2,N,C>& r, int row, int col)
{
  typedef typename UnaryReturn<PMatrixREG<T1,N,C>, FnPokeSpinMatrix>::Type_t  Return_t;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      pokeSpin(l.elem(i,j),r.elem(i,j),row,col);
  return static_cast<Return_t&>(l);
}



//! dest = 0
template<class T, int N, template<class,int> class C> 
inline void 
zero_rep(PMatrixREG<T,N,C>& dest) 
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      zero_rep(dest.elem(i,j));
}


//! dest = (mask) ? s1 : dest
template<class T, class T1, int N, template<class,int> class C> 
inline void 
copymask(PMatrixREG<T,N,C>& d, const PScalarREG<T1>& mask, const PMatrixREG<T,N,C>& s1) 
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      copymask(d.elem(i,j),mask.elem(),s1.elem(i,j));
}


//! dest [some type] = source [some type]
template<class T, class T1, int N, template<class,int> class C>
inline void 
copy_site(PMatrixREG<T,N,C>& d, int isite, const PMatrixREG<T1,N,C>& s1)
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      copy_site(d.elem(i,j), isite, s1.elem(i,j));
}

//! dest [some type] = source [some type]
template<class T, class T1, int N, template<class,int> class C>
inline void 
copy_site(PMatrixREG<T,N,C>& d, int isite, const PScalarREG<T1>& s1)
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      copy_site(d.elem(i,j), isite, s1.elem());
}


//! gather several inner sites together
template<class T, class T1, int N, template<class,int> class C>
inline void 
gather_sites(PMatrixREG<T,N,C>& d, 
	     const PMatrixREG<T1,N,C>& s0, int i0, 
	     const PMatrixREG<T1,N,C>& s1, int i1,
	     const PMatrixREG<T1,N,C>& s2, int i2,
	     const PMatrixREG<T1,N,C>& s3, int i3)
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
template<class T, int N, template<class,int> class C, class T1, class T2, class T3>
inline void
fill_random(PMatrixREG<T,N,C>& d, T1& seed, T2& skewed_seed, const T3& seed_mult)
{
  // The skewed_seed is the starting seed to use
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      fill_random(d.elem(i,j), seed, skewed_seed, seed_mult);
}

//! dest  = gaussian
template<class T, int N, template<class,int> class C>
inline void
fill_gaussian(PMatrixREG<T,N,C>& d, PMatrixREG<T,N,C>& r1, PMatrixREG<T,N,C>& r2)
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      fill_gaussian(d.elem(i,j), r1.elem(i,j), r2.elem(i,j));
}



#if 0
// Global sum over site indices only
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnSum> {
  typedef C<typename UnaryReturn<T, FnSum>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnSum>::Type_t
sum(const PMatrixREG<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixREG<T,N,C>, FnSum>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = sum(s1.elem(i,j));

  return d;
}
#endif


// InnerProduct (norm-seq) global sum = sum(tr(adj(s1)*s1))
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnNorm2> {
  typedef PScalarREG<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrixREG<T,N,C>, FnLocalNorm2> {
  typedef PScalarREG<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrixREG<T,N,C>, FnLocalNorm2>::Type_t
localNorm2(const PMatrixREG<T,N,C>& s1)
{
  typename UnaryReturn<PMatrixREG<T,N,C>, FnLocalNorm2>::Type_t  d;

  d.elem() = localNorm2(s1.elem(0,0));
  for(int j=1; j < N; ++j)
    d.elem() += localNorm2(s1.elem(0,j));

  for(int i=1; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem() += localNorm2(s1.elem(i,j));

  return d;
}


//! PScalarREG = innerProduct(PMatrixREG,PMatrixREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnInnerProduct> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

//! PScalarREG = localInnerProduct(PMatrixREG,PMatrixREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnLocalInnerProduct> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnLocalInnerProduct>::Type_t
localInnerProduct(const PMatrixREG<T1,N,C>& s1, const PMatrixREG<T2,N,C>& s2)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnLocalInnerProduct>::Type_t  d;

  d.elem() = localInnerProduct(s1.elem(0,0), s2.elem(0,0));
  for(int k=1; k < N; ++k)
    d.elem() += localInnerProduct(s1.elem(k,0), s2.elem(k,0));

  for(int j=1; j < N; ++j)
    for(int k=0; k < N; ++k)
      d.elem() += localInnerProduct(s1.elem(k,j), s2.elem(k,j));

  return d;
}

//! PScalarREG = localInnerProduct(PMatrixREG,PScalarREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, FnLocalInnerProduct> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(const PMatrixREG<T1,N,C>& s1, const PScalarREG<T2>& s2)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, FnLocalInnerProduct>::Type_t  d;

  d.elem() = localInnerProduct(s1.elem(0,0), s2.elem());
  for(int k=1; k < N; ++k)
    d.elem() += localInnerProduct(s1.elem(k,k), s2.elem());

  return d;
}

//! PScalarREG = localInnerProduct(PScalarREG,PMatrixREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, FnLocalInnerProduct> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, FnLocalInnerProduct>::Type_t
localInnerProduct(const PScalarREG<T1>& s1, const PMatrixREG<T2,N,C>& s2)
{
  typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, FnLocalInnerProduct>::Type_t  d;

  d.elem() = localInnerProduct(s1.elem(), s2.elem(0,0));
  for(int k=1; k < N; ++k)
    d.elem() += localInnerProduct(s1.elem(), s2.elem(k,k));

  return d;
}


//! PScalarREG = innerProductReal(PMatrixREG,PMatrixREG)
/*!
 * return  realpart of InnerProduct(adj(s1)*s2)
 */
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnInnerProductReal > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

//! PScalarREG = innerProductReal(PMatrixREG,PMatrixREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnLocalInnerProductReal > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const PMatrixREG<T1,N,C>& s1, const PMatrixREG<T2,N,C>& s2)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PMatrixREG<T2,N,C>, FnLocalInnerProductReal>::Type_t  d;

  d.elem() = localInnerProductReal(s1.elem(0,0), s2.elem(0,0));
  for(int k=1; k < N; ++k)
    d.elem() += localInnerProductReal(s1.elem(k,0), s2.elem(k,0));

  for(int j=1; j < N; ++j)
    for(int k=0; k < N; ++k)
      d.elem() += localInnerProductReal(s1.elem(k,j), s2.elem(k,j));

  return d;
}

//! PScalarREG = localInnerProductReal(PMatrixREG,PScalarREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, FnLocalInnerProductReal > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const PMatrixREG<T1,N,C>& s1, const PScalarREG<T2>& s2)
{
  typename BinaryReturn<PMatrixREG<T1,N,C>, PScalarREG<T2>, FnLocalInnerProductReal>::Type_t  d;

  d.elem() = localInnerProductReal(s1.elem(0,0), s2.elem());
  for(int k=1; k < N; ++k)
    d.elem() += localInnerProductReal(s1.elem(k,0), s2.elem(k,k));

  return d;
}

//! PScalarREG = localInnerProductReal(PScalarREG,PMatrixREG)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, FnLocalInnerProductReal > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const PScalarREG<T1>& s1, const PMatrixREG<T2,N,C>& s2)
{
  typename BinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, FnLocalInnerProductReal>::Type_t  d;

  d.elem() = localInnerProductReal(s1.elem(), s2.elem(0,0));
  for(int k=1; k < N; ++k)
    d.elem() += localInnerProductReal(s1.elem(), s2.elem(k,k));

  return d;
}


//! PMatrixREG<T> = where(PScalarREG, PMatrixREG, PMatrixREG)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3, int N, template<class,int> class C>
struct TrinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, PMatrixREG<T3,N,C>, FnWhere> {
  typedef C<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t, N>  Type_t;
};

template<class T1, class T2, class T3, int N, template<class,int> class C>
inline typename TrinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, PMatrixREG<T3,N,C>, FnWhere>::Type_t
where(const PScalarREG<T1>& a, const PMatrixREG<T2,N,C>& b, const PMatrixREG<T3,N,C>& c)
{
  typename TrinaryReturn<PScalarREG<T1>, PMatrixREG<T2,N,C>, PMatrixREG<T3,N,C>, FnWhere>::Type_t  d;

  // Not optimal - want to have where outside assignment
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = where(a.elem(), b.elem(i,j), c.elem(i,j));

  return d;
}


template<class T, int N, template<class,int> class C>
inline void 
qdpPHI(PMatrixREG<T,N,C>& d, 
       const PMatrixREG<T,N,C>& phi0, llvm::BasicBlock* bb0 ,
       const PMatrixREG<T,N,C>& phi1, llvm::BasicBlock* bb1 )
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      qdpPHI(d.elem(i,j),
	     phi0.elem(i,j),bb0,
	     phi1.elem(i,j),bb1);
}




/*! @} */  // end of group primmatrix

} // namespace QDP

#endif
