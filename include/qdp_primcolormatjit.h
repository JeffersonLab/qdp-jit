// -*- C++ -*-

/*! \file
 * \brief Primitive Color Matrix
 */


#ifndef QDP_PRIMCOLORMATJIT_H
#define QDP_PRIMCOLORMATJIT_H

namespace QDP {


//-------------------------------------------------------------------------------------
/*! \addtogroup primcolormatrix Color matrix primitive
 * \ingroup primmatrix
 *
 * Primitive type that transforms like a Color Matrix
 *
 * @{
 */


//! Primitive color Matrix class 
template <class T, int N> class PColorMatrixJIT : public PMatrixJIT<T, N, PColorMatrixJIT>
{
public:

  PColorMatrixJIT(curry_t c): PMatrixJIT<T, N, PColorMatrixJIT>(c) {}
  PColorMatrixJIT(newspace_t n): PMatrixJIT<T, N, PColorMatrixJIT>(n) {}
  PColorMatrixJIT(newspace_t n,PColorMatrixJIT* orig): PMatrixJIT<T, N, PColorMatrixJIT>(n,orig) { }



  //! PColorMatrixJIT = PScalarJIT
  /*! Fill with primitive scalar */
  template<class T1>
  inline
  PColorMatrixJIT& operator=(const PScalarJIT<T1>& rhs)
    {
      this->assign(rhs);
      return *this;
    }

  //! PColorMatrixJIT = PColorMatrixJIT
  /*! Set equal to another PMatrix */
  template<class T1>
  inline
  PColorMatrixJIT& operator=(const PColorMatrixJIT<T1,N>& rhs) 
    {
      this->assign(rhs);
      return *this;
    }



  PColorMatrixJIT& operator=(const PColorMatrixJIT& rhs) 
    {
      this->assign(rhs);
      return *this;
    }

};

/*! @} */   // end of group primcolormatrix

//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------


// Underlying word type
template<class T1, int N>
struct WordType<PColorMatrixJIT<T1,N> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

// Fixed Precisions
template<class T1, int N>
struct SinglePrecType<PColorMatrixJIT<T1,N> >
{
  typedef PColorMatrixJIT<typename SinglePrecType<T1>::Type_t, N> Type_t;
};

template<class T1, int N>
struct DoublePrecType<PColorMatrixJIT<T1,N> >
{
  typedef PColorMatrixJIT<typename DoublePrecType<T1>::Type_t, N> Type_t;
};



// Internally used scalars
template<class T, int N>
struct InternalScalar<PColorMatrixJIT<T,N> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive into a scalar leaving grid along
template<class T, int N>
struct PrimitiveScalar<PColorMatrixJIT<T,N> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices along
template<class T, int N>
struct LatticeScalar<PColorMatrixJIT<T,N> > {
  typedef PColorMatrixJIT<typename LatticeScalar<T>::Type_t, N>  Type_t;
};

//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PColorMatrixJIT) -> PColorMatrixJIT
template<class T1, int N, class Op>
struct UnaryReturn<PColorMatrixJIT<T1,N>, Op> {
  typedef PColorMatrixJIT<typename UnaryReturn<T1, Op>::Type_t, N>  Type_t;
};

// Default binary(PScalarJIT,PColorMatrixJIT) -> PColorMatrixJIT
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2,N>, Op> {
  typedef PColorMatrixJIT<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PColorMatrixJIT,PColorMatrixJIT) -> PColorMatrixJIT
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PColorMatrixJIT<T2,N>, Op> {
  typedef PColorMatrixJIT<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PColorMatrixJIT,PScalarJIT) -> PColorMatrixJIT
template<class T1, int N, class T2, class Op>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PScalarJIT<T2>, Op> {
  typedef PColorMatrixJIT<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};


// Assignment is different
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PColorMatrixJIT<T2,N>, OpAssign > {
  typedef PColorMatrixJIT<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PColorMatrixJIT<T2,N>, OpAddAssign > {
  typedef PColorMatrixJIT<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PColorMatrixJIT<T2,N>, OpSubtractAssign > {
  typedef PColorMatrixJIT<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PColorMatrixJIT<T2,N>, OpMultiplyAssign > {
  typedef PColorMatrixJIT<T1,N> &Type_t;
};
 

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PScalarJIT<T2>, OpAssign > {
  typedef PColorMatrixJIT<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PScalarJIT<T2>, OpAddAssign > {
  typedef PColorMatrixJIT<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PScalarJIT<T2>, OpSubtractAssign > {
  typedef PColorMatrixJIT<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PScalarJIT<T2>, OpMultiplyAssign > {
  typedef PColorMatrixJIT<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PScalarJIT<T2>, OpDivideAssign > {
  typedef PColorMatrixJIT<T1,N> &Type_t;
};
 


// ColorMatrix
template<class T, int N>
struct UnaryReturn<PColorMatrixJIT<T,N>, FnTrace > {
  typedef PScalarJIT<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PColorMatrixJIT<T,N>, FnRealTrace > {
  typedef PScalarJIT<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PColorMatrixJIT<T,N>, FnImagTrace > {
  typedef PScalarJIT<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PColorMatrixJIT<T,N>, FnNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PColorMatrixJIT<T,N>, FnLocalNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PColorMatrixJIT<T2,N>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PScalarJIT<T2>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2,N>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PColorMatrixJIT<T2,N>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PScalarJIT<T2>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2,N>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PColorMatrixJIT<T2,N>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PScalarJIT<T2>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2,N>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PColorMatrixJIT<T2,N>, FnInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PScalarJIT<T2>, FnInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2,N>, FnInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PColorMatrixJIT<T2,N>, FnLocalInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PScalarJIT<T2>, FnLocalInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2,N>, FnLocalInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};





//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

/*! \addtogroup primcolormatrix */
/*! @{ */

// trace = traceColor(source1)
/*! This only acts on color indices and is diagonal in all other indices */
template<class T, int N>
struct UnaryReturn<PColorMatrixJIT<T,N>, FnTraceColor > {
  typedef PScalarJIT<typename UnaryReturn<T, FnTraceColor>::Type_t>  Type_t;
};

template<class T, int N>
inline typename UnaryReturn<PColorMatrixJIT<T,N>, FnTraceColor>::Type_t
traceColor(const PColorMatrixJIT<T,N>& s1)
{
  typename UnaryReturn<PColorMatrixJIT<T,N>, FnTraceColor>::Type_t  d(s1.func());

  // Since the color index is eaten, do not need to pass on function by
  // calling trace(...) again
  d.elem() = s1.elem(0,0);
  for(int i=1; i < N; ++i)
    d.elem() += s1.elem(i,i);

  return d;
}


//! PScalarJIT = traceColorMultiply(PColorMatrixJIT,PColorMatrixJIT)
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PColorMatrixJIT<T2,N>, FnTraceColorMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
inline typename BinaryReturn<PColorMatrixJIT<T1,N>, PColorMatrixJIT<T2,N>, FnTraceColorMultiply>::Type_t
traceColorMultiply(const PColorMatrixJIT<T1,N>& l, const PColorMatrixJIT<T2,N>& r)
{
  typename BinaryReturn<PColorMatrixJIT<T1,N>, PColorMatrixJIT<T2,N>, FnTraceColorMultiply>::Type_t  d(l.func());

  // The traceColor is eaten here
  d.elem() = l.elem(0,0) * r.elem(0,0);
  for(int k=1; k < N; ++k)
    d.elem() += l.elem(0,k) * r.elem(k,0);

  for(int j=1; j < N; ++j)
    for(int k=0; k < N; ++k)
      d.elem() += l.elem(j,k) * r.elem(k,j);

  return d;
}

//! PScalarJIT = traceColorMultiply(PColorMatrixJIT,PScalarJIT)
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PScalarJIT<T2>, FnTraceColorMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PColorMatrixJIT<T1,N>, PScalarJIT<T2>, FnTraceColorMultiply>::Type_t
traceColorMultiply(const PColorMatrixJIT<T1,N>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PColorMatrixJIT<T1,N>, PScalarJIT<T2>, FnTraceColorMultiply>::Type_t  d(l.func());

  // The traceColor is eaten here
  d.elem() = l.elem(0,0) * r.elem();
  for(int k=1; k < N; ++k)
    d.elem() += l.elem(k,k) * r.elem();

  return d;
}

// PScalarJIT = traceColorMultiply(PScalarJIT,PColorMatrixJIT)
template<class T1, class T2, int N>
struct BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2,N>, FnTraceColorMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
inline typename BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2,N>, FnTraceColorMultiply>::Type_t
traceColorMultiply(const PScalarJIT<T1>& l, const PColorMatrixJIT<T2,N>& r)
{
  typename BinaryReturn<PScalarJIT<T1>, PColorMatrixJIT<T2,N>, FnTraceColorMultiply>::Type_t  d(l.func());

  // The traceColor is eaten here
  d.elem() = l.elem() * r.elem(0,0);
  for(int k=1; k < N; ++k)
    d.elem() += l.elem() * r.elem(k,k);

  return d;
}


/*! Specialise the return type */
template <class T, int N>
struct UnaryReturn<PColorMatrixJIT<T,N>, FnTransposeColor > {
  typedef PColorMatrixJIT<typename UnaryReturn<T, FnTransposeColor>::Type_t, N> Type_t;
};

//! PColorMatrixJIT = transposeColor(PColorMatrixJIT) 
/*! t = transposeColor(source1) - ColorMatrix specialization -- where the work is actually done */
template<class T, int N>
inline typename UnaryReturn<PColorMatrixJIT<T,N>, FnTransposeColor >::Type_t
transposeColor(const PColorMatrixJIT<T,N>& s1)
{
  typename UnaryReturn<PColorMatrixJIT<T,N>, FnTransposeColor>::Type_t d(s1.func());;
 
  for(int i=0; i < N; i++) { 
    for(int j=0; j < N; j++) { 
      // Transpose, so flip indices
      d.elem(i,j) = s1.elem(j,i);
    }
  }
  return d;
}


//-----------------------------------------------
// OuterProduct must be handled specially for each color and spin
// The problem is the traits class - I have no way to say to PVector's
//  transform into a PMatrix but downcast the trait to a PColorMatrixJIT or PSpinMatrix

//! PColorMatrixJIT = outerProduct(PColorVectorJIT, PColorVectorJIT)
template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorJIT<T1,N>, PColorVectorJIT<T2,N>, FnOuterProduct> {
  typedef PColorMatrixJIT<typename BinaryReturn<T1, T2, FnOuterProduct>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N>
inline typename BinaryReturn<PColorVectorJIT<T1,N>, PColorVectorJIT<T2,N>, FnOuterProduct>::Type_t
outerProduct(const PColorVectorJIT<T1,N>& l, const PColorVectorJIT<T2,N>& r)
{
  typename BinaryReturn<PColorVectorJIT<T1,N>, PColorVectorJIT<T2,N>, FnOuterProduct>::Type_t  d(l.func());

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = outerProduct(l.elem(i),r.elem(j));

  return d;
}


//-----------------------------------------------
// Peeking and poking
//! Extract color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T, int N>
struct UnaryReturn<PColorMatrixJIT<T,N>, FnPeekColorMatrixJIT > {
  typedef PScalarJIT<typename UnaryReturn<T, FnPeekColorMatrixJIT>::Type_t>  Type_t;
};

template<class T, int N>
inline typename UnaryReturn<PColorMatrixJIT<T,N>, FnPeekColorMatrixJIT>::Type_t
peekColor(const PColorMatrixJIT<T,N>& l, int row, int col)
{
  typename UnaryReturn<PColorMatrixJIT<T,N>, FnPeekColorMatrixJIT>::Type_t  d( l.func() );
  d.elem() = l.getRegElem(row,col);
  return d;
}

//! Insert color matrix components
template<class T1, class T2, int N>
inline PColorMatrixJIT<T1,N>&
pokeColor(PColorMatrixJIT<T1,N>& l, const PScalarJIT<T2>& r, int row, int col)
{
  // Note, do not need to propagate down since the function is eaten at this level
  l.getRegElem(row,col) = r.elem();
  return l;
}


//-----------------------------------------------------------------------------
// Contraction for color matrices
// colorContract 
template<class T1, class T2, class T3, int N>
struct TrinaryReturn<PColorMatrixJIT<T1,N>, PColorMatrixJIT<T2,N>, PColorMatrixJIT<T3,N>, FnColorContract> {
  typedef PScalarJIT<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};

//! dest  = colorContract(Qprop1,Qprop2,Qprop3)
/*!
 * Performs:
 *  \f$dest = \sum_{i1,i2,i3,j1,j2,j3} \epsilon^{i1,j1,k1}\epsilon^{i2,j2,k2} Q1^{i1,i2} Q2^{j1,j2} Q3^{k1,k2}\f$
 *
 * This routine is completely unrolled for 3 colors
 */
template<class T1, class T2, class T3>
inline typename TrinaryReturn<PColorMatrixJIT<T1,3>, PColorMatrixJIT<T2,3>, PColorMatrixJIT<T3,3>, FnColorContract>::Type_t
colorContract(const PColorMatrixJIT<T1,3>& s1, const PColorMatrixJIT<T2,3>& s2, const PColorMatrixJIT<T3,3>& s3)
{
  typename TrinaryReturn<PColorMatrixJIT<T1,3>, PColorMatrixJIT<T2,3>, PColorMatrixJIT<T3,3>, FnColorContract>::Type_t  d(s1.func());

  // Permutations: +(0,1,2)+(1,2,0)+(2,0,1)-(1,0,2)-(0,2,1)-(2,1,0)

  // d = \epsilon^{i1,j1,k1}\epsilon^{i2,j2,k2} Q1^{i1,i2} Q2^{j1,j2} Q3^{k1,k2}
  d.elem() = (s1.elem(0,0)*s2.elem(1,1)
           -  s1.elem(1,0)*s2.elem(0,1)
           -  s1.elem(0,1)*s2.elem(1,0)
           +  s1.elem(1,1)*s2.elem(0,0))*s3.elem(2,2)
           + (s1.elem(1,1)*s2.elem(2,2)
           -  s1.elem(2,1)*s2.elem(1,2)
           -  s1.elem(1,2)*s2.elem(2,1)
           +  s1.elem(2,2)*s2.elem(1,1))*s3.elem(0,0)
           + (s1.elem(2,2)*s2.elem(0,0)
           -  s1.elem(0,2)*s2.elem(2,0)
           -  s1.elem(2,0)*s2.elem(0,2)
           +  s1.elem(0,0)*s2.elem(2,2))*s3.elem(1,1)

           + (s1.elem(1,0)*s2.elem(2,1)
           -  s1.elem(2,0)*s2.elem(1,1)
           -  s1.elem(1,1)*s2.elem(2,0)
           +  s1.elem(2,1)*s2.elem(1,0))*s3.elem(0,2)
           + (s1.elem(1,2)*s2.elem(2,0)
           -  s1.elem(2,2)*s2.elem(1,0)
           -  s1.elem(1,0)*s2.elem(2,2)
           +  s1.elem(2,0)*s2.elem(1,2))*s3.elem(0,1)

           + (s1.elem(2,0)*s2.elem(0,1)
           -  s1.elem(0,0)*s2.elem(2,1)
           -  s1.elem(2,1)*s2.elem(0,0)
           +  s1.elem(0,1)*s2.elem(2,0))*s3.elem(1,2)
           + (s1.elem(2,1)*s2.elem(0,2)
           -  s1.elem(0,1)*s2.elem(2,2)
           -  s1.elem(2,2)*s2.elem(0,1)
           +  s1.elem(0,2)*s2.elem(2,1))*s3.elem(1,0)

           + (s1.elem(0,1)*s2.elem(1,2)
           -  s1.elem(1,1)*s2.elem(0,2)
           -  s1.elem(0,2)*s2.elem(1,1)
           +  s1.elem(1,2)*s2.elem(0,1))*s3.elem(2,0)
           + (s1.elem(0,2)*s2.elem(1,0)
           -  s1.elem(1,2)*s2.elem(0,0)
           -  s1.elem(0,0)*s2.elem(1,2)
           +  s1.elem(1,0)*s2.elem(0,2))*s3.elem(2,1);

  return d;
}


//! dest  = colorContract(Qprop1,Qprop2,Qprop3)
/*!
 * Performs:
 *  \f$dest = \sum_{i1,i2,i3,j1,j2,j3} \epsilon^{i1,j1,k1}\epsilon^{i2,j2,k2} Q1^{i1,i2} Q2^{j1,j2} Q3^{k1,k2}\f$
 *
 *  These are some place holders for Nc = 1
 *  These routine are actually used in the 
 *  baryon routines. Seperate baryon routines
 *  should be written for every number of colors.
 */
template<class T1, class T2, class T3>
inline typename TrinaryReturn<PColorMatrixJIT<T1,1>, PColorMatrixJIT<T2,1>, PColorMatrixJIT<T3,1>, FnColorContract>::Type_t
colorContract(const PColorMatrixJIT<T1,1>& s1, const PColorMatrixJIT<T2,1>& s2, const PColorMatrixJIT<T3,1>& s3)
{
  typename TrinaryReturn<PColorMatrixJIT<T1,1>, PColorMatrixJIT<T2,1>, PColorMatrixJIT<T3,1>, FnColorContract>::Type_t  d(s1.func());

  // not written 
  QDPIO::cerr << __func__ << ": not written for Nc=1" << endl;
  QDP_abort(1);

  return d ; 
}


//! dest  = colorContract(Qprop1,Qprop2,Qprop3)
/*!
 * Performs:
 *  \f$dest = \sum_{i1,i2,i3,j1,j2,j3} \epsilon^{i1,j1,k1}\epsilon^{i2,j2,k2} Q1^{i1,i2} Q2^{j1,j2} Q3^{k1,k2}\f$
 *
 *  These are some place holders for Nc = 2
 *  These routine are actually used in the 
 *  baryon routines. Seperate baryon routines
 *  should be written for every number of colors.
 */
template<class T1, class T2, class T3>
inline typename TrinaryReturn<PColorMatrixJIT<T1,2>, PColorMatrixJIT<T2,2>, PColorMatrixJIT<T3,2>, FnColorContract>::Type_t
colorContract(const PColorMatrixJIT<T1,2>& s1, const PColorMatrixJIT<T2,2>& s2, const PColorMatrixJIT<T3,2>& s3)
{
  typename TrinaryReturn<PColorMatrixJIT<T1,2>, PColorMatrixJIT<T2,2>, PColorMatrixJIT<T3,2>, FnColorContract>::Type_t  d(s1.func());

  // not written 
  QDPIO::cerr << __func__ << ": not written for Nc=2" << endl;
  QDP_abort(1);

  return d ; 
}


//! dest  = colorContract(Qprop1,Qprop2,Qprop3)
/*!
 * Performs:
 *  \f$dest = \sum_{i1,i2,i3,j1,j2,j3} \epsilon^{i1,j1,k1}\epsilon^{i2,j2,k2} Q1^{i1,i2} Q2^{j1,j2} Q3^{k1,k2}\f$
 *
 *  These are some place holders for Nc = 4
 *  These routine are actually used in the 
 *  baryon routines. Seperate baryon routines
 *  should be written for every number of colors.
 */
template<class T1, class T2, class T3>
inline typename TrinaryReturn<PColorMatrixJIT<T1,4>, PColorMatrixJIT<T2,4>, PColorMatrixJIT<T3,4>, FnColorContract>::Type_t
colorContract(const PColorMatrixJIT<T1,4>& s1, const PColorMatrixJIT<T2,4>& s2, const PColorMatrixJIT<T3,4>& s3)
{
  typename TrinaryReturn<PColorMatrixJIT<T1,4>, PColorMatrixJIT<T2,4>, PColorMatrixJIT<T3,4>, FnColorContract>::Type_t  d(s1.func());

  // not written 
  QDPIO::cerr << __func__ << ": not written for Nc=4" << endl;
  QDP_abort(1);

  return d ; 
}


//-----------------------------------------------------------------------------
// Contraction for quark propagators
// QuarkContract 
//! dest  = QuarkContractXX(Qprop1,Qprop2)
/*!
 * Performs:
 *  \f$dest^{k2,k1} = \sum_{i1,i2,j1,j2} \epsilon^{i1,j1,k1}\epsilon^{i2,j2,k2} Q1^{i1,i2} Q2^{j1,j2}\f$
 *
 * This routine is completely unrolled for 3 colors
 */
template<class T1, class T2>
inline typename BinaryReturn<PColorMatrixJIT<T1,3>, PColorMatrixJIT<T2,3>, FnQuarkContractXX>::Type_t
quarkContractXX(const PColorMatrixJIT<T1,3>& s1, const PColorMatrixJIT<T2,3>& s2)
{
  typename BinaryReturn<PColorMatrixJIT<T1,3>, PColorMatrixJIT<T2,3>, FnQuarkContractXX>::Type_t  d(s1.func());

  // Permutations: +(0,1,2)+(1,2,0)+(2,0,1)-(1,0,2)-(0,2,1)-(2,1,0)

  // k1 = 0, k2 = 0
  // d(0,0) = eps^{i1,j1,0}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(1,2,0),-(2,1,0)    +(1,2,0),-(2,1,0)
  d.elem(0,0) = s1.elem(1,1)*s2.elem(2,2)
              - s1.elem(1,2)*s2.elem(2,1)
              - s1.elem(2,1)*s2.elem(1,2)
              + s1.elem(2,2)*s2.elem(1,1);

  // k1 = 1, k2 = 0
  // d(0,1) = eps^{i1,j1,1}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(2,0,1),-(0,2,1)    +(1,2,0),-(2,1,0)    
  d.elem(0,1) = s1.elem(2,1)*s2.elem(0,2)
              - s1.elem(2,2)*s2.elem(0,1)
              - s1.elem(0,1)*s2.elem(2,2)
              + s1.elem(0,2)*s2.elem(2,1);

  // k1 = 2, k2 = 0
  // d(0,2) = eps^{i1,j1,2}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(0,1,2),-(1,0,2)    +(1,2,0),-(2,1,0)    
  d.elem(0,2) = s1.elem(0,1)*s2.elem(1,2)
              - s1.elem(0,2)*s2.elem(1,1)
              - s1.elem(1,1)*s2.elem(0,2)
              + s1.elem(1,2)*s2.elem(0,1);

  // k1 = 0, k2 = 1
  // d(1,0) = eps^{i1,j1,0}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(1,2,0),-(2,1,0)    +(2,0,1),-(0,2,1)
  d.elem(1,0) = s1.elem(1,2)*s2.elem(2,0)
              - s1.elem(1,0)*s2.elem(2,2)
              - s1.elem(2,2)*s2.elem(1,0)
              + s1.elem(2,0)*s2.elem(1,2);

  // k1 = 1, k2 = 1
  // d(1,1) = eps^{i1,j1,1}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(2,0,1),-(0,2,1)    +(2,0,1),-(0,2,1)
  d.elem(1,1) = s1.elem(2,2)*s2.elem(0,0)
              - s1.elem(2,0)*s2.elem(0,2)
              - s1.elem(0,2)*s2.elem(2,0)
              + s1.elem(0,0)*s2.elem(2,2);

  // k1 = 2, k2 = 1
  // d(1,1) = eps^{i1,j1,2}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(0,1,2),-(1,0,2)    +(2,0,1),-(0,2,1)
  d.elem(1,2) = s1.elem(0,2)*s2.elem(1,0)
              - s1.elem(0,0)*s2.elem(1,2)
              - s1.elem(1,2)*s2.elem(0,0)
              + s1.elem(1,0)*s2.elem(0,2);

  // k1 = 0, k2 = 2
  // d(2,0) = eps^{i1,j1,0}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(1,2,0),-(2,1,0)    +(0,1,2),-(1,0,2)
  d.elem(2,0) = s1.elem(1,0)*s2.elem(2,1)
              - s1.elem(1,1)*s2.elem(2,0)
              - s1.elem(2,0)*s2.elem(1,1)
              + s1.elem(2,1)*s2.elem(1,0);

  // k1 = 1, k2 = 2
  // d(2,1) = eps^{i1,j1,1}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(2,0,1),-(0,2,1)    +(0,1,2),-(1,0,2)
  d.elem(2,1) = s1.elem(2,0)*s2.elem(0,1)
              - s1.elem(2,1)*s2.elem(0,0)
              - s1.elem(0,0)*s2.elem(2,1)
              + s1.elem(0,1)*s2.elem(2,0);

  // k1 = 2, k2 = 2
  // d(2,2) = eps^{i1,j1,2}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(0,1,2),-(1,0,2)    +(0,1,2),-(1,0,2)
  d.elem(2,2) = s1.elem(0,0)*s2.elem(1,1)
              - s1.elem(0,1)*s2.elem(1,0)
              - s1.elem(1,0)*s2.elem(0,1)
              + s1.elem(1,1)*s2.elem(0,0);

  return d;
}


// Contraction for quark propagators
// QuarkContract 
//! dest  = QuarkContractXX(Qprop1,Qprop2)
/*!
 * Performs:
 *  \f$dest^{k2,k1} = \sum_{i1,i2,j1,j2} \epsilon^{i1,j1,k1}\epsilon^{i2,j2,k2} Q1^{i1,i2} Q2^{j1,j2}\f$
 *
 *  These are some place holders for Nc = 2
 *  These routine are actually used in the 
 *  baryon routines. Seperate baryon routines
 *  should be written for every number of colors.
 */
template<class T1, class T2>
inline typename BinaryReturn<PColorMatrixJIT<T1,1>, PColorMatrixJIT<T2,1>, FnQuarkContractXX>::Type_t
quarkContractXX(const PColorMatrixJIT<T1,1>& s1, const PColorMatrixJIT<T2,1>& s2)
{
  typename BinaryReturn<PColorMatrixJIT<T1,1>, PColorMatrixJIT<T2,1>, FnQuarkContractXX>::Type_t  d(s1.func());

  // not yet written 
  QDPIO::cerr << __func__ << ": not written for Nc=1" << endl;
  QDP_abort(1);

  return d ; 
}


// Contraction for quark propagators
// QuarkContract 
//! dest  = QuarkContractXX(Qprop1,Qprop2)
/*!
 * Performs:
 *  \f$dest^{k2,k1} = \sum_{i1,i2,j1,j2} \epsilon^{i1,j1,k1}\epsilon^{i2,j2,k2} Q1^{i1,i2} Q2^{j1,j2}\f$
 *
 *  These are some place holders for Nc = 2
 *  These routine are actually used in the 
 *  baryon routines. Seperate baryon routines
 *  should be written for every number of colors.
 */
template<class T1, class T2>
inline typename BinaryReturn<PColorMatrixJIT<T1,2>, PColorMatrixJIT<T2,2>, FnQuarkContractXX>::Type_t
quarkContractXX(const PColorMatrixJIT<T1,2>& s1, const PColorMatrixJIT<T2,2>& s2)
{
  typename BinaryReturn<PColorMatrixJIT<T1,2>, PColorMatrixJIT<T2,2>, FnQuarkContractXX>::Type_t  d(s1.func());

  // not yet written 
  QDPIO::cerr << __func__ << ": not written for Nc=2" << endl;
  QDP_abort(1);

  return d ; 
}


// Contraction for quark propagators
// QuarkContract 
//! dest  = QuarkContractXX(Qprop1,Qprop2)
/*!
 * Performs:
 *  \f$dest^{k2,k1} = \sum_{i1,i2,j1,j2} \epsilon^{i1,j1,k1}\epsilon^{i2,j2,k2} Q1^{i1,i2} Q2^{j1,j2}\f$
 *
 *  These are some place holders for Nc = 4 
 *  These routine are actually used in the 
 *  baryon routines. Seperate baryon routines
 *  should be written for every number of colors.
 */
template<class T1, class T2>
inline typename BinaryReturn<PColorMatrixJIT<T1,4>, PColorMatrixJIT<T2,4>, FnQuarkContractXX>::Type_t
quarkContractXX(const PColorMatrixJIT<T1,4>& s1, const PColorMatrixJIT<T2,4>& s2)
{
  typename BinaryReturn<PColorMatrixJIT<T1,4>, PColorMatrixJIT<T2,4>, FnQuarkContractXX>::Type_t  d(s1.func());

  // not yet written 
  QDPIO::cerr << __func__ << ": not written for Nc=4" << endl;
  QDP_abort(1);

  return d ; 
}


/*! @} */   // end of group primcolormatrix

} // namespace QDP

#endif
