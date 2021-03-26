// -*- C++ -*-

/*! \file
 * \brief Primitive Color Matrix
 */


#ifndef QDP_PRIMCOLORMATREG_H
#define QDP_PRIMCOLORMATREG_H

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
template <class T, int N> class PColorMatrixREG : public PMatrixREG<T, N, PColorMatrixREG>
{
public:

  void setup( const typename JITType< PColorMatrixREG >::Type_t& j ) {
    for (int i = 0 ; i < N ; i++ ) 
      for (int q = 0 ; q < N ; q++ ) 
	this->elem(i,q).setup( j.elem(i,q) );
  }

  void setup_value( const typename JITType< PColorMatrixREG >::Type_t& j ) {
    for (int i = 0 ; i < N ; i++ ) 
      for (int q = 0 ; q < N ; q++ ) 
	this->elem(i,q).setup_value( j.elem(i,q) );
  }

  
  PColorMatrixREG() {}

  PColorMatrixREG( const typename JITType< PColorMatrixREG >::Type_t& rhs ) {
    setup(rhs);
  }

  //! PColorMatrixREG = PScalarREG
  /*! Fill with primitive scalar */
  template<class T1>
  inline
  PColorMatrixREG& operator=(const PScalarREG<T1>& rhs)
    {
      this->assign(rhs);
      return *this;
    }

  //! PColorMatrixREG = PColorMatrixREG
  /*! Set equal to another PMatrix */
  template<class T1>
  inline
  PColorMatrixREG& operator=(const PColorMatrixREG<T1,N>& rhs) 
    {
      this->assign(rhs);
      return *this;
    }



  PColorMatrixREG& operator=(const PColorMatrixREG& rhs) 
    {
      this->assign(rhs);
      return *this;
    }

};





/*! @} */   // end of group primcolormatrix

//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

  template <class T, int N>
  struct JITType< PColorMatrixREG<T,N> >
  {
    typedef PColorMatrixJIT< typename JITType<T>::Type_t,N >  Type_t;
  };



// Underlying word type
template<class T1, int N>
struct WordType<PColorMatrixREG<T1,N> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

// Fixed Precisions
template<class T1, int N>
struct SinglePrecType<PColorMatrixREG<T1,N> >
{
  typedef PColorMatrixREG<typename SinglePrecType<T1>::Type_t, N> Type_t;
};

template<class T1, int N>
struct DoublePrecType<PColorMatrixREG<T1,N> >
{
  typedef PColorMatrixREG<typename DoublePrecType<T1>::Type_t, N> Type_t;
};



// Internally used scalars
template<class T, int N>
struct InternalScalar<PColorMatrixREG<T,N> > {
  typedef PScalarREG<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive into a scalar leaving grid along
template<class T, int N>
struct PrimitiveScalar<PColorMatrixREG<T,N> > {
  typedef PScalarREG<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices along
template<class T, int N>
struct LatticeScalar<PColorMatrixREG<T,N> > {
  typedef PColorMatrixREG<typename LatticeScalar<T>::Type_t, N>  Type_t;
};

//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PColorMatrixREG) -> PColorMatrixREG
template<class T1, int N, class Op>
struct UnaryReturn<PColorMatrixREG<T1,N>, Op> {
  typedef PColorMatrixREG<typename UnaryReturn<T1, Op>::Type_t, N>  Type_t;
};

// Default binary(PScalarREG,PColorMatrixREG) -> PColorMatrixREG
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PScalarREG<T1>, PColorMatrixREG<T2,N>, Op> {
  typedef PColorMatrixREG<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PColorMatrixREG,PColorMatrixREG) -> PColorMatrixREG
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PColorMatrixREG<T1,N>, PColorMatrixREG<T2,N>, Op> {
  typedef PColorMatrixREG<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PColorMatrixREG,PScalarREG) -> PColorMatrixREG
template<class T1, int N, class T2, class Op>
struct BinaryReturn<PColorMatrixREG<T1,N>, PScalarREG<T2>, Op> {
  typedef PColorMatrixREG<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};


// Assignment is different
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PColorMatrixREG<T2,N>, OpAssign > {
  typedef PColorMatrixREG<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PColorMatrixREG<T2,N>, OpAddAssign > {
  typedef PColorMatrixREG<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PColorMatrixREG<T2,N>, OpSubtractAssign > {
  typedef PColorMatrixREG<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PColorMatrixREG<T2,N>, OpMultiplyAssign > {
  typedef PColorMatrixREG<T1,N> &Type_t;
};
 

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PScalarREG<T2>, OpAssign > {
  typedef PColorMatrixREG<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PScalarREG<T2>, OpAddAssign > {
  typedef PColorMatrixREG<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PScalarREG<T2>, OpSubtractAssign > {
  typedef PColorMatrixREG<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PScalarREG<T2>, OpMultiplyAssign > {
  typedef PColorMatrixREG<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PScalarREG<T2>, OpDivideAssign > {
  typedef PColorMatrixREG<T1,N> &Type_t;
};
 


// ColorMatrix
template<class T, int N>
struct UnaryReturn<PColorMatrixREG<T,N>, FnTrace > {
  typedef PScalarREG<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PColorMatrixREG<T,N>, FnRealTrace > {
  typedef PScalarREG<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PColorMatrixREG<T,N>, FnImagTrace > {
  typedef PScalarREG<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PColorMatrixREG<T,N>, FnNorm2 > {
  typedef PScalarREG<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PColorMatrixREG<T,N>, FnLocalNorm2 > {
  typedef PScalarREG<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PColorMatrixREG<T2,N>, FnTraceMultiply> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PScalarREG<T2>, FnTraceMultiply> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PScalarREG<T1>, PColorMatrixREG<T2,N>, FnTraceMultiply> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PColorMatrixREG<T2,N>, FnInnerProduct> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PScalarREG<T2>, FnInnerProduct> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PScalarREG<T1>, PColorMatrixREG<T2,N>, FnInnerProduct> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PColorMatrixREG<T2,N>, FnLocalInnerProduct> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PScalarREG<T2>, FnLocalInnerProduct> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PScalarREG<T1>, PColorMatrixREG<T2,N>, FnLocalInnerProduct> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PColorMatrixREG<T2,N>, FnInnerProductReal> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PScalarREG<T2>, FnInnerProductReal> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PScalarREG<T1>, PColorMatrixREG<T2,N>, FnInnerProductReal> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PColorMatrixREG<T2,N>, FnLocalInnerProductReal> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PScalarREG<T2>, FnLocalInnerProductReal> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PScalarREG<T1>, PColorMatrixREG<T2,N>, FnLocalInnerProductReal> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};





template<class T1, int N>
struct UnaryReturn<PColorMatrixREG<T1,N>, FnIsFinite> {
  typedef PScalarREG< typename UnaryReturn<T1, FnIsFinite >::Type_t > Type_t;
};


template<class T1, int N>
inline typename UnaryReturn<PColorMatrixREG<T1,N>, FnIsFinite>::Type_t
isfinite(const PColorMatrixREG<T1,N>& l)
{
  typedef typename UnaryReturn<PColorMatrixREG<T1,N>, FnIsFinite>::Type_t Ret_t;
  Ret_t d(true);

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem() &= isfinite(l.elem(i,j));

  return d;
}





//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

/*! \addtogroup primcolormatrix */
/*! @{ */

// trace = traceColor(source1)
/*! This only acts on color indices and is diagonal in all other indices */
template<class T, int N>
struct UnaryReturn<PColorMatrixREG<T,N>, FnTraceColor > {
  typedef PScalarREG<typename UnaryReturn<T, FnTraceColor>::Type_t>  Type_t;
};

template<class T, int N>
inline typename UnaryReturn<PColorMatrixREG<T,N>, FnTraceColor>::Type_t
traceColor(const PColorMatrixREG<T,N>& s1)
{
  typename UnaryReturn<PColorMatrixREG<T,N>, FnTraceColor>::Type_t  d;

  // Since the color index is eaten, do not need to pass on function by
  // calling trace(...) again
  d.elem() = s1.elem(0,0);
  for(int i=1; i < N; ++i)
    d.elem() += s1.elem(i,i);

  return d;
}


//! PScalarREG = traceColorMultiply(PColorMatrixREG,PColorMatrixREG)
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PColorMatrixREG<T2,N>, FnTraceColorMultiply> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
inline typename BinaryReturn<PColorMatrixREG<T1,N>, PColorMatrixREG<T2,N>, FnTraceColorMultiply>::Type_t
traceColorMultiply(const PColorMatrixREG<T1,N>& l, const PColorMatrixREG<T2,N>& r)
{
  typename BinaryReturn<PColorMatrixREG<T1,N>, PColorMatrixREG<T2,N>, FnTraceColorMultiply>::Type_t  d;

  // The traceColor is eaten here
  d.elem() = l.elem(0,0) * r.elem(0,0);
  for(int k=1; k < N; ++k)
    d.elem() += l.elem(0,k) * r.elem(k,0);

  for(int j=1; j < N; ++j)
    for(int k=0; k < N; ++k)
      d.elem() += l.elem(j,k) * r.elem(k,j);

  return d;
}

//! PScalarREG = traceColorMultiply(PColorMatrixREG,PScalarREG)
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrixREG<T1,N>, PScalarREG<T2>, FnTraceColorMultiply> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PColorMatrixREG<T1,N>, PScalarREG<T2>, FnTraceColorMultiply>::Type_t
traceColorMultiply(const PColorMatrixREG<T1,N>& l, const PScalarREG<T2>& r)
{
  typename BinaryReturn<PColorMatrixREG<T1,N>, PScalarREG<T2>, FnTraceColorMultiply>::Type_t  d;

  // The traceColor is eaten here
  d.elem() = l.elem(0,0) * r.elem();
  for(int k=1; k < N; ++k)
    d.elem() += l.elem(k,k) * r.elem();

  return d;
}

// PScalarREG = traceColorMultiply(PScalarREG,PColorMatrixREG)
template<class T1, class T2, int N>
struct BinaryReturn<PScalarREG<T1>, PColorMatrixREG<T2,N>, FnTraceColorMultiply> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
inline typename BinaryReturn<PScalarREG<T1>, PColorMatrixREG<T2,N>, FnTraceColorMultiply>::Type_t
traceColorMultiply(const PScalarREG<T1>& l, const PColorMatrixREG<T2,N>& r)
{
  typename BinaryReturn<PScalarREG<T1>, PColorMatrixREG<T2,N>, FnTraceColorMultiply>::Type_t  d;

  // The traceColor is eaten here
  d.elem() = l.elem() * r.elem(0,0);
  for(int k=1; k < N; ++k)
    d.elem() += l.elem() * r.elem(k,k);

  return d;
}


/*! Specialise the return type */
template <class T, int N>
struct UnaryReturn<PColorMatrixREG<T,N>, FnTransposeColor > {
  typedef PColorMatrixREG<typename UnaryReturn<T, FnTransposeColor>::Type_t, N> Type_t;
};

//! PColorMatrixREG = transposeColor(PColorMatrixREG) 
/*! t = transposeColor(source1) - ColorMatrix specialization -- where the work is actually done */
template<class T, int N>
inline typename UnaryReturn<PColorMatrixREG<T,N>, FnTransposeColor >::Type_t
transposeColor(const PColorMatrixREG<T,N>& s1)
{
  typename UnaryReturn<PColorMatrixREG<T,N>, FnTransposeColor>::Type_t d;;
 
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
//  transform into a PMatrix but downcast the trait to a PColorMatrixREG or PSpinMatrix

//! PColorMatrixREG = outerProduct(PColorVectorREG, PColorVectorREG)
template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorREG<T1,N>, PColorVectorREG<T2,N>, FnOuterProduct> {
  typedef PColorMatrixREG<typename BinaryReturn<T1, T2, FnOuterProduct>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N>
inline typename BinaryReturn<PColorVectorREG<T1,N>, PColorVectorREG<T2,N>, FnOuterProduct>::Type_t
outerProduct(const PColorVectorREG<T1,N>& l, const PColorVectorREG<T2,N>& r)
{
  typename BinaryReturn<PColorVectorREG<T1,N>, PColorVectorREG<T2,N>, FnOuterProduct>::Type_t  d;

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
struct UnaryReturn<PColorMatrixREG<T,N>, FnPeekColorMatrixREG > {
  typedef PScalarREG<typename UnaryReturn<T, FnPeekColorMatrixREG>::Type_t>  Type_t;
};

template<class T, int N>
inline typename UnaryReturn<PColorMatrixREG<T,N>, FnPeekColorMatrixREG>::Type_t
peekColor(const PColorMatrixREG<T,N>& l, llvm::Value* row, llvm::Value* col)
{
  typename UnaryReturn<PColorMatrixREG<T,N>, FnPeekColorMatrixREG>::Type_t  d;

  typedef typename JITType< PColorMatrixREG<T,N> >::Type_t TTjit;

  llvm::Value* ptr_local = llvm_alloca( llvm_get_type<typename WordType<T>::Type_t>() , TTjit::Size_t );

  TTjit dj;
  dj.setup( ptr_local, JitDeviceLayout::Scalar );
  dj=l;

  d.elem() = dj.getRegElem(row,col);
  return d;
}

//! Insert color matrix components
template<class T1, class T2, int N>
inline PColorMatrixREG<T1,N>&
pokeColor(PColorMatrixREG<T1,N>& l, const PScalarREG<T2>& r, int row, int col)
{
  // Note, do not need to propagate down since the function is eaten at this level
  l.getRegElem(row,col) = r.elem();
  return l;
}


//-----------------------------------------------------------------------------
// Contraction for color matrices
// colorContract 
template<class T1, class T2, class T3, int N>
struct TrinaryReturn<PColorMatrixREG<T1,N>, PColorMatrixREG<T2,N>, PColorMatrixREG<T3,N>, FnColorContract> {
  typedef PScalarREG<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};

//! dest  = colorContract(Qprop1,Qprop2,Qprop3)
/*!
 * Performs:
 *  \f$dest = \sum_{i1,i2,i3,j1,j2,j3} \epsilon^{i1,j1,k1}\epsilon^{i2,j2,k2} Q1^{i1,i2} Q2^{j1,j2} Q3^{k1,k2}\f$
 *
 * This routine is completely unrolled for 3 colors
 */
template<class T1, class T2, class T3>
inline typename TrinaryReturn<PColorMatrixREG<T1,3>, PColorMatrixREG<T2,3>, PColorMatrixREG<T3,3>, FnColorContract>::Type_t
colorContract(const PColorMatrixREG<T1,3>& s1, const PColorMatrixREG<T2,3>& s2, const PColorMatrixREG<T3,3>& s3)
{
  typename TrinaryReturn<PColorMatrixREG<T1,3>, PColorMatrixREG<T2,3>, PColorMatrixREG<T3,3>, FnColorContract>::Type_t  d;

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
inline typename TrinaryReturn<PColorMatrixREG<T1,1>, PColorMatrixREG<T2,1>, PColorMatrixREG<T3,1>, FnColorContract>::Type_t
colorContract(const PColorMatrixREG<T1,1>& s1, const PColorMatrixREG<T2,1>& s2, const PColorMatrixREG<T3,1>& s3)
{
  typename TrinaryReturn<PColorMatrixREG<T1,1>, PColorMatrixREG<T2,1>, PColorMatrixREG<T3,1>, FnColorContract>::Type_t  d;

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
inline typename TrinaryReturn<PColorMatrixREG<T1,2>, PColorMatrixREG<T2,2>, PColorMatrixREG<T3,2>, FnColorContract>::Type_t
colorContract(const PColorMatrixREG<T1,2>& s1, const PColorMatrixREG<T2,2>& s2, const PColorMatrixREG<T3,2>& s3)
{
  typename TrinaryReturn<PColorMatrixREG<T1,2>, PColorMatrixREG<T2,2>, PColorMatrixREG<T3,2>, FnColorContract>::Type_t  d;

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
inline typename TrinaryReturn<PColorMatrixREG<T1,4>, PColorMatrixREG<T2,4>, PColorMatrixREG<T3,4>, FnColorContract>::Type_t
colorContract(const PColorMatrixREG<T1,4>& s1, const PColorMatrixREG<T2,4>& s2, const PColorMatrixREG<T3,4>& s3)
{
  typename TrinaryReturn<PColorMatrixREG<T1,4>, PColorMatrixREG<T2,4>, PColorMatrixREG<T3,4>, FnColorContract>::Type_t  d;

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
inline typename BinaryReturn<PColorMatrixREG<T1,3>, PColorMatrixREG<T2,3>, FnQuarkContractXX>::Type_t
quarkContractXX(const PColorMatrixREG<T1,3>& s1, const PColorMatrixREG<T2,3>& s2)
{
  typename BinaryReturn<PColorMatrixREG<T1,3>, PColorMatrixREG<T2,3>, FnQuarkContractXX>::Type_t  d;
#if 1

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
#else
  
  typedef typename BinaryReturn<PColorMatrixREG<T1,3>, PColorMatrixREG<T2,3>, FnQuarkContractXX>::Type_t::Sub_t  T_return;

  JitStackMatrix< T1 , 3 > s1_a( s1 );
  JitStackMatrix< T2 , 3 > s2_a( s2 );
  JitStackMatrix< T_return , 3 > d_a;

  JitForLoop loop_j(0,3);
  JitForLoop loop_i(0,3);

  d_a.elemJIT( loop_i.index() , loop_j.index() ) =
				   s1_a.elemREG( llvm_epsilon_1st( 1 , loop_j.index() ) , llvm_epsilon_2nd( 1 , loop_i.index() ) ) *
				   s2_a.elemREG( llvm_epsilon_1st( 2 , loop_j.index() ) , llvm_epsilon_2nd( 2 , loop_i.index() ) )
				 - s1_a.elemREG( llvm_epsilon_1st( 1 , loop_j.index() ) , llvm_epsilon_2nd( 2 , loop_i.index() ) ) *
				   s2_a.elemREG( llvm_epsilon_1st( 2 , loop_j.index() ) , llvm_epsilon_2nd( 1 , loop_i.index() ) )
				 - s1_a.elemREG( llvm_epsilon_1st( 2 , loop_j.index() ) , llvm_epsilon_2nd( 1 , loop_i.index() ) ) *
				   s2_a.elemREG( llvm_epsilon_1st( 1 , loop_j.index() ) , llvm_epsilon_2nd( 2 , loop_i.index() ) )
				 + s1_a.elemREG( llvm_epsilon_1st( 2 , loop_j.index() ) , llvm_epsilon_2nd( 2 , loop_i.index() ) ) *
      				   s2_a.elemREG( llvm_epsilon_1st( 1 , loop_j.index() ) , llvm_epsilon_2nd( 1 , loop_i.index() ) );
				 
  loop_i.end();
  loop_j.end();

  for(int i=0; i < 3; ++i)
    for(int j=0; j < 3; ++j)
      d.elem(i,j).setup( d_a.elemJIT(i,j) );


#endif
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
inline typename BinaryReturn<PColorMatrixREG<T1,1>, PColorMatrixREG<T2,1>, FnQuarkContractXX>::Type_t
quarkContractXX(const PColorMatrixREG<T1,1>& s1, const PColorMatrixREG<T2,1>& s2)
{
  typename BinaryReturn<PColorMatrixREG<T1,1>, PColorMatrixREG<T2,1>, FnQuarkContractXX>::Type_t  d;

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
inline typename BinaryReturn<PColorMatrixREG<T1,2>, PColorMatrixREG<T2,2>, FnQuarkContractXX>::Type_t
quarkContractXX(const PColorMatrixREG<T1,2>& s1, const PColorMatrixREG<T2,2>& s2)
{
  typename BinaryReturn<PColorMatrixREG<T1,2>, PColorMatrixREG<T2,2>, FnQuarkContractXX>::Type_t  d;

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
inline typename BinaryReturn<PColorMatrixREG<T1,4>, PColorMatrixREG<T2,4>, FnQuarkContractXX>::Type_t
quarkContractXX(const PColorMatrixREG<T1,4>& s1, const PColorMatrixREG<T2,4>& s2)
{
  typename BinaryReturn<PColorMatrixREG<T1,4>, PColorMatrixREG<T2,4>, FnQuarkContractXX>::Type_t  d;

  // not yet written 
  QDPIO::cerr << __func__ << ": not written for Nc=4" << endl;
  QDP_abort(1);

  return d ; 
}




/*! @} */   // end of group primcolormatrix

} // namespace QDP

#endif
