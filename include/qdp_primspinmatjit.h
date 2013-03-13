// -*- C++ -*-

/*! \file
 * \brief Primitive Spin Matrix
 */

#ifndef QDP_PRIMSPINMATJIT_H
#define QDP_PRIMSPINMATJIT_H

namespace QDP {


//-------------------------------------------------------------------------------------
/*! \addtogroup primspinmatrix Spin matrix primitive
 * \ingroup primmatrix
 *
 * Primitive type that transforms like a Spin Matrix
 *
 * @{
 */


//! Primitive Spin Matrix class
/*! 
   * Spin matrix class support gamma matrix algebra 
   *
   * NOTE: the class is mostly empty - it is the specialized versions below
   * that know for a fixed size how gamma matrices (constants) should act
   * on the spin vectors.
   */
template <class T, int N> class PSpinMatrixJIT : public PMatrixJIT<T, N, PSpinMatrixJIT>
{
  //PSpinMatrixJIT(const PSpinMatrixJIT& a);
public:
  PSpinMatrixJIT(){}


  template<class T1>
  PSpinMatrixJIT(const PSpinMatrixREG<T1,N>& a)
  {
    this->assign(a);
  }

  //! PSpinMatrixJIT = PScalarJIT
  /*! Fill with primitive scalar */
  template<class T1>
  inline
  PSpinMatrixJIT& operator=(const PScalarREG<T1>& rhs)
    {
      this->assign(rhs);
      return *this;
    }

  //! PSpinMatrixJIT = PSpinMatrixJIT
  /*! Set equal to another PSpinMatrixJIT */
  template<class T1>
  inline
  PSpinMatrixJIT& operator=(const PSpinMatrixREG<T1,N>& rhs) 
    {
      this->assign(rhs);
      return *this;
    }

};

/*! @} */   // end of group primspinmatrix





//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

template<class T1, int N>
struct REGType<PSpinMatrixJIT<T1,N> > 
{
  typedef PSpinMatrixREG<typename REGType<T1>::Type_t,N>  Type_t;
};


// Underlying word type
template<class T1, int N>
struct WordType<PSpinMatrixJIT<T1,N> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

template<class T1, int N>
struct SinglePrecType<PSpinMatrixJIT<T1, N> >
{
  typedef PSpinMatrixJIT< typename SinglePrecType<T1>::Type_t , N > Type_t;
};

template<class T1, int N>
struct DoublePrecType<PSpinMatrixJIT<T1, N> >
{
  typedef PSpinMatrixJIT< typename DoublePrecType<T1>::Type_t , N > Type_t;
};

// Internally used scalars
template<class T, int N>
struct InternalScalar<PSpinMatrixJIT<T,N> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive into a scalar leaving grid alone
template<class T, int N>
struct PrimitiveScalar<PSpinMatrixJIT<T,N> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T, int N>
struct LatticeScalar<PSpinMatrixJIT<T,N> > {
  typedef PSpinMatrixJIT<typename LatticeScalar<T>::Type_t, N>  Type_t;
};


//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PSpinMatrixJIT) -> PSpinMatrixJIT
template<class T1, int N, class Op>
struct UnaryReturn<PSpinMatrixJIT<T1,N>, Op> {
  typedef PSpinMatrixJIT<typename UnaryReturn<T1, Op>::Type_t, N>  Type_t;
};

// Default binary(PScalarJIT,PSpinMatrixJIT) -> PSpinMatrixJIT
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2,N>, Op> {
  typedef PSpinMatrixJIT<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PSpinMatrixJIT,PSpinMatrixJIT) -> PSpinMatrixJIT
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinMatrixJIT<T2,N>, Op> {
  typedef PSpinMatrixJIT<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PSpinMatrixJIT,PScalarJIT) -> PSpinMatrixJIT
template<class T1, int N, class T2, class Op>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PScalarJIT<T2>, Op> {
  typedef PSpinMatrixJIT<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};


#if 0
template<class T1, class T2>
struct UnaryReturn<PSpinMatrixJIT<T2,N>, OpCast<T1> > {
  typedef PScalarJIT<typename UnaryReturn<T, OpCast>::Type_t, N>  Type_t;
//  typedef T1 Type_t;
};
#endif


// Assignment is different
template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinMatrixJIT<T2,N>, OpAssign > {
  typedef PSpinMatrixJIT<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinMatrixJIT<T2,N>, OpAddAssign > {
  typedef PSpinMatrixJIT<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinMatrixJIT<T2,N>, OpSubtractAssign > {
  typedef PSpinMatrixJIT<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinMatrixJIT<T2,N>, OpMultiplyAssign > {
  typedef PSpinMatrixJIT<T1,N> &Type_t;
};
 

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PScalarJIT<T2>, OpAssign > {
  typedef PSpinMatrixJIT<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PScalarJIT<T2>, OpAddAssign > {
  typedef PSpinMatrixJIT<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PScalarJIT<T2>, OpSubtractAssign > {
  typedef PSpinMatrixJIT<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PScalarJIT<T2>, OpMultiplyAssign > {
  typedef PSpinMatrixJIT<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PScalarJIT<T2>, OpDivideAssign > {
  typedef PSpinMatrixJIT<T1,N> &Type_t;
};
 


// SpinMatrix
template<class T, int N>
struct UnaryReturn<PSpinMatrixJIT<T,N>, FnTrace > {
  typedef PScalarJIT<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinMatrixJIT<T,N>, FnRealTrace > {
  typedef PScalarJIT<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinMatrixJIT<T,N>, FnImagTrace > {
  typedef PScalarJIT<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinMatrixJIT<T,N>, FnNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinMatrixJIT<T,N>, FnLocalNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinMatrixJIT<T2,N>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PScalarJIT<T2>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2,N>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinMatrixJIT<T2,N>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PScalarJIT<T2>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2,N>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinMatrixJIT<T2,N>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PScalarJIT<T2>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2,N>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinMatrixJIT<T2,N>, FnInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PScalarJIT<T2>, FnInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2,N>, FnInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinMatrixJIT<T2,N>, FnLocalInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PScalarJIT<T2>, FnLocalInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2,N>, FnLocalInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};



// Gamma algebra
template<int m, class T2, int N, class OpGammaConstMultiply>
struct BinaryReturn<GammaConst<N,m>, PSpinMatrixJIT<T2,N>, OpGammaConstMultiply> {
  typedef PSpinMatrixJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConst>
struct BinaryReturn<PSpinMatrixJIT<T2,N>, GammaConst<N,m>, OpMultiplyGammaConst> {
  typedef PSpinMatrixJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};

template<class T2, int N, class OpGammaTypeMultiply>
struct BinaryReturn<GammaType<N>, PSpinMatrixJIT<T2,N>, OpGammaTypeMultiply> {
  typedef PSpinMatrixJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaType>
struct BinaryReturn<PSpinMatrixJIT<T2,N>, GammaType<N>, OpMultiplyGammaType> {
  typedef PSpinMatrixJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};


// Gamma algebra
template<int m, class T2, int N, class OpGammaConstDPMultiply>
struct BinaryReturn<GammaConstDP<N,m>, PSpinMatrixJIT<T2,N>, OpGammaConstDPMultiply> {
  typedef PSpinMatrixJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConstDP>
struct BinaryReturn<PSpinMatrixJIT<T2,N>, GammaConstDP<N,m>, OpMultiplyGammaConstDP> {
  typedef PSpinMatrixJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};

template<class T2, int N, class OpGammaTypeDPMultiply>
struct BinaryReturn<GammaTypeDP<N>, PSpinMatrixJIT<T2,N>, OpGammaTypeDPMultiply> {
  typedef PSpinMatrixJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaTypeDP>
struct BinaryReturn<PSpinMatrixJIT<T2,N>, GammaTypeDP<N>, OpMultiplyGammaTypeDP> {
  typedef PSpinMatrixJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};




  template<class T0,class T1,class T2, int N >
  inline typename TrinaryReturn<PScalarJIT<T0>, PSpinMatrixJIT<T1,N>, PSpinMatrixJIT<T2,N>, FnWhere >::Type_t
  do_where(const PScalarJIT<T0> &a, const PSpinMatrixJIT<T1,N> &b, const PSpinMatrixJIT<T2,N> &c)
{
  int pred;
  get_pred( pred , a );

  typename TrinaryReturn<PScalarJIT<T0>, PSpinMatrixJIT<T1,N>, PSpinMatrixJIT<T2,N>, FnWhere >::Type_t ret(a.func());

  a.func().addCondBranchPred_if( pred );
  ret = b;
  a.func().addCondBranchPred_else();
  ret = c;
  a.func().addCondBranchPred_fi();

  return ret;
}




//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------
/*! \addtogroup primspinmatrix */
/*! @{ */

// SpinMatrix class primitive operations

// trace = traceSpin(source1)
/*! This only acts on spin indices and is diagonal in all other indices */
template<class T, int N>
struct UnaryReturn<PSpinMatrixJIT<T,N>, FnTraceSpin > {
  typedef PScalarJIT<typename UnaryReturn<T, FnTraceSpin>::Type_t>  Type_t;
};

template<class T, int N>
inline typename UnaryReturn<PSpinMatrixJIT<T,N>, FnTraceSpin>::Type_t
traceSpin(const PSpinMatrixJIT<T,N>& s1)
{
  typename UnaryReturn<PSpinMatrixJIT<T,N>, FnTraceSpin>::Type_t  d(s1.func());
  
  // Since the spin index is eaten, do not need to pass on function by
  // calling trace(...) again
  d.elem() = s1.elem(0,0);
  for(int i=1; i < N; ++i)
    d.elem() += s1.elem(i,i);

  return d;
}

//! traceSpinMultiply(source1,source2)
template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinMatrixJIT<T2,N>, FnTraceSpinMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinMatrixJIT<T2,N>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(const PSpinMatrixJIT<T1,N>& l, const PSpinMatrixJIT<T2,N>& r)
{
  typename BinaryReturn<PSpinMatrixJIT<T1,N>, PSpinMatrixJIT<T2,N>, FnTraceSpinMultiply>::Type_t  d(l.func());

  // The traceSpin is eaten here
  d.elem() = l.elem(0,0) * r.elem(0,0);
  for(int k=1; k < N; ++k)
    d.elem() += l.elem(0,k) * r.elem(k,0);

  for(int j=1; j < N; ++j)
    for(int k=0; k < N; ++k)
      d.elem() += l.elem(j,k) * r.elem(k,j);

  return d;
}

//! PScalarJIT = traceSpinMultiply(PSpinMatrixJIT,PScalarJIT)
template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrixJIT<T1,N>, PScalarJIT<T2>, FnTraceSpinMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PSpinMatrixJIT<T1,N>, PScalarJIT<T2>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(const PSpinMatrixJIT<T1,N>& l, const PScalarJIT<T2>& r)
{
  typename BinaryReturn<PSpinMatrixJIT<T1,N>, PScalarJIT<T2>, FnTraceSpinMultiply>::Type_t  d(l.func());

  // The traceSpin is eaten here
  d.elem() = l.elem(0,0) * r.elem();
  for(int k=1; k < N; ++k)
    d.elem() += l.elem(k,k) * r.elem();

  return d;
}

// PScalarJIT = traceSpinMultiply(PScalarJIT,PSpinMatrixJIT)
template<class T1, class T2, int N>
struct BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2,N>, FnTraceSpinMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
inline typename BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2,N>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(const PScalarJIT<T1>& l, const PSpinMatrixJIT<T2,N>& r)
{
  typename BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2,N>, FnTraceSpinMultiply>::Type_t  d(l.func());

  // The traceSpin is eaten here
  d.elem() = l.elem() * r.elem(0,0);
  for(int k=1; k < N; ++k)
    d.elem() += l.elem() * r.elem(k,k);

  return d;
}



/*! Specialise the return type */
template <class T, int N>
struct UnaryReturn<PSpinMatrixJIT<T,N>, FnTransposeSpin > {
  typedef PSpinMatrixJIT<typename UnaryReturn<T, FnTransposeSpin>::Type_t, N> Type_t;
};

//! PSpinMatrixJIT = transposeSpin(PSpinMatrixJIT) 
/*! t = transposeSpin(source1) - SpinMatrix specialization -- where the work is actually done */
template<class T, int N>
inline typename UnaryReturn<PSpinMatrixJIT<T,N>, FnTransposeSpin >::Type_t
transposeSpin(const PSpinMatrixJIT<T,N>& s1)
{
  typename UnaryReturn<PSpinMatrixJIT<T,N>, FnTransposeSpin>::Type_t d(s1.func());
 
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
// The problem is the traits class - I have no way to say to PVectorJIT's
//  transform into a PMatrixJIT but downcast the trait to a PColorMatrix or
//  PSpinMatrixJIT

//! PSpinMatrixJIT = outerProduct(PSpinVectorJIT, PSpinVectorJIT)
template<class T1, class T2, int N>
struct BinaryReturn<PSpinVectorJIT<T1,N>, PSpinVectorJIT<T2,N>, FnOuterProduct> {
  typedef PSpinMatrixJIT<typename BinaryReturn<T1, T2, FnOuterProduct>::Type_t, N>  Type_t;
};

template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinVectorJIT<T1,N>, PSpinVectorJIT<T2,N>, FnOuterProduct>::Type_t
outerProduct(const PSpinVectorJIT<T1,N>& l, const PSpinVectorJIT<T2,N>& r)
{
  typename BinaryReturn<PSpinVectorJIT<T1,N>, PSpinVectorJIT<T2,N>, FnOuterProduct>::Type_t  d(l.func());

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = outerProduct(l.elem(i),r.elem(j));

  return d;
}


//-----------------------------------------------
// Optimization of traceSpin(outerProduct(PSpinVectorJIT, PSpinVectorJIT))

//! PScalarJIT = traceSpinOuterProduct(PSpinVectorJIT, PSpinVectorJIT)
template<class T1, class T2, int N>
struct BinaryReturn<PSpinVectorJIT<T1,N>, PSpinVectorJIT<T2,N>, FnTraceSpinOuterProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnOuterProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
inline typename BinaryReturn<PSpinVectorJIT<T1,N>, PSpinVectorJIT<T2,N>, FnTraceSpinOuterProduct>::Type_t
traceSpinOuterProduct(const PSpinVectorJIT<T1,N>& l, const PSpinVectorJIT<T2,N>& r)
{
  typename BinaryReturn<PSpinVectorJIT<T1,N>, PSpinVectorJIT<T2,N>, FnTraceSpinOuterProduct>::Type_t  d(l.func());

  d.elem() = outerProduct(l.elem(0),r.elem(0));
  for(int i=1; i < N; ++i)
    d.elem() += outerProduct(l.elem(i),r.elem(i));

  return d;
}


//-----------------------------------------------
// Peeking and poking
//! Extract spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T, int N>
struct UnaryReturn<PSpinMatrixJIT<T,N>, FnPeekSpinMatrixREG > {
  typedef PScalarJIT<typename UnaryReturn<T, FnPeekSpinMatrixREG >::Type_t>  Type_t;
};

// template<class T, int N>
// inline typename UnaryReturn<PSpinMatrixJIT<T,N>, FnPeekSpinMatrixJIT >::Type_t
// peekSpin(const PSpinMatrixJIT<T,N>& l, int row, int col)
// {
//   typename UnaryReturn<PSpinMatrixJIT<T,N>, FnPeekSpinMatrixJIT >::Type_t  d(l.func());

//   // Note, do not need to propagate down since the function is eaten at this level
//   d.elem() = l.getRegElem(row,col);
//   return d;
// }

// //! Insert spin matrix components
// template<class T1, class T2, int N>
// inline PSpinMatrixJIT<T1,N>&
// pokeSpin(PSpinMatrixJIT<T1,N>& l, const PScalarJIT<T2>& r, int row, int col)
// {
//   // Note, do not need to propagate down since the function is eaten at this level
//   l.getRegElem(row,col) = r.elem();
//   return l;
// }


template<class T1, class T2, int N>
inline PSpinMatrixJIT<T1,N>&
pokeSpin(PSpinMatrixJIT<T1,N>& l, const PScalarREG<T2>& r, jit_value_t row, jit_value_t col)
{
  l.getJitElem(row,col) = r.elem();
  return l;
}



//-----------------------------------------------

// SpinMatrix<4> = Gamma<4,m> * SpinMatrix<4>
// There are 16 cases here for Nd=4
template<class T2>
inline typename BinaryReturn<GammaConst<4,0>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,0>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,0>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());
  
  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = r.elem(0,i);
    d.elem(1,i) = r.elem(1,i);
    d.elem(2,i) = r.elem(2,i);
    d.elem(3,i) = r.elem(3,i);
  }

  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,1>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,1>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,1>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());
  
  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesI(r.elem(3,i));
    d.elem(1,i) = timesI(r.elem(2,i));
    d.elem(2,i) = timesMinusI(r.elem(1,i));
    d.elem(3,i) = timesMinusI(r.elem(0,i));
  }

  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,2>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,2>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,2>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = -r.elem(3,i);
    d.elem(1,i) = r.elem(2,i);
    d.elem(2,i) = r.elem(1,i);
    d.elem(3,i) = -r.elem(0,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,3>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,3>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,3>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesMinusI(r.elem(0,i));
    d.elem(1,i) = timesI(r.elem(1,i));
    d.elem(2,i) = timesMinusI(r.elem(2,i));
    d.elem(3,i) = timesI(r.elem(3,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,4>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,4>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,4>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesI(r.elem(2,i));
    d.elem(1,i) = timesMinusI(r.elem(3,i));
    d.elem(2,i) = timesMinusI(r.elem(0,i));
    d.elem(3,i) = timesI(r.elem(1,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,5>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,5>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,5>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = -r.elem(1,i);
    d.elem(1,i) = r.elem(0,i);
    d.elem(2,i) = -r.elem(3,i);
    d.elem(3,i) = r.elem(2,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,6>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,6>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,6>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesMinusI(r.elem(1,i));
    d.elem(1,i) = timesMinusI(r.elem(0,i));
    d.elem(2,i) = timesMinusI(r.elem(3,i));
    d.elem(3,i) = timesMinusI(r.elem(2,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,7>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,7>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,7>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = r.elem(2,i);
    d.elem(1,i) = r.elem(3,i);
    d.elem(2,i) = -r.elem(0,i);
    d.elem(3,i) = -r.elem(1,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,8>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,8>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,8>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = r.elem(2,i);
    d.elem(1,i) = r.elem(3,i);
    d.elem(2,i) = r.elem(0,i);
    d.elem(3,i) = r.elem(1,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,9>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,9>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,9>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesI(r.elem(1,i));
    d.elem(1,i) = timesI(r.elem(0,i));
    d.elem(2,i) = timesMinusI(r.elem(3,i));
    d.elem(3,i) = timesMinusI(r.elem(2,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,10>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,10>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,10>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = -r.elem(1,i);
    d.elem(1,i) = r.elem(0,i);
    d.elem(2,i) = r.elem(3,i);
    d.elem(3,i) = -r.elem(2,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,11>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,11>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,11>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesMinusI(r.elem(2,i));
    d.elem(1,i) = timesI(r.elem(3,i));
    d.elem(2,i) = timesMinusI(r.elem(0,i));
    d.elem(3,i) = timesI(r.elem(1,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,12>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,12>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,12>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesI(r.elem(0,i));
    d.elem(1,i) = timesMinusI(r.elem(1,i));
    d.elem(2,i) = timesMinusI(r.elem(2,i));
    d.elem(3,i) = timesI(r.elem(3,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,13>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,13>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,13>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = -r.elem(3,i);
    d.elem(1,i) = r.elem(2,i);
    d.elem(2,i) = -r.elem(1,i);
    d.elem(3,i) = r.elem(0,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,14>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,14>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,14>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesMinusI(r.elem(3,i));
    d.elem(1,i) = timesMinusI(r.elem(2,i));
    d.elem(2,i) = timesMinusI(r.elem(1,i));
    d.elem(3,i) = timesMinusI(r.elem(0,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,15>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,15>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,15>, PSpinMatrixJIT<T2,4>, OpGammaConstMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = r.elem(0,i);
    d.elem(1,i) = r.elem(1,i);
    d.elem(2,i) = -r.elem(2,i);
    d.elem(3,i) = -r.elem(3,i);
  }
  
  return d;
}


// SpinMatrix<4> = SpinMatrix<4> * Gamma<4,m>
// There are 16 cases here for Nd=4
template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,0>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,0>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,0>, OpGammaConstMultiply>::Type_t  d(l.func()); 

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,0);
    d.elem(i,1) =  l.elem(i,1);
    d.elem(i,2) =  l.elem(i,2);
    d.elem(i,3) =  l.elem(i,3);
  }
 
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,1>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,1>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,1>, OpGammaConstMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesMinusI(l.elem(i,3));
    d.elem(i,1) = timesMinusI(l.elem(i,2));
    d.elem(i,2) = timesI(l.elem(i,1));
    d.elem(i,3) = timesI(l.elem(i,0));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,2>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,2>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,2>, OpGammaConstMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = -l.elem(i,3);
    d.elem(i,1) =  l.elem(i,2);
    d.elem(i,2) =  l.elem(i,1);
    d.elem(i,3) = -l.elem(i,0);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,3>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,3>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,3>, OpGammaConstMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesMinusI(l.elem(i,0));
    d.elem(i,1) = timesI(l.elem(i,1));
    d.elem(i,2) = timesMinusI(l.elem(i,2));
    d.elem(i,3) = timesI(l.elem(i,3));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,4>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,4>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,4>, OpGammaConstMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesMinusI(l.elem(i,2));
    d.elem(i,1) = timesI(l.elem(i,3));
    d.elem(i,2) = timesI(l.elem(i,0));
    d.elem(i,3) = timesMinusI(l.elem(i,1));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,5>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,5>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,5>, OpGammaConstMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,1);
    d.elem(i,1) = -l.elem(i,0);
    d.elem(i,2) =  l.elem(i,3);
    d.elem(i,3) = -l.elem(i,2);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,6>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,6>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,6>, OpGammaConstMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesMinusI(l.elem(i,1));
    d.elem(i,1) = timesMinusI(l.elem(i,0));
    d.elem(i,2) = timesMinusI(l.elem(i,3));
    d.elem(i,3) = timesMinusI(l.elem(i,2));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,7>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,7>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,7>, OpGammaConstMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = -l.elem(i,2);
    d.elem(i,1) = -l.elem(i,3);
    d.elem(i,2) =  l.elem(i,0);
    d.elem(i,3) =  l.elem(i,1);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,8>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,8>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,8>, OpGammaConstMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,2);
    d.elem(i,1) =  l.elem(i,3);
    d.elem(i,2) =  l.elem(i,0);
    d.elem(i,3) =  l.elem(i,1);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,9>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,9>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,9>, OpGammaConstMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesI(l.elem(i,1));
    d.elem(i,1) = timesI(l.elem(i,0));
    d.elem(i,2) = timesMinusI(l.elem(i,3));
    d.elem(i,3) = timesMinusI(l.elem(i,2));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,10>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,10>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,10>, OpGammaConstMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,1);
    d.elem(i,1) = -l.elem(i,0);
    d.elem(i,2) = -l.elem(i,3);
    d.elem(i,3) =  l.elem(i,2);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,11>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,11>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,11>, OpGammaConstMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesMinusI(l.elem(i,2));
    d.elem(i,1) = timesI(l.elem(i,3));
    d.elem(i,2) = timesMinusI(l.elem(i,0));
    d.elem(i,3) = timesI(l.elem(i,1));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,12>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,12>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,12>, OpGammaConstMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesI(l.elem(i,0));
    d.elem(i,1) = timesMinusI(l.elem(i,1));
    d.elem(i,2) = timesMinusI(l.elem(i,2));
    d.elem(i,3) = timesI(l.elem(i,3));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,13>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,13>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,13>, OpGammaConstMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,3);
    d.elem(i,1) = -l.elem(i,2);
    d.elem(i,2) =  l.elem(i,1);
    d.elem(i,3) = -l.elem(i,0);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,14>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,14>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,14>, OpGammaConstMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesMinusI(l.elem(i,3));
    d.elem(i,1) = timesMinusI(l.elem(i,2));
    d.elem(i,2) = timesMinusI(l.elem(i,1));
    d.elem(i,3) = timesMinusI(l.elem(i,0));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,15>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConst<4,15>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConst<4,15>, OpGammaConstMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,0);
    d.elem(i,1) =  l.elem(i,1);
    d.elem(i,2) = -l.elem(i,2);
    d.elem(i,3) = -l.elem(i,3);
  }
  
  return d;
}


//-----------------------------------------------

// SpinMatrix<4> = GammaDP<4,m> * SpinMatrix<4>
// There are 16 cases here for Nd=4
template<class T2>
inline typename BinaryReturn<GammaConstDP<4,0>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,0>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,0>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());
  
  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = r.elem(0,i);
    d.elem(1,i) = r.elem(1,i);
    d.elem(2,i) = r.elem(2,i);
    d.elem(3,i) = r.elem(3,i);
  }

  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,1>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,1>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,1>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());
  
  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesMinusI(r.elem(3,i));
    d.elem(1,i) = timesMinusI(r.elem(2,i));
    d.elem(2,i) = timesI(r.elem(1,i));
    d.elem(3,i) = timesI(r.elem(0,i));
  }

  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,2>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,2>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,2>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = -r.elem(3,i);
    d.elem(1,i) =  r.elem(2,i);
    d.elem(2,i) =  r.elem(1,i);
    d.elem(3,i) = -r.elem(0,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,3>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,3>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,3>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesI(r.elem(0,i));
    d.elem(1,i) = timesMinusI(r.elem(1,i));
    d.elem(2,i) = timesI(r.elem(2,i));
    d.elem(3,i) = timesMinusI(r.elem(3,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,4>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,4>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,4>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesMinusI(r.elem(2,i));
    d.elem(1,i) = timesI(r.elem(3,i));
    d.elem(2,i) = timesI(r.elem(0,i));
    d.elem(3,i) = timesMinusI(r.elem(1,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,5>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,5>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,5>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = -r.elem(1,i);
    d.elem(1,i) =  r.elem(0,i);
    d.elem(2,i) = -r.elem(3,i);
    d.elem(3,i) =  r.elem(2,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,6>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,6>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,6>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesI(r.elem(1,i));
    d.elem(1,i) = timesI(r.elem(0,i));
    d.elem(2,i) = timesI(r.elem(3,i));
    d.elem(3,i) = timesI(r.elem(2,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,7>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,7>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,7>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) =  r.elem(2,i);
    d.elem(1,i) =  r.elem(3,i);
    d.elem(2,i) = -r.elem(0,i);
    d.elem(3,i) = -r.elem(1,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,8>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,8>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,8>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) =  r.elem(0,i);
    d.elem(1,i) =  r.elem(1,i);
    d.elem(2,i) = -r.elem(2,i);
    d.elem(3,i) = -r.elem(3,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,9>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,9>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,9>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesI(r.elem(3,i));
    d.elem(1,i) = timesI(r.elem(2,i));
    d.elem(2,i) = timesI(r.elem(1,i));
    d.elem(3,i) = timesI(r.elem(0,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,10>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,10>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,10>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) =  r.elem(3,i);
    d.elem(1,i) = -r.elem(2,i);
    d.elem(2,i) = -r.elem(1,i);
    d.elem(3,i) =  r.elem(0,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,11>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,11>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,11>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesI(r.elem(0,i));
    d.elem(1,i) = timesMinusI(r.elem(1,i));
    d.elem(2,i) = timesMinusI(r.elem(2,i));
    d.elem(3,i) = timesI(r.elem(3,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,12>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,12>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,12>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesI(r.elem(2,i));
    d.elem(1,i) = timesMinusI(r.elem(3,i));
    d.elem(2,i) = timesI(r.elem(0,i));
    d.elem(3,i) = timesMinusI(r.elem(1,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,13>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,13>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,13>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = -r.elem(1,i);
    d.elem(1,i) =  r.elem(0,i);
    d.elem(2,i) =  r.elem(3,i);
    d.elem(3,i) = -r.elem(2,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,14>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,14>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,14>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = timesI(r.elem(1,i));
    d.elem(1,i) = timesI(r.elem(0,i));
    d.elem(2,i) = timesMinusI(r.elem(3,i));
    d.elem(3,i) = timesMinusI(r.elem(2,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConstDP<4,15>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t
operator*(const GammaConstDP<4,15>&, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<GammaConstDP<4,15>, PSpinMatrixJIT<T2,4>, OpGammaConstDPMultiply>::Type_t  d(r.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = -r.elem(2,i);
    d.elem(1,i) = -r.elem(3,i);
    d.elem(2,i) = -r.elem(0,i);
    d.elem(3,i) = -r.elem(1,i);
  }
  
  return d;
}


// SpinMatrix<4> = SpinMatrix<4> * GammaDP<4,m>
// There are 16 cases here for Nd=4
template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,0>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,0>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,0>, OpGammaConstDPMultiply>::Type_t  d(l.func()); 

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,0);
    d.elem(i,1) =  l.elem(i,1);
    d.elem(i,2) =  l.elem(i,2);
    d.elem(i,3) =  l.elem(i,3);
  }
 
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,1>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,1>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,1>, OpGammaConstDPMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesMinusI(l.elem(i,3));
    d.elem(i,1) = timesMinusI(l.elem(i,2));
    d.elem(i,2) = timesI(l.elem(i,1));
    d.elem(i,3) = timesI(l.elem(i,0));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,2>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,2>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,2>, OpGammaConstDPMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = -l.elem(i,3);
    d.elem(i,1) =  l.elem(i,2);
    d.elem(i,2) =  l.elem(i,1);
    d.elem(i,3) = -l.elem(i,0);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,3>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,3>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,3>, OpGammaConstDPMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesI(l.elem(i,0));
    d.elem(i,1) = timesMinusI(l.elem(i,1));
    d.elem(i,2) = timesI(l.elem(i,2));
    d.elem(i,3) = timesMinusI(l.elem(i,3));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,4>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,4>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,4>, OpGammaConstDPMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesMinusI(l.elem(i,2));
    d.elem(i,1) = timesI(l.elem(i,3));
    d.elem(i,2) = timesI(l.elem(i,0));
    d.elem(i,3) = timesMinusI(l.elem(i,1));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,5>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,5>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,5>, OpGammaConstDPMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = -l.elem(i,1);
    d.elem(i,1) =  l.elem(i,0);
    d.elem(i,2) = -l.elem(i,3);
    d.elem(i,3) =  l.elem(i,2);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,6>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,6>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,6>, OpGammaConstDPMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesI(l.elem(i,1));
    d.elem(i,1) = timesI(l.elem(i,0));
    d.elem(i,2) = timesI(l.elem(i,3));
    d.elem(i,3) = timesI(l.elem(i,2));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,7>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,7>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,7>, OpGammaConstDPMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,2);
    d.elem(i,1) =  l.elem(i,3);
    d.elem(i,2) = -l.elem(i,0);
    d.elem(i,3) = -l.elem(i,1);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,8>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,8>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,8>, OpGammaConstDPMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,0);
    d.elem(i,1) =  l.elem(i,1);
    d.elem(i,2) = -l.elem(i,2);
    d.elem(i,3) = -l.elem(i,3);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,9>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,9>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,9>, OpGammaConstDPMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesI(l.elem(i,3));
    d.elem(i,1) = timesI(l.elem(i,2));
    d.elem(i,2) = timesI(l.elem(i,1));
    d.elem(i,3) = timesI(l.elem(i,0));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,10>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,10>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,10>, OpGammaConstDPMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,3);
    d.elem(i,1) = -l.elem(i,2);
    d.elem(i,2) = -l.elem(i,1);
    d.elem(i,3) =  l.elem(i,0);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,11>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,11>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,11>, OpGammaConstDPMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesI(l.elem(i,0));
    d.elem(i,1) = timesMinusI(l.elem(i,1));
    d.elem(i,2) = timesMinusI(l.elem(i,2));
    d.elem(i,3) = timesI(l.elem(i,3));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,12>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,12>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,12>, OpGammaConstDPMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesI(l.elem(i,2));
    d.elem(i,1) = timesMinusI(l.elem(i,3));
    d.elem(i,2) = timesI(l.elem(i,0));
    d.elem(i,3) = timesMinusI(l.elem(i,1));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,13>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,13>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,13>, OpGammaConstDPMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = -l.elem(i,1);
    d.elem(i,1) =  l.elem(i,0);
    d.elem(i,2) =  l.elem(i,3);
    d.elem(i,3) = -l.elem(i,2);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,14>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,14>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,14>, OpGammaConstDPMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = timesI(l.elem(i,1));
    d.elem(i,1) = timesI(l.elem(i,0));
    d.elem(i,2) = timesMinusI(l.elem(i,3));
    d.elem(i,3) = timesMinusI(l.elem(i,2));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,15>, OpGammaConstDPMultiply>::Type_t
operator*(const PSpinMatrixJIT<T2,4>& l, const GammaConstDP<4,15>&)
{
  typename BinaryReturn<PSpinMatrixJIT<T2,4>, GammaConstDP<4,15>, OpGammaConstDPMultiply>::Type_t  d(l.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = -l.elem(i,2);
    d.elem(i,1) = -l.elem(i,3);
    d.elem(i,2) = -l.elem(i,0);
    d.elem(i,3) = -l.elem(i,1);
  }
  
  return d;
}


//-----------------------------------------------------------------------------
//! PSpinVectorJIT<T,4> = P_+ * PSpinVectorJIT<T,4>
template<class T>
inline typename UnaryReturn<PSpinMatrixJIT<T,4>, FnChiralProjectPlus>::Type_t
chiralProjectPlus(const PSpinMatrixJIT<T,4>& s1)
{
  typename UnaryReturn<PSpinMatrixJIT<T,4>, FnChiralProjectPlus>::Type_t  d(s1.func());

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = s1.elem(0,i);
    d.elem(1,i) = s1.elem(1,i);
    zero_rep(d.elem(2,i));
    zero_rep(d.elem(3,i));
  }

  return d;
}

//! PSpinVectorJIT<T,4> = P_- * PSpinVectorJIT<T,4>
template<class T>
inline typename UnaryReturn<PSpinMatrixJIT<T,4>, FnChiralProjectMinus>::Type_t
chiralProjectMinus(const PSpinMatrixJIT<T,4>& s1)
{
  typename UnaryReturn<PSpinMatrixJIT<T,4>, FnChiralProjectMinus>::Type_t  d(s1.func());

  for(int i=0; i < 4; ++i)
  {
    zero_rep(d.elem(0,i));
    zero_rep(d.elem(1,i));
    d.elem(2,i) = s1.elem(2,i);
    d.elem(3,i) = s1.elem(3,i);
  }

  return d;
}

//------------------------------------------
// PScalarJIT = traceSpinQuarkContract13(PSpinMatrixJIT,PSpinMatrixJIT)
template<class T1, class T2>
struct BinaryReturn<PSpinMatrixJIT<T1,4>, PSpinMatrixJIT<T2,4>, FnTraceSpinQuarkContract13> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceSpinQuarkContract13>::Type_t>  Type_t;
};

//! PScalarJIT = traceSpinQuarkContract13(PSpinMatrixJIT,PSpinMatrixJIT)
template<class T1, class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T1,4>, PSpinMatrixJIT<T2,4>, FnTraceSpinQuarkContract13>::Type_t
traceSpinQuarkContract13(const PSpinMatrixJIT<T1,4>& l, const PSpinMatrixJIT<T2,4>& r)
{
  typename BinaryReturn<PSpinMatrixJIT<T1,4>, PSpinMatrixJIT<T2,4>, FnTraceSpinQuarkContract13>::Type_t  d(l.func());

  d.elem() = quarkContractXX(l.elem(0,0), r.elem(0,0));
  for(int k=1; k < 4; ++k)
    d.elem() += quarkContractXX(l.elem(k,0), r.elem(k,0));

  for(int j=1; j < 4; ++j)
    for(int k=0; k < 4; ++k)
      d.elem() += quarkContractXX(l.elem(k,j), r.elem(k,j));

  return d;
}


// quark propagator contraction
template<class T1, class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T1,4>, PSpinMatrixJIT<T2,4>, FnQuarkContract13>::Type_t
quarkContract13(const PSpinMatrixJIT<T1,4>& s1, const PSpinMatrixJIT<T2,4>& s2)
{
  typename BinaryReturn<PSpinMatrixJIT<T1,4>, PSpinMatrixJIT<T2,4>, FnQuarkContract13>::Type_t  d(s1.func());

  for(int j=0; j < 4; ++j)
    for(int i=0; i < 4; ++i)
    {
      d.elem(i,j) = quarkContractXX(s1.elem(0,i), s2.elem(0,j));
      for(int k=1; k < 4; ++k)
	d.elem(i,j) += quarkContractXX(s1.elem(k,i), s2.elem(k,j));
    }

  return d;
}

template<class T1, class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T1,4>, PSpinMatrixJIT<T2,4>, FnQuarkContract14>::Type_t
quarkContract14(const PSpinMatrixJIT<T1,4>& s1, const PSpinMatrixJIT<T2,4>& s2)
{
  typename BinaryReturn<PSpinMatrixJIT<T1,4>, PSpinMatrixJIT<T2,4>, FnQuarkContract14>::Type_t  d(s1.func());

  for(int j=0; j < 4; ++j)
    for(int i=0; i < 4; ++i)
    {
      d.elem(i,j) = quarkContractXX(s1.elem(0,i), s2.elem(j,0));
      for(int k=1; k < 4; ++k)
	d.elem(i,j) += quarkContractXX(s1.elem(k,i), s2.elem(j,k));
    }

  return d;
}

template<class T1, class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T1,4>, PSpinMatrixJIT<T2,4>, FnQuarkContract23>::Type_t
quarkContract23(const PSpinMatrixJIT<T1,4>& s1, const PSpinMatrixJIT<T2,4>& s2)
{
  typename BinaryReturn<PSpinMatrixJIT<T1,4>, PSpinMatrixJIT<T2,4>, FnQuarkContract23>::Type_t  d(s1.func());

  for(int j=0; j < 4; ++j)
    for(int i=0; i < 4; ++i)
    {
      d.elem(i,j) = quarkContractXX(s1.elem(i,0), s2.elem(0,j));
      for(int k=1; k < 4; ++k)
	d.elem(i,j) += quarkContractXX(s1.elem(i,k), s2.elem(k,j));
    }

  return d;
}

template<class T1, class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T1,4>, PSpinMatrixJIT<T2,4>, FnQuarkContract24>::Type_t
quarkContract24(const PSpinMatrixJIT<T1,4>& s1, const PSpinMatrixJIT<T2,4>& s2)
{
  typename BinaryReturn<PSpinMatrixJIT<T1,4>, PSpinMatrixJIT<T2,4>, FnQuarkContract24>::Type_t  d(s1.func());

  for(int j=0; j < 4; ++j)
    for(int i=0; i < 4; ++i)
    {
      d.elem(i,j) = quarkContractXX(s1.elem(i,0), s2.elem(j,0));
      for(int k=1; k < 4; ++k)
	d.elem(i,j) += quarkContractXX(s1.elem(i,k), s2.elem(j,k));
    }

  return d;
}

template<class T1, class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T1,4>, PSpinMatrixJIT<T2,4>, FnQuarkContract12>::Type_t
quarkContract12(const PSpinMatrixJIT<T1,4>& s1, const PSpinMatrixJIT<T2,4>& s2)
{
  typename BinaryReturn<PSpinMatrixJIT<T1,4>, PSpinMatrixJIT<T2,4>, FnQuarkContract12>::Type_t  d(s1.func());

  for(int j=0; j < 4; ++j)
    for(int i=0; i < 4; ++i)
    {
      d.elem(i,j) = quarkContractXX(s1.elem(0,0), s2.elem(i,j));
      for(int k=1; k < 4; ++k)
	d.elem(i,j) += quarkContractXX(s1.elem(k,k), s2.elem(i,j));
    }

  return d;
}

template<class T1, class T2>
inline typename BinaryReturn<PSpinMatrixJIT<T1,4>, PSpinMatrixJIT<T2,4>, FnQuarkContract34>::Type_t
quarkContract34(const PSpinMatrixJIT<T1,4>& s1, const PSpinMatrixJIT<T2,4>& s2)
{
  typename BinaryReturn<PSpinMatrixJIT<T1,4>, PSpinMatrixJIT<T2,4>, FnQuarkContract34>::Type_t  d(s1.func());

  for(int j=0; j < 4; ++j)
    for(int i=0; i < 4; ++i)
    {
      d.elem(i,j) = quarkContractXX(s1.elem(i,j), s2.elem(0,0));
      for(int k=1; k < 4; ++k)
	d.elem(i,j) += quarkContractXX(s1.elem(i,j), s2.elem(k,k));
    }

  return d;
}

/*! @} */   // end of group primspinmatrix

} // namespace QDP

#endif
