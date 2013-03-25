// -*- C++ -*-

/*! \file
 * \brief Primitive Color Vector
 */


#ifndef QDP_PRIMCOLORVECJIT_H
#define QDP_PRIMCOLORVECJIT_H

namespace QDP {

//-------------------------------------------------------------------------------------
/*! \addtogroup primcolorvector Color vector primitive
 * \ingroup primvector
 *
 * Primitive type that transforms like a Color vector
 *
 * @{
 */

//! Primitive color Vector class
template <class T, int N> class PColorVectorJIT : public PVectorJIT<T, N, PColorVectorJIT>
{
public:

  // Default constructing should be possible
  // then there is no need for MPL index when
  // construction a PMatrix<T,N>
  PColorVectorJIT(){}

  //! PColorVectorJIT = PColorVectorJIT
  /*! Set equal to another PColorVectorJIT */
  template<class T1>
  inline
  PColorVectorJIT& operator=(const PColorVectorREG<T1,N>& rhs) 
    {
      this->assign(rhs);
      return *this;
    }

  template<class T1>
  inline
  PColorVectorJIT& operator+=(const PColorVectorREG<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	this->elem(i) += rhs.elem(i);

      return *this;
    }

  //! PVectorJIT -= PVectorJIT
  template<class T1>
  inline
  PColorVectorJIT& operator-=(const PColorVectorREG<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	this->elem(i) -= rhs.elem(i);

      return *this;
    }


};

/*! @} */  // end of group primcolorvector

//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------


template<class T1, int N>
struct REGType<PColorVectorJIT<T1,N> > 
{
  typedef PColorVectorREG<typename REGType<T1>::Type_t,N>  Type_t;
};


// Underlying word type
template<class T1, int N>
struct WordType<PColorVectorJIT<T1,N> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};



template<class T1, int N>
struct SinglePrecType< PColorVectorJIT<T1,N> >
{
  typedef PColorVectorJIT< typename SinglePrecType<T1>::Type_t, N> Type_t;
};


template<class T1, int N>
struct DoublePrecType< PColorVectorJIT<T1,N> >
{
  typedef PColorVectorJIT< typename DoublePrecType<T1>::Type_t, N> Type_t;
};


// Internally used scalars
template<class T, int N>
struct InternalScalar<PColorVectorJIT<T,N> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive scalar leaving other indices along
template<class T, int N>
struct PrimitiveScalar<PColorVectorJIT<T,N> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T, int N>
struct LatticeScalar<PColorVectorJIT<T,N> > {
  typedef PColorVectorJIT<typename LatticeScalar<T>::Type_t, N>  Type_t;
};

//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PColorVectorJIT) -> PColorVectorJIT
template<class T1, int N, class Op>
struct UnaryReturn<PColorVectorJIT<T1,N>, Op> {
  typedef PColorVectorJIT<typename UnaryReturn<T1, Op>::Type_t, N>  Type_t;
};
// Default binary(PScalarJIT,PColorVectorJIT) -> PColorVectorJIT
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PScalarJIT<T1>, PColorVectorJIT<T2,N>, Op> {
  typedef PColorVectorJIT<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PColorMatrixJIT,PColorVectorJIT) -> PColorVectorJIT
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PColorMatrixJIT<T1,N>, PColorVectorJIT<T2,N>, Op> {
  typedef PColorVectorJIT<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PColorVectorJIT,PScalarJIT) -> PColorVectorJIT
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PColorVectorJIT<T1,N>, PScalarJIT<T2>, Op> {
  typedef PColorVectorJIT<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PColorVectorJIT,PColorVectorJIT) -> PColorVectorJIT
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PColorVectorJIT<T1,N>, PColorVectorJIT<T2,N>, Op> {
  typedef PColorVectorJIT<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};


#if 0
template<class T1, class T2>
struct UnaryReturn<PScalarJIT<T2>, OpCast<T1> > {
  typedef PScalarJIT<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif


// Assignment is different
template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorJIT<T1,N>, PColorVectorJIT<T2,N>, OpAssign > {
  typedef PColorVectorJIT<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorJIT<T1,N>, PColorVectorJIT<T2,N>, OpAddAssign > {
  typedef PColorVectorJIT<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorJIT<T1,N>, PColorVectorJIT<T2,N>, OpSubtractAssign > {
  typedef PColorVectorJIT<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorJIT<T1,N>, PScalarJIT<T2>, OpMultiplyAssign > {
  typedef PColorVectorJIT<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorJIT<T1,N>, PScalarJIT<T2>, OpDivideAssign > {
  typedef PColorVectorJIT<T1,N> &Type_t;
};
 

// ColorVector
template<class T, int N>
struct UnaryReturn<PColorVectorJIT<T,N>, FnNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PColorVectorJIT<T,N>, FnLocalNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorJIT<T1,N>, PColorVectorJIT<T2,N>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorJIT<T1,N>, PColorVectorJIT<T2,N>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorJIT<T1,N>, PColorVectorJIT<T2,N>, FnInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorJIT<T1,N>, PColorVectorJIT<T2,N>, FnLocalInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};




//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

// Peeking and poking
//! Extract color vector components 
template<class T, int N>
struct UnaryReturn<PColorVectorJIT<T,N>, FnPeekColorVectorREG > {
  typedef PScalarJIT<typename UnaryReturn<T, FnPeekColorVectorREG >::Type_t>  Type_t;
};

// template<class T, int N>
// inline typename UnaryReturn<PColorVectorJIT<T,N>, FnPeekColorVectorJIT >::Type_t
// peekColor(const PColorVectorJIT<T,N>& l, int row)
// {
//   typename UnaryReturn<PColorVectorJIT<T,N>, FnPeekColorVectorJIT >::Type_t  d(l.func());

//   // Note, do not need to propagate down since the function is eaten at this level
//   d.elem() = l.getRegElem(row);
//   return d;
// }

// //! Insert color vector components
// template<class T1, class T2, int N>
// inline PColorVectorJIT<T1,N>&
// pokeColor(PColorVectorJIT<T1,N>& l, const PScalarJIT<T2>& r, int row)
// {
//   // Note, do not need to propagate down since the function is eaten at this level
//   l.getRegElem(row) = r.elem();
//   return l;
// }


//! Insert color vector components
template<class T1, class T2, int N>
inline PColorVectorJIT<T1,N>&
pokeColor(PColorVectorJIT<T1,N>& l, const PScalarREG<T2>& r, jit_value_t row)
{
  l.getJitElem(row) = r.elem();
  return l;
}


//-----------------------------------------------------------------------------
// Contraction for color vectors
// colorContract 
template<class T1, class T2, class T3, int N>
struct TrinaryReturn<PColorVectorJIT<T1,N>, PColorVectorJIT<T2,N>, PColorVectorJIT<T3,N>, FnColorContract> {
  typedef PScalarJIT<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};

//! dest  = colorContract(Qvec1,Qvec2,Qvec3)
/*!
 * Performs:
 *  \f$dest = \sum_{i,j,k} \epsilon^{i,j,k} V1^{i} V2^{j} V3^{k}\f$
 *
 * This routine is completely unrolled for 3 colors
 */
template<class T1, class T2, class T3>
inline typename TrinaryReturn<PColorVectorJIT<T1,3>, PColorVectorJIT<T2,3>, PColorVectorJIT<T3,3>, FnColorContract>::Type_t
colorContract(const PColorVectorJIT<T1,3>& s1, const PColorVectorJIT<T2,3>& s2, const PColorVectorJIT<T3,3>& s3)
{
  typename TrinaryReturn<PColorVectorJIT<T1,3>, PColorVectorJIT<T2,3>, PColorVectorJIT<T3,3>, FnColorContract>::Type_t  d(s1.func());

  // Permutations: +(0,1,2)+(1,2,0)+(2,0,1)-(1,0,2)-(0,2,1)-(2,1,0)

  // d = \epsilon^{i,j,k} V1^{i} V2^{j} V3^{k}
  d.elem() = (s1.elem(0)*s2.elem(1)
           -  s1.elem(1)*s2.elem(0))*s3.elem(2)
           + (s1.elem(1)*s2.elem(2)
           -  s1.elem(2)*s2.elem(1))*s3.elem(0)
           + (s1.elem(2)*s2.elem(0)
           -  s1.elem(0)*s2.elem(2))*s3.elem(1);

  return d;
}


//-----------------------------------------------------------------------------
// Contraction for color vectors
// colorContract 
template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorJIT<T1,N>, PColorVectorJIT<T2,N>, FnColorVectorContract> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnColorVectorContract>::Type_t>  Type_t;
};

//! dest  = colorVectorContract(Qvec1,Qvec2)
/*!
 * Performs:
 *  \f$dest = \sum_{i} V1^{i} V2^{i}\f$
 */
template<class T1, class T2, int N>
inline typename BinaryReturn<PColorVectorJIT<T1,N>, PColorVectorJIT<T2,N>, FnColorVectorContract>::Type_t
colorVectorContract(const PColorVectorJIT<T1,N>& s1, const PColorVectorJIT<T2,N>& s2)
{
  typename BinaryReturn<PColorVectorJIT<T1,N>, PColorVectorJIT<T2,N>, FnColorVectorContract>::Type_t  d(s1.func());

  // d = V1^{i} V2^{i}
  d.elem() = s1.elem(0)*s2.elem(0);
  for(int i=1; i < N; ++i)
    d.elem() += s1.elem(i)*s2.elem(i);

  return d;
}


//-----------------------------------------------------------------------------
// diquark color cross product   s1 X s2
//! Contraction for color vectors
template<class T1, class T2>
struct BinaryReturn<PColorVectorJIT<T1,3>, PColorVectorJIT<T2,3>, FnColorCrossProduct> {
  typedef PColorVectorJIT<typename BinaryReturn<T1, T2, FnColorCrossProduct>::Type_t, 3>  Type_t;
};

//! dest  = colorCrossProduct(Qvec1,Qvec2)
/*!
 * Performs:
 *  \f$dest^{i} = \sum_{j,k} \epsilon^{i,j,k} V1^{j} V2^{k}\f$
 *
 * This routine is completely unrolled for 3 colors
 */
template<class T1, class T2>
inline typename BinaryReturn<PColorVectorJIT<T1,3>, PColorVectorJIT<T2,3>, FnColorCrossProduct>::Type_t
colorCrossProduct(const PColorVectorJIT<T1,3>& s1, const PColorVectorJIT<T2,3>& s2)
{
  typename BinaryReturn<PColorVectorJIT<T1,3>, PColorVectorJIT<T2,3>, FnColorCrossProduct>::Type_t  d(s1.func());
  
  d.elem(0) = s1.elem(1)*s2.elem(2) - s1.elem(2)*s2.elem(1);
  d.elem(1) = s1.elem(2)*s2.elem(0) - s1.elem(0)*s2.elem(2);
  d.elem(2) = s1.elem(0)*s2.elem(1) - s1.elem(1)*s2.elem(0);

 return d;
}



} // namespace QDP

#endif

