// -*- C++ -*-

/*! \file
 * \brief Primitive Color Vector
 */


#ifndef QDP_PRIMCOLORVECREG_H
#define QDP_PRIMCOLORVECREG_H

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
template <class T, int N> class PColorVectorREG : public PVectorREG<T, N, PColorVectorREG>
{
public:

  void setup( const typename JITType< PColorVectorREG >::Type_t& j ) {
    for (int i = 0 ; i < N ; i++ )
      this->elem(i).setup( j.elem(i) );
  }


  //! PColorVectorREG = PColorVectorREG
  /*! Set equal to another PColorVectorREG */
  template<class T1>
  inline
  PColorVectorREG& operator=(const PColorVectorREG<T1,N>& rhs) 
    {
      this->assign(rhs);
      return *this;
    }

  PColorVectorREG& operator=(const PColorVectorREG& rhs) 
    {
      this->assign(rhs);
      return *this;
    }

};


template <class T, int N>
jit_function_t getFunc(const PColorVectorREG<T,N>& l) {
  return getFunc(l.elem(0));
}



/*! @} */  // end of group primcolorvector

//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------


template<class T1, int N>
struct JITType<PColorVectorREG<T1,N> > 
{
  typedef PColorVectorJIT<typename JITType<T1>::Type_t,N>  Type_t;
};



// Underlying word type
template<class T1, int N>
struct WordType<PColorVectorREG<T1,N> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};



template<class T1, int N>
struct SinglePrecType< PColorVectorREG<T1,N> >
{
  typedef PColorVectorREG< typename SinglePrecType<T1>::Type_t, N> Type_t;
};


template<class T1, int N>
struct DoublePrecType< PColorVectorREG<T1,N> >
{
  typedef PColorVectorREG< typename DoublePrecType<T1>::Type_t, N> Type_t;
};


// Internally used scalars
template<class T, int N>
struct InternalScalar<PColorVectorREG<T,N> > {
  typedef PScalarREG<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive scalar leaving other indices along
template<class T, int N>
struct PrimitiveScalar<PColorVectorREG<T,N> > {
  typedef PScalarREG<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T, int N>
struct LatticeScalar<PColorVectorREG<T,N> > {
  typedef PColorVectorREG<typename LatticeScalar<T>::Type_t, N>  Type_t;
};

//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PColorVectorREG) -> PColorVectorREG
template<class T1, int N, class Op>
struct UnaryReturn<PColorVectorREG<T1,N>, Op> {
  typedef PColorVectorREG<typename UnaryReturn<T1, Op>::Type_t, N>  Type_t;
};
// Default binary(PScalarREG,PColorVectorREG) -> PColorVectorREG
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PScalarREG<T1>, PColorVectorREG<T2,N>, Op> {
  typedef PColorVectorREG<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PColorMatrixREG,PColorVectorREG) -> PColorVectorREG
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PColorMatrixREG<T1,N>, PColorVectorREG<T2,N>, Op> {
  typedef PColorVectorREG<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PColorVectorREG,PScalarREG) -> PColorVectorREG
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PColorVectorREG<T1,N>, PScalarREG<T2>, Op> {
  typedef PColorVectorREG<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PColorVectorREG,PColorVectorREG) -> PColorVectorREG
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PColorVectorREG<T1,N>, PColorVectorREG<T2,N>, Op> {
  typedef PColorVectorREG<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
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
struct BinaryReturn<PColorVectorREG<T1,N>, PColorVectorREG<T2,N>, OpAssign > {
  typedef PColorVectorREG<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorREG<T1,N>, PColorVectorREG<T2,N>, OpAddAssign > {
  typedef PColorVectorREG<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorREG<T1,N>, PColorVectorREG<T2,N>, OpSubtractAssign > {
  typedef PColorVectorREG<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorREG<T1,N>, PScalarREG<T2>, OpMultiplyAssign > {
  typedef PColorVectorREG<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorREG<T1,N>, PScalarREG<T2>, OpDivideAssign > {
  typedef PColorVectorREG<T1,N> &Type_t;
};
 

// ColorVector
template<class T, int N>
struct UnaryReturn<PColorVectorREG<T,N>, FnNorm2 > {
  typedef PScalarREG<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PColorVectorREG<T,N>, FnLocalNorm2 > {
  typedef PScalarREG<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorREG<T1,N>, PColorVectorREG<T2,N>, FnInnerProduct> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorREG<T1,N>, PColorVectorREG<T2,N>, FnLocalInnerProduct> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorREG<T1,N>, PColorVectorREG<T2,N>, FnInnerProductReal> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorVectorREG<T1,N>, PColorVectorREG<T2,N>, FnLocalInnerProductReal> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};




//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

// Peeking and poking
//! Extract color vector components 
template<class T, int N>
struct UnaryReturn<PColorVectorREG<T,N>, FnPeekColorVectorREG > {
  typedef PScalarREG<typename UnaryReturn<T, FnPeekColorVectorREG >::Type_t>  Type_t;
};

template<class T, int N>
inline typename UnaryReturn<PColorVectorREG<T,N>, FnPeekColorVectorREG >::Type_t
peekColor(const PColorVectorREG<T,N>& l, jit_value_t row)
{
  typename UnaryReturn<PColorVectorREG<T,N>, FnPeekColorVectorREG >::Type_t  d;

  typedef typename JITType< PColorVectorREG<T,N> >::Type_t TTjit;

  jit_value_t ptr_local = jit_allocate_local( getFunc(l), 
					      jit_type<typename WordType<T>::Type_t>::value , 
					      TTjit::Size_t );

  TTjit dj;
  dj.setup( getFunc(l) , ptr_local, 1 , 0);
  dj=l;

  d.elem() = dj.getRegElem(row);
  return d;
}

//! Insert color vector components
template<class T1, class T2, int N>
inline PColorVectorREG<T1,N>&
pokeColor(PColorVectorREG<T1,N>& l, const PScalarREG<T2>& r, int row)
{
  // Note, do not need to propagate down since the function is eaten at this level
  l.getRegElem(row) = r.elem();
  return l;
}


//-----------------------------------------------------------------------------
// Contraction for color vectors
// colorContract 
template<class T1, class T2, class T3, int N>
struct TrinaryReturn<PColorVectorREG<T1,N>, PColorVectorREG<T2,N>, PColorVectorREG<T3,N>, FnColorContract> {
  typedef PScalarREG<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};

//! dest  = colorContract(Qvec1,Qvec2,Qvec3)
/*!
 * Performs:
 *  \f$dest = \sum_{i,j,k} \epsilon^{i,j,k} V1^{i} V2^{j} V3^{k}\f$
 *
 * This routine is completely unrolled for 3 colors
 */
template<class T1, class T2, class T3>
inline typename TrinaryReturn<PColorVectorREG<T1,3>, PColorVectorREG<T2,3>, PColorVectorREG<T3,3>, FnColorContract>::Type_t
colorContract(const PColorVectorREG<T1,3>& s1, const PColorVectorREG<T2,3>& s2, const PColorVectorREG<T3,3>& s3)
{
  typename TrinaryReturn<PColorVectorREG<T1,3>, PColorVectorREG<T2,3>, PColorVectorREG<T3,3>, FnColorContract>::Type_t  d;

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
struct BinaryReturn<PColorVectorREG<T1,N>, PColorVectorREG<T2,N>, FnColorVectorContract> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnColorVectorContract>::Type_t>  Type_t;
};

//! dest  = colorVectorContract(Qvec1,Qvec2)
/*!
 * Performs:
 *  \f$dest = \sum_{i} V1^{i} V2^{i}\f$
 */
template<class T1, class T2, int N>
inline typename BinaryReturn<PColorVectorREG<T1,N>, PColorVectorREG<T2,N>, FnColorVectorContract>::Type_t
colorVectorContract(const PColorVectorREG<T1,N>& s1, const PColorVectorREG<T2,N>& s2)
{
  typename BinaryReturn<PColorVectorREG<T1,N>, PColorVectorREG<T2,N>, FnColorVectorContract>::Type_t  d;

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
struct BinaryReturn<PColorVectorREG<T1,3>, PColorVectorREG<T2,3>, FnColorCrossProduct> {
  typedef PColorVectorREG<typename BinaryReturn<T1, T2, FnColorCrossProduct>::Type_t, 3>  Type_t;
};

//! dest  = colorCrossProduct(Qvec1,Qvec2)
/*!
 * Performs:
 *  \f$dest^{i} = \sum_{j,k} \epsilon^{i,j,k} V1^{j} V2^{k}\f$
 *
 * This routine is completely unrolled for 3 colors
 */
template<class T1, class T2>
inline typename BinaryReturn<PColorVectorREG<T1,3>, PColorVectorREG<T2,3>, FnColorCrossProduct>::Type_t
colorCrossProduct(const PColorVectorREG<T1,3>& s1, const PColorVectorREG<T2,3>& s2)
{
  typename BinaryReturn<PColorVectorREG<T1,3>, PColorVectorREG<T2,3>, FnColorCrossProduct>::Type_t  d;
  
  d.elem(0) = s1.elem(1)*s2.elem(2) - s1.elem(2)*s2.elem(1);
  d.elem(1) = s1.elem(2)*s2.elem(0) - s1.elem(0)*s2.elem(2);
  d.elem(2) = s1.elem(0)*s2.elem(1) - s1.elem(1)*s2.elem(0);

 return d;
}



} // namespace QDP

#endif

