// -*- C++ -*-

/*! \file
 * \brief Primitive Seed
 */


#ifndef QDP_PRIMSEEDJIT_H
#define QDP_PRIMSEEDJIT_H

namespace QDP {

//-------------------------------------------------------------------------------------
/*! \addtogroup primseed Seed primitive
 * \ingroup fiber
 *
 * Primitive type for supporting random numbers. This is really a 
 * big integer class.
 *
 * @{
 */


//! Primitive Seed class
/*!
   * Seed primitives exist to facilitate seed multiplication - namely
   * multiplication of a large integer represented as 4 smaller integers
   *
   * NOTE: the class is still treated as a template since there may be
   * an inner lattice so the type could be represented as an length 4
   * array of lattice integers
   */
template <class T> class PSeedJIT : public BaseJIT<T,4>
{
  private:
    template<class T1>
    PSeedJIT& operator=( const PSeedJIT<T1>& rhs);
    PSeedJIT& operator=( const PSeedJIT& rhs);

public:

  // Default constructing should be possible
  // then there is no need for MPL index when
  // construction a PMatrix<T,N>
  PSeedJIT() {}
  ~PSeedJIT() {}
  
  // template<class T1>
  // PSeedJIT& operator=( const PSeedREG<T1>& rhs) {
  //   for(int i=0; i < 4; ++i)
  //     elem(i) = rhs.elem(i);
  //   return *this;
  // }


  //! PSeedJIT = PScalarJIT
  /*! Set equal to input scalar (an integer) */
  template<class T1>
  inline
  PSeedJIT& assign(const PScalarREG<T1>& rhs) 
    {
      typedef typename InternalScalar<T1>::Type_t  S;

      elem(0) = rhs.elem() & S(4095);
      elem(1) = (rhs.elem() >> S(12)) & S(4095);
      elem(2) = (rhs.elem() >> S(24)) & S(4095);
//      elem(3) = (rhs.elem() >> S(36)) & S(2047);  // This probably will never be nonzero
      zero_rep(elem(3));    // assumes 32 bit integers

      return *this;
    }

  //! PSeedJIT = PScalarJIT
  /*! Set equal to input scalar (an integer) */
  template<class T1>
  inline
  PSeedJIT& operator=(const PScalarREG<T1>& rhs) 
    {
      return assign(rhs);
    }

  //! PSeedJIT = PSeedJIT
  /*! Set equal to another PSeedJIT */
  template<class T1>
  inline
  PSeedJIT& operator=(const PSeedREG<T1>& rhs) 
    {
      for(int i=0; i < 4; ++i)
	elem(i) = rhs.elem(i);

      return *this;
    }


public:
        T& elem(int i)       {return this->arrayF(i);}
  const T& elem(int i) const {return this->arrayF(i);}
};



//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

template<class T>
struct ScalarType<PSeedJIT<T> >
{
  typedef PSeedJIT< typename ScalarType<T>::Type_t > Type_t;
};
  
template<class T> 
struct REGType< PSeedJIT<T> >
{
  typedef PSeedREG<typename REGType<T>::Type_t>  Type_t;
};

template<class T> 
struct BASEType< PSeedJIT<T> >
{
  typedef PSeed<typename BASEType<T>::Type_t>  Type_t;
};


// Underlying word type
template<class T1>
struct WordType<PSeedJIT<T1> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

// Fixed Precision versions (do these even make sense? )

template<class T1>
struct SinglePrecType<PSeedJIT<T1> >
{
  typedef PSeedJIT< typename SinglePrecType<T1>::Type_t > Type_t;
};

template<class T1>
struct DoublePrecType<PSeedJIT<T1> >
{
  typedef PSeedJIT< typename DoublePrecType<T1>::Type_t > Type_t;
};


// Internally used scalars
template<class T>
struct InternalScalar<PSeedJIT<T> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive scalar leaving grid alone
template<class T>
struct PrimitiveScalar<PSeedJIT<T> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T>
struct LatticeScalar<PSeedJIT<T> > {
  typedef PSeedJIT<typename LatticeScalar<T>::Type_t>  Type_t;
};


//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Assignment is different
template<class T1, class T2 >
struct BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpAssign > {
  typedef PSeedJIT<T1> &Type_t;
};

 

//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

// PScalarJIT = (PSeedJIT == PSeedJIT)
template<class T1, class T2>
struct BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpEQ> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpEQ>::Type_t
operator==(const PSeedJIT<T1>& l, const PSeedJIT<T2>& r)
{
  return 
    (l.elem(0) == r.elem(0)) && 
    (l.elem(1) == r.elem(1)) && 
    (l.elem(2) == r.elem(2)) && 
    (l.elem(3) == r.elem(3));
}


// PScalarJIT = (Seed != Seed)
template<class T1, class T2>
struct BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpNE> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpNE>::Type_t
operator!=(const PSeedJIT<T1>& l, const PSeedJIT<T2>& r)
{
  return 
    (l.elem(0) != r.elem(0)) ||
    (l.elem(1) != r.elem(1)) || 
    (l.elem(2) != r.elem(2)) || 
    (l.elem(3) != r.elem(3));
}


/*! \addtogroup primseed
 * @{ 
 */

// Primitive Seeds


template<class T1, class T2>
struct BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpMultiply> {
  typedef PSeedJIT<typename BinaryReturn<T1, T2, OpMultiply>::Type_t>  Type_t;
};



template<class T1, class T2>
struct BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpBitwiseOr> {
  typedef PSeedJIT<typename BinaryReturn<T1, T2, OpBitwiseOr>::Type_t>  Type_t;
};


// Mixed versions
template<class T1, class T2>
struct BinaryReturn<PSeedJIT<T1>, PScalarJIT<T2>, OpBitwiseOr> {
  typedef PSeedJIT<typename BinaryReturn<T1, T2, OpBitwiseOr>::Type_t>  Type_t;
};
 

/*! 
 * This left shift implementation will not work properly for shifts
 * greater than 12
 */
template<class T1, class T2>
struct BinaryReturn<PSeedJIT<T1>, PScalarJIT<T2>, OpLeftShift> {
  typedef PSeedJIT<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};



//! dest [float type] = source [seed type]
template<class T>
struct UnaryReturn<PSeedJIT<T>, FnSeedToFloat> {
  typedef PScalarJIT<typename UnaryReturn<T, FnSeedToFloat>::Type_t>  Type_t;
};



//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T>
struct UnaryReturn<PSeedJIT<T>, FnGetSite> {
  typedef PSeedJIT<typename UnaryReturn<T, FnGetSite>::Type_t>  Type_t;
};


// Functions
//! dest = 0
template<class T> 
inline void 
zero_rep(PSeedJIT<T> dest) 
{
  for(int i=0; i < 4; ++i)
    zero_rep(dest.elem(i));
}



/*! @} */

} // namespace QDP

#endif
