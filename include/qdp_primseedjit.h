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
public:

  //! construct dest = const
  template<class T1>
  PSeedJIT(const PScalarJIT<T1>& rhs)
  {
    assign(rhs);
  }

#if 0
  PSeedJIT(const PSeedJIT& a): JV<T,4>(newspace_t(a.func()),&a)
  {
    assign(a);
  }

  template<class T1>
  PSeedJIT(const PSeedJIT<T1>& a): JV<T,4>(newspace_t(a.func()))
  {
    assign(a);
  }
#endif

  //! PSeedJIT = PScalarJIT
  /*! Set equal to input scalar (an integer) */
  template<class T1>
  inline
  PSeedJIT& assign(const PScalarJIT<T1>& rhs) 
    {
      typedef typename InternalScalar<T1>::Type_t  S;

      // elem(0) = rhs.elem() & S(rhs.func(),4095);
      // elem(1) = (rhs.elem() >> S(rhs.func(),12)) & S(rhs.func(),4095);
      // elem(2) = (rhs.elem() >> S(rhs.func(),24)) & S(rhs.func(),4095);

      S s4095(rhs.func());
      S s12(rhs.func());
      S s24(rhs.func());
      s4095 = 4095;
      s12 = 12;
      s24 = 24;

      elem(0) = rhs.elem() & S(rhs.func(),4095);
      elem(1) = (rhs.elem() >> s12) & s4095;
      elem(2) = (rhs.elem() >> s24) & s4095;
//      elem(3) = (rhs.elem() >> S(36)) & S(2047);  // This probably will never be nonzero
      zero_rep(elem(3));    // assumes 32 bit integers

      return *this;
    }

  //! PSeedJIT = PScalarJIT
  /*! Set equal to input scalar (an integer) */
  template<class T1>
  inline
  PSeedJIT& operator=(const PScalarJIT<T1>& rhs) 
    {
      return assign(rhs);
    }

  //! PSeedJIT = PSeedJIT
  /*! Set equal to another PSeedJIT */
  template<class T1>
  inline
  PSeedJIT& operator=(const PSeedJIT<T1>& rhs) 
    {
      for(int i=0; i < 4; ++i)
	elem(i) = rhs.elem(i);

      return *this;
    }



  PSeedJIT& operator=(const PSeedJIT& rhs) 
    {
      for(int i=0; i < 4; ++i)
	elem(i) = rhs.elem(i);

      return *this;
    }


public:
        T& elem(int i)       {return this->arrayF(i);}
  const T& elem(int i) const {return this->arrayF(i);}

  // T& elem(int i)             {return JV<T,4>::getF()[i]; }
  // const T& elem(int i) const {return JV<T,4>::getF()[i]; }
};



//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

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

//! PSeedJIT<T> = PSeedJIT<T> * PSeedJIT<T>
/*!
 * A 47 bit seed multiplication is represented as the multiplication
 * of three 12bit and one 11bit integer
 *
 * i3 = s1(3)*s2(0) + s1(2)*s2(1)
 *    + s1(1)*s2(2) + s1(0)*s2(3);
 * i2 = s1(2)*s2(0) + s1(1)*s2(1)
 *    + s1(0)*s2(2);
 * i1 = s1(1)*s2(0) + s1(0)*s2(1);
 * i0 = s1(0)*s2(0);
 *
 * dest(0) = mod(i0, 4096);
 * i1      = i1 + i0/4096;
 * dest(1) = mod(i1, 4096);
 * i2      = i2 + i1/4096;
 * dest(2) = mod(i2, 4096);
 * i3      = i3 + i2/4096
 * dest(3) = mod(i3, 2048);
 */
template<class T1, class T2>
struct BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpMultiply> {
  typedef PSeedJIT<typename BinaryReturn<T1, T2, OpMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpMultiply>::Type_t
operator*(const PSeedJIT<T1>& s1, const PSeedJIT<T2>& s2)
{
  typename BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpMultiply>::Type_t  d(s1.func());
  typedef typename BinaryReturn<T1, T2, OpMultiply>::Type_t  T;
  typedef typename InternalScalar<T>::Type_t  S;
  T  i0(s1.func()), i1(s1.func()), i2(s1.func()), i3(s1.func());

  S s4095(s1.func());
  S s2047(s1.func());
  S s12(s1.func());
  s4095 = 4095;
  s2047 = 2047;
  s12 = 12;


  /* i3 = s1(3)*s2(0) + s1(2)*s2(1) + s1(1)*s2(2) + s1(0)*s2(3) */
  i3  = s1.elem(3) * s2.elem(0);
  i3 += s1.elem(2) * s2.elem(1);
  i3 += s1.elem(1) * s2.elem(2);
  i3 += s1.elem(0) * s2.elem(3);

  /* i2 = s1(2)*s2(0) + s1(1)*s2(1) + s1(0)*s2(2) */
  i2  = s1.elem(2) * s2.elem(0);
  i2 += s1.elem(1) * s2.elem(1);
  i2 += s1.elem(0) * s2.elem(2);

  /* i1 = s1(1)*s2(0) + s1(0)*s2(1) */
  i1  = s1.elem(1) * s2.elem(0);
  i1 += s1.elem(0) * s2.elem(1);

  /* i0 = s1(0)*s2(0) */
  i0 = s1.elem(0) * s2.elem(0);
  
  /* dest(0) = mod(i0, 4096) */
  //  d.elem(0) = i0 & S(s1.func(),4095);
  d.elem(0) = i0 & s4095;

  /* i1 = i1 + i0/4096 */
  i1 += i0 >> s12;
  //i1 += i0 >> S(s1.func(),12);

  /* dest(1) = mod(i1, 4096) */
  d.elem(1) = i1 & s4095;
  //  d.elem(1) = i1 & S(s1.func(),4095);

  /* i2 = i2 + i1/4096 */
  //  i2 += i1 >> S(s1.func(),12);
  i2 += i1 >> s12;

  /* dest(2) = mod(i2, 4096) */
  d.elem(2) = i2 & s4095;
  //d.elem(2) = i2 & S(s1.func(),4095);
  /* i3 = i3 + i2/4096 */
  i3 += i2 >> s12;
  //  i3 += i2 >> S(s1.func(),12);

  /* dest(3) = mod(i3, 2048) */
  d.elem(3) = i3 & s2047;
  //d.elem(3) = i3 & S(s1.func(),2047);
  
  return d;
}


template<class T1, class T2>
struct BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpBitwiseOr> {
  typedef PSeedJIT<typename BinaryReturn<T1, T2, OpBitwiseOr>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpBitwiseOr>::Type_t
operator|(const PSeedJIT<T1>& l, const PSeedJIT<T2>& r)
{
  typename BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpBitwiseOr>::Type_t  d(l.func());

  d.elem(0) = l.elem(0) | r.elem(0);
  d.elem(1) = l.elem(1) | r.elem(1);
  d.elem(2) = l.elem(2) | r.elem(2);
  d.elem(3) = l.elem(3) | r.elem(3);

  return d;
}



// Mixed versions
template<class T1, class T2>
struct BinaryReturn<PSeedJIT<T1>, PScalarJIT<T2>, OpBitwiseOr> {
  typedef PSeedJIT<typename BinaryReturn<T1, T2, OpBitwiseOr>::Type_t>  Type_t;
};
 
template<class T1, class T2>
inline typename BinaryReturn<PSeedJIT<T1>, PScalarJIT<T2>, OpBitwiseOr>::Type_t
operator|(const PSeedJIT<T1>& l, const PScalarJIT<T2>& r)
{
  // Lazy implementation

  PSeedJIT<T2>  d(l.func());
  d = r;

  return (l | d);
}



/*! 
 * This left shift implementation will not work properly for shifts
 * greater than 12
 */
template<class T1, class T2>
struct BinaryReturn<PSeedJIT<T1>, PScalarJIT<T2>, OpLeftShift> {
  typedef PSeedJIT<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PSeedJIT<T1>, PScalarJIT<T2>, OpLeftShift>::Type_t
operator<<(const PSeedJIT<T1>& s1, const PScalarJIT<T2>& s2)
{
  typename BinaryReturn<PSeedJIT<T1>, PScalarJIT<T2>, OpLeftShift>::Type_t  d(s1.func());
  typedef typename BinaryReturn<T1, T2, OpLeftShift>::Type_t  T;
  typedef typename InternalScalar<T>::Type_t  S;
  T  i0, i1, i2, i3;

  i0 = s1.elem(0) << s2.elem();
  i1 = s1.elem(1) << s2.elem();
  i2 = s1.elem(2) << s2.elem();
  i3 = s1.elem(3) << s2.elem();

  d.elem(0) = i0 & S(4095);
  i0 >>= S(12);
  i1 |= i0 & S(4095);
  d.elem(1) = i1 & S(4095);
  i1 >>= S(12);
  i2 |= i1 & S(4095);
  d.elem(2) = i2 & S(4095);
  i2 >>= S(12);
  i3 |= i2 & S(4095);
  d.elem(3) = i3 & S(2047);

  return d;
}


//! dest [float type] = source [seed type]
template<class T>
struct UnaryReturn<PSeedJIT<T>, FnSeedToFloat> {
  typedef PScalarJIT<typename UnaryReturn<T, FnSeedToFloat>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<PSeedJIT<T>, FnSeedToFloat>::Type_t
seedToFloat(const PSeedJIT<T>& s1)
{
  typename UnaryReturn<PSeedJIT<T>, FnSeedToFloat>::Type_t  d(s1.func());
  typedef typename RealScalar<T>::Type_t  S;

  S  twom11(s1.func(),1.0 / 2048.0);
  S  twom12(s1.func(),1.0 / 4096.0);
  S  fs1(s1.func()), fs2(s1.func());

//  recast_rep(fs1, s1.elem(0));
  fs1 = S(s1.elem(0));
  d.elem() = twom12 * S(s1.elem(0));

//  recast_rep(fs1, s1.elem(1));
  fs1 = S(s1.elem(1));
  fs2 = fs1 + d.elem();
  d.elem() = twom12 * fs2;

//  recast_rep(fs1, s1.elem(2));
  fs1 = S(s1.elem(2));
  fs2 = fs1 + d.elem();
  d.elem() = twom12 * fs2;

//  recast_rep(fs1, s1.elem(3));
  fs1 = S(s1.elem(3));
  fs2 = fs1 + d.elem();
  d.elem() = twom11 * fs2;

  return d;
}


//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T>
struct UnaryReturn<PSeedJIT<T>, FnGetSite> {
  typedef PSeedJIT<typename UnaryReturn<T, FnGetSite>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<PSeedJIT<T>, FnGetSite>::Type_t
getSite(const PSeedJIT<T>& s1, int innersite)
{ 
  typename UnaryReturn<PSeedJIT<T>, FnGetSite>::Type_t  d(s1.func());

  for(int i=0; i < 4; ++i)
    d.elem(i) = getSite(s1.elem(i), innersite);

  return d;
}


// Functions
//! dest = 0
template<class T> 
inline void 
zero_rep(PSeedJIT<T>& dest) 
{
  for(int i=0; i < 4; ++i)
    zero_rep(dest.elem(i));
}


//! dest = (mask) ? s1 : dest
template<class T, class T1> 
inline void 
copymask(PSeedJIT<T>& d, const PScalarJIT<T1>& mask, const PSeedJIT<T>& s1) 
{
  for(int i=0; i < 4; ++i)
    copymask(d.elem(i),mask.elem(),s1.elem(i));
}

/*! @} */

} // namespace QDP

#endif
