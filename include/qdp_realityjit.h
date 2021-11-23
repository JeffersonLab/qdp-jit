// -*- C++ -*-

/*! \file
 * \brief Reality
 */


#ifndef QDP_REALITYJIT_H
#define QDP_REALITYJIT_H


namespace QDP {

template<class T>
class RScalarJIT : public BaseJIT<T,1>
{
public:

  // Default constructing should be possible
  RScalarJIT() {}
  ~RScalarJIT() {}

  template<class T1>
  RScalarJIT& operator=( const RScalarREG<T1>& rhs) {
    elem() = rhs.elem();
    return *this;
  }


  RScalarJIT(const T& rhs) {
    this->setup( rhs.getBaseReg() , JitDeviceLayout::Scalar );
    elem() = rhs;
  }



  template<class T1>
  RScalarJIT& operator=(const typename REGType<T1>::Type_t& rhs) 
  {
    elem() = rhs.elem();
    return *this;
  }



  //! RScalarJIT += RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator+=(const RScalarREG<T1>& rhs) 
    {
      elem() += rhs.elem();
      return *this;
    }

  RScalarJIT& operator+=(const typename REGType<RScalarJIT>::Type_t& rhs) 
    {
      elem() += rhs.elem();
      return *this;
    }

  //! RScalarJIT -= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator-=(const RScalarREG<T1>& rhs) 
    {
      elem() -= rhs.elem();
      return *this;
    }

  RScalarJIT& operator-=(const typename REGType<RScalarJIT>::Type_t& rhs) 
    {
      elem() -= rhs.elem();
      return *this;
    }


  //! RScalarJIT *= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator*=(const RScalarREG<T1>& rhs) 
    {
      elem() *= rhs.elem();
      return *this;
    }

  RScalarJIT& operator*=(const typename REGType<RScalarJIT>::Type_t& rhs) 
    {
      elem() *= rhs.elem();
      return *this;
    }


  //! RScalarJIT /= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator/=(const RScalarREG<T1>& rhs) 
    {
      elem() /= rhs.elem();
      return *this;
    }

  RScalarJIT& operator/=(const typename REGType<RScalarJIT>::Type_t& rhs) 
    {
      elem() /= rhs.elem();
      return *this;
    }

  //! RScalarJIT %= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator%=(const RScalarREG<T1>& rhs) 
    {
      elem() %= rhs.elem();
      return *this;
    }

  //! RScalarJIT |= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator|=(const RScalarREG<T1>& rhs) 
    {
      elem() |= rhs.elem();
      return *this;
    }

  //! RScalarJIT &= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator&=(const RScalarREG<T1>& rhs) 
    {
      elem() &= rhs.elem();
      return *this;
    }

  //! RScalarJIT ^= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator^=(const RScalarREG<T1>& rhs) 
    {
      elem() ^= rhs.elem();
      return *this;
    }

  //! RScalarJIT <<= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator<<=(const RScalarREG<T1>& rhs) 
    {
      elem() <<= rhs.elem();
      return *this;
    }


  //! RScalarJIT >>= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator>>=(const RScalarREG<T1>& rhs) 
    {
      elem() >>= rhs.elem();
      return *this;
    }


public:
  inline       T& elem()       { return this->arrayF(0); }
  inline const T& elem() const { return this->arrayF(0); }
};



//-------------------------------------------------------------------------------------
/*! \addtogroup rcomplex Complex reality
 * \ingroup fiber
 *
 * Reality Complex is a type for objects that hold a real and imaginary part
 *
 * @{
 */

template<class T>
class RComplexJIT: public BaseJIT<T,2>
{
public:

  // Default constructing should be possible
  // then there is no need for MPL index when
  // construction a PMatrix<T,N>
  RComplexJIT() {}
  ~RComplexJIT() {}


  //! Construct from two scalars
  //RComplexJIT(Jit& j,const typename WordType<T>::Type_t& re, const typename WordType<T>::Type_t& im): JV<T,2>(j,re,im) {}


  RComplexJIT(const T& re,const T& im) {
    real() = re;
    imag() = im;
  }


  //! RComplexJIT += RScalarJIT
  template<class T1>
  inline
  RComplexJIT& operator+=(const RScalarREG<T1>& rhs) 
    {
      real() += rhs.elem();
      return *this;
    }


  //! RComplexJIT -= RScalarJIT
  template<class T1>
  inline
  RComplexJIT& operator-=(const RScalarREG<T1>& rhs) 
    {
      real() -= rhs.elem();
      return *this;
    }


  //! RComplexJIT *= RScalarJIT
  template<class T1>
  inline
  RComplexJIT& operator*=(const RScalarREG<T1>& rhs) 
    {
      real() *= rhs.elem();
      imag() *= rhs.elem();
      return *this;
    }

  //! RComplexJIT /= RScalarJIT
  template<class T1>
  inline
  RComplexJIT& operator/=(const RScalarREG<T1>& rhs) 
    {
      real() /= rhs.elem();
      imag() /= rhs.elem();
      return *this;
    }

  //! RComplexJIT += RComplexJIT
  template<class T1>
  inline
  RComplexJIT& operator+=(const RComplexREG<T1>& rhs) 
    {
      real() += rhs.real();
      imag() += rhs.imag();
      return *this;
    }

  RComplexJIT& operator+=(const typename REGType<RComplexJIT>::Type_t& rhs) 
    {
      real() += rhs.real();
      imag() += rhs.imag();
      return *this;
    }

  //! RComplexJIT -= RComplexJIT
  template<class T1>
  inline
  RComplexJIT& operator-=(const RComplexREG<T1>& rhs) 
    {
      real() -= rhs.real();
      imag() -= rhs.imag();
      return *this;
    }

  RComplexJIT& operator-=(const typename REGType<RComplexJIT>::Type_t& rhs) 
    {
      real() -= rhs.real();
      imag() -= rhs.imag();
      return *this;
    }

  //! RComplexJIT *= RComplexJIT
  template<class T1>
  inline
  RComplexJIT& operator*=(const RComplexREG<T1>& rhs) 
    {
      typename REGType<RComplexJIT>::Type_t me(*this);
      *this = me * rhs;
      return *this;
    }

  RComplexJIT& operator*=(const typename REGType<RComplexJIT>::Type_t& rhs) 
    {
      typename REGType<RComplexJIT>::Type_t me(*this);
      *this = me * rhs;
      return *this;
    }

  //! RComplexJIT /= RComplexJIT
  template<class T1>
  inline
  RComplexJIT& operator/=(const RComplexREG<T1>& rhs) 
    {
      typename REGType<RComplexJIT>::Type_t me(*this);
      *this = me / rhs;
      return *this;
    }

  RComplexJIT& operator/=(const typename REGType<RComplexJIT>::Type_t& rhs) 
    {
      typename REGType<RComplexJIT>::Type_t me(*this);
      *this = me / rhs;
      return *this;
    }

  template<class T1>
  RComplexJIT& operator=(const RComplexREG<T1>& rhs) 
    {
      real() = rhs.real();
      imag() = rhs.imag();
      return *this;
    }

  RComplexJIT& operator=(const typename REGType<RComplexJIT>::Type_t& rhs) 
    {
      real() = rhs.real();
      imag() = rhs.imag();
      return *this;
    }


  template<class T1>
  inline
  RComplexJIT& operator=(const RScalarREG<T1>& rhs) 
    {
      real() = rhs.elem();
      zero_rep(imag());
      return *this;
    }


public:
  inline       T& real()       { return this->arrayF(0); }
  inline const T& real() const { return this->arrayF(0); }

  inline       T& imag()       { return this->arrayF(1); }
  inline const T& imag() const { return this->arrayF(1); }
};




/*! @} */   // end of group rcomplex

//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

template<class T>
struct ScalarType<RScalarJIT<T> >
{
  typedef RScalarJIT< typename ScalarType<T>::Type_t > Type_t;
};


template<class T>
struct ScalarType<RComplexJIT<T> >
{
  typedef RComplexJIT< typename ScalarType<T>::Type_t > Type_t;
};


  
template<class T>
struct REGType< RScalarJIT<T> >
{
  typedef RScalarREG<typename REGType<T>::Type_t>  Type_t;
};

template<class T>
struct REGType< RComplexJIT<T> >
{
  typedef RComplexREG<typename REGType<T>::Type_t>  Type_t;
};


template<class T>
struct BASEType< RScalarJIT<T> >
{
  typedef RScalar<typename BASEType<T>::Type_t>  Type_t;
};

template<class T>
struct BASEType< RComplexJIT<T> >
{
  typedef RComplex<typename BASEType<T>::Type_t>  Type_t;
};


// Underlying word type
template<class T>
struct WordType<RScalarJIT<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

template<class T>
struct WordType<RComplexJIT<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

// Fixed types
template<class T> 
struct SinglePrecType<RScalarJIT<T> >
{
  typedef RScalarJIT<typename SinglePrecType<T>::Type_t>  Type_t;
};

template<class T> 
struct SinglePrecType<RComplexJIT<T> >
{
  typedef RComplexJIT<typename SinglePrecType<T>::Type_t>  Type_t;
};

template<class T> 
struct DoublePrecType<RScalarJIT<T> >
{
  typedef RScalarJIT<typename DoublePrecType<T>::Type_t>  Type_t;
};

template<class T> 
struct DoublePrecType<RComplexJIT<T> >
{
  typedef RComplexJIT<typename DoublePrecType<T>::Type_t>  Type_t;
};


// Internally used scalars
template<class T>
struct InternalScalar<RScalarJIT<T> > {
  typedef RScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};

template<class T>
struct InternalScalar<RComplexJIT<T> > {
  typedef RScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};


// Makes a primitive scalar leaving grid alone
template<class T>
struct PrimitiveScalar<RScalarJIT<T> > {
  typedef RScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

template<class T>
struct PrimitiveScalar<RComplexJIT<T> > {
  typedef RScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T>
struct LatticeScalar<RScalarJIT<T> > {
  typedef RScalarJIT<typename LatticeScalar<T>::Type_t>  Type_t;
};

template<class T>
struct LatticeScalar<RComplexJIT<T> > {
  typedef RComplexJIT<typename LatticeScalar<T>::Type_t>  Type_t;
};


// Internally used real scalars
template<class T>
struct RealScalar<RScalarJIT<T> > {
  typedef RScalarJIT<typename RealScalar<T>::Type_t>  Type_t;
};

template<class T>
struct RealScalar<RComplexJIT<T> > {
  typedef RScalarJIT<typename RealScalar<T>::Type_t>  Type_t;
};


//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------




template<class T> 
inline
void zero_rep(RScalarJIT<T> dest) 
{
  zero_rep(dest.elem());
}


template<class T> 
inline
void zero_rep(RComplexJIT<T> dest) 
{
  zero_rep(dest.real());
  zero_rep(dest.imag());
}


//! dest  = random  
template<class T, class T1, class T2, class T3>
inline void
fill_random_jit(RScalarJIT<T> d, T1 seed, T2 skewed_seed, const T3& seed_mult)
{
  fill_random_jit(d.elem(), seed, skewed_seed, seed_mult);
}


template<class T, class T1, class T2, class T3>
inline void
fill_random_jit(RComplexJIT<T> d, T1 seed, T2 skewed_seed, const T3& seed_mult)
{
  fill_random_jit(d.real(), seed, skewed_seed, seed_mult);
  fill_random_jit(d.imag(), seed, skewed_seed, seed_mult);
}


template<class T,class T2>
inline void
fill_gaussian(RScalarJIT<T> d, RScalarREG<T2>& r1, RScalarREG<T2>& r2)
{
  typedef typename InternalScalar<T2>::Type_t  S;

  // r1 and r2 are the input random numbers needed

  /* Stage 2: get the cos of the second number  */
  T2  g_r;

  r2.elem() *= S(6.283185307);
  g_r = cos(r2.elem());
    
  /* Stage 4: get  sqrt(-2.0 * log(u1)) */
  r1.elem() = sqrt(-S(2.0) * log(r1.elem()));

  /* Stage 5:   g_r = sqrt(-2*log(u1))*cos(2*pi*u2) */
  /* Stage 5:   g_i = sqrt(-2*log(u1))*sin(2*pi*u2) */
  d.elem() = r1.elem() * g_r;
}


template<class T,class T2>
inline void
fill_gaussian(RComplexJIT<T> d, RComplexREG<T2>& r1, RComplexREG<T2>& r2)
{
  typedef typename InternalScalar<T2>::Type_t  S;

  // r1 and r2 are the input random numbers needed

  /* Stage 2: get the cos of the second number  */
  T2  g_r, g_i;

  r2.real() *= S(6.283185307);
  g_r = cos(r2.real());
  g_i = sin(r2.real());
    
  /* Stage 4: get  sqrt(-2.0 * log(u1)) */
  r1.real() = sqrt(-S(2.0) * log(r1.real()));

  /* Stage 5:   g_r = sqrt(-2*log(u1))*cos(2*pi*u2) */
  /* Stage 5:   g_i = sqrt(-2*log(u1))*sin(2*pi*u2) */
  d.real() = r1.real() * g_r;
  d.imag() = r1.real() * g_i;
}



/*! @} */  // end of group rcomplex

} // namespace QDP

#endif
