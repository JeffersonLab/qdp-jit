// -*- C++ -*-

/*! \file
 * \brief Primitive Scalar
 */

#ifndef QDP_PRIMSCALARJIT_H
#define QDP_PRIMSCALARJIT_H

namespace QDP {


//-------------------------------------------------------------------------------------
/*! \addtogroup primscalar Scalar primitive
 * \ingroup fiber
 *
 * Primitive Scalar is a placeholder for no primitive structure
 *
 * @{
 */

//! Primitive Scalar
/*! Placeholder for no primitive structure */
  template<class T> class PScalarJIT : public BaseJIT<T,1>
  {
  public:

    // Default constructing should be possible
    PScalarJIT() {}

    // Copy constructor
    PScalarJIT(const PScalarJIT& rhs): BaseJIT<T,1>(rhs)
    {}
    

    //~PScalarJIT() {}

    template<class T1>
    PScalarJIT& operator=( const PScalarREG<T1>& rhs) {
      elem() = rhs.elem();
      return *this;
    }



    //! PScalarJIT += PScalarJIT
    template<class T1>
    inline
    PScalarJIT& operator+=(const PScalarREG<T1>& rhs) 
    {
      elem() += rhs.elem();
      return *this;
    }

    //! PScalarJIT -= PScalarJIT
    template<class T1>
    inline
    PScalarJIT& operator-=(const PScalarREG<T1>& rhs) 
    {
      elem() -= rhs.elem();
      return *this;
    }

    //! PScalarJIT *= PScalarJIT
    template<class T1>
    inline
    PScalarJIT& operator*=(const PScalarREG<T1>& rhs) 
    {
      elem() *= rhs.elem();
      return *this;
    }

    //! PScalarJIT /= PScalarJIT
    template<class T1>
    inline
    PScalarJIT& operator/=(const PScalarREG<T1>& rhs) 
    {
      elem() /= rhs.elem();
      return *this;
    }

    //! PScalarJIT %= PScalarJIT
    template<class T1>
    inline
    PScalarJIT& operator%=(const PScalarREG<T1>& rhs) 
    {
      elem() %= rhs.elem();
      return *this;
    }

    //! PScalarJIT |= PScalarJIT
    template<class T1>
    inline
    PScalarJIT& operator|=(const PScalarREG<T1>& rhs) 
    {
      elem() |= rhs.elem();
      return *this;
    }

    //! PScalarJIT &= PScalarJIT
    template<class T1>
    inline
    PScalarJIT& operator&=(const PScalarREG<T1>& rhs) 
    {
      elem() &= rhs.elem();
      return *this;
    }

    //! PScalarJIT ^= PScalarJIT
    template<class T1>
    inline
    PScalarJIT& operator^=(const PScalarREG<T1>& rhs) 
    {
      elem() ^= rhs.elem();
      return *this;
    }

    //! PScalarJIT <<= PScalarJIT
    template<class T1>
    inline
    PScalarJIT& operator<<=(const PScalarREG<T1>& rhs) 
    {
      elem() <<= rhs.elem();
      return *this;
    }

    //! PScalarJIT >>= PScalarJIT
    template<class T1>
    inline
    PScalarJIT& operator>>=(const PScalarREG<T1>& rhs) 
    {
      elem() >>= rhs.elem();
      return *this;
    }

  
  public:
    inline       T& elem()       { return this->arrayF(0); }
    inline const T& elem() const { return this->arrayF(0); }
  };




// Input
//! Ascii input
template<class T>
inline
istream& operator>>(istream& s, PScalarJIT<T>& d)
{
  return s >> d.elem();
}

//! Ascii input
template<class T>
inline
StandardInputStream& operator>>(StandardInputStream& s, PScalarJIT<T>& d)
{
  return s >> d.elem();
}

// Output
//! Ascii output
template<class T>
inline
ostream& operator<<(ostream& s, const PScalarJIT<T>& d)
{
  return s << d.elem();
}

//! Ascii output
template<class T>
inline
StandardOutputStream& operator<<(StandardOutputStream& s, const PScalarJIT<T>& d)
{
  return s << d.elem();
}

//! Text input
template<class T>
inline
TextReader& operator>>(TextReader& txt, PScalarJIT<T>& d)
{
  return txt >> d.elem();
}

//! Text output
template<class T>
inline
TextWriter& operator<<(TextWriter& txt, const PScalarJIT<T>& d)
{
  return txt << d.elem();
}

#ifndef QDP_NO_LIBXML2
//! XML output
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, const PScalarJIT<T>& d)
{
  return xml << d.elem();
}

//! XML input
template<class T>
inline
void read(XMLReader& xml, const string& path, PScalarJIT<T>& d)
{
  read(xml, path, d.elem());
}
#endif

/*! @} */  // end of group primscalar


//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------


template<class T>
struct ScalarType<PScalarJIT<T> >
{
  typedef PScalarJIT< typename ScalarType<T>::Type_t > Type_t;
};


template<class T> 
struct REGType< PScalarJIT<T> >
{
  typedef PScalarREG<typename REGType<T>::Type_t>  Type_t;
};

template<class T> 
struct BASEType< PScalarJIT<T> >
{
  typedef PScalar<typename BASEType<T>::Type_t>  Type_t;
};


// Underlying word type
template<class T>
struct WordType<PScalarJIT<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

// Fixed Precision Types 
template<class T>
struct SinglePrecType<PScalarJIT<T> >
{
  typedef PScalarJIT< typename SinglePrecType<T>::Type_t > Type_t;
};

template<class T>
struct DoublePrecType<PScalarJIT<T> >
{
  typedef PScalarJIT< typename DoublePrecType<T>::Type_t > Type_t;
};

// Internally used scalars
template<class T>
struct InternalScalar<PScalarJIT<T> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};

// Internally used real scalars
template<class T>
struct RealScalar<PScalarJIT<T> > {
  typedef PScalarJIT<typename RealScalar<T>::Type_t>  Type_t;
};

// Makes a primitive scalar leaving grid alone
template<class T>
struct PrimitiveScalar<PScalarJIT<T> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T>
struct LatticeScalar<PScalarJIT<T> > {
  typedef PScalarJIT<typename LatticeScalar<T>::Type_t>  Type_t;
};



template<class T1, class T2>
inline PScalarJIT<T1>&
pokeColor(PScalarJIT<T1>& l, const PScalarREG<T2>& r, llvm::Value * row)
{
  pokeColor(l.elem(),r.elem(),row);
  return l;
}

template<class T1, class T2>
inline PScalarJIT<T1>&
pokeColor(PScalarJIT<T1>& l, const PScalarREG<T2>& r, llvm::Value * row, llvm::Value * col)
{
  pokeColor(l.elem(),r.elem(),row,col);
  return l;
}


//! dest = 0
template<class T> 
inline void 
zero_rep(PScalarJIT<T> dest) 
{
  zero_rep(dest.elem());
}


//! dest  = random  
template<class T, class T1, class T2,class T3>
inline void
fill_random_jit( PScalarJIT<T> d, T1 seed, T2 skewed_seed, const T3& seed_mult)
{
  fill_random_jit(d.elem(), seed, skewed_seed, seed_mult);
}


//! dest  = gaussian  
template<class T,class T2>
inline void
fill_gaussian(PScalarJIT<T> d, PScalarREG<T2>& r1, PScalarREG<T2>& r2)
{
  fill_gaussian(d.elem(), r1.elem(), r2.elem());
}



/*! @} */  // end of group primscalar

} // namespace QDP

#endif
