// -*- C++ -*-

/*! \file
 * \brief Reality
 */


#ifndef QDP_REALITYREG_H
#define QDP_REALITYREG_H

#include <sstream>

namespace QDP {




template<class T>
class RScalarREG //: public BaseREG<T,1,RScalarREG<T> >
{
  T F;
public:

  void setup(const RScalarJIT< typename JITType<T>::Type_t >& rhs ) {
    F.setup( rhs.elem() );
  }

  void setup_value(const RScalarJIT< typename JITType<T>::Type_t >& rhs ) {
    F.setup_value( rhs.elem() );
  }

  
  RScalarREG(const RScalarJIT< typename JITType<T>::Type_t >& rhs ) {
    setup( rhs.elem() );
  }

  RScalarREG(const typename WordType<T>::Type_t& rhs): F(rhs) {}

  
  // RScalarREG& operator=( const RScalarJIT< typename JITType<T>::Type_t >& rhs) {
  //   setup(rhs);
  //   return *this;
  // }


  // Default constructing should be possible
  // then there is no need for MPL index when
  // construction a PMatrix<T,N>
  RScalarREG() {}
  ~RScalarREG() {}

  template<class T1>
  RScalarREG& operator=( const RScalarREG<T1>& rhs) {
    elem() = rhs.elem();
    return *this;
  }

  RScalarREG& operator=( const RScalarREG& rhs) {
    elem() = rhs.elem();
    return *this;
  }

  RScalarREG& operator=( typename WordType<T>::Type_t rhs) {
    elem() = rhs;
    return *this;
  }


#if 0
  //---------------------------------------------------------
  //! construct dest = const
#endif

  //RScalarREG(const typename WordType<T>::Type_t& rhs) : JV<T,1>( NULL , rhs ) {}


  //! construct dest = rhs
  template<class T1>
  RScalarREG(const RScalarREG<T1>& rhs)  {
    elem() = rhs.elem();
  }

  RScalarREG(const RScalarREG& rhs)  {
    elem() = rhs.elem();
  }

  RScalarREG(const T& rhs) {
    elem() = rhs;
  }



  //! RScalarREG += RScalarREG
  template<class T1>
  inline
  RScalarREG& operator+=(const RScalarREG<T1>& rhs) 
    {
      elem() += rhs.elem();
      return *this;
    }

  RScalarREG& operator+=(const RScalarREG& rhs) 
    {
      elem() += rhs.elem();
      return *this;
    }

  //! RScalarREG -= RScalarREG
  template<class T1>
  inline
  RScalarREG& operator-=(const RScalarREG<T1>& rhs) 
    {
      elem() -= rhs.elem();
      return *this;
    }

  RScalarREG& operator-=(const RScalarREG& rhs) 
    {
      elem() -= rhs.elem();
      return *this;
    }

  //! RScalarREG *= RScalarREG
  template<class T1>
  inline
  RScalarREG& operator*=(const RScalarREG<T1>& rhs) 
    {
      elem() *= rhs.elem();
      return *this;
    }

  RScalarREG& operator*=(const RScalarREG& rhs) 
    {
      elem() *= rhs.elem();
      return *this;
    }

  //! RScalarREG /= RScalarREG
  template<class T1>
  inline
  RScalarREG& operator/=(const RScalarREG<T1>& rhs) 
    {
      elem() /= rhs.elem();
      return *this;
    }

  RScalarREG& operator/=(const RScalarREG& rhs) 
    {
      elem() /= rhs.elem();
      return *this;
    }

  //! RScalarREG %= RScalarREG
  template<class T1>
  inline
  RScalarREG& operator%=(const RScalarREG<T1>& rhs) 
    {
      elem() %= rhs.elem();
      return *this;
    }

  RScalarREG& operator%=(const RScalarREG& rhs) 
    {
      elem() %= rhs.elem();
      return *this;
    }

  //! RScalarREG |= RScalarREG
  template<class T1>
  inline
  RScalarREG& operator|=(const RScalarREG<T1>& rhs) 
    {
      elem() |= rhs.elem();
      return *this;
    }

  RScalarREG& operator|=(const RScalarREG& rhs) 
    {
      elem() |= rhs.elem();
      return *this;
    }

  //! RScalarREG &= RScalarREG
  template<class T1>
  inline
  RScalarREG& operator&=(const RScalarREG<T1>& rhs) 
    {
      elem() &= rhs.elem();
      return *this;
    }

  RScalarREG& operator&=(const RScalarREG& rhs) 
    {
      elem() &= rhs.elem();
      return *this;
    }

  //! RScalarREG ^= RScalarREG
  template<class T1>
  inline
  RScalarREG& operator^=(const RScalarREG<T1>& rhs) 
    {
      elem() ^= rhs.elem();
      return *this;
    }

  RScalarREG& operator^=(const RScalarREG& rhs) 
    {
      elem() ^= rhs.elem();
      return *this;
    }

  //! RScalarREG <<= RScalarREG
  template<class T1>
  inline
  RScalarREG& operator<<=(const RScalarREG<T1>& rhs) 
    {
      elem() <<= rhs.elem();
      return *this;
    }

  RScalarREG& operator<<=(const RScalarREG& rhs) 
    {
      elem() <<= rhs.elem();
      return *this;
    }

  //! RScalarREG >>= RScalarREG
  template<class T1>
  inline
  RScalarREG& operator>>=(const RScalarREG<T1>& rhs) 
    {
      elem() >>= rhs.elem();
      return *this;
    }

  RScalarREG& operator>>=(const RScalarREG& rhs) 
    {
      elem() >>= rhs.elem();
      return *this;
    }


public:
  inline       T& elem()       { return F; }
  inline const T& elem() const { return F; }

  // inline       T& elem()       { return this->arrayF(0); }
  // inline const T& elem() const { return this->arrayF(0); }
};

 
// Input
//! Ascii input
template<class T>
inline
istream& operator>>(istream& s, RScalarREG<T>& d)
{
  return s >> d.elem();
}

//! Ascii input
template<class T>
inline
StandardInputStream& operator>>(StandardInputStream& s, RScalarREG<T>& d)
{
  return s >> d.elem();
}

//! Ascii output
template<class T> 
inline  
ostream& operator<<(ostream& s, const RScalarREG<T>& d)
{
  return s << d.elem();
}

//! Ascii output
template<class T> 
inline  
StandardOutputStream& operator<<(StandardOutputStream& s, const RScalarREG<T>& d)
{
  return s << d.elem();
}


//! Text input
template<class T>
inline
TextReader& operator>>(TextReader& s, RScalarREG<T>& d)
{
  return s >> d.elem();
}

//! Text output
template<class T> 
inline  
TextWriter& operator<<(TextWriter& s, const RScalarREG<T>& d)
{
  return s << d.elem();
}

#ifndef QDP_NO_LIBXML2
//! XML output
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, const RScalarREG<T>& d)
{
  return xml << d.elem();
}

//! XML input
template<class T>
inline
void read(XMLReader& xml, const string& path, RScalarREG<T>& d)
{
  read(xml, path, d.elem());
}
#endif

/*! @} */  // end of group rscalar


//-------------------------------------------------------------------------------------
/*! \addtogroup rcomplex Complex reality
 * \ingroup fiber
 *
 * Reality Complex is a type for objects that hold a real and imaginary part
 *
 * @{
 */

template<class T>
class RComplexREG //: public BaseREG<T,2,RComplexREG<T> >
{
  T re,im;
public:

  void setup(const RComplexJIT< typename JITType<T>::Type_t >& rhs ) {
    re.setup( rhs.real() );
    im.setup( rhs.imag() );
  }

  void setup_value(const RComplexJIT< typename JITType<T>::Type_t >& rhs ) {
    re.setup_value( rhs.real() );
    im.setup_value( rhs.imag() );
  }

  
  // RComplexREG& operator=( const RComplexJIT< typename JITType<T>::Type_t >& rhs) {
  //   setup(rhs);
  //   return *this;
  // }

  RComplexREG( const RComplexJIT< typename JITType<T>::Type_t >& rhs) {
    setup(rhs);
  }


  // Default constructing should be possible
  // then there is no need for MPL index when
  // construction a PMatrix<T,N>
  RComplexREG() {}
  ~RComplexREG() {}

  //! Construct from two reality scalars
  template<class T1, class T2>
  RComplexREG(const RScalarREG<T1>& _re, const RScalarREG<T2>& _im) {
    real() = _re.elem();
    imag() = _im.elem();
  }

  //! Construct from two scalars
  //RComplexREG(Jit& j,const typename WordType<T>::Type_t& re, const typename WordType<T>::Type_t& im): JV<T,2>(j,re,im) {}


  RComplexREG(const T& re,const T& im) {
    real() = re;
    imag() = im;
  }


  //! RComplexREG += RScalarREG
  template<class T1>
  inline
  RComplexREG& operator+=(const RScalarREG<T1>& rhs) 
    {
      real() += rhs.elem();
      return *this;
    }

  //! RComplexREG -= RScalarREG
  template<class T1>
  inline
  RComplexREG& operator-=(const RScalarREG<T1>& rhs) 
    {
      real() -= rhs.elem();
      return *this;
    }

  //! RComplexREG *= RScalarREG
  template<class T1>
  inline
  RComplexREG& operator*=(const RScalarREG<T1>& rhs) 
    {
      real() *= rhs.elem();
      imag() *= rhs.elem();
      return *this;
    }

  //! RComplexREG /= RScalarREG
  template<class T1>
  inline
  RComplexREG& operator/=(const RScalarREG<T1>& rhs) 
    {
      real() /= rhs.elem();
      imag() /= rhs.elem();
      return *this;
    }

  //! RComplexREG += RComplexREG
  template<class T1>
  inline
  RComplexREG& operator+=(const RComplexREG<T1>& rhs) 
    {
      real() += rhs.real();
      imag() += rhs.imag();
      return *this;
    }

  RComplexREG& operator+=(const RComplexREG& rhs) 
    {
      real() += rhs.real();
      imag() += rhs.imag();
      return *this;
    }

  //! RComplexREG -= RComplexREG
  template<class T1>
  inline
  RComplexREG& operator-=(const RComplexREG<T1>& rhs) 
    {
      real() -= rhs.real();
      imag() -= rhs.imag();
      return *this;
    }

  RComplexREG& operator-=(const RComplexREG& rhs) 
    {
      real() -= rhs.real();
      imag() -= rhs.imag();
      return *this;
    }

  //! RComplexREG *= RComplexREG
  template<class T1>
  inline
  RComplexREG& operator*=(const RComplexREG<T1>& rhs) 
    {
      RComplexREG<T> d;
      d = *this * rhs;

      real() = d.real();
      imag() = d.imag();
      return *this;
    }


  RComplexREG& operator*=(const RComplexREG& rhs)
    {
      RComplexREG<T> d;
      d = *this * rhs;

      real() = d.real();
      imag() = d.imag();
      return *this;
    }

  //! RComplexREG /= RComplexREG
  template<class T1>
  inline
  RComplexREG& operator/=(const RComplexREG<T1>& rhs) 
    {
      RComplexREG<T> d;
      d = *this / rhs;

      real() = d.real();
      imag() = d.imag();
      return *this;
    }

  RComplexREG& operator/=(const RComplexREG& rhs) 
    {
      RComplexREG<T> d;
      d = *this / rhs;

      real() = d.real();
      imag() = d.imag();
      return *this;
    }

  template<class T1>
  RComplexREG& operator=(const RComplexREG<T1>& rhs) 
    {
      real() = rhs.real();
      imag() = rhs.imag();
      return *this;
    }


  RComplexREG& operator=(const RComplexREG& rhs) 
    {
      real() = rhs.real();
      imag() = rhs.imag();
      return *this;
    }

  template<class T1>
  inline
  RComplexREG& operator=(const RScalarREG<T1>& rhs) 
    {
      real() = rhs.elem();
      zero_rep(imag());
      return *this;
    }


public:
  inline       T& real()       { return re; }
  inline const T& real() const { return re; }

  inline       T& imag()       { return im; }
  inline const T& imag() const { return im; }
};






//! Stream output
template<class T>
inline
ostream& operator<<(ostream& s, const RComplexREG<T>& d)
{
  s << "( " << d.real() << " , " << d.imag() << " )";
  return s;
}

//! Stream output
template<class T>
inline
StandardOutputStream& operator<<(StandardOutputStream& s, const RComplexREG<T>& d)
{
  s << "( " << d.real() << " , " << d.imag() << " )";
  return s;
}

//! Text input
template<class T>
inline
TextReader& operator>>(TextReader& s, RComplexREG<T>& d)
{
  return s >> d.real() >> d.imag();
}

//! Text output
template<class T> 
inline  
TextWriter& operator<<(TextWriter& s, const RComplexREG<T>& d)
{
  return s << d.real() << d.imag();
}

#ifndef QDP_NO_LIBXML2
//! XML output
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, const RComplexREG<T>& d)
{
  xml.openTag("re");
  xml << d.real();
  xml.closeTag();
  xml.openTag("im");
  xml << d.imag();
  xml.closeTag();

  return xml;
}

//! XML input
template<class T>
inline
void read(XMLReader& xml, const string& xpath, RComplexREG<T>& d)
{
  std::ostringstream error_message;
  
  // XPath for the real part 
  string path_real = xpath + "/re";
	
  // XPath for the imaginary part.
  string path_imag = xpath + "/im";
	
  // Try and recursively get the real part
  try { 
    read(xml, path_real, d.real());
  }
  catch(const string &e) {
    error_message << "XPath Query: " << xpath << " Error: "
		  << "Failed to match real part of RComplexREG Object with self constructed path: " << path_real;
    
    throw error_message.str();
  }
	
  // Try and recursively get the imaginary part
  try {
    read(xml, path_imag, d.imag());
  }
  catch(const string &e) {
    error_message << "XPath Query: " << xpath <<" Error:"
		  <<"Failed to match imaginary part of RComplexREG Object with self constructed path: " << path_imag;
    
    throw error_message.str();
  }
}
#endif

/*! @} */   // end of group rcomplex

//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

template<class T> 
struct JITType< RScalarREG<T> >
{
  typedef RScalarJIT<typename JITType<T>::Type_t>  Type_t;
};

template<class T> 
struct JITType< RComplexREG<T> >
{
  typedef RComplexJIT<typename JITType<T>::Type_t>  Type_t;
};


// Underlying word type
template<class T>
struct WordType<RScalarREG<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

template<class T>
struct WordType<RComplexREG<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

// Fixed types
template<class T> 
struct SinglePrecType<RScalarREG<T> >
{
  typedef RScalarREG<typename SinglePrecType<T>::Type_t>  Type_t;
};

template<class T> 
struct SinglePrecType<RComplexREG<T> >
{
  typedef RComplexREG<typename SinglePrecType<T>::Type_t>  Type_t;
};

template<class T> 
struct DoublePrecType<RScalarREG<T> >
{
  typedef RScalarREG<typename DoublePrecType<T>::Type_t>  Type_t;
};

template<class T> 
struct DoublePrecType<RComplexREG<T> >
{
  typedef RComplexREG<typename DoublePrecType<T>::Type_t>  Type_t;
};


// Internally used scalars
template<class T>
struct InternalScalar<RScalarREG<T> > {
  typedef RScalarREG<typename InternalScalar<T>::Type_t>  Type_t;
};

template<class T>
struct InternalScalar<RComplexREG<T> > {
  typedef RScalarREG<typename InternalScalar<T>::Type_t>  Type_t;
};


// Makes a primitive scalar leaving grid alone
template<class T>
struct PrimitiveScalar<RScalarREG<T> > {
  typedef RScalarREG<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

template<class T>
struct PrimitiveScalar<RComplexREG<T> > {
  typedef RScalarREG<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T>
struct LatticeScalar<RScalarREG<T> > {
  typedef RScalarREG<typename LatticeScalar<T>::Type_t>  Type_t;
};

template<class T>
struct LatticeScalar<RComplexREG<T> > {
  typedef RComplexREG<typename LatticeScalar<T>::Type_t>  Type_t;
};


// Internally used real scalars
template<class T>
struct RealScalar<RScalarREG<T> > {
  typedef RScalarREG<typename RealScalar<T>::Type_t>  Type_t;
};

template<class T>
struct RealScalar<RComplexREG<T> > {
  typedef RScalarREG<typename RealScalar<T>::Type_t>  Type_t;
};


//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(RScalarREG) -> RScalarREG
template<class T1, class Op>
struct UnaryReturn<RScalarREG<T1>, Op> {
  typedef RScalarREG<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

// Default unary(RComplexREG) -> RComplexREG
template<class T1, class Op>
struct UnaryReturn<RComplexREG<T1>, Op> {
  typedef RComplexREG<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

// Default binary(RScalarREG,RScalarREG) -> RScalarREG
template<class T1, class T2, class Op>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, Op> {
  typedef RScalarREG<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Default binary(RComplexREG,RComplexREG) -> RComplexREG
template<class T1, class T2, class Op>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, Op> {
  typedef RComplexREG<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Default binary(RScalarREG,RComplexREG) -> RComplexREG
template<class T1, class T2, class Op>
struct BinaryReturn<RScalarREG<T1>, RComplexREG<T2>, Op> {
  typedef RComplexREG<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Default binary(RComplexREG,RScalarREG) -> RComplexREG
template<class T1, class T2, class Op>
struct BinaryReturn<RComplexREG<T1>, RScalarREG<T2>, Op> {
  typedef RComplexREG<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};




// RScalarREG
#if 0
template<class T1, class T2>
struct UnaryReturn<RScalarREG<T2>, OpCast<T1> > {
  typedef RScalarREG<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif


template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpAddAssign > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpSubtractAssign > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpMultiplyAssign > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpDivideAssign > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpModAssign > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpBitwiseOrAssign > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpBitwiseAndAssign > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpBitwiseXorAssign > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpLeftShiftAssign > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpRightShiftAssign > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2, class T3>
struct TrinaryReturn<RScalarREG<T1>, RScalarREG<T2>, RScalarREG<T3>, FnColorContract> {
  typedef RScalarREG<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};

// RScalarREG
// Gamma algebra
template<int N, int m, class T2, class OpGammaConstMultiply>
struct BinaryReturn<GammaConst<N,m>, RScalarREG<T2>, OpGammaConstMultiply> {
  typedef RScalarREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConst>
struct BinaryReturn<RScalarREG<T2>, GammaConst<N,m>, OpMultiplyGammaConst> {
  typedef RScalarREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeMultiply>
struct BinaryReturn<GammaType<N>, RScalarREG<T2>, OpGammaTypeMultiply> {
  typedef RScalarREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaType>
struct BinaryReturn<RScalarREG<T2>, GammaType<N>, OpMultiplyGammaType> {
  typedef RScalarREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};


// RScalarREG
// Gamma algebra
template<int N, int m, class T2, class OpGammaConstDPMultiply>
struct BinaryReturn<GammaConstDP<N,m>, RScalarREG<T2>, OpGammaConstDPMultiply> {
  typedef RScalarREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConstDP>
struct BinaryReturn<RScalarREG<T2>, GammaConstDP<N,m>, OpMultiplyGammaConstDP> {
  typedef RScalarREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeDPMultiply>
struct BinaryReturn<GammaTypeDP<N>, RScalarREG<T2>, OpGammaTypeDPMultiply> {
  typedef RScalarREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaTypeDP>
struct BinaryReturn<RScalarREG<T2>, GammaTypeDP<N>, OpMultiplyGammaTypeDP> {
  typedef RScalarREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};



// RComplexREG
// Gamma algebra
template<int N, int m, class T2, class OpGammaConstMultiply>
struct BinaryReturn<GammaConst<N,m>, RComplexREG<T2>, OpGammaConstMultiply> {
  typedef RComplexREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConst>
struct BinaryReturn<RComplexREG<T2>, GammaConst<N,m>, OpMultiplyGammaConst> {
  typedef RComplexREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeMultiply>
struct BinaryReturn<GammaType<N>, RComplexREG<T2>, OpGammaTypeMultiply> {
  typedef RComplexREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaType>
struct BinaryReturn<RComplexREG<T2>, GammaType<N>, OpMultiplyGammaType> {
  typedef RComplexREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};


// Gamma algebra
template<int N, int m, class T2, class OpGammaConstDPMultiply>
struct BinaryReturn<GammaConstDP<N,m>, RComplexREG<T2>, OpGammaConstDPMultiply> {
  typedef RComplexREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConstDP>
struct BinaryReturn<RComplexREG<T2>, GammaConstDP<N,m>, OpMultiplyGammaConstDP> {
  typedef RComplexREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeDPMultiply>
struct BinaryReturn<GammaTypeDP<N>, RComplexREG<T2>, OpGammaTypeDPMultiply> {
  typedef RComplexREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaTypeDP>
struct BinaryReturn<RComplexREG<T2>, GammaTypeDP<N>, OpMultiplyGammaTypeDP> {
  typedef RComplexREG<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};


// Assignment is different
template<class T1, class T2 >
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpAssign > {
//  typedef RComplexREG<T1> &Type_t;
  typedef RComplexREG<typename BinaryReturn<T1, T2, OpAssign>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpAddAssign > {
  typedef RComplexREG<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpSubtractAssign > {
  typedef RComplexREG<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpMultiplyAssign > {
  typedef RComplexREG<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpDivideAssign > {
  typedef RComplexREG<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpModAssign > {
  typedef RComplexREG<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpBitwiseOrAssign > {
  typedef RComplexREG<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpBitwiseAndAssign > {
  typedef RComplexREG<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpBitwiseXorAssign > {
  typedef RComplexREG<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpLeftShiftAssign > {
  typedef RComplexREG<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpRightShiftAssign > {
  typedef RComplexREG<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2, class T3>
struct TrinaryReturn<RComplexREG<T1>, RComplexREG<T2>, RComplexREG<T3>, FnColorContract> {
  typedef RComplexREG<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};






//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

/*! \addtogroup rscalar
 * @{ 
 */

// Scalar Reality
template<class T>
struct UnaryReturn<RScalarREG<T>, OpNot > {
  typedef RScalarREG<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, OpNot>::Type_t
operator!(const RScalarREG<T1>& l)
{
  return ! l.elem();
}


template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, OpUnaryPlus>::Type_t
operator+(const RScalarREG<T1>& l)
{
  return +l.elem();
}


template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, OpUnaryMinus>::Type_t
operator-(const RScalarREG<T1>& l)
{
  return -l.elem();
}


template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpAdd>::Type_t
operator+(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() + r.elem();
}


template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpSubtract>::Type_t
operator-(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() - r.elem();
}


template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpMultiply>::Type_t
operator*(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpMultiply>::Type_t ret;
  ret = l.elem() * r.elem();
  return ret;
}



// Optimized  adj(RScalarREG)*RScalarREG
template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpAdjMultiply>::Type_t
adjMultiply(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  /*! NOTE: removed transpose here !!!!!  */

//  return transpose(l.elem()) * r.elem();
  return l.elem() * r.elem();
}

// Optimized  RScalarREG*adj(RScalarREG)
template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  /*! NOTE: removed transpose here !!!!!  */

//  return l.elem() * transpose(r.elem());
  return l.elem() * r.elem();
}

// Optimized  adj(RScalarREG)*adj(RScalarREG)
template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  /*! NOTE: removed transpose here !!!!!  */

//  return transpose(l.elem()) * transpose(r.elem());
  return l.elem() * r.elem();
}


template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpDivide>::Type_t
operator/(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() / r.elem();
}



template<class T1, class T2 >
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpLeftShift > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpLeftShift>::Type_t
operator<<(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() << r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpRightShift > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpRightShift>::Type_t
operator>>(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() >> r.elem();
}


template<class T1, class T2 >
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpMod>::Type_t
operator%(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() % r.elem();
}

template<class T1, class T2 >
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpBitwiseXor>::Type_t
operator^(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() ^ r.elem();
}

template<class T1, class T2 >
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpBitwiseAnd>::Type_t
operator&(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() & r.elem();
}

template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpBitwiseOr>::Type_t
operator|(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() | r.elem();
}



// Comparisons
template<class T1, class T2 >
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpLT > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpLT>::Type_t
operator<(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() < r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpLE > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpLE>::Type_t
operator<=(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() <= r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpGT > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpGT>::Type_t
operator>(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() > r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpGE > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpGE>::Type_t
operator>=(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() >= r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpEQ > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpEQ>::Type_t
operator==(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() == r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpNE > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpNE>::Type_t
operator!=(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() != r.elem();
}


template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpAnd > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpAnd>::Type_t
operator&&(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() && r.elem();
}


template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpOr > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, OpOr>::Type_t
operator||(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() || r.elem();
}



//-----------------------------------------------------------------------------
// Functions

// Adjoint
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnAdjoint>::Type_t
adj(const RScalarREG<T1>& s1)
{
  /*! NOTE: removed transpose here !!!!!  */

//  return transpose(s1.elem()); // The complex nature has been eaten here
  return s1.elem(); // The complex nature has been eaten here
}


// Conjugate
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnConjugate>::Type_t
conj(const RScalarREG<T1>& s1)
{
  return s1.elem();  // The complex nature has been eaten here
}


// Transpose
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnTranspose>::Type_t
transpose(const RScalarREG<T1>& s1)
{
  /*! NOTE: removed transpose here !!!!!  */

//  return transpose(s1.elem());
  return s1.elem();
}



// TRACE
// trace = Trace(source1)
template<class T>
struct UnaryReturn<RScalarREG<T>, FnTrace > {
  typedef RScalarREG<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnTrace>::Type_t
trace(const RScalarREG<T1>& s1)
{
//  return trace(s1.elem());

  /*! NOTE: removed trace here !!!!!  */
  return s1.elem();
}


// trace = Re(Trace(source1))
template<class T>
struct UnaryReturn<RScalarREG<T>, FnRealTrace > {
  typedef RScalarREG<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnRealTrace>::Type_t
realTrace(const RScalarREG<T1>& s1)
{
//  return trace_real(s1.elem());

  /*! NOTE: removed trace here !!!!!  */
  return s1.elem();
}


// trace = Im(Trace(source1))
template<class T>
struct UnaryReturn<RScalarREG<T>, FnImagTrace > {
  typedef RScalarREG<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnImagTrace>::Type_t
imagTrace(const RScalarREG<T1>& s1)
{
//  return trace_imag(s1.elem());

  /*! NOTE: removed trace here !!!!!  */
  return s1.elem();
}

//! RScalarREG = trace(RScalarREG * RScalarREG)
template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, FnTraceMultiply>::Type_t
traceMultiply(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
//  return traceMultiply(l.elem(), r.elem());

  /*! NOTE: removed trace here !!!!!  */
  return l.elem() * r.elem();
}


// RScalarREG = Re(RScalarREG)  [identity]
template<class T>
inline typename UnaryReturn<RScalarREG<T>, FnReal>::Type_t
real(const RScalarREG<T>& s1)
{
  return s1.elem();
}


// RScalarREG = Im(RScalarREG) [this is zero]
template<class T>
inline typename UnaryReturn<RScalarREG<T>, FnImag>::Type_t
imag(const RScalarREG<T>& s1)
{
  typedef typename InternalScalar<T>::Type_t  S;
  return S(0);
}


// ArcCos
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnArcCos>::Type_t
acos(const RScalarREG<T1>& s1)
{
  return acos(s1.elem());
}

// ArcSin
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnArcSin>::Type_t
asin(const RScalarREG<T1>& s1)
{
  return asin(s1.elem());
}

// ArcTan
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnArcTan>::Type_t
atan(const RScalarREG<T1>& s1)
{
  return atan(s1.elem());
}

// Ceil(ing)
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnCeil>::Type_t
ceil(const RScalarREG<T1>& s1)
{
  return ceil(s1.elem());
}

// Cos
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnCos>::Type_t
cos(const RScalarREG<T1>& s1)
{
  return cos(s1.elem());
}

// Cosh
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnHypCos>::Type_t
cosh(const RScalarREG<T1>& s1)
{
  return cosh(s1.elem());
}

// Exp
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnExp>::Type_t
exp(const RScalarREG<T1>& s1)
{
  return exp(s1.elem());
}

// Fabs
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnFabs>::Type_t
fabs(const RScalarREG<T1>& s1)
{
  return fabs(s1.elem());
}

// Floor
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnFloor>::Type_t
floor(const RScalarREG<T1>& s1)
{
  return floor(s1.elem());
}

// Log
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnLog>::Type_t
log(const RScalarREG<T1>& s1)
{
  return log(s1.elem());
}

// Log10
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnLog10>::Type_t
log10(const RScalarREG<T1>& s1)
{
  return log10(s1.elem());
}

// Sin
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnSin>::Type_t
sin(const RScalarREG<T1>& s1)
{
  return sin(s1.elem());
}

// Sinh
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnHypSin>::Type_t
sinh(const RScalarREG<T1>& s1)
{
  return sinh(s1.elem());
}

// Sqrt
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnSqrt>::Type_t
sqrt(const RScalarREG<T1>& s1)
{
  return sqrt(s1.elem());
}


template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnIsFinite>::Type_t
isfinite(const RScalarREG<T1>& s1)
{
  return isfinite(s1.elem());
}


  

// Tan
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnTan>::Type_t
tan(const RScalarREG<T1>& s1)
{
  return tan(s1.elem());
}

// Tanh
template<class T1>
inline typename UnaryReturn<RScalarREG<T1>, FnHypTan>::Type_t
tanh(const RScalarREG<T1>& s1)
{
  return tanh(s1.elem());
}


//! RScalarREG<T> = pow(RScalarREG<T> , RScalarREG<T>)
template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, FnPow>::Type_t
pow(const RScalarREG<T1>& s1, const RScalarREG<T2>& s2)
{
  return pow(s1.elem(), s2.elem());
}

//! RScalarREG<T> = atan2(RScalarREG<T> , RScalarREG<T>)
template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, FnArcTan2>::Type_t
atan2(const RScalarREG<T1>& s1, const RScalarREG<T2>& s2)
{
  return atan2(s1.elem(), s2.elem());
}


//! RScalarREG = outerProduct(RScalarREG, RScalarREG)
template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, FnOuterProduct>::Type_t
outerProduct(const RScalarREG<T1>& l, const RScalarREG<T2>& r)
{
  return l.elem() * r.elem();
}



//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T>
inline typename UnaryReturn<RScalarREG<T>, FnGetSite>::Type_t
getSite(const RScalarREG<T>& s1, int innersite)
{
  return getSite(s1.elem(), innersite);
}

//! Extract color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T>
inline typename UnaryReturn<RScalarREG<T>, FnPeekColorVector>::Type_t
peekColor(const RScalarREG<T>& l, int row)
{
  return peekColor(l.elem(),row);
}

//! Extract color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T>
inline typename UnaryReturn<RScalarREG<T>, FnPeekColorMatrix>::Type_t
peekColor(const RScalarREG<T>& l, int row, int col)
{
  return peekColor(l.elem(),row,col);
}

//! Extract spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
inline typename UnaryReturn<RScalarREG<T>, FnPeekSpinVector>::Type_t
peekSpin(const RScalarREG<T>& l, int row)
{
  return peekSpin(l.elem(),row);
}

//! Extract spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
inline typename UnaryReturn<RScalarREG<T>, FnPeekSpinMatrix>::Type_t
peekSpin(const RScalarREG<T>& l, int row, int col)
{
  return peekSpin(l.elem(),row,col);
}


//------------------------------------------
//! dest = (mask) ? s1 : dest
template<class T, class T1> 
inline
void copymask(RScalarREG<T>& d, const RScalarREG<T1>& mask, const RScalarREG<T>& s1) 
{
  copymask(d.elem(),mask.elem(),s1.elem());
}

//! dest [float type] = source [int type]
template<class T, class T1>
inline
void cast_rep(T& d, const RScalarREG<T1>& s1)
{
  cast_rep(d, s1.elem());
}


//! dest [float type] = source [int type]
template<class T, class T1>
inline
void recast_rep(RScalarREG<T>& d, const RScalarREG<T1>& s1)
{
  cast_rep(d.elem(), s1.elem());
}


//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
copy_site(RScalarREG<T>& d, int isite, const RScalarREG<T1>& s1)
{
  copy_site(d.elem(), isite, s1.elem());
}


//! gather several inner sites together
template<class T, class T1>
inline void 
gather_sites(RScalarREG<T>& d, 
	     const RScalarREG<T1>& s0, int i0, 
	     const RScalarREG<T1>& s1, int i1,
	     const RScalarREG<T1>& s2, int i2,
	     const RScalarREG<T1>& s3, int i3)
{
  gather_sites(d.elem(), 
	       s0.elem(), i0, 
	       s1.elem(), i1, 
	       s2.elem(), i2, 
	       s3.elem(), i3);
}


#if 1
// Global sum over site indices only
template<class T>
struct UnaryReturn<RScalarREG<T>, FnSum > {
  typedef RScalarREG<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalarREG<T>, FnSum>::Type_t
sum(const RScalarREG<T>& s1)
{
  return sum(s1.elem());
}
#endif


// Global max
template<class T>
struct UnaryReturn<RScalarREG<T>, FnGlobalMax> {
  typedef RScalarREG<typename UnaryReturn<T, FnGlobalMax>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalarREG<T>, FnGlobalMax>::Type_t
globalMax(const RScalarREG<T>& s1)
{
  return globalMax(s1.elem());
}


// Global min
template<class T>
struct UnaryReturn<RScalarREG<T>, FnGlobalMin> {
  typedef RScalarREG<typename UnaryReturn<T, FnGlobalMin>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalarREG<T>, FnGlobalMin>::Type_t
globalMin(const RScalarREG<T>& s1)
{
  return globalMin(s1.elem());
}



//------------------------------------------
// InnerProduct (norm-seq) global sum = sum(tr(adj(s1)*s1))
template<class T>
struct UnaryReturn<RScalarREG<T>, FnNorm2 > {
  typedef RScalarREG<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<RScalarREG<T>, FnLocalNorm2 > {
  typedef RScalarREG<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalarREG<T>, FnLocalNorm2>::Type_t
localNorm2(const RScalarREG<T>& s1)
{
  return localNorm2(s1.elem());
}



//! RScalarREG<T> = InnerProduct(adj(RScalarREG<T1>)*RScalarREG<T2>)
template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, FnInnerProduct > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, FnLocalInnerProduct > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(const RScalarREG<T1>& s1, const RScalarREG<T2>& s2)
{
  return localInnerProduct(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, FnLocalColorInnerProduct>::Type_t
localColorInnerProduct(const RScalarREG<T1>& s1, const RScalarREG<T2>& s2)
{
  return localColorInnerProduct(s1.elem(), s2.elem());
}


//! RScalarREG<T> = InnerProductReal(adj(PMatrix<T1>)*PMatrix<T1>)
// Real-ness is eaten at this level
template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, FnInnerProductReal > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, FnLocalInnerProductReal > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const RScalarREG<T1>& s1, const RScalarREG<T2>& s2)
{
  return localInnerProduct(s1.elem(), s2.elem());
}


//! RScalarREG<T> = where(RScalarREG, RScalarREG, RScalarREG)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3>
struct TrinaryReturn<RScalarREG<T1>, RScalarREG<T2>, RScalarREG<T3>, FnWhere> {
  typedef RScalarREG<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
inline typename TrinaryReturn<RScalarREG<T1>, RScalarREG<T2>, RScalarREG<T3>, FnWhere>::Type_t
where(const RScalarREG<T1>& a, const RScalarREG<T2>& b, const RScalarREG<T3>& c)
{
  return where(a.elem(), b.elem(), c.elem());
}



//-----------------------------------------------------------------------------
// Broadcast operations
//! dest = 0
template<class T> 
inline
void zero_rep(RScalarREG<T>& dest) 
{
  zero_rep(dest.elem());
}


//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
copy_site(RComplexREG<T>& d, int isite, const RComplexREG<T1>& s1)
{
  copy_site(d.real(), isite, s1.real());
  copy_site(d.imag(), isite, s1.imag());
}

#if 0
//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
copy_site(RComplexREG<T>& d, int isite, const RScalarREG<T1>& s1)
{
  copy_site(d.real(), isite, s1.elem());
  zero_rep(d.imag());   // this is wrong - want zero only at a site. Fix when needed.
}
#endif


//! gather several inner sites together
template<class T, class T1>
inline void 
gather_sites(RComplexREG<T>& d, 
	     const RComplexREG<T1>& s0, int i0, 
	     const RComplexREG<T1>& s1, int i1,
	     const RComplexREG<T1>& s2, int i2,
	     const RComplexREG<T1>& s3, int i3)
{
  gather_sites(d.real(), 
	       s0.real(), i0, 
	       s1.real(), i1, 
	       s2.real(), i2, 
	       s3.real(), i3);

  gather_sites(d.imag(), 
	       s0.imag(), i0, 
	       s1.imag(), i1, 
	       s2.imag(), i2, 
	       s3.imag(), i3);
}


//! dest  = random  
template<class T, class T1, class T2, class T3>
inline void
fill_random(RScalarREG<T>& d, T1& seed, T2& skewed_seed, const T3& seed_mult)
{
  fill_random(d.elem(), seed, skewed_seed, seed_mult);
}



//! dest  = gaussian  
/*! Real form of complex polar method */
template<class T>
inline void
fill_gaussian( RScalarREG<T>& d, RScalarREG<T>& r1, RScalarREG<T>& r2)
{
  T w_2pi;
  T w_2;
  T w_g_r;
  T w_r1;
  T w_r2;

  w_r1 = r1.elem();
  w_r2 = r2.elem();

  w_2pi = (float)6.283185307;
  w_2 = (float)2.0;

  w_r2 *= w_2pi;
  w_g_r = cos(w_r2);

  w_r1 = sqrt( -w_2 * log(w_r1) );

  d.elem() = w_r1 * w_g_r;

  //  fill_gaussian(d.elem(), r1.elem(), r2.elem());
}

/*! @} */   // end of group rscalar



//-----------------------------------------------------------------------------
// Complex Reality
//-----------------------------------------------------------------------------

/*! \addtogroup rcomplex 
 * @{ 
 */

//! RComplexREG = +RComplexREG
template<class T1>
inline typename UnaryReturn<RComplexREG<T1>, OpUnaryPlus>::Type_t
operator+(const RComplexREG<T1>& l)
{
  typedef typename UnaryReturn<RComplexREG<T1>, OpUnaryPlus>::Type_t  Ret_t;

  return Ret_t(+l.real(),
	       +l.imag());
}


//! RComplexREG = -RComplexREG
template<class T1>
inline typename UnaryReturn<RComplexREG<T1>, OpUnaryMinus>::Type_t
operator-(const RComplexREG<T1>& l)
{
  typedef typename UnaryReturn<RComplexREG<T1>, OpUnaryMinus>::Type_t  Ret_t;

  return Ret_t(-l.real(),
	       -l.imag());
}



//! RComplexREG = RComplexREG - RComplexREG
template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpAdd>::Type_t
operator+(const RComplexREG<T1>& l, const RComplexREG<T2>& r)
{
  typedef typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpAdd>::Type_t  Ret_t;

  return Ret_t(l.real() + r.real(),
	       l.imag() + r.imag());
}


//! RComplexREG = RComplexREG + RScalarREG
template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RScalarREG<T2>, OpAdd>::Type_t
operator+(const RComplexREG<T1>& l, const RScalarREG<T2>& r)
{
  typedef typename BinaryReturn<RComplexREG<T1>, RScalarREG<T2>, OpAdd>::Type_t  Ret_t;

  return Ret_t(l.real()+r.elem(),
	       l.imag());
}

//! RComplexREG = RScalarREG + RComplexREG
template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RComplexREG<T2>, OpAdd>::Type_t
operator+(const RScalarREG<T1>& l, const RComplexREG<T2>& r)
{
  typedef typename BinaryReturn<RScalarREG<T1>, RComplexREG<T2>, OpAdd>::Type_t  Ret_t;

  return Ret_t(l.elem()+r.real(),
	       r.imag());
}


//! RComplexREG = RComplexREG - RComplexREG
template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpSubtract>::Type_t
operator-(const RComplexREG<T1>& l, const RComplexREG<T2>& r)
{
  typedef typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpSubtract>::Type_t  Ret_t;

  return Ret_t(l.real() - r.real(),
	       l.imag() - r.imag());
}

//! RComplexREG = RComplexREG - RScalarREG
template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RScalarREG<T2>, OpSubtract>::Type_t
operator-(const RComplexREG<T1>& l, const RScalarREG<T2>& r)
{
  typedef typename BinaryReturn<RComplexREG<T1>, RScalarREG<T2>, OpSubtract>::Type_t  Ret_t;

  return Ret_t(l.real() - r.elem(),
	       l.imag());
}

//! RComplexREG = RScalarREG - RComplexREG
template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RComplexREG<T2>, OpSubtract>::Type_t
operator-(const RScalarREG<T1>& l, const RComplexREG<T2>& r)
{
  typedef typename BinaryReturn<RScalarREG<T1>, RComplexREG<T2>, OpSubtract>::Type_t  Ret_t;

  return Ret_t(l.elem() - r.real(),
	       - r.imag());
}





template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpMultiply>::Type_t
operator*(const RComplexREG<T1>& l, const RComplexREG<T2>& r) 
{
#if 1
  typedef typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpMultiply>::Type_t  Ret_t;

  return Ret_t(l.real()*r.real() - l.imag()*r.imag(),
	       l.real()*r.imag() + l.imag()*r.real());
#else
  typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpMultiply>::Type_t ret;
  ret.real() = l.real()*r.real() - l.imag()*r.imag();
  ret.imag() = l.real()*r.imag() + l.imag()*r.real();
  return ret;
#endif
}




//! RComplexREG = RScalarREG * RComplexREG
template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RComplexREG<T2>, OpMultiply>::Type_t
operator*(const RScalarREG<T1>& l, const RComplexREG<T2>& r)
{
  typedef typename BinaryReturn<RScalarREG<T1>, RComplexREG<T2>, OpMultiply>::Type_t  Ret_t;

  return Ret_t(l.elem()*r.real(), 
	       l.elem()*r.imag());
}

//! RComplexREG = RComplexREG * RScalarREG
template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RScalarREG<T2>, OpMultiply>::Type_t
operator*(const RComplexREG<T1>& l, const RScalarREG<T2>& r)
{
  typedef typename BinaryReturn<RComplexREG<T1>, RScalarREG<T2>, OpMultiply>::Type_t  Ret_t;

  return Ret_t(l.real()*r.elem(), 
	       l.imag()*r.elem());
}


// Optimized  adj(RComplexREG)*RComplexREG
template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpAdjMultiply>::Type_t
adjMultiply(const RComplexREG<T1>& l, const RComplexREG<T2>& r)
{
  typedef typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpAdjMultiply>::Type_t  Ret_t;

  // The complex conjugate nature has been eaten here leaving simple multiples
  // involving transposes - which are probably null
  
//  d.real() = transpose(l.real())*r.real() + transpose(l.imag())*r.imag();
//  d.imag() = transpose(l.real())*r.imag() - transpose(l.imag())*r.real();
//  return d;

  /*! NOTE: removed transpose here !!!!!  */
  return Ret_t(l.real()*r.real() + l.imag()*r.imag(),
	       l.real()*r.imag() - l.imag()*r.real());
}


template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(const RComplexREG<T1>& l, const RComplexREG<T2>& r)
{
  typedef typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpMultiplyAdj>::Type_t  Ret_t;

  return Ret_t(l.real()*r.real() + l.imag()*r.imag(),
	       l.imag()*r.real() - l.real()*r.imag());
}


// Optimized  adj(RComplexREG)*adj(RComplexREG)
template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const RComplexREG<T1>& l, const RComplexREG<T2>& r)
{
  typedef typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpAdjMultiplyAdj>::Type_t  Ret_t;

  // The complex conjugate nature has been eaten here leaving simple multiples
  // involving transposes - which are probably null
//  d.real() = transpose(l.real())*transpose(r.real()) - transpose(l.imag())*transpose(r.imag());
//  d.imag() = -(transpose(l.real())*transpose(r.imag()) + transpose(l.imag())*transpose(r.real()));
//  return d;

  /*! NOTE: removed transpose here !!!!!  */
  return Ret_t(l.real()*r.real() - l.imag()*r.imag(),
	       -(l.real()*r.imag() + l.imag()*r.real()));
}


//! RComplexREG = RComplexREG / RComplexREG
template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpDivide>::Type_t
operator/(const RComplexREG<T1>& l, const RComplexREG<T2>& r)
{
  typedef typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpDivide>::Type_t  Ret_t;

  T2 tmp = T2(1.0) / (r.real()*r.real() + r.imag()*r.imag());

  return Ret_t((l.real()*r.real() + l.imag()*r.imag()) * tmp,
	       (l.imag()*r.real() - l.real()*r.imag()) * tmp);
}

//! RComplexREG = RComplexREG / RScalarREG
template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RScalarREG<T2>, OpDivide>::Type_t
operator/(const RComplexREG<T1>& l, const RScalarREG<T2>& r)
{
  typedef typename BinaryReturn<RComplexREG<T1>, RScalarREG<T2>, OpDivide>::Type_t  Ret_t;

  T2 tmp = T2(1.0) / r.elem();

  return Ret_t(l.real() * tmp, 
	       l.imag() * tmp);
}

//! RComplexREG = RScalarREG / RComplexREG
template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RComplexREG<T2>, OpDivide>::Type_t
operator/(const RScalarREG<T1>& l, const RComplexREG<T2>& r)
{
  typedef typename BinaryReturn<RScalarREG<T1>, RComplexREG<T2>, OpDivide>::Type_t  Ret_t;

  T2 tmp = T2(1.0) / (r.real()*r.real() + r.imag()*r.imag());

  return Ret_t(l.elem() * r.real() * tmp,
	       -l.elem() * r.imag() * tmp);
}



//-----------------------------------------------------------------------------
// Functions

// Adjoint
template<class T1>
inline typename UnaryReturn<RComplexREG<T1>, FnAdjoint>::Type_t
adj(const RComplexREG<T1>& l)
{
  typedef typename UnaryReturn<RComplexREG<T1>, FnAdjoint>::Type_t  Ret_t;

  // The complex conjugate nature has been eaten here leaving transpose
//  d.real() = transpose(l.real());
//  d.imag() = -transpose(l.imag());
//  return d;

  /*! NOTE: removed transpose here !!!!!  */
  return Ret_t(l.real(),
	       -l.imag());
}

// Conjugate
template<class T1>
inline typename UnaryReturn<RComplexREG<T1>, FnConjugate>::Type_t
conj(const RComplexREG<T1>& l)
{
  typedef typename UnaryReturn<RComplexREG<T1>, FnConjugate>::Type_t  Ret_t;

  return Ret_t(l.real(),
	       -l.imag());
}

// Transpose
template<class T1>
inline typename UnaryReturn<RComplexREG<T1>, FnTranspose>::Type_t
transpose(const RComplexREG<T1>& l)
{
  typedef typename UnaryReturn<RComplexREG<T1>, FnTranspose>::Type_t  Ret_t;

//  d.real() = transpose(l.real());
//  d.imag() = transpose(l.imag());
//  return d;

  /*! NOTE: removed transpose here !!!!!  */
  return Ret_t(l.real(), 
	       l.imag());
}

// TRACE
// trace = Trace(source1)
template<class T>
struct UnaryReturn<RComplexREG<T>, FnTrace > {
  typedef RComplexREG<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RComplexREG<T1>, FnTrace>::Type_t
trace(const RComplexREG<T1>& s1)
{
  typedef typename UnaryReturn<RComplexREG<T1>, FnTrace>::Type_t  Ret_t;

  /*! NOTE: removed trace here !!!!!  */
  return Ret_t(s1.real(),
	       s1.imag());
}


// trace = Re(Trace(source1))
template<class T>
struct UnaryReturn<RComplexREG<T>, FnRealTrace > {
  typedef RScalarREG<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RComplexREG<T1>, FnRealTrace>::Type_t
realTrace(const RComplexREG<T1>& s1)
{
  /*! NOTE: removed trace here !!!!!  */
  return s1.real();
}


// trace = Im(Trace(source1))
template<class T>
struct UnaryReturn<RComplexREG<T>, FnImagTrace > {
  typedef RScalarREG<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RComplexREG<T1>, FnImagTrace>::Type_t
imagTrace(const RComplexREG<T1>& s1)
{
  /*! NOTE: removed trace here !!!!!  */
  return s1.imag();
}

//! RComplexREG = trace(RComplexREG * RComplexREG)
template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpMultiply>::Type_t
traceMultiply(const RComplexREG<T1>& l, const RComplexREG<T2>& r)
{
//  return traceMultiply(l.elem(), r.elem());

  /*! NOTE: removed trace here !!!!!  */
  typedef typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, OpMultiply>::Type_t  Ret_t;

  return Ret_t(l.real()*r.real() - l.imag()*r.imag(),
	       l.real()*r.imag() + l.imag()*r.real());
}


// RScalarREG = Re(RComplexREG)
template<class T>
struct UnaryReturn<RComplexREG<T>, FnReal > {
  typedef RScalarREG<typename UnaryReturn<T, FnReal>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RComplexREG<T1>, FnReal>::Type_t
real(const RComplexREG<T1>& s1)
{
  return s1.real();
}

// RScalarREG = Im(RComplexREG)
template<class T>
struct UnaryReturn<RComplexREG<T>, FnImag > {
  typedef RScalarREG<typename UnaryReturn<T, FnImag>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RComplexREG<T1>, FnImag>::Type_t
imag(const RComplexREG<T1>& s1)
{
  return s1.imag();
}


//! RComplexREG<T> = (RScalarREG<T> , RScalarREG<T>)
template<class T1, class T2>
struct BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, FnCmplx > {
  typedef RComplexREG<typename BinaryReturn<T1, T2, FnCmplx>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, FnCmplx>::Type_t
cmplx(const RScalarREG<T1>& s1, const RScalarREG<T2>& s2)
{
  typedef typename BinaryReturn<RScalarREG<T1>, RScalarREG<T2>, FnCmplx>::Type_t  Ret_t;

  return Ret_t(s1.elem(),
	       s2.elem());
}



// RComplexREG = i * RScalarREG
template<class T>
struct UnaryReturn<RScalarREG<T>, FnTimesI > {
  typedef RComplexREG<typename UnaryReturn<T, FnTimesI>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalarREG<T>, FnTimesI>::Type_t
timesI(const RScalarREG<T>& s1)
{
  typename UnaryReturn<RScalarREG<T>, FnTimesI>::Type_t  d;

  zero_rep(d.real());
  d.imag() = s1.elem();
  return d;
}

// RComplexREG = i * RComplexREG
template<class T>
inline typename UnaryReturn<RComplexREG<T>, FnTimesI>::Type_t
timesI(const RComplexREG<T>& s1)
{
  typedef typename UnaryReturn<RComplexREG<T>, FnTimesI>::Type_t  Ret_t;

  return Ret_t(-s1.imag(),
	       s1.real());
}


// RComplexREG = -i * RScalarREG
template<class T>
struct UnaryReturn<RScalarREG<T>, FnTimesMinusI > {
  typedef RComplexREG<typename UnaryReturn<T, FnTimesMinusI>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalarREG<T>, FnTimesMinusI>::Type_t
timesMinusI(const RScalarREG<T>& s1)
{
  typename UnaryReturn<RScalarREG<T>, FnTimesMinusI>::Type_t  d;

  zero_rep(d.real());
  d.imag() = -s1.elem();
  return d;
}


// RComplexREG = -i * RComplexREG
template<class T>
inline typename UnaryReturn<RComplexREG<T>, FnTimesMinusI>::Type_t
timesMinusI(const RComplexREG<T>& s1)
{
  typedef typename UnaryReturn<RComplexREG<T>, FnTimesMinusI>::Type_t  Ret_t;

  return Ret_t(s1.imag(),
	       -s1.real());
}


//! RComplexREG = outerProduct(RComplexREG, RComplexREG)
template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, FnOuterProduct>::Type_t
outerProduct(const RComplexREG<T1>& l, const RComplexREG<T2>& r)
{
  typedef typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, FnOuterProduct>::Type_t  Ret_t;

  // Return   l*conj(r)
  return Ret_t(l.real()*r.real() + l.imag()*r.imag(),
	       l.imag()*r.real() - l.real()*r.imag());
}

//! RComplexREG = outerProduct(RComplexREG, RScalarREG)
template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RScalarREG<T2>, FnOuterProduct>::Type_t
outerProduct(const RComplexREG<T1>& l, const RScalarREG<T2>& r)
{
  typedef typename BinaryReturn<RComplexREG<T1>, RScalarREG<T2>, FnOuterProduct>::Type_t  Ret_t;

  // Return   l*conj(r)
  return Ret_t(l.real()*r.elem(),
	       l.imag()*r.elem());
}

//! RComplexREG = outerProduct(RScalarREG, RComplexREG)
template<class T1, class T2>
inline typename BinaryReturn<RScalarREG<T1>, RComplexREG<T2>, FnOuterProduct>::Type_t
outerProduct(const RScalarREG<T1>& l, const RComplexREG<T2>& r)
{
  typedef typename BinaryReturn<RScalarREG<T1>, RComplexREG<T2>, FnOuterProduct>::Type_t  Ret_t;

  // Return   l*conj(r)
  return Ret_t( l.elem()*r.real(),
	       -l.elem()*r.imag());
}


//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T>
inline typename UnaryReturn<RComplexREG<T>, FnGetSite>::Type_t
getSite(const RComplexREG<T>& s1, int innersite)
{
  typedef typename UnaryReturn<RComplexREG<T>, FnGetSite>::Type_t  Ret_t;

  return Ret_t(getSite(s1.real(), innersite), 
	       getSite(s1.imag(), innersite));
}


//! dest = (mask) ? s1 : dest
template<class T, class T1> 
inline
void copymask(RComplexREG<T>& d, const RScalarREG<T1>& mask, const RComplexREG<T>& s1) 
{
  copymask(d.real(),mask.elem(),s1.real());
  copymask(d.imag(),mask.elem(),s1.imag());
}


#if 1
// Global sum over site indices only
template<class T>
struct UnaryReturn<RComplexREG<T>, FnSum> {
  typedef RComplexREG<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RComplexREG<T>, FnSum>::Type_t
sum(const RComplexREG<T>& s1)
{
  typedef typename UnaryReturn<RComplexREG<T>, FnSum>::Type_t  Ret_t;

  return Ret_t(sum(s1.real()),
	       sum(s1.imag()));
}
#endif


// InnerProduct (norm-seq) global sum = sum(tr(adj(s1)*s1))
template<class T>
struct UnaryReturn<RComplexREG<T>, FnNorm2 > {
  typedef RScalarREG<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<RComplexREG<T>, FnLocalNorm2 > {
  typedef RScalarREG<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RComplexREG<T>, FnLocalNorm2>::Type_t
localNorm2(const RComplexREG<T>& s1)
{
  return localNorm2(s1.real()) + localNorm2(s1.imag());
}



//! RComplexREG<T> = InnerProduct(adj(RComplexREG<T1>)*RComplexREG<T2>)
template<class T1, class T2>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, FnInnerProduct > {
  typedef RComplexREG<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, FnLocalInnerProduct > {
  typedef RComplexREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(const RComplexREG<T1>& l, const RComplexREG<T2>& r)
{
  typedef typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, FnLocalInnerProduct>::Type_t  Ret_t;

  return Ret_t(localInnerProduct(l.real(),r.real()) + localInnerProduct(l.imag(),r.imag()),
	       localInnerProduct(l.real(),r.imag()) - localInnerProduct(l.imag(),r.real()));
}



template<class T1, class T2>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, FnLocalColorInnerProduct > {
  typedef RComplexREG<typename BinaryReturn<T1, T2, FnLocalColorInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, FnLocalColorInnerProduct>::Type_t
localColorInnerProduct(const RComplexREG<T1>& l, const RComplexREG<T2>& r)
{
  typedef typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, FnLocalColorInnerProduct>::Type_t  Ret_t;

  return Ret_t(localColorInnerProduct(l.real(),r.real()) + localColorInnerProduct(l.imag(),r.imag()),
	       localColorInnerProduct(l.real(),r.imag()) - localColorInnerProduct(l.imag(),r.real()));
}

  

//! RScalarREG<T> = InnerProductReal(adj(RComplexREG<T1>)*RComplexREG<T1>)
// Real-ness is eaten at this level
template<class T1, class T2>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, FnInnerProductReal > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, FnLocalInnerProductReal > {
  typedef RScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RComplexREG<T1>, RComplexREG<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const RComplexREG<T1>& l, const RComplexREG<T2>& r)
{
  return localInnerProduct(l.real(),r.real()) + localInnerProduct(l.imag(),r.imag());
}


//! RComplexREG<T> = where(RScalarREG, RComplexREG, RComplexREG)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3>
struct TrinaryReturn<RScalarREG<T1>, RComplexREG<T2>, RComplexREG<T3>, FnWhere> {
  typedef RComplexREG<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
inline typename TrinaryReturn<RScalarREG<T1>, RComplexREG<T2>, RComplexREG<T3>, FnWhere>::Type_t
where(const RScalarREG<T1>& a, const RComplexREG<T2>& b, const RComplexREG<T3>& c)
{
  typedef typename TrinaryReturn<RScalarREG<T1>, RComplexREG<T2>, RComplexREG<T3>, FnWhere>::Type_t  Ret_t;

  // Not optimal - want to have where outside assignment
  return Ret_t(where(a.elem(), b.real(), c.real()),
	       where(a.elem(), b.imag(), c.imag()));
}

//! RComplexREG<T> = where(RScalarREG, RComplexREG, RScalarREG)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3>
struct TrinaryReturn<RScalarREG<T1>, RComplexREG<T2>, RScalarREG<T3>, FnWhere> {
  typedef RComplexREG<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
inline typename TrinaryReturn<RScalarREG<T1>, RComplexREG<T2>, RComplexREG<T3>, FnWhere>::Type_t
where(const RScalarREG<T1>& a, const RComplexREG<T2>& b, const RScalarREG<T3>& c)
{
  typedef typename TrinaryReturn<RScalarREG<T1>, RComplexREG<T2>, RScalarREG<T3>, FnWhere>::Type_t  Ret_t;
  typedef typename InternalScalar<T3>::Type_t  S;

  // Not optimal - want to have where outside assignment
  return Ret_t(where(a.elem(), b.real(), c.real()),
	       where(a.elem(), b.imag(), S(0)));
}

//! RComplexREG<T> = where(RScalarREG, RScalarREG, RComplexREG)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3>
struct TrinaryReturn<RScalarREG<T1>, RScalarREG<T2>, RComplexREG<T3>, FnWhere> {
  typedef RComplexREG<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
inline typename TrinaryReturn<RScalarREG<T1>, RScalarREG<T2>, RComplexREG<T3>, FnWhere>::Type_t
where(const RScalarREG<T1>& a, const RScalarREG<T2>& b, const RComplexREG<T3>& c)
{
  typedef typename TrinaryReturn<RScalarREG<T1>, RScalarREG<T2>, RComplexREG<T3>, FnWhere>::Type_t  Ret_t;
  typedef typename InternalScalar<T2>::Type_t  S;

  // Not optimal - want to have where outside assignment
  return Ret_t(where(a.elem(), b.real(), c.real()),
	       where(a.elem(), S(0), c.imag()));
}


//-----------------------------------------------------------------------------
// Broadcast operations
//! dest = 0
template<class T> 
inline
void zero_rep(RComplexREG<T>& dest) 
{
  zero_rep(dest.real());
  zero_rep(dest.imag());
}


//! dest  = random  
template<class T, class T1, class T2, class T3>
inline void
fill_random(RComplexREG<T>& d, T1& seed, T2& skewed_seed, const T3& seed_mult)
{
  //d.func().insert_label("before_real");    
  fill_random(d.real(), seed, skewed_seed, seed_mult);
  //d.func().insert_label("before_imag");    
  fill_random(d.imag(), seed, skewed_seed, seed_mult);
}


template<class T>
inline void
get_pred(int& pred, const RScalarREG<T>& d)
{
  get_pred(pred , d.elem() );
}



//! dest  = gaussian
/*! RComplexREG polar method */
template<class T>
inline void
fill_gaussian(RComplexREG<T>& d, RComplexREG<T>& r1, RComplexREG<T>& r2)
{
  T w_2pi;
  T w_2;
  T w_g_r;
  T w_g_i;
  T w_r1_r;
  T w_r2_r;
  //T w_r1_i(d.func());
  //T w_r2_i(d.func());

  w_r1_r = r1.real();
  w_r2_r = r2.real();
  //w_r1_i = r1.imag();
  //w_r2_i = r2.imag();

  w_2pi = (float)6.283185307;
  w_2 = (float)2.0;

  w_r2_r *= w_2pi;
  w_g_r = cos(w_r2_r);
  w_g_i = sin(w_r2_r);

  w_r1_r = sqrt( -w_2 * log(w_r1_r) );

  d.real() = w_r1_r * w_g_r;
  d.imag() = w_r1_r * w_g_i;

#if 0
  typedef typename InternalScalar<T>::Type_t  S;

  // r1 and r2 are the input random numbers needed

  /* Stage 2: get the cos of the second number  */
  T  g_r, g_i;

  r2.real() *= S(6.283185307);
  g_r = cos(r2.real());
  g_i = sin(r2.real());
    
  /* Stage 4: get  sqrt(-2.0 * log(u1)) */
  r1.real() = sqrt(-S(2.0) * log(r1.real()));

  /* Stage 5:   g_r = sqrt(-2*log(u1))*cos(2*pi*u2) */
  /* Stage 5:   g_i = sqrt(-2*log(u1))*sin(2*pi*u2) */
  d.real() = r1.real() * g_r;
  d.imag() = r1.real() * g_i;
#endif
}




template<class T1>
struct UnaryReturn<RComplexREG<T1>, FnIsFinite> {
  typedef RScalarREG<typename UnaryReturn<T1, FnIsFinite>::Type_t>  Type_t;
};


template<class T1>
inline typename UnaryReturn<RComplexREG<T1>, FnIsFinite>::Type_t
isfinite(const RComplexREG<T1>& s1)
{
  return isfinite(s1.real()) && isfinite(s1.imag());
}




/*! @} */  // end of group rcomplex

} // namespace QDP

#endif
