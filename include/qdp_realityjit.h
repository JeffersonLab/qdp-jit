// -*- C++ -*-

/*! \file
 * \brief Reality
 */


#ifndef QDP_REALITYJIT_H
#define QDP_REALITYJIT_H

#include <sstream>

namespace QDP {




template<class T>
class RScalarJIT : public JV<T,1>
{
public:

  RScalarJIT(Jit& j,int r , int of , int ol): JV<T,1>(j,r,of,ol) {}
  RScalarJIT(Jit& j): JV<T,1>(j) {}


  template<class T1>
  RScalarJIT& operator=( const RScalarJIT<T1>& rhs) {
    elem() = rhs.elem();
    return *this;
  }

  RScalarJIT& operator=( const RScalarJIT& rhs) {
    elem() = rhs.elem();
    return *this;
  }


#if 0
  RScalarJIT() {}
  ~RScalarJIT() {}

  //---------------------------------------------------------
  //! construct dest = const
  RScalarJIT(const typename WordType<T>::Type_t& rhs) : F(rhs) {}

  //! construct dest = rhs
  template<class T1>
  RScalarJIT(const RScalarJIT<T1>& rhs) : F(rhs.elem()) {}

  //! construct dest = rhs
  template<class T1>
  RScalarJIT(const T1& rhs) : F(rhs) {}
#endif




  //! RScalarJIT += RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator+=(const RScalarJIT<T1>& rhs) 
    {
      elem() += rhs.elem();
      return *this;
    }

  //! RScalarJIT -= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator-=(const RScalarJIT<T1>& rhs) 
    {
      elem() -= rhs.elem();
      return *this;
    }

  //! RScalarJIT *= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator*=(const RScalarJIT<T1>& rhs) 
    {
      elem() *= rhs.elem();
      return *this;
    }

  //! RScalarJIT /= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator/=(const RScalarJIT<T1>& rhs) 
    {
      elem() /= rhs.elem();
      return *this;
    }

  //! RScalarJIT %= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator%=(const RScalarJIT<T1>& rhs) 
    {
      elem() %= rhs.elem();
      return *this;
    }

  //! RScalarJIT |= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator|=(const RScalarJIT<T1>& rhs) 
    {
      elem() |= rhs.elem();
      return *this;
    }

  //! RScalarJIT &= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator&=(const RScalarJIT<T1>& rhs) 
    {
      elem() &= rhs.elem();
      return *this;
    }

  //! RScalarJIT ^= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator^=(const RScalarJIT<T1>& rhs) 
    {
      elem() ^= rhs.elem();
      return *this;
    }

  //! RScalarJIT <<= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator<<=(const RScalarJIT<T1>& rhs) 
    {
      elem() <<= rhs.elem();
      return *this;
    }

  //! RScalarJIT >>= RScalarJIT
  template<class T1>
  inline
  RScalarJIT& operator>>=(const RScalarJIT<T1>& rhs) 
    {
      elem() >>= rhs.elem();
      return *this;
    }

#if 0
  RScalarJIT(const RScalarJIT& a) : JV<T,1>::JV(a) {
    std::cout << "RScalarJIT copy c-tor " << (void*)this << "\n";
  }
#endif


public:
  inline       T& elem()       { return JV<T,1>::getF()[0]; }
  inline const T& elem() const { return JV<T,1>::getF()[0]; }

private:
  //RScalarJIT(const RScalarJIT& a);
};

 
// Input
//! Ascii input
template<class T>
inline
istream& operator>>(istream& s, RScalarJIT<T>& d)
{
  return s >> d.elem();
}

//! Ascii input
template<class T>
inline
StandardInputStream& operator>>(StandardInputStream& s, RScalarJIT<T>& d)
{
  return s >> d.elem();
}

//! Ascii output
template<class T> 
inline  
ostream& operator<<(ostream& s, const RScalarJIT<T>& d)
{
  return s << d.elem();
}

//! Ascii output
template<class T> 
inline  
StandardOutputStream& operator<<(StandardOutputStream& s, const RScalarJIT<T>& d)
{
  return s << d.elem();
}


//! Text input
template<class T>
inline
TextReader& operator>>(TextReader& s, RScalarJIT<T>& d)
{
  return s >> d.elem();
}

//! Text output
template<class T> 
inline  
TextWriter& operator<<(TextWriter& s, const RScalarJIT<T>& d)
{
  return s << d.elem();
}

#ifndef QDP_NO_LIBXML2
//! XML output
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, const RScalarJIT<T>& d)
{
  return xml << d.elem();
}

//! XML input
template<class T>
inline
void read(XMLReader& xml, const string& path, RScalarJIT<T>& d)
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
class RComplexJIT: public JV<T,2>
{
public:

  RComplexJIT(Jit& j,int r , int of , int ol): JV<T,2>(j,r,of,ol) {}
  RComplexJIT(Jit& j): JV<T,2>(j) {}

#if 0
  RComplexJIT() {}
  ~RComplexJIT() {}

  //! Construct from two reality scalars
  template<class T1, class T2>
  RComplexJIT(const RScalarJIT<T1>& _re, const RScalarJIT<T2>& _im): re(_re.elem()), im(_im.elem()) {}

  //! Construct from two scalars
  template<class T1, class T2>
  RComplexJIT(const T1& _re, const T2& _im): re(_re), im(_im) {}
#endif


  //! RComplexJIT += RScalarJIT
  template<class T1>
  inline
  RComplexJIT& operator+=(const RScalarJIT<T1>& rhs) 
    {
      real() += rhs.elem();
      return *this;
    }

  //! RComplexJIT -= RScalarJIT
  template<class T1>
  inline
  RComplexJIT& operator-=(const RScalarJIT<T1>& rhs) 
    {
      real() -= rhs.elem();
      return *this;
    }

  //! RComplexJIT *= RScalarJIT
  template<class T1>
  inline
  RComplexJIT& operator*=(const RScalarJIT<T1>& rhs) 
    {
      real() *= rhs.elem();
      imag() *= rhs.elem();
      return *this;
    }

  //! RComplexJIT /= RScalarJIT
  template<class T1>
  inline
  RComplexJIT& operator/=(const RScalarJIT<T1>& rhs) 
    {
      real() /= rhs.elem();
      imag() /= rhs.elem();
      return *this;
    }

  //! RComplexJIT += RComplexJIT
  template<class T1>
  inline
  RComplexJIT& operator+=(const RComplexJIT<T1>& rhs) 
    {
      real() += rhs.real();
      imag() += rhs.imag();
      return *this;
    }

  //! RComplexJIT -= RComplexJIT
  template<class T1>
  inline
  RComplexJIT& operator-=(const RComplexJIT<T1>& rhs) 
    {
      real() -= rhs.real();
      imag() -= rhs.imag();
      return *this;
    }

  //! RComplexJIT *= RComplexJIT
  template<class T1>
  inline
  RComplexJIT& operator*=(const RComplexJIT<T1>& rhs) 
    {
      RComplexJIT<T> d;
      d = *this * rhs;

      real() = d.real();
      imag() = d.imag();
      return *this;
    }

  //! RComplexJIT /= RComplexJIT
  template<class T1>
  inline
  RComplexJIT& operator/=(const RComplexJIT<T1>& rhs) 
    {
      RComplexJIT<T> d;
      d = *this / rhs;

      real() = d.real();
      imag() = d.imag();
      return *this;
    }

  template<class T1>
  RComplexJIT& operator=(const RComplexJIT<T1>& rhs) 
    {
      real() = rhs.real();
      imag() = rhs.imag();
      return *this;
    }


  RComplexJIT& operator=(const RComplexJIT& rhs) 
    {
      real() = rhs.real();
      imag() = rhs.imag();
      return *this;
    }

  template<class T1>
  inline
  RComplexJIT& operator=(const RScalarJIT<T1>& rhs) 
    {
      real() = rhs.elem();
      zero_rep(imag());
      return *this;
    }

#if 0
  RComplexJIT(const RComplexJIT& a) : JV<T,2>::JV(a) {
    std::cout << "RComplexJIT copy c-tor " << (void*)this << "\n";
  }
#endif

public:
  inline       T& real()       { return JV<T,2>::getF()[0]; }
  inline const T& real() const { return JV<T,2>::getF()[0]; }

  inline       T& imag()       { return JV<T,2>::getF()[1]; }
  inline const T& imag() const { return JV<T,2>::getF()[1]; }
};



//! Stream output
template<class T>
inline
ostream& operator<<(ostream& s, const RComplexJIT<T>& d)
{
  s << "( " << d.real() << " , " << d.imag() << " )";
  return s;
}

//! Stream output
template<class T>
inline
StandardOutputStream& operator<<(StandardOutputStream& s, const RComplexJIT<T>& d)
{
  s << "( " << d.real() << " , " << d.imag() << " )";
  return s;
}

//! Text input
template<class T>
inline
TextReader& operator>>(TextReader& s, RComplexJIT<T>& d)
{
  return s >> d.real() >> d.imag();
}

//! Text output
template<class T> 
inline  
TextWriter& operator<<(TextWriter& s, const RComplexJIT<T>& d)
{
  return s << d.real() << d.imag();
}

#ifndef QDP_NO_LIBXML2
//! XML output
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, const RComplexJIT<T>& d)
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
void read(XMLReader& xml, const string& xpath, RComplexJIT<T>& d)
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
		  << "Failed to match real part of RComplexJIT Object with self constructed path: " << path_real;
    
    throw error_message.str();
  }
	
  // Try and recursively get the imaginary part
  try {
    read(xml, path_imag, d.imag());
  }
  catch(const string &e) {
    error_message << "XPath Query: " << xpath <<" Error:"
		  <<"Failed to match imaginary part of RComplexJIT Object with self constructed path: " << path_imag;
    
    throw error_message.str();
  }
}
#endif

/*! @} */   // end of group rcomplex

//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

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

// Default unary(RScalarJIT) -> RScalarJIT
template<class T1, class Op>
struct UnaryReturn<RScalarJIT<T1>, Op> {
  typedef RScalarJIT<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

// Default unary(RComplexJIT) -> RComplexJIT
template<class T1, class Op>
struct UnaryReturn<RComplexJIT<T1>, Op> {
  typedef RComplexJIT<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

// Default binary(RScalarJIT,RScalarJIT) -> RScalarJIT
template<class T1, class T2, class Op>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, Op> {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Default binary(RComplexJIT,RComplexJIT) -> RComplexJIT
template<class T1, class T2, class Op>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, Op> {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Default binary(RScalarJIT,RComplexJIT) -> RComplexJIT
template<class T1, class T2, class Op>
struct BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, Op> {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Default binary(RComplexJIT,RScalarJIT) -> RComplexJIT
template<class T1, class T2, class Op>
struct BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, Op> {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};




// RScalarJIT
#if 0
template<class T1, class T2>
struct UnaryReturn<RScalarJIT<T2>, OpCast<T1> > {
  typedef RScalarJIT<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif


template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpAddAssign > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpSubtractAssign > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpMultiplyAssign > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpDivideAssign > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpModAssign > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpBitwiseOrAssign > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpBitwiseAndAssign > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpBitwiseXorAssign > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpLeftShiftAssign > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpRightShiftAssign > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2, class T3>
struct TrinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, RScalarJIT<T3>, FnColorContract> {
  typedef RScalarJIT<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};

// RScalarJIT
// Gamma algebra
template<int N, int m, class T2, class OpGammaConstMultiply>
struct BinaryReturn<GammaConst<N,m>, RScalarJIT<T2>, OpGammaConstMultiply> {
  typedef RScalarJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConst>
struct BinaryReturn<RScalarJIT<T2>, GammaConst<N,m>, OpMultiplyGammaConst> {
  typedef RScalarJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeMultiply>
struct BinaryReturn<GammaType<N>, RScalarJIT<T2>, OpGammaTypeMultiply> {
  typedef RScalarJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaType>
struct BinaryReturn<RScalarJIT<T2>, GammaType<N>, OpMultiplyGammaType> {
  typedef RScalarJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};


// RScalarJIT
// Gamma algebra
template<int N, int m, class T2, class OpGammaConstDPMultiply>
struct BinaryReturn<GammaConstDP<N,m>, RScalarJIT<T2>, OpGammaConstDPMultiply> {
  typedef RScalarJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConstDP>
struct BinaryReturn<RScalarJIT<T2>, GammaConstDP<N,m>, OpMultiplyGammaConstDP> {
  typedef RScalarJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeDPMultiply>
struct BinaryReturn<GammaTypeDP<N>, RScalarJIT<T2>, OpGammaTypeDPMultiply> {
  typedef RScalarJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaTypeDP>
struct BinaryReturn<RScalarJIT<T2>, GammaTypeDP<N>, OpMultiplyGammaTypeDP> {
  typedef RScalarJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};



// RComplexJIT
// Gamma algebra
template<int N, int m, class T2, class OpGammaConstMultiply>
struct BinaryReturn<GammaConst<N,m>, RComplexJIT<T2>, OpGammaConstMultiply> {
  typedef RComplexJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConst>
struct BinaryReturn<RComplexJIT<T2>, GammaConst<N,m>, OpMultiplyGammaConst> {
  typedef RComplexJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeMultiply>
struct BinaryReturn<GammaType<N>, RComplexJIT<T2>, OpGammaTypeMultiply> {
  typedef RComplexJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaType>
struct BinaryReturn<RComplexJIT<T2>, GammaType<N>, OpMultiplyGammaType> {
  typedef RComplexJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};


// Gamma algebra
template<int N, int m, class T2, class OpGammaConstDPMultiply>
struct BinaryReturn<GammaConstDP<N,m>, RComplexJIT<T2>, OpGammaConstDPMultiply> {
  typedef RComplexJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConstDP>
struct BinaryReturn<RComplexJIT<T2>, GammaConstDP<N,m>, OpMultiplyGammaConstDP> {
  typedef RComplexJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeDPMultiply>
struct BinaryReturn<GammaTypeDP<N>, RComplexJIT<T2>, OpGammaTypeDPMultiply> {
  typedef RComplexJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaTypeDP>
struct BinaryReturn<RComplexJIT<T2>, GammaTypeDP<N>, OpMultiplyGammaTypeDP> {
  typedef RComplexJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};


// Assignment is different
template<class T1, class T2 >
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpAssign > {
//  typedef RComplexJIT<T1> &Type_t;
  typedef RComplexJIT<typename BinaryReturn<T1, T2, OpAssign>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpAddAssign > {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpSubtractAssign > {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpMultiplyAssign > {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpDivideAssign > {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpModAssign > {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpBitwiseOrAssign > {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpBitwiseAndAssign > {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpBitwiseXorAssign > {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpLeftShiftAssign > {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpRightShiftAssign > {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2, class T3>
struct TrinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, RComplexJIT<T3>, FnColorContract> {
  typedef RComplexJIT<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};






//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

/*! \addtogroup rscalar
 * @{ 
 */

// Scalar Reality
template<class T>
struct UnaryReturn<RScalarJIT<T>, OpNot > {
  typedef RScalarJIT<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, OpNot>::Type_t
operator!(const RScalarJIT<T1>& l)
{
  return ! l.elem();
}


template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, OpUnaryPlus>::Type_t
operator+(const RScalarJIT<T1>& l)
{
  return +l.elem();
}


template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, OpUnaryMinus>::Type_t
operator-(const RScalarJIT<T1>& l)
{
  return -l.elem();
}


template<class T1, class T2>
inline void
addRep(const typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpAdd>::Type_t& dest, const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  addRep( dest.elem() , l.elem() , r.elem() );
}


template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpSubtract>::Type_t
operator-(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() - r.elem();
}


template<class T1, class T2>
void mulRep(const typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpMultiply>::Type_t& dest, const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  mulRep(dest.elem(),l.elem(),r.elem());
}

// Optimized  adj(RScalarJIT)*RScalarJIT
template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpAdjMultiply>::Type_t
adjMultiply(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  /*! NOTE: removed transpose here !!!!!  */

//  return transpose(l.elem()) * r.elem();
  return l.elem() * r.elem();
}

// Optimized  RScalarJIT*adj(RScalarJIT)
template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  /*! NOTE: removed transpose here !!!!!  */

//  return l.elem() * transpose(r.elem());
  return l.elem() * r.elem();
}

// Optimized  adj(RScalarJIT)*adj(RScalarJIT)
template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  /*! NOTE: removed transpose here !!!!!  */

//  return transpose(l.elem()) * transpose(r.elem());
  return l.elem() * r.elem();
}


template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpDivide>::Type_t
operator/(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() / r.elem();
}



template<class T1, class T2 >
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpLeftShift > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpLeftShift>::Type_t
operator<<(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() << r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpRightShift > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpRightShift>::Type_t
operator>>(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() >> r.elem();
}


template<class T1, class T2 >
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpMod>::Type_t
operator%(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() % r.elem();
}

template<class T1, class T2 >
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpBitwiseXor>::Type_t
operator^(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() ^ r.elem();
}

template<class T1, class T2 >
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpBitwiseAnd>::Type_t
operator&(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() & r.elem();
}

template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpBitwiseOr>::Type_t
operator|(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() | r.elem();
}



// Comparisons
template<class T1, class T2 >
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpLT > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpLT>::Type_t
operator<(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() < r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpLE > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpLE>::Type_t
operator<=(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() <= r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpGT > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpGT>::Type_t
operator>(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() > r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpGE > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpGE>::Type_t
operator>=(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() >= r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpEQ > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpEQ>::Type_t
operator==(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() == r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpNE > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpNE>::Type_t
operator!=(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() != r.elem();
}


template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpAnd > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpAnd>::Type_t
operator&&(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() && r.elem();
}


template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpOr > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, OpOr>::Type_t
operator||(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() || r.elem();
}



//-----------------------------------------------------------------------------
// Functions

// Adjoint
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnAdjoint>::Type_t
adj(const RScalarJIT<T1>& s1)
{
  /*! NOTE: removed transpose here !!!!!  */

//  return transpose(s1.elem()); // The complex nature has been eaten here
  return s1.elem(); // The complex nature has been eaten here
}


// Conjugate
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnConjugate>::Type_t
conj(const RScalarJIT<T1>& s1)
{
  return s1.elem();  // The complex nature has been eaten here
}


// Transpose
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnTranspose>::Type_t
transpose(const RScalarJIT<T1>& s1)
{
  /*! NOTE: removed transpose here !!!!!  */

//  return transpose(s1.elem());
  return s1.elem();
}



// TRACE
// trace = Trace(source1)
template<class T>
struct UnaryReturn<RScalarJIT<T>, FnTrace > {
  typedef RScalarJIT<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnTrace>::Type_t
trace(const RScalarJIT<T1>& s1)
{
//  return trace(s1.elem());

  /*! NOTE: removed trace here !!!!!  */
  return s1.elem();
}


// trace = Re(Trace(source1))
template<class T>
struct UnaryReturn<RScalarJIT<T>, FnRealTrace > {
  typedef RScalarJIT<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnRealTrace>::Type_t
realTrace(const RScalarJIT<T1>& s1)
{
//  return trace_real(s1.elem());

  /*! NOTE: removed trace here !!!!!  */
  return s1.elem();
}


// trace = Im(Trace(source1))
template<class T>
struct UnaryReturn<RScalarJIT<T>, FnImagTrace > {
  typedef RScalarJIT<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnImagTrace>::Type_t
imagTrace(const RScalarJIT<T1>& s1)
{
//  return trace_imag(s1.elem());

  /*! NOTE: removed trace here !!!!!  */
  return s1.elem();
}

//! RScalarJIT = trace(RScalarJIT * RScalarJIT)
template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnTraceMultiply>::Type_t
traceMultiply(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
//  return traceMultiply(l.elem(), r.elem());

  /*! NOTE: removed trace here !!!!!  */
  return l.elem() * r.elem();
}


// RScalarJIT = Re(RScalarJIT)  [identity]
template<class T>
inline typename UnaryReturn<RScalarJIT<T>, FnReal>::Type_t
real(const RScalarJIT<T>& s1)
{
  return s1.elem();
}


// RScalarJIT = Im(RScalarJIT) [this is zero]
template<class T>
inline typename UnaryReturn<RScalarJIT<T>, FnImag>::Type_t
imag(const RScalarJIT<T>& s1)
{
  typedef typename InternalScalar<T>::Type_t  S;
  return S(0);
}


// ArcCos
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnArcCos>::Type_t
acos(const RScalarJIT<T1>& s1)
{
  return acos(s1.elem());
}

// ArcSin
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnArcSin>::Type_t
asin(const RScalarJIT<T1>& s1)
{
  return asin(s1.elem());
}

// ArcTan
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnArcTan>::Type_t
atan(const RScalarJIT<T1>& s1)
{
  return atan(s1.elem());
}

// Ceil(ing)
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnCeil>::Type_t
ceil(const RScalarJIT<T1>& s1)
{
  return ceil(s1.elem());
}

// Cos
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnCos>::Type_t
cos(const RScalarJIT<T1>& s1)
{
  return cos(s1.elem());
}

// Cosh
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnHypCos>::Type_t
cosh(const RScalarJIT<T1>& s1)
{
  return cosh(s1.elem());
}

// Exp
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnExp>::Type_t
exp(const RScalarJIT<T1>& s1)
{
  return exp(s1.elem());
}

// Fabs
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnFabs>::Type_t
fabs(const RScalarJIT<T1>& s1)
{
  return fabs(s1.elem());
}

// Floor
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnFloor>::Type_t
floor(const RScalarJIT<T1>& s1)
{
  return floor(s1.elem());
}

// Log
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnLog>::Type_t
log(const RScalarJIT<T1>& s1)
{
  return log(s1.elem());
}

// Log10
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnLog10>::Type_t
log10(const RScalarJIT<T1>& s1)
{
  return log10(s1.elem());
}

// Sin
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnSin>::Type_t
sin(const RScalarJIT<T1>& s1)
{
  return sin(s1.elem());
}

// Sinh
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnHypSin>::Type_t
sinh(const RScalarJIT<T1>& s1)
{
  return sinh(s1.elem());
}

// Sqrt
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnSqrt>::Type_t
sqrt(const RScalarJIT<T1>& s1)
{
  return sqrt(s1.elem());
}

// Tan
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnTan>::Type_t
tan(const RScalarJIT<T1>& s1)
{
  return tan(s1.elem());
}

// Tanh
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnHypTan>::Type_t
tanh(const RScalarJIT<T1>& s1)
{
  return tanh(s1.elem());
}


//! RScalarJIT<T> = pow(RScalarJIT<T> , RScalarJIT<T>)
template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnPow>::Type_t
pow(const RScalarJIT<T1>& s1, const RScalarJIT<T2>& s2)
{
  return pow(s1.elem(), s2.elem());
}

//! RScalarJIT<T> = atan2(RScalarJIT<T> , RScalarJIT<T>)
template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnArcTan2>::Type_t
atan2(const RScalarJIT<T1>& s1, const RScalarJIT<T2>& s2)
{
  return atan2(s1.elem(), s2.elem());
}


//! RScalarJIT = outerProduct(RScalarJIT, RScalarJIT)
template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnOuterProduct>::Type_t
outerProduct(const RScalarJIT<T1>& l, const RScalarJIT<T2>& r)
{
  return l.elem() * r.elem();
}


//! dest [float type] = source [seed type]
template<class T1>
inline typename UnaryReturn<RScalarJIT<T1>, FnSeedToFloat>::Type_t
seedToFloat(const RScalarJIT<T1>& s1)
{
  return seedToFloat(s1.elem());
}

//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T>
inline typename UnaryReturn<RScalarJIT<T>, FnGetSite>::Type_t
getSite(const RScalarJIT<T>& s1, int innersite)
{
  return getSite(s1.elem(), innersite);
}

//! Extract color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T>
inline typename UnaryReturn<RScalarJIT<T>, FnPeekColorVector>::Type_t
peekColor(const RScalarJIT<T>& l, int row)
{
  return peekColor(l.elem(),row);
}

//! Extract color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T>
inline typename UnaryReturn<RScalarJIT<T>, FnPeekColorMatrix>::Type_t
peekColor(const RScalarJIT<T>& l, int row, int col)
{
  return peekColor(l.elem(),row,col);
}

//! Extract spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
inline typename UnaryReturn<RScalarJIT<T>, FnPeekSpinVector>::Type_t
peekSpin(const RScalarJIT<T>& l, int row)
{
  return peekSpin(l.elem(),row);
}

//! Extract spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
inline typename UnaryReturn<RScalarJIT<T>, FnPeekSpinMatrix>::Type_t
peekSpin(const RScalarJIT<T>& l, int row, int col)
{
  return peekSpin(l.elem(),row,col);
}


//------------------------------------------
//! dest = (mask) ? s1 : dest
template<class T, class T1> 
inline
void copymask(RScalarJIT<T>& d, const RScalarJIT<T1>& mask, const RScalarJIT<T>& s1) 
{
  copymask(d.elem(),mask.elem(),s1.elem());
}

//! dest [float type] = source [int type]
template<class T, class T1>
inline
void cast_rep(T& d, const RScalarJIT<T1>& s1)
{
  cast_rep(d, s1.elem());
}


//! dest [float type] = source [int type]
template<class T, class T1>
inline
void recast_rep(RScalarJIT<T>& d, const RScalarJIT<T1>& s1)
{
  cast_rep(d.elem(), s1.elem());
}


//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
copy_site(RScalarJIT<T>& d, int isite, const RScalarJIT<T1>& s1)
{
  copy_site(d.elem(), isite, s1.elem());
}


//! gather several inner sites together
template<class T, class T1>
inline void 
gather_sites(RScalarJIT<T>& d, 
	     const RScalarJIT<T1>& s0, int i0, 
	     const RScalarJIT<T1>& s1, int i1,
	     const RScalarJIT<T1>& s2, int i2,
	     const RScalarJIT<T1>& s3, int i3)
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
struct UnaryReturn<RScalarJIT<T>, FnSum > {
  typedef RScalarJIT<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalarJIT<T>, FnSum>::Type_t
sum(const RScalarJIT<T>& s1)
{
  return sum(s1.elem());
}
#endif


// Global max
template<class T>
struct UnaryReturn<RScalarJIT<T>, FnGlobalMax> {
  typedef RScalarJIT<typename UnaryReturn<T, FnGlobalMax>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalarJIT<T>, FnGlobalMax>::Type_t
globalMax(const RScalarJIT<T>& s1)
{
  return globalMax(s1.elem());
}


// Global min
template<class T>
struct UnaryReturn<RScalarJIT<T>, FnGlobalMin> {
  typedef RScalarJIT<typename UnaryReturn<T, FnGlobalMin>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalarJIT<T>, FnGlobalMin>::Type_t
globalMin(const RScalarJIT<T>& s1)
{
  return globalMin(s1.elem());
}



//------------------------------------------
// InnerProduct (norm-seq) global sum = sum(tr(adj(s1)*s1))
template<class T>
struct UnaryReturn<RScalarJIT<T>, FnNorm2 > {
  typedef RScalarJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<RScalarJIT<T>, FnLocalNorm2 > {
  typedef RScalarJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalarJIT<T>, FnLocalNorm2>::Type_t
localNorm2(const RScalarJIT<T>& s1)
{
  return localNorm2(s1.elem());
}



//! RScalarJIT<T> = InnerProduct(adj(RScalarJIT<T1>)*RScalarJIT<T2>)
template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnInnerProduct > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnLocalInnerProduct > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(const RScalarJIT<T1>& s1, const RScalarJIT<T2>& s2)
{
  return localInnerProduct(s1.elem(), s2.elem());
}


//! RScalarJIT<T> = InnerProductReal(adj(PMatrix<T1>)*PMatrix<T1>)
// Real-ness is eaten at this level
template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnInnerProductReal > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnLocalInnerProductReal > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const RScalarJIT<T1>& s1, const RScalarJIT<T2>& s2)
{
  return localInnerProduct(s1.elem(), s2.elem());
}


//! RScalarJIT<T> = where(RScalarJIT, RScalarJIT, RScalarJIT)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3>
struct TrinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, RScalarJIT<T3>, FnWhere> {
  typedef RScalarJIT<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
inline typename TrinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, RScalarJIT<T3>, FnWhere>::Type_t
where(const RScalarJIT<T1>& a, const RScalarJIT<T2>& b, const RScalarJIT<T3>& c)
{
  return where(a.elem(), b.elem(), c.elem());
}



//-----------------------------------------------------------------------------
// Broadcast operations
//! dest = 0
template<class T> 
inline
void zero_rep(RScalarJIT<T>& dest) 
{
  zero_rep(dest.elem());
}


//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
copy_site(RComplexJIT<T>& d, int isite, const RComplexJIT<T1>& s1)
{
  copy_site(d.real(), isite, s1.real());
  copy_site(d.imag(), isite, s1.imag());
}

#if 0
//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
copy_site(RComplexJIT<T>& d, int isite, const RScalarJIT<T1>& s1)
{
  copy_site(d.real(), isite, s1.elem());
  zero_rep(d.imag());   // this is wrong - want zero only at a site. Fix when needed.
}
#endif


//! gather several inner sites together
template<class T, class T1>
inline void 
gather_sites(RComplexJIT<T>& d, 
	     const RComplexJIT<T1>& s0, int i0, 
	     const RComplexJIT<T1>& s1, int i1,
	     const RComplexJIT<T1>& s2, int i2,
	     const RComplexJIT<T1>& s3, int i3)
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
template<class T, class T1, class T2>
inline void
fill_random(RScalarJIT<T>& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  fill_random(d.elem(), seed, skewed_seed, seed_mult);
}



//! dest  = gaussian  
/*! Real form of complex polar method */
template<class T>
inline void
fill_gaussian(RScalarJIT<T>& d, RScalarJIT<T>& r1, RScalarJIT<T>& r2)
{
  typedef typename InternalScalar<T>::Type_t  S;

  // r1 and r2 are the input random numbers needed

  /* Stage 2: get the cos of the second number  */
  T  g_r;

  r2.elem() *= S(6.283185307);
  g_r = cos(r2.elem());
    
  /* Stage 4: get  sqrt(-2.0 * log(u1)) */
  r1.elem() = sqrt(-S(2.0) * log(r1.elem()));

  /* Stage 5:   g_r = sqrt(-2*log(u1))*cos(2*pi*u2) */
  /* Stage 5:   g_i = sqrt(-2*log(u1))*sin(2*pi*u2) */
  d.elem() = r1.elem() * g_r;
}

/*! @} */   // end of group rscalar



//-----------------------------------------------------------------------------
// Complex Reality
//-----------------------------------------------------------------------------

/*! \addtogroup rcomplex 
 * @{ 
 */

//! RComplexJIT = +RComplexJIT
template<class T1>
inline typename UnaryReturn<RComplexJIT<T1>, OpUnaryPlus>::Type_t
operator+(const RComplexJIT<T1>& l)
{
  typedef typename UnaryReturn<RComplexJIT<T1>, OpUnaryPlus>::Type_t  Ret_t;

  return Ret_t(+l.real(),
	       +l.imag());
}


//! RComplexJIT = -RComplexJIT
template<class T1>
inline typename UnaryReturn<RComplexJIT<T1>, OpUnaryMinus>::Type_t
operator-(const RComplexJIT<T1>& l)
{
  typedef typename UnaryReturn<RComplexJIT<T1>, OpUnaryMinus>::Type_t  Ret_t;

  return Ret_t(-l.real(),
	       -l.imag());
}


//! RComplexJIT = RComplexJIT + RComplexJIT
template<class T1, class T2>
inline void
addRep(const typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpAdd>::Type_t& dest, const RComplexJIT<T1>& l, const RComplexJIT<T2>& r)
{
  addRep( dest.real() , l.real() , r.real() );
  addRep( dest.imag() , l.imag() , r.imag() );
}

//! RComplexJIT = RComplexJIT + RScalarJIT
template<class T1, class T2>
inline typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, OpAdd>::Type_t
operator+(const RComplexJIT<T1>& l, const RScalarJIT<T2>& r)
{
  typedef typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, OpAdd>::Type_t  Ret_t;

  return Ret_t(l.real()+r.elem(),
	       l.imag());
}

//! RComplexJIT = RScalarJIT + RComplexJIT
template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, OpAdd>::Type_t
operator+(const RScalarJIT<T1>& l, const RComplexJIT<T2>& r)
{
  typedef typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, OpAdd>::Type_t  Ret_t;

  return Ret_t(l.elem()+r.real(),
	       r.imag());
}


//! RComplexJIT = RComplexJIT - RComplexJIT
template<class T1, class T2>
inline typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpSubtract>::Type_t
operator-(const RComplexJIT<T1>& l, const RComplexJIT<T2>& r)
{
  typedef typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpSubtract>::Type_t  Ret_t;

  return Ret_t(l.real() - r.real(),
	       l.imag() - r.imag());
}

//! RComplexJIT = RComplexJIT - RScalarJIT
template<class T1, class T2>
inline typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, OpSubtract>::Type_t
operator-(const RComplexJIT<T1>& l, const RScalarJIT<T2>& r)
{
  typedef typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, OpSubtract>::Type_t  Ret_t;

  return Ret_t(l.real() - r.elem(),
	       l.imag());
}

//! RComplexJIT = RScalarJIT - RComplexJIT
template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, OpSubtract>::Type_t
operator-(const RScalarJIT<T1>& l, const RComplexJIT<T2>& r)
{
  typedef typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, OpSubtract>::Type_t  Ret_t;

  return Ret_t(l.elem() - r.real(),
	       - r.imag());
}



template<class T1, class T2>
inline void
mulRep(const typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpMultiply>::Type_t& dest, const RComplexJIT<T1>& l, const RComplexJIT<T2>& r)
{
#if 0
  RScalarJIT<T1> t0(dest.getFunc());
  RScalarJIT<T1> t1(dest.getFunc());

  mulRep( t0.elem() , l.imag() , r.imag() );
  negRep( t0.elem() , t0.elem() );
  fmaRep( dest.real() , l.real() , r.real() , t0.elem() );

  mulRep( t1.elem() , l.imag() , r.real() );
  fmaRep( dest.imag() , l.real() , r.imag() , t1.elem() );
#else
  RScalarJIT<T1> t0(dest.func());

  mulRep( t0.elem() , l.imag() , r.imag() );
  mulRep( dest.real() , l.real() , r.real() );
  subRep( dest.real() , dest.real() , t0.elem() );

  mulRep( t0.elem() , l.imag() , r.real() );
  mulRep( dest.imag() , l.real() , r.imag() );
  addRep( dest.imag() , dest.imag() , t0.elem() );
#endif
}




//! RComplexJIT = RScalarJIT * RComplexJIT
template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, OpMultiply>::Type_t
operator*(const RScalarJIT<T1>& l, const RComplexJIT<T2>& r)
{
  typedef typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, OpMultiply>::Type_t  Ret_t;

  return Ret_t(l.elem()*r.real(), 
	       l.elem()*r.imag());
}

//! RComplexJIT = RComplexJIT * RScalarJIT
template<class T1, class T2>
inline typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, OpMultiply>::Type_t
operator*(const RComplexJIT<T1>& l, const RScalarJIT<T2>& r)
{
  typedef typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, OpMultiply>::Type_t  Ret_t;

  return Ret_t(l.real()*r.elem(), 
	       l.imag()*r.elem());
}


// Optimized  adj(RComplexJIT)*RComplexJIT
template<class T1, class T2>
inline typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpAdjMultiply>::Type_t
adjMultiply(const RComplexJIT<T1>& l, const RComplexJIT<T2>& r)
{
  typedef typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpAdjMultiply>::Type_t  Ret_t;

  // The complex conjugate nature has been eaten here leaving simple multiples
  // involving transposes - which are probably null
  
//  d.real() = transpose(l.real())*r.real() + transpose(l.imag())*r.imag();
//  d.imag() = transpose(l.real())*r.imag() - transpose(l.imag())*r.real();
//  return d;

  /*! NOTE: removed transpose here !!!!!  */
  return Ret_t(l.real()*r.real() + l.imag()*r.imag(),
	       l.real()*r.imag() - l.imag()*r.real());
}


// template<class T1, class T2>
// inline typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpMultiplyAdj>::Type_t
// multiplyAdj(const RComplexJIT<T1>& l, const RComplexJIT<T2>& r)
// {
//   typedef typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpMultiplyAdj>::Type_t  Ret_t;

//   return Ret_t(l.real()*r.real() + l.imag()*r.imag(),
// 	       l.imag()*r.real() - l.real()*r.imag());
// }
template<class T1, class T2>
inline void
multiplyAdjRep(const typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpMultiplyAdj>::Type_t& d, const RComplexJIT<T1>& l, const RComplexJIT<T2>& r)
{
  typename BinaryReturn<T1, T2, OpMultiplyAdj>::Type_t tmp(d.func());
  mulRep( d.real() , l.real(), r.real() );
  mulRep( tmp , l.imag() , r.imag() );
  addRep( d.real() , d.real() , tmp );

  mulRep( d.imag() , l.imag() , r.real() );
  mulRep( tmp , l.real() , r.imag() );
  subRep( d.imag() , d.imag() , tmp );
}

// Optimized  adj(RComplexJIT)*adj(RComplexJIT)
template<class T1, class T2>
inline typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const RComplexJIT<T1>& l, const RComplexJIT<T2>& r)
{
  typedef typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpAdjMultiplyAdj>::Type_t  Ret_t;

  // The complex conjugate nature has been eaten here leaving simple multiples
  // involving transposes - which are probably null
//  d.real() = transpose(l.real())*transpose(r.real()) - transpose(l.imag())*transpose(r.imag());
//  d.imag() = -(transpose(l.real())*transpose(r.imag()) + transpose(l.imag())*transpose(r.real()));
//  return d;

  /*! NOTE: removed transpose here !!!!!  */
  return Ret_t(l.real()*r.real() - l.imag()*r.imag(),
	       -(l.real()*r.imag() + l.imag()*r.real()));
}


//! RComplexJIT = RComplexJIT / RComplexJIT
template<class T1, class T2>
inline typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpDivide>::Type_t
operator/(const RComplexJIT<T1>& l, const RComplexJIT<T2>& r)
{
  typedef typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpDivide>::Type_t  Ret_t;

  T2 tmp = T2(1.0) / (r.real()*r.real() + r.imag()*r.imag());

  return Ret_t((l.real()*r.real() + l.imag()*r.imag()) * tmp,
	       (l.imag()*r.real() - l.real()*r.imag()) * tmp);
}

//! RComplexJIT = RComplexJIT / RScalarJIT
template<class T1, class T2>
inline typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, OpDivide>::Type_t
operator/(const RComplexJIT<T1>& l, const RScalarJIT<T2>& r)
{
  typedef typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, OpDivide>::Type_t  Ret_t;

  T2 tmp = T2(1.0) / r.elem();

  return Ret_t(l.real() * tmp, 
	       l.imag() * tmp);
}

//! RComplexJIT = RScalarJIT / RComplexJIT
template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, OpDivide>::Type_t
operator/(const RScalarJIT<T1>& l, const RComplexJIT<T2>& r)
{
  typedef typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, OpDivide>::Type_t  Ret_t;

  T2 tmp = T2(1.0) / (r.real()*r.real() + r.imag()*r.imag());

  return Ret_t(l.elem() * r.real() * tmp,
	       -l.elem() * r.imag() * tmp);
}



//-----------------------------------------------------------------------------
// Functions

// Adjoint
template<class T1>
inline typename UnaryReturn<RComplexJIT<T1>, FnAdjoint>::Type_t
adj(const RComplexJIT<T1>& l)
{
  typedef typename UnaryReturn<RComplexJIT<T1>, FnAdjoint>::Type_t  Ret_t;

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
inline typename UnaryReturn<RComplexJIT<T1>, FnConjugate>::Type_t
conj(const RComplexJIT<T1>& l)
{
  typedef typename UnaryReturn<RComplexJIT<T1>, FnConjugate>::Type_t  Ret_t;

  return Ret_t(l.real(),
	       -l.imag());
}

// Transpose
template<class T1>
inline typename UnaryReturn<RComplexJIT<T1>, FnTranspose>::Type_t
transpose(const RComplexJIT<T1>& l)
{
  typedef typename UnaryReturn<RComplexJIT<T1>, FnTranspose>::Type_t  Ret_t;

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
struct UnaryReturn<RComplexJIT<T>, FnTrace > {
  typedef RComplexJIT<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RComplexJIT<T1>, FnTrace>::Type_t
trace(const RComplexJIT<T1>& s1)
{
  typedef typename UnaryReturn<RComplexJIT<T1>, FnTrace>::Type_t  Ret_t;

  /*! NOTE: removed trace here !!!!!  */
  return Ret_t(s1.real(),
	       s1.imag());
}


// trace = Re(Trace(source1))
template<class T>
struct UnaryReturn<RComplexJIT<T>, FnRealTrace > {
  typedef RScalarJIT<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RComplexJIT<T1>, FnRealTrace>::Type_t
realTrace(const RComplexJIT<T1>& s1)
{
  /*! NOTE: removed trace here !!!!!  */
  return s1.real();
}


// trace = Im(Trace(source1))
template<class T>
struct UnaryReturn<RComplexJIT<T>, FnImagTrace > {
  typedef RScalarJIT<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RComplexJIT<T1>, FnImagTrace>::Type_t
imagTrace(const RComplexJIT<T1>& s1)
{
  /*! NOTE: removed trace here !!!!!  */
  return s1.imag();
}

//! RComplexJIT = trace(RComplexJIT * RComplexJIT)
template<class T1, class T2>
inline typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpMultiply>::Type_t
traceMultiply(const RComplexJIT<T1>& l, const RComplexJIT<T2>& r)
{
//  return traceMultiply(l.elem(), r.elem());

  /*! NOTE: removed trace here !!!!!  */
  typedef typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, OpMultiply>::Type_t  Ret_t;

  return Ret_t(l.real()*r.real() - l.imag()*r.imag(),
	       l.real()*r.imag() + l.imag()*r.real());
}


// RScalarJIT = Re(RComplexJIT)
template<class T>
struct UnaryReturn<RComplexJIT<T>, FnReal > {
  typedef RScalarJIT<typename UnaryReturn<T, FnReal>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RComplexJIT<T1>, FnReal>::Type_t
real(const RComplexJIT<T1>& s1)
{
  return s1.real();
}

// RScalarJIT = Im(RComplexJIT)
template<class T>
struct UnaryReturn<RComplexJIT<T>, FnImag > {
  typedef RScalarJIT<typename UnaryReturn<T, FnImag>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RComplexJIT<T1>, FnImag>::Type_t
imag(const RComplexJIT<T1>& s1)
{
  return s1.imag();
}


//! RComplexJIT<T> = (RScalarJIT<T> , RScalarJIT<T>)
template<class T1, class T2>
struct BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnCmplx > {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, FnCmplx>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnCmplx>::Type_t
cmplx(const RScalarJIT<T1>& s1, const RScalarJIT<T2>& s2)
{
  typedef typename BinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, FnCmplx>::Type_t  Ret_t;

  return Ret_t(s1.elem(),
	       s2.elem());
}



// RComplexJIT = i * RScalarJIT
template<class T>
struct UnaryReturn<RScalarJIT<T>, FnTimesI > {
  typedef RComplexJIT<typename UnaryReturn<T, FnTimesI>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalarJIT<T>, FnTimesI>::Type_t
timesI(const RScalarJIT<T>& s1)
{
  typename UnaryReturn<RScalarJIT<T>, FnTimesI>::Type_t  d;

  zero_rep(d.real());
  d.imag() = s1.elem();
  return d;
}

// RComplexJIT = i * RComplexJIT
template<class T>
inline typename UnaryReturn<RComplexJIT<T>, FnTimesI>::Type_t
timesI(const RComplexJIT<T>& s1)
{
  typedef typename UnaryReturn<RComplexJIT<T>, FnTimesI>::Type_t  Ret_t;

  return Ret_t(-s1.imag(),
	       s1.real());
}


// RComplexJIT = -i * RScalarJIT
template<class T>
struct UnaryReturn<RScalarJIT<T>, FnTimesMinusI > {
  typedef RComplexJIT<typename UnaryReturn<T, FnTimesMinusI>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalarJIT<T>, FnTimesMinusI>::Type_t
timesMinusI(const RScalarJIT<T>& s1)
{
  typename UnaryReturn<RScalarJIT<T>, FnTimesMinusI>::Type_t  d;

  zero_rep(d.real());
  d.imag() = -s1.elem();
  return d;
}


// RComplexJIT = -i * RComplexJIT
template<class T>
inline typename UnaryReturn<RComplexJIT<T>, FnTimesMinusI>::Type_t
timesMinusI(const RComplexJIT<T>& s1)
{
  typedef typename UnaryReturn<RComplexJIT<T>, FnTimesMinusI>::Type_t  Ret_t;

  return Ret_t(s1.imag(),
	       -s1.real());
}


//! RComplexJIT = outerProduct(RComplexJIT, RComplexJIT)
template<class T1, class T2>
inline typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, FnOuterProduct>::Type_t
outerProduct(const RComplexJIT<T1>& l, const RComplexJIT<T2>& r)
{
  typedef typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, FnOuterProduct>::Type_t  Ret_t;

  // Return   l*conj(r)
  return Ret_t(l.real()*r.real() + l.imag()*r.imag(),
	       l.imag()*r.real() - l.real()*r.imag());
}

//! RComplexJIT = outerProduct(RComplexJIT, RScalarJIT)
template<class T1, class T2>
inline typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, FnOuterProduct>::Type_t
outerProduct(const RComplexJIT<T1>& l, const RScalarJIT<T2>& r)
{
  typedef typename BinaryReturn<RComplexJIT<T1>, RScalarJIT<T2>, FnOuterProduct>::Type_t  Ret_t;

  // Return   l*conj(r)
  return Ret_t(l.real()*r.elem(),
	       l.imag()*r.elem());
}

//! RComplexJIT = outerProduct(RScalarJIT, RComplexJIT)
template<class T1, class T2>
inline typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, FnOuterProduct>::Type_t
outerProduct(const RScalarJIT<T1>& l, const RComplexJIT<T2>& r)
{
  typedef typename BinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, FnOuterProduct>::Type_t  Ret_t;

  // Return   l*conj(r)
  return Ret_t( l.elem()*r.real(),
	       -l.elem()*r.imag());
}


//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T>
inline typename UnaryReturn<RComplexJIT<T>, FnGetSite>::Type_t
getSite(const RComplexJIT<T>& s1, int innersite)
{
  typedef typename UnaryReturn<RComplexJIT<T>, FnGetSite>::Type_t  Ret_t;

  return Ret_t(getSite(s1.real(), innersite), 
	       getSite(s1.imag(), innersite));
}


//! dest = (mask) ? s1 : dest
template<class T, class T1> 
inline
void copymask(RComplexJIT<T>& d, const RScalarJIT<T1>& mask, const RComplexJIT<T>& s1) 
{
  copymask(d.real(),mask.elem(),s1.real());
  copymask(d.imag(),mask.elem(),s1.imag());
}


#if 1
// Global sum over site indices only
template<class T>
struct UnaryReturn<RComplexJIT<T>, FnSum> {
  typedef RComplexJIT<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RComplexJIT<T>, FnSum>::Type_t
sum(const RComplexJIT<T>& s1)
{
  typedef typename UnaryReturn<RComplexJIT<T>, FnSum>::Type_t  Ret_t;

  return Ret_t(sum(s1.real()),
	       sum(s1.imag()));
}
#endif


// InnerProduct (norm-seq) global sum = sum(tr(adj(s1)*s1))
template<class T>
struct UnaryReturn<RComplexJIT<T>, FnNorm2 > {
  typedef RScalarJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<RComplexJIT<T>, FnLocalNorm2 > {
  typedef RScalarJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RComplexJIT<T>, FnLocalNorm2>::Type_t
localNorm2(const RComplexJIT<T>& s1)
{
  return localNorm2(s1.real()) + localNorm2(s1.imag());
}



//! RComplexJIT<T> = InnerProduct(adj(RComplexJIT<T1>)*RComplexJIT<T2>)
template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, FnInnerProduct > {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, FnLocalInnerProduct > {
  typedef RComplexJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(const RComplexJIT<T1>& l, const RComplexJIT<T2>& r)
{
  typedef typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, FnLocalInnerProduct>::Type_t  Ret_t;

  return Ret_t(localInnerProduct(l.real(),r.real()) + localInnerProduct(l.imag(),r.imag()),
	       localInnerProduct(l.real(),r.imag()) - localInnerProduct(l.imag(),r.real()));
}


//! RScalarJIT<T> = InnerProductReal(adj(RComplexJIT<T1>)*RComplexJIT<T1>)
// Real-ness is eaten at this level
template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, FnInnerProductReal > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, FnLocalInnerProductReal > {
  typedef RScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RComplexJIT<T1>, RComplexJIT<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const RComplexJIT<T1>& l, const RComplexJIT<T2>& r)
{
  return localInnerProduct(l.real(),r.real()) + localInnerProduct(l.imag(),r.imag());
}


//! RComplexJIT<T> = where(RScalarJIT, RComplexJIT, RComplexJIT)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3>
struct TrinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, RComplexJIT<T3>, FnWhere> {
  typedef RComplexJIT<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
inline typename TrinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, RComplexJIT<T3>, FnWhere>::Type_t
where(const RScalarJIT<T1>& a, const RComplexJIT<T2>& b, const RComplexJIT<T3>& c)
{
  typedef typename TrinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, RComplexJIT<T3>, FnWhere>::Type_t  Ret_t;

  // Not optimal - want to have where outside assignment
  return Ret_t(where(a.elem(), b.real(), c.real()),
	       where(a.elem(), b.imag(), c.imag()));
}

//! RComplexJIT<T> = where(RScalarJIT, RComplexJIT, RScalarJIT)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3>
struct TrinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, RScalarJIT<T3>, FnWhere> {
  typedef RComplexJIT<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
inline typename TrinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, RComplexJIT<T3>, FnWhere>::Type_t
where(const RScalarJIT<T1>& a, const RComplexJIT<T2>& b, const RScalarJIT<T3>& c)
{
  typedef typename TrinaryReturn<RScalarJIT<T1>, RComplexJIT<T2>, RScalarJIT<T3>, FnWhere>::Type_t  Ret_t;
  typedef typename InternalScalar<T3>::Type_t  S;

  // Not optimal - want to have where outside assignment
  return Ret_t(where(a.elem(), b.real(), c.real()),
	       where(a.elem(), b.imag(), S(0)));
}

//! RComplexJIT<T> = where(RScalarJIT, RScalarJIT, RComplexJIT)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3>
struct TrinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, RComplexJIT<T3>, FnWhere> {
  typedef RComplexJIT<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
inline typename TrinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, RComplexJIT<T3>, FnWhere>::Type_t
where(const RScalarJIT<T1>& a, const RScalarJIT<T2>& b, const RComplexJIT<T3>& c)
{
  typedef typename TrinaryReturn<RScalarJIT<T1>, RScalarJIT<T2>, RComplexJIT<T3>, FnWhere>::Type_t  Ret_t;
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
void zero_rep(RComplexJIT<T>& dest) 
{
  zero_rep(dest.real());
  zero_rep(dest.imag());
}


//! dest  = random  
template<class T, class T1, class T2>
inline void
fill_random(RComplexJIT<T>& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  fill_random(d.real(), seed, skewed_seed, seed_mult);
  fill_random(d.imag(), seed, skewed_seed, seed_mult);
}


//! dest  = gaussian
/*! RComplexJIT polar method */
template<class T>
inline void
fill_gaussian(RComplexJIT<T>& d, RComplexJIT<T>& r1, RComplexJIT<T>& r2)
{
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
}

/*! @} */  // end of group rcomplex

} // namespace QDP

#endif
