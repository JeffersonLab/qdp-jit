// -*- C++ -*-

/*! \file
 * \brief Reality
 */


#ifndef QDP_WORD_H
#define QDP_WORD_H


#include <sstream>


namespace QDP {



template<class T> class Word
{
public:
  //  enum { Size = 1 };
  
  Word() {}

  
  ~Word() {}

  //---------------------------------------------------------
  //! construct dest = const
  
  Word(const typename WordType<T>::Type_t& rhs) : F(rhs) {}

  //! construct dest = rhs
  template<class T1>
  
  Word(const Word<T1>& rhs) : F(rhs.elem()) {}

  //! construct dest = rhs
  // template<class T1>
  // Word(const T1& rhs) : F(rhs) {}

  //---------------------------------------------------------
#if 0
  //! dest = const
  /*! Fill with a constant. Will be promoted to underlying word type */
  inline
  Word& operator=(const typename WordType<T>::Type_t& rhs)
    {
      elem() = rhs;
      return *this;
    }
#endif

  //! Word = Word
  /*! Set equal to another Word */
  template<class T1>
   inline
  Word& operator=(const Word<T1>& rhs) 
    {
      elem() = rhs.elem();
      return *this;
    }

  //! Word += Word
  template<class T1>
   inline
  Word& operator+=(const Word<T1>& rhs) 
    {
      elem() += rhs.elem();
      return *this;
    }

  //! Word -= Word
  template<class T1>
   inline
  Word& operator-=(const Word<T1>& rhs) 
    {
      elem() -= rhs.elem();
      return *this;
    }

  //! Word *= Word
  template<class T1>
   inline
  Word& operator*=(const Word<T1>& rhs) 
    {
      elem() *= rhs.elem();
      return *this;
    }

  //! Word /= Word
  template<class T1>
   inline
  Word& operator/=(const Word<T1>& rhs) 
    {
      elem() /= rhs.elem();
      return *this;
    }

  //! Word %= Word
  template<class T1>
   inline
  Word& operator%=(const Word<T1>& rhs) 
    {
      elem() %= rhs.elem();
      return *this;
    }

  //! Word |= Word
  template<class T1>
   inline
  Word& operator|=(const Word<T1>& rhs) 
    {
      elem() |= rhs.elem();
      return *this;
    }

  //! Word &= Word
  template<class T1>
   inline
  Word& operator&=(const Word<T1>& rhs) 
    {
      elem() &= rhs.elem();
      return *this;
    }

  //! Word ^= Word
  template<class T1>
   inline
  Word& operator^=(const Word<T1>& rhs) 
    {
      elem() ^= rhs.elem();
      return *this;
    }

  //! Word <<= Word
  template<class T1>
   inline
  Word& operator<<=(const Word<T1>& rhs) 
    {
      elem() <<= rhs.elem();
      return *this;
    }

  //! Word >>= Word
  template<class T1>
   inline
  Word& operator>>=(const Word<T1>& rhs) 
    {
      elem() >>= rhs.elem();
      return *this;
    }


  //! Do deep copies here
   Word(const Word& a): F(a.F) {}

public:
   T& elem() {return F;}
   const T& elem() const {return F;}

private:
  T F;
};




template<class T> 
struct JITType<Word<T> >
{
  typedef WordJIT<typename JITType<T>::Type_t>  Type_t;
};



// Input
//! Ascii input
template<class T>
inline
istream& operator>>(istream& s, Word<T>& d)
{
  return s >> d.elem();
}

//! Ascii input
template<class T>
inline
StandardInputStream& operator>>(StandardInputStream& s, Word<T>& d)
{
  return s >> d.elem();
}

//! Ascii output
template<class T> 
inline  
ostream& operator<<(ostream& s, const Word<T>& d)
{
  return s << d.elem();
}

//! Ascii output
template<class T> 
inline  
StandardOutputStream& operator<<(StandardOutputStream& s, const Word<T>& d)
{
  return s << d.elem();
}


//! Text input
template<class T>
inline
TextReader& operator>>(TextReader& s, Word<T>& d)
{
  return s >> d.elem();
}

//! Text output
template<class T> 
inline  
TextWriter& operator<<(TextWriter& s, const Word<T>& d)
{
  return s << d.elem();
}

#ifndef QDP_NO_LIBXML2
//! XML output
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, const Word<T>& d)
{
  return xml << d.elem();
}

//! XML input
template<class T>
inline
void read(XMLReader& xml, const string& path, Word<T>& d)
{
  read(xml, path, d.elem());
}
#endif




// Underlying word type
template<class T>
struct WordType<Word<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

// Fixed types
template<class T> 
struct SinglePrecType<Word<T> >
{
  typedef Word<typename SinglePrecType<T>::Type_t>  Type_t;
};

template<class T> 
struct DoublePrecType<Word<T> >
{
  typedef Word<typename DoublePrecType<T>::Type_t>  Type_t;
};


// Internally used scalars
template<class T>
struct InternalScalar<Word<T> > {
  typedef Word<typename InternalScalar<T>::Type_t>  Type_t;
};



// Makes a primitive scalar leaving grid alone
template<class T>
struct PrimitiveScalar<Word<T> > {
  typedef Word<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T>
struct LatticeScalar<Word<T> > {
  typedef Word<typename LatticeScalar<T>::Type_t>  Type_t;
};


// Internally used real scalars
template<class T>
struct RealScalar<Word<T> > {
  typedef Word<typename RealScalar<T>::Type_t>  Type_t;
};


//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(Word) -> Word
template<class T1, class Op>
struct UnaryReturn<Word<T1>, Op> {
  typedef Word<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

// Default binary(Word,Word) -> Word
template<class T1, class T2, class Op>
struct BinaryReturn<Word<T1>, Word<T2>, Op> {
  typedef Word<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Word
#if 0
template<class T1, class T2>
struct UnaryReturn<Word<T2>, OpCast<T1> > {
  typedef Word<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif


template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, OpAddAssign > {
  typedef Word<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, OpSubtractAssign > {
  typedef Word<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, OpMultiplyAssign > {
  typedef Word<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, OpDivideAssign > {
  typedef Word<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, OpModAssign > {
  typedef Word<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, OpBitwiseOrAssign > {
  typedef Word<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, OpBitwiseAndAssign > {
  typedef Word<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, OpBitwiseXorAssign > {
  typedef Word<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, OpLeftShiftAssign > {
  typedef Word<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, OpRightShiftAssign > {
  typedef Word<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2, class T3>
struct TrinaryReturn<Word<T1>, Word<T2>, Word<T3>, FnColorContract> {
  typedef Word<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};

// Word
// Gamma algebra
template<int N, int m, class T2, class OpGammaConstMultiply>
struct BinaryReturn<GammaConst<N,m>, Word<T2>, OpGammaConstMultiply> {
  typedef Word<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConst>
struct BinaryReturn<Word<T2>, GammaConst<N,m>, OpMultiplyGammaConst> {
  typedef Word<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeMultiply>
struct BinaryReturn<GammaType<N>, Word<T2>, OpGammaTypeMultiply> {
  typedef Word<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaType>
struct BinaryReturn<Word<T2>, GammaType<N>, OpMultiplyGammaType> {
  typedef Word<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};


// Word
// Gamma algebra
template<int N, int m, class T2, class OpGammaConstDPMultiply>
struct BinaryReturn<GammaConstDP<N,m>, Word<T2>, OpGammaConstDPMultiply> {
  typedef Word<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConstDP>
struct BinaryReturn<Word<T2>, GammaConstDP<N,m>, OpMultiplyGammaConstDP> {
  typedef Word<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeDPMultiply>
struct BinaryReturn<GammaTypeDP<N>, Word<T2>, OpGammaTypeDPMultiply> {
  typedef Word<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaTypeDP>
struct BinaryReturn<Word<T2>, GammaTypeDP<N>, OpMultiplyGammaTypeDP> {
  typedef Word<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};






// Scalar Reality
template<class T>
struct UnaryReturn<Word<T>, OpNot > {
  typedef Word<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
};

template<class T1>
 inline typename UnaryReturn<Word<T1>, OpNot>::Type_t
operator!(const Word<T1>& l)
{
  return ! l.elem();
}


template<class T1>
 inline typename UnaryReturn<Word<T1>, OpUnaryPlus>::Type_t
operator+(const Word<T1>& l)
{
  return +l.elem();
}


template<class T1>
 inline typename UnaryReturn<Word<T1>, OpUnaryMinus>::Type_t
operator-(const Word<T1>& l)
{
  return -l.elem();
}


template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpAdd>::Type_t
operator+(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem()+r.elem();
}


template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpSubtract>::Type_t
operator-(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() - r.elem();
}


template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpMultiply>::Type_t
operator*(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() * r.elem();
}


// Optimized  adj(Word)*Word
template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpAdjMultiply>::Type_t
adjMultiply(const Word<T1>& l, const Word<T2>& r)
{
  /*! NOTE: removed transpose here !!!!!  */

//  return transpose(l.elem()) * r.elem();
  return l.elem() * r.elem();
}

// Optimized  Word*adj(Word)
template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(const Word<T1>& l, const Word<T2>& r)
{
  /*! NOTE: removed transpose here !!!!!  */

//  return l.elem() * transpose(r.elem());
  return l.elem() * r.elem();
}

// Optimized  adj(Word)*adj(Word)
template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const Word<T1>& l, const Word<T2>& r)
{
  /*! NOTE: removed transpose here !!!!!  */

//  return transpose(l.elem()) * transpose(r.elem());
  return l.elem() * r.elem();
}


template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpDivide>::Type_t
operator/(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() / r.elem();
}



template<class T1, class T2 >
struct BinaryReturn<Word<T1>, Word<T2>, OpLeftShift > {
  typedef Word<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpLeftShift>::Type_t
operator<<(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() << r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<Word<T1>, Word<T2>, OpRightShift > {
  typedef Word<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpRightShift>::Type_t
operator>>(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() >> r.elem();
}


template<class T1, class T2 >
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpMod>::Type_t
operator%(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() % r.elem();
}

template<class T1, class T2 >
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpBitwiseXor>::Type_t
operator^(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() ^ r.elem();
}

template<class T1, class T2 >
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpBitwiseAnd>::Type_t
operator&(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() & r.elem();
}

template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpBitwiseOr>::Type_t
operator|(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() | r.elem();
}



// Comparisons
template<class T1, class T2 >
struct BinaryReturn<Word<T1>, Word<T2>, OpLT > {
  typedef Word<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpLT>::Type_t
operator<(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() < r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<Word<T1>, Word<T2>, OpLE > {
  typedef Word<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpLE>::Type_t
operator<=(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() <= r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<Word<T1>, Word<T2>, OpGT > {
  typedef Word<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpGT>::Type_t
operator>(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() > r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<Word<T1>, Word<T2>, OpGE > {
  typedef Word<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpGE>::Type_t
operator>=(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() >= r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<Word<T1>, Word<T2>, OpEQ > {
  typedef Word<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpEQ>::Type_t
operator==(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() == r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<Word<T1>, Word<T2>, OpNE > {
  typedef Word<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpNE>::Type_t
operator!=(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() != r.elem();
}


template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, OpAnd > {
  typedef Word<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpAnd>::Type_t
operator&&(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() && r.elem();
}


template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, OpOr > {
  typedef Word<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, OpOr>::Type_t
operator||(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() || r.elem();
}



//-----------------------------------------------------------------------------
// Functions

// Adjoint
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnAdjoint>::Type_t
adj(const Word<T1>& s1)
{
  /*! NOTE: removed transpose here !!!!!  */

//  return transpose(s1.elem()); // The complex nature has been eaten here
  return s1.elem(); // The complex nature has been eaten here
}


// Conjugate
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnConjugate>::Type_t
conj(const Word<T1>& s1)
{
  return s1.elem();  // The complex nature has been eaten here
}


// Transpose
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnTranspose>::Type_t
transpose(const Word<T1>& s1)
{
  /*! NOTE: removed transpose here !!!!!  */

//  return transpose(s1.elem());
  return s1.elem();
}



// TRACE
// trace = Trace(source1)
template<class T>
struct UnaryReturn<Word<T>, FnTrace > {
  typedef Word<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T1>
 inline typename UnaryReturn<Word<T1>, FnTrace>::Type_t
trace(const Word<T1>& s1)
{
//  return trace(s1.elem());

  /*! NOTE: removed trace here !!!!!  */
  return s1.elem();
}


// trace = Re(Trace(source1))
template<class T>
struct UnaryReturn<Word<T>, FnRealTrace > {
  typedef Word<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T1>
 inline typename UnaryReturn<Word<T1>, FnRealTrace>::Type_t
realTrace(const Word<T1>& s1)
{
//  return trace_real(s1.elem());

  /*! NOTE: removed trace here !!!!!  */
  return s1.elem();
}


// trace = Im(Trace(source1))
template<class T>
struct UnaryReturn<Word<T>, FnImagTrace > {
  typedef Word<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T1>
 inline typename UnaryReturn<Word<T1>, FnImagTrace>::Type_t
imagTrace(const Word<T1>& s1)
{
//  return trace_imag(s1.elem());

  /*! NOTE: removed trace here !!!!!  */
  return s1.elem();
}

//! Word = trace(Word * Word)
template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, FnTraceMultiply>::Type_t
traceMultiply(const Word<T1>& l, const Word<T2>& r)
{
//  return traceMultiply(l.elem(), r.elem());

  /*! NOTE: removed trace here !!!!!  */
  return l.elem() * r.elem();
}


// Word = Re(Word)  [identity]
template<class T>
 inline typename UnaryReturn<Word<T>, FnReal>::Type_t
real(const Word<T>& s1)
{
  return s1.elem();
}


// Word = Im(Word) [this is zero]
template<class T>
 inline typename UnaryReturn<Word<T>, FnImag>::Type_t
imag(const Word<T>& s1)
{
  typedef typename InternalScalar<T>::Type_t  S;
  return S(0);
}


// ArcCos
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnArcCos>::Type_t
acos(const Word<T1>& s1)
{
  return acos(s1.elem());
}

// ArcSin
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnArcSin>::Type_t
asin(const Word<T1>& s1)
{
  return asin(s1.elem());
}

// ArcTan
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnArcTan>::Type_t
atan(const Word<T1>& s1)
{
  return atan(s1.elem());
}

// Ceil(ing)
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnCeil>::Type_t
ceil(const Word<T1>& s1)
{
  return ceil(s1.elem());
}

// Cos
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnCos>::Type_t
cos(const Word<T1>& s1)
{
  return cos(s1.elem());
}

// Cosh
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnHypCos>::Type_t
cosh(const Word<T1>& s1)
{
  return cosh(s1.elem());
}

// Exp
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnExp>::Type_t
exp(const Word<T1>& s1)
{
  return exp(s1.elem());
}

// Fabs
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnFabs>::Type_t
fabs(const Word<T1>& s1)
{
  return fabs(s1.elem());
}

// Floor
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnFloor>::Type_t
floor(const Word<T1>& s1)
{
  return floor(s1.elem());
}

// Log
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnLog>::Type_t
log(const Word<T1>& s1)
{
  return log(s1.elem());
}

// Log10
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnLog10>::Type_t
log10(const Word<T1>& s1)
{
  return log10(s1.elem());
}

// Sin
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnSin>::Type_t
sin(const Word<T1>& s1)
{
  return sin(s1.elem());
}

// Sinh
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnHypSin>::Type_t
sinh(const Word<T1>& s1)
{
  return sinh(s1.elem());
}

// Sqrt
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnSqrt>::Type_t
sqrt(const Word<T1>& s1)
{
  return sqrt(s1.elem());
}

// Tan
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnTan>::Type_t
tan(const Word<T1>& s1)
{
  return tan(s1.elem());
}

// Tanh
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnHypTan>::Type_t
tanh(const Word<T1>& s1)
{
  return tanh(s1.elem());
}


//-----------------------------------------------------------------------------
// These functions always return bool
//! isnan
template<class T1>
inline bool
isnan(const Word<T1>& s1)
{
  return isnan(s1.elem());
}

//! isinf
template<class T1>
inline bool
isinf(const Word<T1>& s1)
{
  return isinf(s1.elem());
}

//! isnormal
template<class T1>
inline bool
isnormal(const Word<T1>& s1)
{
  return isnormal(s1.elem());
}

//! isfinite
template<class T1>
inline bool
isfinite(const Word<T1>& s1)
{
  return isfinite(s1.elem());
}




//! Word<T> = pow(Word<T> , Word<T>)
template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, FnPow>::Type_t
pow(const Word<T1>& s1, const Word<T2>& s2)
{
  return pow(s1.elem(), s2.elem());
}

//! Word<T> = atan2(Word<T> , Word<T>)
template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, FnArcTan2>::Type_t
atan2(const Word<T1>& s1, const Word<T2>& s2)
{
  return atan2(s1.elem(), s2.elem());
}


//! Word = outerProduct(Word, Word)
template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, FnOuterProduct>::Type_t
outerProduct(const Word<T1>& l, const Word<T2>& r)
{
  return l.elem() * r.elem();
}


//! dest [float type] = source [seed type]
template<class T1>
 inline typename UnaryReturn<Word<T1>, FnSeedToFloat>::Type_t
seedToFloat(const Word<T1>& s1)
{
  return seedToFloat(s1.elem());
}

//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T>
 inline typename UnaryReturn<Word<T>, FnGetSite>::Type_t
getSite(const Word<T>& s1, int innersite)
{
  return getSite(s1.elem(), innersite);
}

//! Extract color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T>
 inline typename UnaryReturn<Word<T>, FnPeekColorVector>::Type_t
peekColor(const Word<T>& l, int row)
{
  return peekColor(l.elem(),row);
}

//! Extract color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T>
 inline typename UnaryReturn<Word<T>, FnPeekColorMatrix>::Type_t
peekColor(const Word<T>& l, int row, int col)
{
  return peekColor(l.elem(),row,col);
}

//! Extract spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
 inline typename UnaryReturn<Word<T>, FnPeekSpinVector>::Type_t
peekSpin(const Word<T>& l, int row)
{
  return peekSpin(l.elem(),row);
}

//! Extract spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
 inline typename UnaryReturn<Word<T>, FnPeekSpinMatrix>::Type_t
peekSpin(const Word<T>& l, int row, int col)
{
  return peekSpin(l.elem(),row,col);
}



template<class T, class T1, class T2>
inline void
fill_random(Word<T>& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  fill_random(d.elem(), seed, skewed_seed, seed_mult);
}




//-----------------------------------------------------------------------------
//! QDP Int to int primitive in conversion routine
template<class T> 
 inline int 
toInt(const Word<T>& s) 
{
  return toInt(s.elem());
}

//! QDP Real to float primitive in conversion routine
template<class T> 
 inline float
toFloat(const Word<T>& s) 
{
  return toFloat(s.elem());
}

//! QDP Double to double primitive in conversion routine
template<class T> 
 inline double
toDouble(const Word<T>& s) 
{
  return toDouble(s.elem());
}

//! QDP Boolean to bool primitive in conversion routine
template<class T> 
 inline bool
toBool(const Word<T>& s) 
{
  return toBool(s.elem());
}

//! QDP Wordtype to primitive wordtype
template<class T> 
 inline typename WordType< Word<T> >::Type_t
toWordType(const Word<T>& s) 
{
  return toWordType(s.elem());
}



//------------------------------------------
//! dest = (mask) ? s1 : dest
template<class T, class T1> 
 inline
void copymask(Word<T>& d, const Word<T1>& mask, const Word<T>& s1) 
{
  copymask(d.elem(),mask.elem(),s1.elem());
}

//! dest [float type] = source [int type]
template<class T, class T1>
 inline
void cast_rep(T& d, const Word<T1>& s1)
{
  cast_rep(d, s1.elem());
}


//! dest [float type] = source [int type]
template<class T, class T1>
 inline
void recast_rep(Word<T>& d, const Word<T1>& s1)
{
  cast_rep(d.elem(), s1.elem());
}


//! dest [some type] = source [some type]
template<class T, class T1>
 inline void 
copy_site(Word<T>& d, int isite, const Word<T1>& s1)
{
  copy_site(d.elem(), isite, s1.elem());
}


//! gather several inner sites together
template<class T, class T1>
 inline void 
gather_sites(Word<T>& d, 
	     const Word<T1>& s0, int i0, 
	     const Word<T1>& s1, int i1,
	     const Word<T1>& s2, int i2,
	     const Word<T1>& s3, int i3)
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
struct UnaryReturn<Word<T>, FnSum > {
  typedef Word<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
 inline typename UnaryReturn<Word<T>, FnSum>::Type_t
sum(const Word<T>& s1)
{
  return sum(s1.elem());
}
#endif


// Global max
template<class T>
struct UnaryReturn<Word<T>, FnGlobalMax> {
  typedef Word<typename UnaryReturn<T, FnGlobalMax>::Type_t>  Type_t;
};

template<class T>
 inline typename UnaryReturn<Word<T>, FnGlobalMax>::Type_t
globalMax(const Word<T>& s1)
{
  return globalMax(s1.elem());
}


// Global min
template<class T>
struct UnaryReturn<Word<T>, FnGlobalMin> {
  typedef Word<typename UnaryReturn<T, FnGlobalMin>::Type_t>  Type_t;
};

template<class T>
 inline typename UnaryReturn<Word<T>, FnGlobalMin>::Type_t
globalMin(const Word<T>& s1)
{
  return globalMin(s1.elem());
}



//------------------------------------------
// InnerProduct (norm-seq) global sum = sum(tr(adj(s1)*s1))
template<class T>
struct UnaryReturn<Word<T>, FnNorm2 > {
  typedef Word<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<Word<T>, FnLocalNorm2 > {
  typedef Word<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T>
 inline typename UnaryReturn<Word<T>, FnLocalNorm2>::Type_t
localNorm2(const Word<T>& s1)
{
  return localNorm2(s1.elem());
}



//! Word<T> = InnerProduct(adj(Word<T1>)*Word<T2>)
template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, FnInnerProduct > {
  typedef Word<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, FnLocalInnerProduct > {
  typedef Word<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(const Word<T1>& s1, const Word<T2>& s2)
{
  return localInnerProduct(s1.elem(), s2.elem());
}


//! Word<T> = InnerProductReal(adj(PMatrix<T1>)*PMatrix<T1>)
// Real-ness is eaten at this level
template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, FnInnerProductReal > {
  typedef Word<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, FnLocalInnerProductReal > {
  typedef Word<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const Word<T1>& s1, const Word<T2>& s2)
{
  return localInnerProduct(s1.elem(), s2.elem());
}


//! Word<T> = where(Word, Word, Word)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3>
struct TrinaryReturn<Word<T1>, Word<T2>, Word<T3>, FnWhere> {
  typedef Word<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
 inline typename TrinaryReturn<Word<T1>, Word<T2>, Word<T3>, FnWhere>::Type_t
where(const Word<T1>& a, const Word<T2>& b, const Word<T3>& c)
{
  return where(a.elem(), b.elem(), c.elem());
}



//-----------------------------------------------------------------------------
// Broadcast operations
//! dest = 0
template<class T> 
 inline
void zero_rep(Word<T>& dest) 
{
  zero_rep(dest.elem());
}


//! RComplex<T> = (Word<T> , Word<T>)
template<class T1, class T2>
struct BinaryReturn<Word<T1>, Word<T2>, FnCmplx > {
  typedef Word<typename BinaryReturn<T1, T2, FnCmplx>::Type_t>  Type_t;
};

template<class T1, class T2>
 inline typename BinaryReturn<Word<T1>, Word<T2>, FnCmplx>::Type_t
cmplx(const Word<T1>& s1, const Word<T2>& s2)
{
  typedef typename BinaryReturn<Word<T1>, Word<T2>, FnCmplx>::Type_t  Ret_t;

  return Ret_t(s1.elem(),
	       s2.elem());
}



// RComplex = i * Word
template<class T>
struct UnaryReturn<Word<T>, FnTimesI > {
  typedef Word<typename UnaryReturn<T, FnTimesI>::Type_t>  Type_t;
};

template<class T>
 inline typename UnaryReturn<Word<T>, FnTimesI>::Type_t
timesI(const Word<T>& s1)
{
  typename UnaryReturn<Word<T>, FnTimesI>::Type_t  d;

  zero_rep(d.real());
  d.imag() = s1.elem();
  return d;
}


// RComplex = -i * Word
template<class T>
struct UnaryReturn<Word<T>, FnTimesMinusI > {
  typedef Word<typename UnaryReturn<T, FnTimesMinusI>::Type_t>  Type_t;
};

template<class T>
 inline typename UnaryReturn<Word<T>, FnTimesMinusI>::Type_t
timesMinusI(const Word<T>& s1)
{
  typename UnaryReturn<Word<T>, FnTimesMinusI>::Type_t  d;

  zero_rep(d.real());
  d.imag() = -s1.elem();
  return d;
}



} // namespace QDP

#endif
