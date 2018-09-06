// -*- C++ -*-

/*! \file
 * \brief Primitive Scalar
 */

#ifndef QDP_PRIMSCALARREG_H
#define QDP_PRIMSCALARREG_H

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
  template<class T> class PScalarREG //: public BaseREG<T,1,PScalarREG<T> >
  {
    T F;
  public:

    void setup(const PScalarJIT< typename JITType<T>::Type_t >& rhs ) {
      F.setup( rhs.elem() );
    }

    PScalarREG(const typename WordType<T>::Type_t& rhs): F(rhs) {}


    // Default constructing should be possible
    // then there is no need for MPL index when
    // construction a PMatrix<T,N>
    PScalarREG() {}
    ~PScalarREG() {}


    PScalarREG(const T& rhs) {
      elem() = rhs;
    }


    //! PScalarREG += PScalarREG
    template<class T1>
    inline
    PScalarREG& operator+=(const PScalarREG<T1>& rhs) 
    {
      elem() += rhs.elem();
      return *this;
    }

    //! PScalarREG -= PScalarREG
    template<class T1>
    inline
    PScalarREG& operator-=(const PScalarREG<T1>& rhs) 
    {
      elem() -= rhs.elem();
      return *this;
    }

    //! PScalarREG *= PScalarREG
    template<class T1>
    inline
    PScalarREG& operator*=(const PScalarREG<T1>& rhs) 
    {
      elem() *= rhs.elem();
      return *this;
    }

    //! PScalarREG /= PScalarREG
    template<class T1>
    inline
    PScalarREG& operator/=(const PScalarREG<T1>& rhs) 
    {
      elem() /= rhs.elem();
      return *this;
    }

    //! PScalarREG %= PScalarREG
    template<class T1>
    inline
    PScalarREG& operator%=(const PScalarREG<T1>& rhs) 
    {
      elem() %= rhs.elem();
      return *this;
    }

    //! PScalarREG |= PScalarREG
    template<class T1>
    inline
    PScalarREG& operator|=(const PScalarREG<T1>& rhs) 
    {
      elem() |= rhs.elem();
      return *this;
    }

    //! PScalarREG &= PScalarREG
    template<class T1>
    inline
    PScalarREG& operator&=(const PScalarREG<T1>& rhs) 
    {
      elem() &= rhs.elem();
      return *this;
    }

    //! PScalarREG ^= PScalarREG
    template<class T1>
    inline
    PScalarREG& operator^=(const PScalarREG<T1>& rhs) 
    {
      elem() ^= rhs.elem();
      return *this;
    }

    //! PScalarREG <<= PScalarREG
    template<class T1>
    inline
    PScalarREG& operator<<=(const PScalarREG<T1>& rhs) 
    {
      elem() <<= rhs.elem();
      return *this;
    }

    //! PScalarREG >>= PScalarREG
    template<class T1>
    inline
    PScalarREG& operator>>=(const PScalarREG<T1>& rhs) 
    {
      elem() >>= rhs.elem();
      return *this;
    }


    inline       T& elem()       { return F; }
    inline const T& elem() const { return F; }

    // inline       T& elem()       { return this->arrayF(0); }
    // inline const T& elem() const { return this->arrayF(0); }
  };




template<class T> 
struct JITType< PScalarREG<T> >
{
  typedef PScalarJIT<typename JITType<T>::Type_t>  Type_t;
};







// Input
//! Ascii input
template<class T>
inline
istream& operator>>(istream& s, PScalarREG<T>& d)
{
  return s >> d.elem();
}

//! Ascii input
template<class T>
inline
StandardInputStream& operator>>(StandardInputStream& s, PScalarREG<T>& d)
{
  return s >> d.elem();
}

// Output
//! Ascii output
template<class T>
inline
ostream& operator<<(ostream& s, const PScalarREG<T>& d)
{
  return s << d.elem();
}

//! Ascii output
template<class T>
inline
StandardOutputStream& operator<<(StandardOutputStream& s, const PScalarREG<T>& d)
{
  return s << d.elem();
}

//! Text input
template<class T>
inline
TextReader& operator>>(TextReader& txt, PScalarREG<T>& d)
{
  return txt >> d.elem();
}

//! Text output
template<class T>
inline
TextWriter& operator<<(TextWriter& txt, const PScalarREG<T>& d)
{
  return txt << d.elem();
}

#ifndef QDP_NO_LIBXML2
//! XML output
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, const PScalarREG<T>& d)
{
  return xml << d.elem();
}

//! XML input
template<class T>
inline
void read(XMLReader& xml, const string& path, PScalarREG<T>& d)
{
  read(xml, path, d.elem());
}
#endif

/*! @} */  // end of group primscalar


//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

// Underlying word type
template<class T>
struct WordType<PScalarREG<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

// Fixed Precision Types 
template<class T>
struct SinglePrecType<PScalarREG<T> >
{
  typedef PScalarREG< typename SinglePrecType<T>::Type_t > Type_t;
};

template<class T>
struct DoublePrecType<PScalarREG<T> >
{
  typedef PScalarREG< typename DoublePrecType<T>::Type_t > Type_t;
};

// Internally used scalars
template<class T>
struct InternalScalar<PScalarREG<T> > {
  typedef PScalarREG<typename InternalScalar<T>::Type_t>  Type_t;
};

// Internally used real scalars
template<class T>
struct RealScalar<PScalarREG<T> > {
  typedef PScalarREG<typename RealScalar<T>::Type_t>  Type_t;
};

// Makes a primitive scalar leaving grid alone
template<class T>
struct PrimitiveScalar<PScalarREG<T> > {
  typedef PScalarREG<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T>
struct LatticeScalar<PScalarREG<T> > {
  typedef PScalarREG<typename LatticeScalar<T>::Type_t>  Type_t;
};


//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PScalarREG) -> PScalarREG
template<class T1, class Op>
struct UnaryReturn<PScalarREG<T1>, Op> {
  typedef PScalarREG<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

// Default binary(PScalarREG,PScalarREG) -> PScalarREG
template<class T1, class T2, class Op>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, Op> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


#if 0
template<class T1, class T2>
struct UnaryReturn<PScalarREG<T2>, OpCast<T1> > {
  typedef PScalarREG<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif

// Assignment is different
template<class T1, class T2 >
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpAssign > {
  typedef PScalarREG<T1> &Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpAddAssign > {
  typedef PScalarREG<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpSubtractAssign > {
  typedef PScalarREG<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpMultiplyAssign > {
  typedef PScalarREG<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpDivideAssign > {
  typedef PScalarREG<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpModAssign > {
  typedef PScalarREG<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpBitwiseOrAssign > {
  typedef PScalarREG<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpBitwiseAndAssign > {
  typedef PScalarREG<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpBitwiseXorAssign > {
  typedef PScalarREG<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpLeftShiftAssign > {
  typedef PScalarREG<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpRightShiftAssign > {
  typedef PScalarREG<T1> &Type_t;
};
 



//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

/*! \addtogroup primscalar */
/*! @{ */

// Primitive Scalars

// ! PScalarREG
template<class T>
struct UnaryReturn<PScalarREG<T>, OpNot > {
  typedef PScalarREG<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, OpNot>::Type_t
operator!(const PScalarREG<T1>& l)
{
  return ! l.elem();
}

// + PScalarREG
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, OpUnaryPlus>::Type_t
operator+(const PScalarREG<T1>& l)
{
  return +l.elem();
}

// - PScalarREG
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, OpUnaryMinus>::Type_t
operator-(const PScalarREG<T1>& l)
{
  return -l.elem();
}

  //PScalarREG + PScalarREG
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpAdd>::Type_t
operator+(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() + r.elem();
}


// PScalarREG - PScalarREG
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpSubtract>::Type_t
operator-(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() - r.elem();
}


  //PScalarREG + PScalarREG
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpMultiply>::Type_t
operator*(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() * r.elem();
}



// Optimized  adj(PMatrix)*PMatrix
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpAdjMultiply>::Type_t
adjMultiply(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return adjMultiply(l.elem(), r.elem());
}


template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return multiplyAdj(l.elem(), r.elem());
}

// Optimized  PMatrix*adj(PMatrix)
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PSpinVector<T2,4>, OpMultiplyAdj>::Type_t
multiplyAdj(const PScalarREG<T1>& l, const PSpinVector<T2,4>& r)
{
  return multiplyAdj(l.elem(), r.elem());
}

// Optimized  adj(PMatrix)*adj(PMatrix)
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return adjMultiplyAdj(l.elem(), r.elem());
}

// PScalarREG / PScalarREG
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpDivide>::Type_t
operator/(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() / r.elem();
}


// PScalarREG << PScalarREG
template<class T1, class T2 >
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpLeftShift > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpLeftShift>::Type_t
operator<<(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() << r.elem();
}

// PScalarREG >> PScalarREG
template<class T1, class T2 >
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpRightShift > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpRightShift>::Type_t
operator>>(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() >> r.elem();
}

// PScalarREG % PScalarREG
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpMod>::Type_t
operator%(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() % r.elem();
}

// PScalarREG ^ PScalarREG
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpBitwiseXor>::Type_t
operator^(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() ^ r.elem();
}

// PScalarREG & PScalarREG
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpBitwiseAnd>::Type_t
operator&(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() & r.elem();
}

// PScalarREG | PScalarREG
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpBitwiseOr>::Type_t
operator|(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() | r.elem();
}


// Comparisons
template<class T1, class T2 >
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpLT > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpLT>::Type_t
operator<(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() < r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpLE > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpLE>::Type_t
operator<=(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() <= r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpGT > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpGT>::Type_t
operator>(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() > r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpGE > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpGE>::Type_t
operator>=(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() >= r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpEQ > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpEQ>::Type_t
operator==(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() == r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpNE > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpNE>::Type_t
operator!=(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() != r.elem();
}


template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpAnd > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpAnd>::Type_t
operator&&(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() && r.elem();
}


template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpOr > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, OpOr>::Type_t
operator||(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return l.elem() || r.elem();
}


//-----------------------------------------------------------------------------
// Functions

// Adjoint
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnAdjoint>::Type_t
adj(const PScalarREG<T1>& s1)
{
  return adj(s1.elem());
}


// Conjugate
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnConjugate>::Type_t
conj(const PScalarREG<T1>& s1)
{
  return conj(s1.elem());
}


// Transpose
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnTranspose>::Type_t
transpose(const PScalarREG<T1>& s1)
{
  return transpose(s1.elem());
}


// TRACE
// trace = Trace(source1)
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnTrace>::Type_t
trace(const PScalarREG<T1>& s1)
{
  return trace(s1.elem());
}


// trace = Re(Trace(source1))
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnRealTrace>::Type_t
realTrace(const PScalarREG<T1>& s1)
{
  return realTrace(s1.elem());
}


// trace = Im(Trace(source1))
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnImagTrace>::Type_t
imagTrace(const PScalarREG<T1>& s1)
{
  return imagTrace(s1.elem());
}


// trace = colorTrace(source1)
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnTraceColor>::Type_t
traceColor(const PScalarREG<T1>& s1)
{
  return traceColor(s1.elem());
}


//! PScalarREG = traceSpin(PScalarREG)
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnTraceSpin>::Type_t
traceSpin(const PScalarREG<T1>& s1)
{
  return traceSpin(s1.elem());
}

//! PScalarREG = transposeSpin(PScalarREG)
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnTransposeSpin>::Type_t
transposeSpin(const PScalarREG<T1>& s1)
{
  return transposeSpin(s1.elem());
}

//! PScalarREG = trace(PScalarREG * PScalarREG)
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnTraceMultiply>::Type_t
traceMultiply(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return traceMultiply(l.elem(), r.elem());
}

//! PScalarREG = traceColor(PScalarREG * PScalarREG)
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnTraceColorMultiply>::Type_t
traceColorMultiply(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return traceMultiply(l.elem(), r.elem());
}

//! PScalarREG = traceSpin(PScalarREG * PScalarREG)
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return traceMultiply(l.elem(), r.elem());
}

//! PScalarREG = traceSpin(outerProduct(PScalarREG, PScalarREG))
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnTraceSpinOuterProduct>::Type_t
traceSpinOuterProduct(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return traceSpinOuterProduct(l.elem(), r.elem());
}

//! PScalarREG = outerProduct(PScalarREG, PScalarREG)
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnOuterProduct>::Type_t
outerProduct(const PScalarREG<T1>& l, const PScalarREG<T2>& r)
{
  return outerProduct(l.elem(),r.elem());
}


//! PScalarREG = Re(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnReal>::Type_t
real(const PScalarREG<T>& s1)
{
  return real(s1.elem());
}


// PScalarREG = Im(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnImag>::Type_t
imag(const PScalarREG<T>& s1)
{
  return imag(s1.elem());
}


// ArcCos
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnArcCos>::Type_t
acos(const PScalarREG<T1>& s1)
{
  return acos(s1.elem());
}

// ArcSin
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnArcSin>::Type_t
asin(const PScalarREG<T1>& s1)
{
  return asin(s1.elem());
}

// ArcTan
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnArcTan>::Type_t
atan(const PScalarREG<T1>& s1)
{
  return atan(s1.elem());
}

// Ceil(ing)
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnCeil>::Type_t
ceil(const PScalarREG<T1>& s1)
{
  return ceil(s1.elem());
}

// Cos
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnCos>::Type_t
cos(const PScalarREG<T1>& s1)
{
  return cos(s1.elem());
}

// Cosh
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnHypCos>::Type_t
cosh(const PScalarREG<T1>& s1)
{
  return cosh(s1.elem());
}

// Exp
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnExp>::Type_t
exp(const PScalarREG<T1>& s1)
{
  return exp(s1.elem());
}

// Fabs
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnFabs>::Type_t
fabs(const PScalarREG<T1>& s1)
{
  return fabs(s1.elem());
}

// Floor
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnFloor>::Type_t
floor(const PScalarREG<T1>& s1)
{
  return floor(s1.elem());
}

// Log
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnLog>::Type_t
log(const PScalarREG<T1>& s1)
{
  return log(s1.elem());
}

// Log10
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnLog10>::Type_t
log10(const PScalarREG<T1>& s1)
{
  return log10(s1.elem());
}

// Sin
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnSin>::Type_t
sin(const PScalarREG<T1>& s1)
{
  return sin(s1.elem());
}

// Sinh
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnHypSin>::Type_t
sinh(const PScalarREG<T1>& s1)
{
  return sinh(s1.elem());
}

// Sqrt
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnSqrt>::Type_t
sqrt(const PScalarREG<T1>& s1)
{
  return sqrt(s1.elem());
}


template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnIsFinite>::Type_t
isfinite(const PScalarREG<T1>& s1)
{
  return isfinite(s1.elem());
}


// Tan
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnTan>::Type_t
tan(const PScalarREG<T1>& s1)
{
  return tan(s1.elem());
}

// Tanh
template<class T1>
inline typename UnaryReturn<PScalarREG<T1>, FnHypTan>::Type_t
tanh(const PScalarREG<T1>& s1)
{
  return tanh(s1.elem());
}



//! PScalarREG<T> = pow(PScalarREG<T> , PScalarREG<T>)
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnPow>::Type_t
pow(const PScalarREG<T1>& s1, const PScalarREG<T2>& s2)
{
  return pow(s1.elem(), s2.elem());
}

//! PScalarREG<T> = atan2(PScalarREG<T> , PScalarREG<T>)
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnArcTan2>::Type_t
atan2(const PScalarREG<T1>& s1, const PScalarREG<T2>& s2)
{
  return atan2(s1.elem(), s2.elem());
}


//! PScalarREG<T> = (PScalarREG<T> , PScalarREG<T>)
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnCmplx>::Type_t
cmplx(const PScalarREG<T1>& s1, const PScalarREG<T2>& s2)
{
  return cmplx(s1.elem(), s2.elem());
}



// Global Functions
// PScalarREG = i * PScalarREG
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnTimesI>::Type_t
timesI(const PScalarREG<T>& s1)
{
  return timesI(s1.elem());
}

// PScalarREG = -i * PScalarREG
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnTimesMinusI>::Type_t
timesMinusI(const PScalarREG<T>& s1)
{
  return timesMinusI(s1.elem());
}


//! dest [float type] = source [seed type]
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSeedToFloat>::Type_t
seedToFloat(const PScalarREG<T>& s1)
{
  return seedToFloat(s1.elem());
}


//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnGetSite>::Type_t
getSite(const PScalarREG<T>& s1, int innersite)
{
  return getSite(s1.elem(), innersite);
}

//! Extract color vector components 
/*! Generically, this is an identity operation. Defined differently under color */

template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnPeekColorVectorREG>::Type_t
peekColor(const PScalarREG<T>& l, llvm::Value * row)
{
  return peekColor(l.elem(),row);
}

//! Extract color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnPeekColorMatrixREG>::Type_t
peekColor(const PScalarREG<T>& l, llvm::Value * row, llvm::Value * col)
{
  return peekColor(l.elem(),row,col);
}

//! Extract spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnPeekSpinVectorREG>::Type_t
peekSpin(const PScalarREG<T>& l, llvm::Value * row)
{
  return peekSpin(l.elem(),row);
}

//! Extract spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnPeekSpinMatrixREG>::Type_t
peekSpin(const PScalarREG<T>& l, llvm::Value * row, llvm::Value * col)
{
  return peekSpin(l.elem(),row,col);
}


//! Insert color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2>
inline PScalarREG<T1>&
pokeColor(PScalarREG<T1>& l, const PScalarREG<T2>& r, llvm::Value * row)
{

  pokeColor(l.elem(),r.elem(),row);
  return l;
}

//! Insert color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2>
inline PScalarREG<T1>&
pokeColor(PScalarREG<T1>& l, const PScalarREG<T2>& r, llvm::Value * row, llvm::Value * col)
{
  pokeColor(l.elem(),r.elem(),row,col);
  return l;
}

//! Insert spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2>
inline PScalarREG<T1>&
pokeSpin(PScalarREG<T1>& l, const PScalarREG<T2>& r, llvm::Value * row)
{
  pokeSpin(l.elem(),r.elem(),row);
  return l;
}

//! Insert spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2>
inline PScalarREG<T1>&
pokeSpin(PScalarREG<T1>& l, const PScalarREG<T2>& r, llvm::Value * row, llvm::Value * col)
{
  pokeSpin(l.elem(),r.elem(),row,col);
  return l;
}


//-----------------------------------------------------------------------------
//! PScalarREG = Gamma<N,m> * PScalarREG
template<class T2, int N, int m>
inline typename BinaryReturn<GammaConst<N,m>, PScalarREG<T2>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<N,m>& l, const PScalarREG<T2>& r)
{
  return l * r.elem();
}

//! PScalarREG = PScalarREG * Gamma<N,m>
template<class T2, int N, int m>
inline typename BinaryReturn<PScalarREG<T2>, GammaConst<N,m>, OpGammaConstMultiply>::Type_t
operator*(const PScalarREG<T2>& l, const GammaConst<N,m>& r)
{
  return l.elem() * r;
}

//-----------------------------------------------------------------------------
//! PScalarREG = SpinProject(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinProjectDir0Minus>::Type_t
spinProjectDir0Minus(const PScalarREG<T>& s1)
{
  return spinProjectDir0Minus(s1.elem());
}

//! PScalarREG = SpinReconstruct(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinReconstructDir0Minus>::Type_t
spinReconstructDir0Minus(const PScalarREG<T>& s1)
{
  return spinReconstructDir0Minus(s1.elem());
}


//! PScalarREG = SpinProject(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinProjectDir1Minus>::Type_t
spinProjectDir1Minus(const PScalarREG<T>& s1)
{
  return spinProjectDir1Minus(s1.elem());
}

//! PScalarREG = SpinReconstruct(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinReconstructDir1Minus>::Type_t
spinReconstructDir1Minus(const PScalarREG<T>& s1)
{
  return spinReconstructDir1Minus(s1.elem());
}


//! PScalarREG = SpinProject(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinProjectDir2Minus>::Type_t
spinProjectDir2Minus(const PScalarREG<T>& s1)
{
  return spinProjectDir2Minus(s1.elem());
}

//! PScalarREG = SpinReconstruct(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinReconstructDir2Minus>::Type_t
spinReconstructDir2Minus(const PScalarREG<T>& s1)
{
  return spinReconstructDir2Minus(s1.elem());
}


//! PScalarREG = SpinProject(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinProjectDir3Minus>::Type_t
spinProjectDir3Minus(const PScalarREG<T>& s1)
{
  return spinProjectDir3Minus(s1.elem());
}

//! PScalarREG = SpinReconstruct(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinReconstructDir3Minus>::Type_t
spinReconstructDir3Minus(const PScalarREG<T>& s1)
{
  return spinReconstructDir3Minus(s1.elem());
}


//! PScalarREG = SpinProject(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinProjectDir0Plus>::Type_t
spinProjectDir0Plus(const PScalarREG<T>& s1)
{
  return spinProjectDir0Plus(s1.elem());
}

//! PScalarREG = SpinReconstruct(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinReconstructDir0Plus>::Type_t
spinReconstructDir0Plus(const PScalarREG<T>& s1)
{
  return spinReconstructDir0Plus(s1.elem());
}


//! PScalarREG = SpinProject(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinProjectDir1Plus>::Type_t
spinProjectDir1Plus(const PScalarREG<T>& s1)
{
  return spinProjectDir1Plus(s1.elem());
}

//! PScalarREG = SpinReconstruct(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinReconstructDir1Plus>::Type_t
spinReconstructDir1Plus(const PScalarREG<T>& s1)
{
  return spinReconstructDir1Plus(s1.elem());
}


//! PScalarREG = SpinProject(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinProjectDir2Plus>::Type_t
spinProjectDir2Plus(const PScalarREG<T>& s1)
{
  return spinProjectDir2Plus(s1.elem());
}

//! PScalarREG = SpinReconstruct(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinReconstructDir2Plus>::Type_t
spinReconstructDir2Plus(const PScalarREG<T>& s1)
{
  return spinReconstructDir2Plus(s1.elem());
}


//! PScalarREG = SpinProject(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinProjectDir3Plus>::Type_t
spinProjectDir3Plus(const PScalarREG<T>& s1)
{
  return spinProjectDir3Plus(s1.elem());
}

//! PScalarREG = SpinReconstruct(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSpinReconstructDir3Plus>::Type_t
spinReconstructDir3Plus(const PScalarREG<T>& s1)
{
  return spinReconstructDir3Plus(s1.elem());
}

//-----------------------------------------------------------------------------
//! PScalarREG = chiralProjectPlus(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnChiralProjectPlus>::Type_t
chiralProjectPlus(const PScalarREG<T>& s1)
{
  return chiralProjectPlus(s1.elem());
}

//! PScalarREG = chiralProjectMinus(PScalarREG)
template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnChiralProjectMinus>::Type_t
chiralProjectMinus(const PScalarREG<T>& s1)
{
  return chiralProjectMinus(s1.elem());
}


//-----------------------------------------------------------------------------
// quark propagator contraction
template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnQuarkContract13>::Type_t
quarkContract13(const PScalarREG<T1>& s1, const PScalarREG<T2>& s2)
{
  return quarkContract13(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnQuarkContract14>::Type_t
quarkContract14(const PScalarREG<T1>& s1, const PScalarREG<T2>& s2)
{
  return quarkContract14(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnQuarkContract23>::Type_t
quarkContract23(const PScalarREG<T1>& s1, const PScalarREG<T2>& s2)
{
  return quarkContract23(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnQuarkContract24>::Type_t
quarkContract24(const PScalarREG<T1>& s1, const PScalarREG<T2>& s2)
{
  return quarkContract24(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnQuarkContract12>::Type_t
quarkContract12(const PScalarREG<T1>& s1, const PScalarREG<T2>& s2)
{
  return quarkContract12(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnQuarkContract34>::Type_t
quarkContract34(const PScalarREG<T1>& s1, const PScalarREG<T2>& s2)
{
  return quarkContract34(s1.elem(), s2.elem());
}


//-----------------------------------------------------------------------------
// Contraction for color matrices
// colorContract 
//! dest  = colorContract(Qprop1,Qprop2,Qprop3)
/*!
 * This routine is completely unrolled for 3 colors
 */
template<class T1, class T2, class T3>
struct TrinaryReturn<PScalarREG<T1>, PScalarREG<T2>, PScalarREG<T3>, FnColorContract> {
  typedef PScalarREG<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
inline typename TrinaryReturn<PScalarREG<T1>, PScalarREG<T2>, PScalarREG<T3>, FnColorContract>::Type_t
colorContract(const PScalarREG<T1>& s1, const PScalarREG<T2>& s2, const PScalarREG<T3>& s3)
{
  return colorContract(s1.elem(), s2.elem(), s3.elem());
}


//-----------------------------------------------------------------------------
// Contraction of two colorvectors
//! dest  = colorVectorContract(Qvec1,Qvec2)
template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnColorVectorContract> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnColorVectorContract>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnColorVectorContract>::Type_t
colorVectorContract(const PScalarREG<T1>& s1, const PScalarREG<T2>& s2)
{
  return colorVectorContract(s1.elem(), s2.elem());
}



//-----------------------------------------------------------------------------
// Cross product for color vectors
//! dest  = colorCrossProduct(Qvec1,Qvec2)
template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnColorCrossProduct> {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnColorCrossProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnColorCrossProduct>::Type_t
colorCrossProduct(const PScalarREG<T1>& s1, const PScalarREG<T2>& s2)
{
  return colorCrossProduct(s1.elem(), s2.elem());
}



//-----------------------------------------------------------------------------
//! dest = (mask) ? s1 : dest
template<class T, class T1> 
inline void 
copymask(PScalarREG<T>& d, const PScalarREG<T1>& mask, const PScalarREG<T>& s1) 
{
  copymask(d.elem(),mask.elem(),s1.elem());
}

//! dest  = random  
template<class T, class T1, class T2,class T3>
inline void
fill_random( PScalarREG<T>& d, T1& seed, T2& skewed_seed, const T3& seed_mult)
{
  fill_random(d.elem(), seed, skewed_seed, seed_mult);
}


template<class T>
inline void
get_pred(int& pred, const PScalarREG<T>& d)
{
  get_pred(pred , d.elem() );
}



//! dest  = gaussian  
template<class T>
inline void
fill_gaussian(PScalarREG<T>& d, PScalarREG<T>& r1, PScalarREG<T>& r2)
{
  fill_gaussian(d.elem(), r1.elem(), r2.elem());
}


#if 1
// Global sum over site indices only
template<class T>
struct UnaryReturn<PScalarREG<T>, FnSum > {
  typedef PScalarREG<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnSum>::Type_t
sum(const PScalarREG<T>& s1)
{
  return sum(s1.elem());
}
#endif


// InnerProduct (norm-seq) global sum = sum(tr(adj(s1)*s1))
template<class T>
struct UnaryReturn<PScalarREG<T>, FnNorm2 > {
  typedef PScalarREG<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PScalarREG<T>, FnLocalNorm2 > {
  typedef PScalarREG<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnLocalNorm2>::Type_t
localNorm2(const PScalarREG<T>& s1)
{
  return localNorm2(s1.elem());
}

// Global max
template<class T>
struct UnaryReturn<PScalarREG<T>, FnGlobalMax> {
  typedef PScalarREG<typename UnaryReturn<T, FnGlobalMax>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnGlobalMax>::Type_t
globalMax(const PScalarREG<T>& s1)
{
  return globalMax(s1.elem());
}


// Global min
template<class T>
struct UnaryReturn<PScalarREG<T>, FnGlobalMin> {
  typedef PScalarREG<typename UnaryReturn<T, FnGlobalMin>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<PScalarREG<T>, FnGlobalMin>::Type_t
globalMin(const PScalarREG<T>& s1)
{
  return globalMin(s1.elem());
}


//! PScalarREG<T> = InnerProduct(adj(PScalarREG<T1>)*PScalarREG<T2>)
template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnInnerProduct > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnLocalInnerProduct > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(const PScalarREG<T1>& s1, const PScalarREG<T2>& s2)
{
  return localInnerProduct(s1.elem(), s2.elem());
}


//! PScalarREG<T> = InnerProductReal(adj(PMatrix<T1>)*PMatrix<T1>)
template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnInnerProductReal > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnLocalInnerProductReal > {
  typedef PScalarREG<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarREG<T1>, PScalarREG<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const PScalarREG<T1>& s1, const PScalarREG<T2>& s2)
{
  return localInnerProductReal(s1.elem(), s2.elem());
}


//! PScalarREG<T> = where(PScalarREG, PScalarREG, PScalarREG)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3>
struct TrinaryReturn<PScalarREG<T1>, PScalarREG<T2>, PScalarREG<T3>, FnWhere> {
  typedef PScalarREG<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
inline typename TrinaryReturn<PScalarREG<T1>, PScalarREG<T2>, PScalarREG<T3>, FnWhere>::Type_t
where(const PScalarREG<T1>& a, const PScalarREG<T2>& b, const PScalarREG<T3>& c)
{
  return where(a.elem(), b.elem(), c.elem());
}



//-----------------------------------------------------------------------------
// Other operations
//! dest = 0
template<class T> 
inline void 
zero_rep(PScalarREG<T>& dest) 
{
  zero_rep(dest.elem());
}

//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
cast_rep(T& d, const PScalarREG<T1>& s1)
{
  cast_rep(d, s1.elem());
}

//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
cast_rep(PScalarREG<T>& d, const PScalarREG<T1>& s1)
{
  cast_rep(d.elem(), s1.elem());
}

//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
copy_site(PScalarREG<T>& d, int isite, const PScalarREG<T1>& s1)
{
  copy_site(d.elem(), isite, s1.elem());
}

//! gather several inner sites together
template<class T, class T1>
inline void 
gather_sites(PScalarREG<T>& d, 
	     const PScalarREG<T1>& s0, int i0, 
	     const PScalarREG<T1>& s1, int i1,
	     const PScalarREG<T1>& s2, int i2,
	     const PScalarREG<T1>& s3, int i3)
{
  gather_sites(d.elem(), s0.elem(), i0, s1.elem(), i1, s2.elem(), i2, s3.elem(), i3);
}


template<class T>
inline void 
qdpPHI(PScalarREG<T>& d, 
       const PScalarREG<T>& phi0, llvm::BasicBlock* bb0 ,
       const PScalarREG<T>& phi1, llvm::BasicBlock* bb1 )
{
  qdpPHI(d.elem(),
	 phi0.elem(),bb0,
	 phi1.elem(),bb1);
}



/*! @} */  // end of group primscalar

} // namespace QDP

#endif
