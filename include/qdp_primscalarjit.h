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
  private:
    template<class T1>
    PScalarJIT& operator=( const PScalarJIT<T1>& rhs);
    PScalarJIT& operator=( const PScalarJIT& rhs);
    //PScalarJIT( const PScalarJIT& rhs);

  public:

    // Default constructing should be possible
    // then there is no need for MPL index when
    // construction a PMatrix<T,N>
    PScalarJIT() {}
    ~PScalarJIT() {}

    template<class T1>
    PScalarJIT& operator=( const PScalarREG<T1>& rhs) {
      elem() = rhs.elem();
      return *this;
    }


#if 0
    //---------------------------------------------------------
    //! construct dest = const
    PScalarJIT(const typename WordType<T>::Type_t& rhs) : F(rhs) {}

    // I keep this deactivated. Takes too much.
#if 0
    template<class T1>
    PScalarJIT(const T1& rhs) : JV<T,1>(newspace_t(rhs.func())) {
      elem() = rhs;
    }
#endif

    PScalarJIT(const T& rhs) : JV<T,1>(newspace_t(rhs.func())) {
      elem() = rhs;
    }


    PScalarJIT(Jit& j,const typename WordType<T>::Type_t& w) : JV<T,1>(j,w) {
    }
#endif



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
struct REGType< PScalarJIT<T> >
{
  typedef PScalarREG<typename REGType<T>::Type_t>  Type_t;
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


//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PScalarJIT) -> PScalarJIT
template<class T1, class Op>
struct UnaryReturn<PScalarJIT<T1>, Op> {
  typedef PScalarJIT<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

// Default binary(PScalarJIT,PScalarJIT) -> PScalarJIT
template<class T1, class T2, class Op>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, Op> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


#if 0
template<class T1, class T2>
struct UnaryReturn<PScalarJIT<T2>, OpCast<T1> > {
  typedef PScalarJIT<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif

// Assignment is different
template<class T1, class T2 >
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpAssign > {
  typedef PScalarJIT<T1> &Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpAddAssign > {
  typedef PScalarJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpSubtractAssign > {
  typedef PScalarJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpMultiplyAssign > {
  typedef PScalarJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpDivideAssign > {
  typedef PScalarJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpModAssign > {
  typedef PScalarJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpBitwiseOrAssign > {
  typedef PScalarJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpBitwiseAndAssign > {
  typedef PScalarJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpBitwiseXorAssign > {
  typedef PScalarJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpLeftShiftAssign > {
  typedef PScalarJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpRightShiftAssign > {
  typedef PScalarJIT<T1> &Type_t;
};
 

#if 0

//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

/*! \addtogroup primscalar */
/*! @{ */

// Primitive Scalars

// ! PScalarJIT
template<class T>
struct UnaryReturn<PScalarJIT<T>, OpNot > {
  typedef PScalarJIT<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, OpNot>::Type_t
operator!(const PScalarJIT<T1>& l)
{
  return ! l.elem();
}

// + PScalarJIT
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, OpUnaryPlus>::Type_t
operator+(const PScalarJIT<T1>& l)
{
  return +l.elem();
}

// - PScalarJIT
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, OpUnaryMinus>::Type_t
operator-(const PScalarJIT<T1>& l)
{
  return -l.elem();
}

  //PScalarJIT + PScalarJIT
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpAdd>::Type_t
operator+(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() + r.elem();
}


// PScalarJIT - PScalarJIT
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpSubtract>::Type_t
operator-(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() - r.elem();
}


  //PScalarJIT + PScalarJIT
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpMultiply>::Type_t
operator*(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
#if 1
  return l.elem() * r.elem();
#else
  typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpMultiply>::Type_t ret(l.func());
  typedef typename BinaryReturn<T1, T2, OpMultiply>::Type_t  T;
  typedef typename InternalScalar<T>::Type_t  S;
  ret = l.elem() * S(l.func(),12);
  return ret;

#endif
}



// Optimized  adj(PMatrix)*PMatrix
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpAdjMultiply>::Type_t
adjMultiply(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return adjMultiply(l.elem(), r.elem());
}


template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return multiplyAdj(l.elem(), r.elem());
}

// Optimized  PMatrix*adj(PMatrix)
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PSpinVector<T2,4>, OpMultiplyAdj>::Type_t
multiplyAdj(const PScalarJIT<T1>& l, const PSpinVector<T2,4>& r)
{
  return multiplyAdj(l.elem(), r.elem());
}

// Optimized  adj(PMatrix)*adj(PMatrix)
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return adjMultiplyAdj(l.elem(), r.elem());
}

// PScalarJIT / PScalarJIT
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpDivide>::Type_t
operator/(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() / r.elem();
}


// PScalarJIT << PScalarJIT
template<class T1, class T2 >
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpLeftShift > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpLeftShift>::Type_t
operator<<(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() << r.elem();
}

// PScalarJIT >> PScalarJIT
template<class T1, class T2 >
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpRightShift > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpRightShift>::Type_t
operator>>(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() >> r.elem();
}

// PScalarJIT % PScalarJIT
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpMod>::Type_t
operator%(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() % r.elem();
}

// PScalarJIT ^ PScalarJIT
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpBitwiseXor>::Type_t
operator^(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() ^ r.elem();
}

// PScalarJIT & PScalarJIT
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpBitwiseAnd>::Type_t
operator&(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() & r.elem();
}

// PScalarJIT | PScalarJIT
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpBitwiseOr>::Type_t
operator|(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() | r.elem();
}


// Comparisons
template<class T1, class T2 >
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpLT > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpLT>::Type_t
operator<(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() < r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpLE > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpLE>::Type_t
operator<=(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() <= r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpGT > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpGT>::Type_t
operator>(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() > r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpGE > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpGE>::Type_t
operator>=(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() >= r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpEQ > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpEQ>::Type_t
operator==(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() == r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpNE > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpNE>::Type_t
operator!=(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() != r.elem();
}


template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpAnd > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpAnd>::Type_t
operator&&(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() && r.elem();
}


template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpOr > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, OpOr>::Type_t
operator||(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return l.elem() || r.elem();
}


//-----------------------------------------------------------------------------
// Functions

// Adjoint
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnAdjoint>::Type_t
adj(const PScalarJIT<T1>& s1)
{
  return adj(s1.elem());
}


// Conjugate
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnConjugate>::Type_t
conj(const PScalarJIT<T1>& s1)
{
  return conj(s1.elem());
}


// Transpose
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnTranspose>::Type_t
transpose(const PScalarJIT<T1>& s1)
{
  return transpose(s1.elem());
}


// TRACE
// trace = Trace(source1)
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnTrace>::Type_t
trace(const PScalarJIT<T1>& s1)
{
  return trace(s1.elem());
}


// trace = Re(Trace(source1))
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnRealTrace>::Type_t
realTrace(const PScalarJIT<T1>& s1)
{
  return realTrace(s1.elem());
}


// trace = Im(Trace(source1))
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnImagTrace>::Type_t
imagTrace(const PScalarJIT<T1>& s1)
{
  return imagTrace(s1.elem());
}


// trace = colorTrace(source1)
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnTraceColor>::Type_t
traceColor(const PScalarJIT<T1>& s1)
{
  return traceColor(s1.elem());
}


//! PScalarJIT = traceSpin(PScalarJIT)
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnTraceSpin>::Type_t
traceSpin(const PScalarJIT<T1>& s1)
{
  return traceSpin(s1.elem());
}

//! PScalarJIT = transposeSpin(PScalarJIT)
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnTransposeSpin>::Type_t
transposeSpin(const PScalarJIT<T1>& s1)
{
  return transposeSpin(s1.elem());
}

//! PScalarJIT = trace(PScalarJIT * PScalarJIT)
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnTraceMultiply>::Type_t
traceMultiply(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return traceMultiply(l.elem(), r.elem());
}

//! PScalarJIT = traceColor(PScalarJIT * PScalarJIT)
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnTraceColorMultiply>::Type_t
traceColorMultiply(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return traceMultiply(l.elem(), r.elem());
}

//! PScalarJIT = traceSpin(PScalarJIT * PScalarJIT)
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return traceMultiply(l.elem(), r.elem());
}

//! PScalarJIT = traceSpin(outerProduct(PScalarJIT, PScalarJIT))
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnTraceSpinOuterProduct>::Type_t
traceSpinOuterProduct(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return traceSpinOuterProduct(l.elem(), r.elem());
}

//! PScalarJIT = outerProduct(PScalarJIT, PScalarJIT)
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnOuterProduct>::Type_t
outerProduct(const PScalarJIT<T1>& l, const PScalarJIT<T2>& r)
{
  return outerProduct(l.elem(),r.elem());
}


//! PScalarJIT = Re(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnReal>::Type_t
real(const PScalarJIT<T>& s1)
{
  return real(s1.elem());
}


// PScalarJIT = Im(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnImag>::Type_t
imag(const PScalarJIT<T>& s1)
{
  return imag(s1.elem());
}


// ArcCos
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnArcCos>::Type_t
acos(const PScalarJIT<T1>& s1)
{
  return acos(s1.elem());
}

// ArcSin
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnArcSin>::Type_t
asin(const PScalarJIT<T1>& s1)
{
  return asin(s1.elem());
}

// ArcTan
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnArcTan>::Type_t
atan(const PScalarJIT<T1>& s1)
{
  return atan(s1.elem());
}

// Ceil(ing)
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnCeil>::Type_t
ceil(const PScalarJIT<T1>& s1)
{
  return ceil(s1.elem());
}

// Cos
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnCos>::Type_t
cos(const PScalarJIT<T1>& s1)
{
  return cos(s1.elem());
}

// Cosh
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnHypCos>::Type_t
cosh(const PScalarJIT<T1>& s1)
{
  return cosh(s1.elem());
}

// Exp
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnExp>::Type_t
exp(const PScalarJIT<T1>& s1)
{
  return exp(s1.elem());
}

// Fabs
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnFabs>::Type_t
fabs(const PScalarJIT<T1>& s1)
{
  return fabs(s1.elem());
}

// Floor
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnFloor>::Type_t
floor(const PScalarJIT<T1>& s1)
{
  return floor(s1.elem());
}

// Log
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnLog>::Type_t
log(const PScalarJIT<T1>& s1)
{
  return log(s1.elem());
}

// Log10
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnLog10>::Type_t
log10(const PScalarJIT<T1>& s1)
{
  return log10(s1.elem());
}

// Sin
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnSin>::Type_t
sin(const PScalarJIT<T1>& s1)
{
  return sin(s1.elem());
}

// Sinh
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnHypSin>::Type_t
sinh(const PScalarJIT<T1>& s1)
{
  return sinh(s1.elem());
}

// Sqrt
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnSqrt>::Type_t
sqrt(const PScalarJIT<T1>& s1)
{
  return sqrt(s1.elem());
}

// Tan
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnTan>::Type_t
tan(const PScalarJIT<T1>& s1)
{
  return tan(s1.elem());
}

// Tanh
template<class T1>
inline typename UnaryReturn<PScalarJIT<T1>, FnHypTan>::Type_t
tanh(const PScalarJIT<T1>& s1)
{
  return tanh(s1.elem());
}



//! PScalarJIT<T> = pow(PScalarJIT<T> , PScalarJIT<T>)
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnPow>::Type_t
pow(const PScalarJIT<T1>& s1, const PScalarJIT<T2>& s2)
{
  return pow(s1.elem(), s2.elem());
}

//! PScalarJIT<T> = atan2(PScalarJIT<T> , PScalarJIT<T>)
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnArcTan2>::Type_t
atan2(const PScalarJIT<T1>& s1, const PScalarJIT<T2>& s2)
{
  return atan2(s1.elem(), s2.elem());
}


//! PScalarJIT<T> = (PScalarJIT<T> , PScalarJIT<T>)
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnCmplx>::Type_t
cmplx(const PScalarJIT<T1>& s1, const PScalarJIT<T2>& s2)
{
  return cmplx(s1.elem(), s2.elem());
}



// Global Functions
// PScalarJIT = i * PScalarJIT
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnTimesI>::Type_t
timesI(const PScalarJIT<T>& s1)
{
  return timesI(s1.elem());
}

// PScalarJIT = -i * PScalarJIT
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnTimesMinusI>::Type_t
timesMinusI(const PScalarJIT<T>& s1)
{
  return timesMinusI(s1.elem());
}


//! dest [float type] = source [seed type]
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSeedToFloat>::Type_t
seedToFloat(const PScalarJIT<T>& s1)
{
  return seedToFloat(s1.elem());
}


//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnGetSite>::Type_t
getSite(const PScalarJIT<T>& s1, int innersite)
{
  return getSite(s1.elem(), innersite);
}
#endif


#if 0
//! Extract color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnPeekColorVectorREG>::Type_t
peekColor(const PScalarJIT<T>& l, int row)
{
  return peekColor(l.elem(),row);
}

//! Extract color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnPeekColorMatrixREG>::Type_t
peekColor(const PScalarJIT<T>& l, int row, int col)
{
  return peekColor(l.elem(),row,col);
}

//! Extract spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnPeekSpinVectorREG>::Type_t
peekSpin(const PScalarJIT<T>& l, int row)
{
  return peekSpin(l.elem(),row);
}

//! Extract spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnPeekSpinMatrixREG>::Type_t
peekSpin(const PScalarJIT<T>& l, int row, int col)
{
  return peekSpin(l.elem(),row,col);
}


//! Insert color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2>
inline PScalarJIT<T1>&
pokeColor(PScalarJIT<T1>& l, const PScalarJIT<T2>& r, int row)
{
  pokeColor(l.elem(),r.elem(),row);
  return l;
}
#endif



template<class T1, class T2>
inline PScalarJIT<T1>&
pokeColor(PScalarJIT<T1>& l, const PScalarREG<T2>& r, jit_value row)
{
  pokeColor(l.elem(),r.elem(),row);
  return l;
}

template<class T1, class T2>
inline PScalarJIT<T1>&
pokeColor(PScalarJIT<T1>& l, const PScalarREG<T2>& r, jit_value row, jit_value col)
{
  pokeColor(l.elem(),r.elem(),row,col);
  return l;
}




#if 0
//! Insert color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2>
inline PScalarJIT<T1>&
pokeColor(PScalarJIT<T1>& l, const PScalarJIT<T2>& r, int row, int col)
{
  pokeColor(l.elem(),r.elem(),row,col);
  return l;
}

//! Insert spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2>
inline PScalarJIT<T1>&
pokeSpin(PScalarJIT<T1>& l, const PScalarJIT<T2>& r, int row)
{
  pokeSpin(l.elem(),r.elem(),row);
  return l;
}

//! Insert spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2>
inline PScalarJIT<T1>&
pokeSpin(PScalarJIT<T1>& l, const PScalarJIT<T2>& r, int row, int col)
{
  pokeSpin(l.elem(),r.elem(),row,col);
  return l;
}
#endif




#if 0
//-----------------------------------------------------------------------------
//! PScalarJIT = Gamma<N,m> * PScalarJIT
template<class T2, int N, int m>
inline typename BinaryReturn<GammaConst<N,m>, PScalarJIT<T2>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<N,m>& l, const PScalarJIT<T2>& r)
{
  return l * r.elem();
}

//! PScalarJIT = PScalarJIT * Gamma<N,m>
template<class T2, int N, int m>
inline typename BinaryReturn<PScalarJIT<T2>, GammaConst<N,m>, OpGammaConstMultiply>::Type_t
operator*(const PScalarJIT<T2>& l, const GammaConst<N,m>& r)
{
  return l.elem() * r;
}

//-----------------------------------------------------------------------------
//! PScalarJIT = SpinProject(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir0Minus>::Type_t
spinProjectDir0Minus(const PScalarJIT<T>& s1)
{
  return spinProjectDir0Minus(s1.elem());
}

//! PScalarJIT = SpinReconstruct(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir0Minus>::Type_t
spinReconstructDir0Minus(const PScalarJIT<T>& s1)
{
  return spinReconstructDir0Minus(s1.elem());
}


//! PScalarJIT = SpinProject(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir1Minus>::Type_t
spinProjectDir1Minus(const PScalarJIT<T>& s1)
{
  return spinProjectDir1Minus(s1.elem());
}

//! PScalarJIT = SpinReconstruct(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir1Minus>::Type_t
spinReconstructDir1Minus(const PScalarJIT<T>& s1)
{
  return spinReconstructDir1Minus(s1.elem());
}


//! PScalarJIT = SpinProject(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir2Minus>::Type_t
spinProjectDir2Minus(const PScalarJIT<T>& s1)
{
  return spinProjectDir2Minus(s1.elem());
}

//! PScalarJIT = SpinReconstruct(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir2Minus>::Type_t
spinReconstructDir2Minus(const PScalarJIT<T>& s1)
{
  return spinReconstructDir2Minus(s1.elem());
}


//! PScalarJIT = SpinProject(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir3Minus>::Type_t
spinProjectDir3Minus(const PScalarJIT<T>& s1)
{
  return spinProjectDir3Minus(s1.elem());
}

//! PScalarJIT = SpinReconstruct(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir3Minus>::Type_t
spinReconstructDir3Minus(const PScalarJIT<T>& s1)
{
  return spinReconstructDir3Minus(s1.elem());
}


//! PScalarJIT = SpinProject(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir0Plus>::Type_t
spinProjectDir0Plus(const PScalarJIT<T>& s1)
{
  return spinProjectDir0Plus(s1.elem());
}

//! PScalarJIT = SpinReconstruct(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir0Plus>::Type_t
spinReconstructDir0Plus(const PScalarJIT<T>& s1)
{
  return spinReconstructDir0Plus(s1.elem());
}


//! PScalarJIT = SpinProject(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir1Plus>::Type_t
spinProjectDir1Plus(const PScalarJIT<T>& s1)
{
  return spinProjectDir1Plus(s1.elem());
}

//! PScalarJIT = SpinReconstruct(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir1Plus>::Type_t
spinReconstructDir1Plus(const PScalarJIT<T>& s1)
{
  return spinReconstructDir1Plus(s1.elem());
}


//! PScalarJIT = SpinProject(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir2Plus>::Type_t
spinProjectDir2Plus(const PScalarJIT<T>& s1)
{
  return spinProjectDir2Plus(s1.elem());
}

//! PScalarJIT = SpinReconstruct(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir2Plus>::Type_t
spinReconstructDir2Plus(const PScalarJIT<T>& s1)
{
  return spinReconstructDir2Plus(s1.elem());
}


//! PScalarJIT = SpinProject(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinProjectDir3Plus>::Type_t
spinProjectDir3Plus(const PScalarJIT<T>& s1)
{
  return spinProjectDir3Plus(s1.elem());
}

//! PScalarJIT = SpinReconstruct(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSpinReconstructDir3Plus>::Type_t
spinReconstructDir3Plus(const PScalarJIT<T>& s1)
{
  return spinReconstructDir3Plus(s1.elem());
}

//-----------------------------------------------------------------------------
//! PScalarJIT = chiralProjectPlus(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnChiralProjectPlus>::Type_t
chiralProjectPlus(const PScalarJIT<T>& s1)
{
  return chiralProjectPlus(s1.elem());
}

//! PScalarJIT = chiralProjectMinus(PScalarJIT)
template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnChiralProjectMinus>::Type_t
chiralProjectMinus(const PScalarJIT<T>& s1)
{
  return chiralProjectMinus(s1.elem());
}


//-----------------------------------------------------------------------------
// quark propagator contraction
template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnQuarkContract13>::Type_t
quarkContract13(const PScalarJIT<T1>& s1, const PScalarJIT<T2>& s2)
{
  return quarkContract13(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnQuarkContract14>::Type_t
quarkContract14(const PScalarJIT<T1>& s1, const PScalarJIT<T2>& s2)
{
  return quarkContract14(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnQuarkContract23>::Type_t
quarkContract23(const PScalarJIT<T1>& s1, const PScalarJIT<T2>& s2)
{
  return quarkContract23(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnQuarkContract24>::Type_t
quarkContract24(const PScalarJIT<T1>& s1, const PScalarJIT<T2>& s2)
{
  return quarkContract24(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnQuarkContract12>::Type_t
quarkContract12(const PScalarJIT<T1>& s1, const PScalarJIT<T2>& s2)
{
  return quarkContract12(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnQuarkContract34>::Type_t
quarkContract34(const PScalarJIT<T1>& s1, const PScalarJIT<T2>& s2)
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
struct TrinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, PScalarJIT<T3>, FnColorContract> {
  typedef PScalarJIT<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
inline typename TrinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, PScalarJIT<T3>, FnColorContract>::Type_t
colorContract(const PScalarJIT<T1>& s1, const PScalarJIT<T2>& s2, const PScalarJIT<T3>& s3)
{
  return colorContract(s1.elem(), s2.elem(), s3.elem());
}


//-----------------------------------------------------------------------------
// Contraction of two colorvectors
//! dest  = colorVectorContract(Qvec1,Qvec2)
template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnColorVectorContract> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnColorVectorContract>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnColorVectorContract>::Type_t
colorVectorContract(const PScalarJIT<T1>& s1, const PScalarJIT<T2>& s2)
{
  return colorVectorContract(s1.elem(), s2.elem());
}



//-----------------------------------------------------------------------------
// Cross product for color vectors
//! dest  = colorCrossProduct(Qvec1,Qvec2)
template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnColorCrossProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnColorCrossProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnColorCrossProduct>::Type_t
colorCrossProduct(const PScalarJIT<T1>& s1, const PScalarJIT<T2>& s2)
{
  return colorCrossProduct(s1.elem(), s2.elem());
}



//-----------------------------------------------------------------------------




template<class T>
inline void
get_pred(int& pred, const PScalarJIT<T>& d)
{
  get_pred(pred , d.elem() );
}





#if 1
// Global sum over site indices only
template<class T>
struct UnaryReturn<PScalarJIT<T>, FnSum > {
  typedef PScalarJIT<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnSum>::Type_t
sum(const PScalarJIT<T>& s1)
{
  return sum(s1.elem());
}
#endif


// InnerProduct (norm-seq) global sum = sum(tr(adj(s1)*s1))
template<class T>
struct UnaryReturn<PScalarJIT<T>, FnNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PScalarJIT<T>, FnLocalNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnLocalNorm2>::Type_t
localNorm2(const PScalarJIT<T>& s1)
{
  return localNorm2(s1.elem());
}

// Global max
template<class T>
struct UnaryReturn<PScalarJIT<T>, FnGlobalMax> {
  typedef PScalarJIT<typename UnaryReturn<T, FnGlobalMax>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnGlobalMax>::Type_t
globalMax(const PScalarJIT<T>& s1)
{
  return globalMax(s1.elem());
}


// Global min
template<class T>
struct UnaryReturn<PScalarJIT<T>, FnGlobalMin> {
  typedef PScalarJIT<typename UnaryReturn<T, FnGlobalMin>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<PScalarJIT<T>, FnGlobalMin>::Type_t
globalMin(const PScalarJIT<T>& s1)
{
  return globalMin(s1.elem());
}


//! PScalarJIT<T> = InnerProduct(adj(PScalarJIT<T1>)*PScalarJIT<T2>)
template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnInnerProduct > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnLocalInnerProduct > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(const PScalarJIT<T1>& s1, const PScalarJIT<T2>& s2)
{
  return localInnerProduct(s1.elem(), s2.elem());
}


//! PScalarJIT<T> = InnerProductReal(adj(PMatrix<T1>)*PMatrix<T1>)
template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnInnerProductReal > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnLocalInnerProductReal > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const PScalarJIT<T1>& s1, const PScalarJIT<T2>& s2)
{
  return localInnerProductReal(s1.elem(), s2.elem());
}


//! PScalarJIT<T> = where(PScalarJIT, PScalarJIT, PScalarJIT)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3>
struct TrinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, PScalarJIT<T3>, FnWhere> {
  typedef PScalarJIT<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
inline typename TrinaryReturn<PScalarJIT<T1>, PScalarJIT<T2>, PScalarJIT<T3>, FnWhere>::Type_t
where(const PScalarJIT<T1>& a, const PScalarJIT<T2>& b, const PScalarJIT<T3>& c)
{
  return where(a.elem(), b.elem(), c.elem());
}



//-----------------------------------------------------------------------------
// Other operations

//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
cast_rep(T& d, const PScalarJIT<T1>& s1)
{
  cast_rep(d, s1.elem());
}

//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
cast_rep(PScalarJIT<T>& d, const PScalarJIT<T1>& s1)
{
  cast_rep(d.elem(), s1.elem());
}

//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
copy_site(PScalarJIT<T>& d, int isite, const PScalarJIT<T1>& s1)
{
  copy_site(d.elem(), isite, s1.elem());
}

//! gather several inner sites together
template<class T, class T1>
inline void 
gather_sites(PScalarJIT<T>& d, 
	     const PScalarJIT<T1>& s0, int i0, 
	     const PScalarJIT<T1>& s1, int i1,
	     const PScalarJIT<T1>& s2, int i2,
	     const PScalarJIT<T1>& s3, int i3)
{
  gather_sites(d.elem(), s0.elem(), i0, s1.elem(), i1, s2.elem(), i2, s3.elem(), i3);
}

#endif


//! dest = (mask) ? s1 : dest
template<class T, class T1, class T2> 
inline void 
copymask(PScalarJIT<T>& d, const PScalarREG<T1>& mask, const PScalarREG<T2>& s1) 
{
  copymask(d.elem(),mask.elem(),s1.elem());
}


//! dest = 0
template<class T> 
inline void 
zero_rep(PScalarJIT<T>& dest) 
{
  zero_rep(dest.elem());
}


//! dest  = random  
template<class T, class T1, class T2,class T3>
inline void
fill_random( PScalarJIT<T>& d, T1& seed, T2& skewed_seed, const T3& seed_mult)
{
  fill_random(d.elem(), seed, skewed_seed, seed_mult);
}


//! dest  = gaussian  
template<class T,class T2>
inline void
fill_gaussian(PScalarJIT<T>& d, PScalarREG<T2>& r1, PScalarREG<T2>& r2)
{
  fill_gaussian(d.elem(), r1.elem(), r2.elem());
}





/*! @} */  // end of group primscalar

} // namespace QDP

#endif
