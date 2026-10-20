#ifndef QDP_REALITY_H
#define QDP_REALITY_H


namespace QDP {


template<class T> class RScalar: public BaseView<T>
{
public:
  typedef T Sub_t;
  enum { ThisSize = 1 };

  Shape shape() const { return BaseView<T>::shape(); }

  RScalar( Shape shape , typename WordType<T>::Type_t *base_ ): BaseView<T>(shape,base_) {}

  RScalar(const RScalar& rhs) : BaseView<T>(rhs) {}

  RScalar operator=(const RScalar& rhs)
  {
    elem() = rhs.elem();
    return *this;
  }

  T elem() const {return BaseView<T>::elem(0);}
};



template<class T>
inline
istream& operator>>(istream& s, RScalar<T> d)
{
  return s >> d.elem();
}


template<class T>
inline
StandardInputStream& operator>>(StandardInputStream& s, RScalar<T> d)
{
  return s >> d.elem();
}


template<class T> 
inline  
ostream& operator<<(ostream& s, RScalar<T> d)
{
  return s << d.elem();
}


template<class T> 
inline  
StandardOutputStream& operator<<(StandardOutputStream& s, RScalar<T> d)
{
  return s << d.elem();
}



template<class T>
inline
TextReader& operator>>(TextReader& s, RScalar<T> d)
{
  return s >> d.elem();
}


template<class T> 
inline  
TextWriter& operator<<(TextWriter& s, RScalar<T> d)
{
  return s << d.elem();
}

#ifndef QDP_NO_LIBXML2
//! XML output
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, RScalar<T> d)
{
  return xml << d.elem();
}

//! XML input
template<class T>
inline
void read(XMLReader& xml, const string& path, RScalar<T> d)
{
  read(xml, path, d.elem());
}
#endif



  
template<class T> class RComplex: public BaseView<T>
{
public:
  typedef T Sub_t;
  enum { ThisSize = 2 };

  Shape shape() const { return BaseView<T>::shape(); }
  
  RComplex( Shape shape , typename WordType<T>::Type_t* f): BaseView<T>(shape,f) {}

  RComplex(const RComplex& a): BaseView<T>(a) {}


  RComplex operator=(const RComplex& rhs)
  {
    real() = rhs.real();
    imag() = rhs.imag();
    return *this;
  }


  T real() const
  {
    return this->elem(0);
  }
  
  T imag() const
  {
    return this->elem(1);
  }
};


  template<class T>
  struct GetLimit<RScalar<T>,0>
  {
    static int limit(Shape shape)
    {
      return 1;
    }
  };


  template<class T>
  struct GetLimit<RComplex<T>,0>
  {
    static int limit(Shape shape)
    {
      return 2;
    }
  };










template<class T> 
struct JITType<RScalar<T> >
{
  typedef RScalarJIT<typename JITType<T>::Type_t>  Type_t;
};


template<class T> 
struct JITType<RComplex<T> >
{
  typedef RComplexJIT<typename JITType<T>::Type_t>  Type_t;
};







template<class T>
inline
ostream& operator<<(ostream& s, RComplex<T> d)
{
  s << "( " << d.real() << " , " << d.imag() << " )";
  return s;
}


template<class T>
inline
StandardOutputStream& operator<<(StandardOutputStream& s, RComplex<T> d)
{
  s << "( " << d.real() << " , " << d.imag() << " )";
  return s;
}


template<class T>
inline
TextReader& operator>>(TextReader& s, RComplex<T> d)
{
  return s >> d.real() >> d.imag();
}


template<class T> 
inline  
TextWriter& operator<<(TextWriter& s, RComplex<T> d)
{
  return s << d.real() << d.imag();
}

#ifndef QDP_NO_LIBXML2
//! XML output
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, RComplex<T> d)
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
void read(XMLReader& xml, const string& xpath, RComplex<T> d)
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
		  << "Failed to match real part of RComplex Object with self constructed path: " << path_real;
    
    throw error_message.str();
  }
	
  // Try and recursively get the imaginary part
  try {
    read(xml, path_imag, d.imag());
  }
  catch(const string &e) {
    error_message << "XPath Query: " << xpath <<" Error:"
		  <<"Failed to match imaginary part of RComplex Object with self constructed path: " << path_imag;
    
    throw error_message.str();
  }
}
#endif


template<class T>
class WordSize< RScalar<T> >
{
public:
  static int value( const Shape& ret )
  {
    return 1 * WordSize<T>::value( ret );
  }
};


template<class T>
class WordSize< RComplex<T> >
{
public:
  static int value( const Shape& ret )
  {
    return 2 * WordSize<T>::value( ret );
  }
};



template<class T>
struct WordType<RScalar<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

template<class T>
struct WordType<RComplex<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};


template<class T> 
struct ScalarType<RScalar<T> >
{
  typedef RScalar<typename ScalarType<T>::Type_t>  Type_t;
};

template<class T> 
struct ScalarType<RComplex<T> >
{
  typedef RComplex<typename ScalarType<T>::Type_t>  Type_t;
};



template<class T> 
struct SinglePrecType<RScalar<T> >
{
  typedef RScalar<typename SinglePrecType<T>::Type_t>  Type_t;
};

template<class T> 
struct SinglePrecType<RComplex<T> >
{
  typedef RComplex<typename SinglePrecType<T>::Type_t>  Type_t;
};

template<class T> 
struct DoublePrecType<RScalar<T> >
{
  typedef RScalar<typename DoublePrecType<T>::Type_t>  Type_t;
};

template<class T> 
struct DoublePrecType<RComplex<T> >
{
  typedef RComplex<typename DoublePrecType<T>::Type_t>  Type_t;
};



template<class T>
struct InternalScalar<RScalar<T> > {
  typedef RScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

template<class T>
struct InternalScalar<RComplex<T> > {
  typedef RScalar<typename InternalScalar<T>::Type_t>  Type_t;
};



template<class T>
struct PrimitiveScalar<RScalar<T> > {
  typedef RScalar<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

template<class T>
struct PrimitiveScalar<RComplex<T> > {
  typedef RScalar<typename PrimitiveScalar<T>::Type_t>  Type_t;
};


template<class T>
struct LatticeScalar<RScalar<T> > {
  typedef RScalar<typename LatticeScalar<T>::Type_t>  Type_t;
};

template<class T>
struct LatticeScalar<RComplex<T> > {
  typedef RComplex<typename LatticeScalar<T>::Type_t>  Type_t;
};



template<class T>
struct RealScalar<RScalar<T> > {
  typedef RScalar<typename RealScalar<T>::Type_t>  Type_t;
};

template<class T>
struct RealScalar<RComplex<T> > {
  typedef RScalar<typename RealScalar<T>::Type_t>  Type_t;
};



template<class T1, class Op>
struct UnaryReturn<RScalar<T1>, Op> {
  typedef RScalar<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};


template<class T1, class Op>
struct UnaryReturn<RComplex<T1>, Op> {
  typedef RComplex<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, Op> {
  typedef RScalar<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, Op> {
  typedef RComplex<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<RScalar<T1>, RComplex<T2>, Op> {
  typedef RComplex<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<RComplex<T1>, RScalar<T2>, Op> {
  typedef RComplex<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};






template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpAddAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpSubtractAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpMultiplyAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpDivideAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpModAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpBitwiseOrAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpBitwiseAndAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpBitwiseXorAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpLeftShiftAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpRightShiftAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
struct TrinaryReturn<RScalar<T1>, RScalar<T2>, RScalar<T3>, FnColorContract> {
  typedef RScalar<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};

 
template<class T1, class T2, class T3>
struct TrinaryReturn<RComplex<T1>, RComplex<T2>, RComplex<T3>, FnColorContract> {
  typedef RComplex<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};


template<class T> 
inline int 
toInt(const RScalar<T>& s) 
{
  return toInt(s.elem());
}


template<class T> 
inline float
toFloat(const RScalar<T>& s) 
{
  return toFloat(s.elem());
}


template<class T> 
inline double
toDouble(const RScalar<T>& s) 
{
  return toDouble(s.elem());
}


template<class T> 
inline bool
toBool(const RScalar<T>& s) 
{
  return toBool(s.elem());
}


template<class T> 
inline typename WordType< RScalar<T> >::Type_t
toWordType(const RScalar<T>& s) 
{
  return toWordType(s.elem());
}



template<class T>
struct UnaryReturn<RScalar<T>, FnSum > {
  typedef RScalar<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};




template<class T>
struct UnaryReturn<RScalar<T>, FnGlobalMax> {
  typedef RScalar<typename UnaryReturn<T, FnGlobalMax>::Type_t>  Type_t;
};



template<class T>
struct UnaryReturn<RScalar<T>, FnGlobalMin> {
  typedef RScalar<typename UnaryReturn<T, FnGlobalMin>::Type_t>  Type_t;
};



template<class T>
struct UnaryReturn<RScalar<T>, FnNorm2 > {
  typedef RScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};


template<class T>
struct UnaryReturn<RScalar<T>, FnLocalNorm2 > {
  typedef RScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};




template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, FnInnerProduct > {
  typedef RScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, FnLocalInnerProduct > {
  typedef RScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, FnLocalColorInnerProduct > {
  typedef RScalar<typename BinaryReturn<T1, T2, FnLocalColorInnerProduct>::Type_t>  Type_t;
};




template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, FnInnerProductReal > {
  typedef RScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, FnLocalInnerProductReal > {
  typedef RScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};




template<class T1, class T2, class T3>
struct TrinaryReturn<RScalar<T1>, RScalar<T2>, RScalar<T3>, FnWhere> {
  typedef RScalar<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};


template<class T>
struct UnaryReturn<RComplex<T>, FnTrace > {
  typedef RComplex<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};



template<class T>
struct UnaryReturn<RComplex<T>, FnRealTrace > {
  typedef RScalar<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};



template<class T>
struct UnaryReturn<RComplex<T>, FnImagTrace > {
  typedef RScalar<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};


template<class T>
struct UnaryReturn<RComplex<T>, FnReal > {
  typedef RScalar<typename UnaryReturn<T, FnReal>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<RComplex<T>, FnImag > {
  typedef RScalar<typename UnaryReturn<T, FnImag>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, FnCmplx > {
  typedef RComplex<typename BinaryReturn<T1, T2, FnCmplx>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<RScalar<T>, FnTimesI > {
  typedef RComplex<typename UnaryReturn<T, FnTimesI>::Type_t>  Type_t;
};



template<class T>
struct UnaryReturn<RScalar<T>, FnTimesMinusI > {
  typedef RComplex<typename UnaryReturn<T, FnTimesMinusI>::Type_t>  Type_t;
};


template<class T>
struct UnaryReturn<RComplex<T>, FnSum> {
  typedef RComplex<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};



template<class T>
struct UnaryReturn<RComplex<T>, FnNorm2 > {
  typedef RScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<RComplex<T>, FnLocalNorm2 > {
  typedef RScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};





template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, FnInnerProduct > {
  typedef RComplex<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, FnLocalInnerProduct > {
  typedef RComplex<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};



template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, FnInnerProductReal > {
  typedef RScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, FnLocalInnerProductReal > {
  typedef RScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
struct TrinaryReturn<RScalar<T1>, RComplex<T2>, RComplex<T3>, FnWhere> {
  typedef RComplex<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
struct TrinaryReturn<RScalar<T1>, RComplex<T2>, RScalar<T3>, FnWhere> {
  typedef RComplex<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
struct TrinaryReturn<RScalar<T1>, RScalar<T2>, RComplex<T3>, FnWhere> {
  typedef RComplex<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};


template<class T>
void zero_rep(RScalar<T> dest)
{
  zero_rep(dest.elem());
}

template<class T>
void zero_rep(RComplex<T> dest)
{
  zero_rep(dest.real());
  zero_rep(dest.imag());
}



} // namespace QDP

#endif
