#ifndef QDP_PRIMSCALAR_H
#define QDP_PRIMSCALAR_H


namespace QDP {


template<class T> class PScalar: public BaseView<T>
{

public:

  typedef T Sub_t;
  enum { ThisSize = 1 };

  Shape shape() const { return BaseView<T>::shape(); }

  
  PScalar( Shape shape , typename WordType<T>::Type_t *base_ ): BaseView<T>(shape,base_) {}


  PScalar(const PScalar& a): BaseView<T>(a)
  {
  }


  PScalar operator=(const PScalar& rhs)
  {
    elem() = rhs.elem();
    return *this;
  }

  
  T elem() const { return BaseView<T>::elem(0); }
};



  template<class T>
  struct GetLimit<PScalar<T>,0>
  {
    static int limit(Shape shape)
    {
      return 1;
    }
  };







template<class T>
class WordSize< PScalar<T> >
{
public:
  static int value( const Shape& ret )
  {
    return 1 * WordSize<T>::value( ret );
  }
};


template<class T> 
struct JITType<PScalar<T> >
{
  typedef PScalarJIT<typename JITType<T>::Type_t>  Type_t;
};


template<class T>
inline
istream& operator>>(istream& s, PScalar<T> d)
{
  return s >> d.elem();
}

template<class T>
inline
StandardInputStream& operator>>(StandardInputStream& s, PScalar<T> d)
{
  return s >> d.elem();
}

template<class T>
inline
ostream& operator<<(ostream& s, PScalar<T> d)
{
  return s << d.elem();
}

template<class T>
inline
StandardOutputStream& operator<<(StandardOutputStream& s, PScalar<T> d)
{
  return s << d.elem();
}

template<class T>
inline
TextReader& operator>>(TextReader& txt, PScalar<T> d)
{
  return txt >> d.elem();
}

template<class T>
inline
TextWriter& operator<<(TextWriter& txt, PScalar<T> d)
{
  return txt << d.elem();
}

#ifndef QDP_NO_LIBXML2
//! XML output
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, PScalar<T> d)
{
  return xml << d.elem();
}

//! XML input
template<class T>
inline
void read(XMLReader& xml, const string& path, PScalar<T> d)
{
  read(xml, path, d.elem());
}
#endif

/*! @} */  // end of group primscalar




template<class T>
struct ScalarType<PScalar<T> >
{
  typedef PScalar< typename ScalarType<T>::Type_t > Type_t;
};


  
template<class T>
struct WordType<PScalar<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

template<class T>
struct SinglePrecType<PScalar<T> >
{
  typedef PScalar< typename SinglePrecType<T>::Type_t > Type_t;
};

template<class T>
struct DoublePrecType<PScalar<T> >
{
  typedef PScalar< typename DoublePrecType<T>::Type_t > Type_t;
};

template<class T>
struct InternalScalar<PScalar<T> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

template<class T>
struct RealScalar<PScalar<T> > {
  typedef PScalar<typename RealScalar<T>::Type_t>  Type_t;
};

template<class T>
struct PrimitiveScalar<PScalar<T> > {
  typedef PScalar<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

template<class T>
struct LatticeScalar<PScalar<T> > {
  typedef PScalar<typename LatticeScalar<T>::Type_t>  Type_t;
};




template<class T1, class Op>
struct UnaryReturn<PScalar<T1>, Op> {
  typedef PScalar<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

template<class T1, class T2, class Op>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, Op> {
  typedef PScalar<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};



template<class T>
void zero_rep(PScalar<T> dest)
{
  zero_rep(dest.elem());
}


template<class T> 
inline int 
toInt(const PScalar<T>& s) 
{
  return toInt(s.elem());
}


template<class T> 
inline float
toFloat(const PScalar<T>& s) 
{
  return toFloat(s.elem());
}


template<class T> 
inline double
toDouble(const PScalar<T>& s) 
{
  return toDouble(s.elem());
}


template<class T> 
inline bool
toBool(const PScalar<T>& s) 
{
  return toBool(s.elem());
}


template<class T> 
inline typename WordType< PScalar<T> >::Type_t
toWordType(const PScalar<T>& s) 
{
  return toWordType(s.elem());
}



} // namespace QDP

#endif
