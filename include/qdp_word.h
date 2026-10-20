#ifndef QDP_WORD_H
#define QDP_WORD_H


namespace QDP {

  template<class T> class Word
  {
    T* f = nullptr;
  public:

    Word(T* f_): f(f_) {}
    Word(Shape dummy, T* f_): f(f_) {}

    Shape shape() const { return Shape(); }

    Word(const Word& a)
    {
      f = a.f;
    }

    Word operator=(const Word& rhs)
    {
      elem() = rhs.elem();
      return *this;
    }


    T& elem() { assert(f); return *f;}
    const T& elem() const { assert(f); return *f;}
  };



  template<class T> 
  struct JITType<Word<T> >
  {
    typedef WordJIT<T>  Type_t;
  };



  template<class T>
  inline
  istream& operator>>(istream& s, Word<T> d)
  {
    return s >> d.elem();
  }


  template<class T>
  inline
  StandardInputStream& operator>>(StandardInputStream& s, Word<T> d)
  {
    return s >> d.elem();
  }


  template<class T> 
  inline  
  ostream& operator<<(ostream& s, Word<T> d)
  {
    return s << d.elem();
  }


  template<class T> 
  inline  
  StandardOutputStream& operator<<(StandardOutputStream& s, Word<T> d)
  {
    return s << d.elem();
  }



  template<class T>
  inline
  TextReader& operator>>(TextReader& s, Word<T> d)
  {
    return s >> d.elem();
  }


  template<class T> 
  inline  
  TextWriter& operator<<(TextWriter& s, Word<T> d)
  {
    return s << d.elem();
  }

#ifndef QDP_NO_LIBXML2
  //! XML output
  template<class T>
  inline
  XMLWriter& operator<<(XMLWriter& xml, Word<T> d)
  {
    return xml << d.elem();
  }

  //! XML input
  template<class T>
  inline
  void read(XMLReader& xml, const string& path, Word<T> d)
  {
    read(xml, path, d.elem());
  }
#endif


  template<class T>
  class WordSize< Word<T> >
  {
  public:
    static int value( const Shape& ret )
    {
      return 1;
    }
  };



  template<class T>
  struct WordType<Word<T> > 
  {
    typedef T Type_t;
  };


  template<class T>
  struct ScalarType<Word<T> > 
  {
    typedef Word<T> Type_t;
  };



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



  template<class T>
  struct InternalScalar<Word<T> > {
    typedef Word<typename InternalScalar<T>::Type_t>  Type_t;
  };


  template<class T>
  struct PrimitiveScalar<Word<T> > {
    typedef Word<typename PrimitiveScalar<T>::Type_t>  Type_t;
  };


  template<class T>
  struct LatticeScalar<Word<T> > {
    typedef Word<typename LatticeScalar<T>::Type_t>  Type_t;
  };


  template<class T>
  struct RealScalar<Word<T> > {
    typedef Word<typename RealScalar<T>::Type_t>  Type_t;
  };



  template<class T1, class Op>
  struct UnaryReturn<Word<T1>, Op> {
    typedef Word<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
  };


  template<class T1, class T2, class Op>
  struct BinaryReturn<Word<T1>, Word<T2>, Op> {
    typedef Word<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };



  template<class T> 
  inline int 
  toInt(const Word<T>& s) 
  {
    return toInt(s.elem());
  }


  template<class T> 
  inline float
  toFloat(const Word<T>& s) 
  {
    return toFloat(s.elem());
  }


  template<class T> 
  inline double
  toDouble(const Word<T>& s) 
  {
    return toDouble(s.elem());
  }


  template<class T> 
  inline bool
  toBool(const Word<T>& s) 
  {
    return toBool(s.elem());
  }


  template<class T> 
  inline typename WordType< Word<T> >::Type_t
  toWordType(const Word<T>& s) 
  {
    return toWordType(s.elem());
  }


  template<class T>
  struct UnaryReturn<Word<T>, FnSum > {
    typedef Word<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
  };



  template<class T>
  struct UnaryReturn<Word<T>, FnGlobalMax> {
    typedef Word<typename UnaryReturn<T, FnGlobalMax>::Type_t>  Type_t;
  };



  template<class T>
  struct UnaryReturn<Word<T>, FnGlobalMin> {
    typedef Word<typename UnaryReturn<T, FnGlobalMin>::Type_t>  Type_t;
  };



  template<class T>
  struct UnaryReturn<Word<T>, FnNorm2 > {
    typedef Word<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
  };

  template<class T>
  struct UnaryReturn<Word<T>, FnLocalNorm2 > {
    typedef Word<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
  };



  template<class T1, class T2>
  struct BinaryReturn<Word<T1>, Word<T2>, FnInnerProduct > {
    typedef Word<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
  };


  template<class T1, class T2>
  struct BinaryReturn<Word<T1>, Word<T2>, FnLocalInnerProduct > {
    typedef Word<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
  };




  template<class T1, class T2>
  struct BinaryReturn<Word<T1>, Word<T2>, FnInnerProductReal > {
    typedef Word<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
  };


  template<class T1, class T2>
  struct BinaryReturn<Word<T1>, Word<T2>, FnLocalInnerProductReal > {
    typedef Word<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
  };


  template<class T1, class T2, class T3>
  struct TrinaryReturn<Word<T1>, Word<T2>, Word<T3>, FnWhere> {
    typedef Word<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
  };


  template<class T>
  void zero_rep(Word<T> dest)
  {
    dest.elem() = 0;
  }


} // namespace QDP

#endif
