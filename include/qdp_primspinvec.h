#ifndef QDP_PRIMSPINVEC_H
#define QDP_PRIMSPINVEC_H


namespace QDP {


template <class T> class PSpinVector : public BaseView<T>
{
public:
  typedef T Sub_t;


  PSpinVector( Shape shape , typename WordType<T>::Type_t *base_ ): BaseView<T>(shape,base_) {}

  
  Shape shape() const { return BaseView<T>::shape(); }

  
  int getN() const
  {
    return shape().ns();
  }

  
  PSpinVector operator=(const PSpinVector& rhs)
  {
    for (int i = 0 ; i < getN() ; ++i )
      elem(i) = rhs.elem(i);
    return *this;
  }


  T elem(int i) const
  {
    return BaseView<T>::elem(i);
  }

};



  template<class T>
  struct GetLimit<PSpinVector<T>,0>
  {
    static int limit(const Shape& shape)
    {
      return shape.extent(Shape::Axis::Spin);
    }
  };



  



  

template<class T>  
inline
istream& operator>>(istream& s, PSpinVector<T>& d)
{
  for(int i=0; i < d.getN(); ++i)
    s >> d.elem(i);

  return s;
}

template<class T>  
inline
StandardInputStream& operator>>(StandardInputStream& s, PSpinVector<T>& d)
{
  for(int i=0; i < d.getN(); ++i)
    s >> d.elem(i);

  return s;
}

template<class T>
inline
ostream& operator<<(ostream& s, const PSpinVector<T>& d)
{
  for(int i=0; i < d.getN(); ++i)
    s << d.elem(i);

  return s;
}


template<class T>
inline
StandardOutputStream& operator<<(StandardOutputStream& s, const PSpinVector<T>& d)
{
  for(int i=0; i < d.getN(); ++i)
    s << d.elem(i);

  return s;
}



template<class T>
inline
TextReader& operator>>(TextReader& txt, PSpinVector<T>& d)
{
  for(int i=0; i < d.getN(); ++i)
    txt >> d.elem(i);

  return txt;
}


template<class T>  
inline
TextWriter& operator<<(TextWriter& txt, const PSpinVector<T>& d)
{
  for(int i=0; i < d.getN(); ++i)
    txt << d.elem(i);

  return txt;
}

#ifndef QDP_NO_LIBXML2
//! XML output
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, const PSpinVector<T>& d)
{
  xml.openTag("SpinVector");

  XMLWriterAPI::AttributeList alist;

  // Copy into another array first
  for(int i=0; i < d.getN(); ++i)
  {
    alist.clear();
    alist.push_back(XMLWriterAPI::Attribute("row", i));

    xml.openTag("elem", alist);
    xml << d.elem(i);
    xml.closeTag();
  }

  xml.closeTag();  // Vector
  return xml;
}
#endif




  template<class T> 
  struct JITType<PSpinVector<T> >
  {
    typedef PSpinVectorJIT<typename JITType<T>::Type_t>  Type_t;
  };


 




template<class T>
inline typename UnaryReturn<PSpinVector<T>, FnGetSite>::Type_t
getSite(const PSpinVector<T>& s1, int innersite)
{ 
  typename UnaryReturn<PSpinVector<T>, FnGetSite>::Type_t  d;int N=0;assert(!"ni");

  for(int i=0; i < N; ++i)
    d.elem(i) = getSite(s1.elem(i), innersite);

  return d;
}




template<class T1, class T2, class T3>
struct TrinaryReturn<PScalar<T1>, PSpinVector<T2>, PSpinVector<T3>, FnWhere> {
  typedef PSpinVector<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};



template<class T>
class WordSize< PSpinVector<T> >
{
public:
  static int value( const Shape& ret )
  {
    return ret.extent(Shape::Axis::Spin) * WordSize<T>::value( ret );
  }
};



template<class T1>
struct WordType<PSpinVector<T1> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

template<class T1>
struct ScalarType< PSpinVector<T1> > 
{
  typedef PSpinVector< typename ScalarType<T1>::Type_t> Type_t;
};



template<class T1>
struct SinglePrecType< PSpinVector<T1> > 
{
  typedef PSpinVector< typename SinglePrecType<T1>::Type_t> Type_t;
};

template<class T1>
struct DoublePrecType< PSpinVector<T1> > 
{
  typedef PSpinVector< typename DoublePrecType<T1>::Type_t> Type_t;
};


template<class T>
struct InternalScalar<PSpinVector<T> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};


template<class T>
struct PrimitiveScalar<PSpinVector<T> > {
  typedef PScalar<typename PrimitiveScalar<T>::Type_t>  Type_t;
};


template<class T>
struct LatticeScalar<PSpinVector<T> > {
  typedef PSpinVector<typename LatticeScalar<T>::Type_t>  Type_t;
};


template<class T1, class Op>
struct UnaryReturn<PSpinVector<T1>, Op> {
  typedef PSpinVector<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<PScalar<T1>, PSpinVector<T2>, Op> {
  typedef PSpinVector<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn< PSpinMatrix<T1>, PSpinVector<T2>, Op> {
  typedef PSpinVector<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};



template<class T1, class T2, template <class> class C1, class Op>
struct BinaryReturn< PMatrix<T1,C1>, PSpinVector<T2>, Op> {
  typedef PSpinVector<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};



template<class T1, class T2, class Op>
struct BinaryReturn<PSpinVector<T1>, PScalar<T2>, Op> {
  typedef PSpinVector<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};



template<class T1, class T2, class Op>
struct BinaryReturn<PSpinVector<T1>, PSpinVector<T2>, Op> {
  typedef PSpinVector<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};




template<class T>
struct UnaryReturn<PSpinVector<T>, FnNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PSpinVector<T>, FnLocalNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinVector<T1>, PSpinVector<T2>, FnInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinVector<T1>, PSpinVector<T2>, FnLocalInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinVector<T1>, PSpinVector<T2>, FnInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinVector<T1>, PSpinVector<T2>, FnLocalInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};



template<class T2>
struct BinaryReturn<GammaType, PSpinVector<T2>, OpGammaTypeMultiply> {
  typedef PSpinVector<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};





template<class T>
struct UnaryReturn<PSpinVector<T>, FnPeekSpinVector > {
  typedef PScalar<typename UnaryReturn<T, FnPeekSpinVector>::Type_t>  Type_t;
};



} // namespace QDP

#endif
