#ifndef QDP_PRIMVECTOR_H
#define QDP_PRIMVECTOR_H



namespace QDP {


template <class T, template<class> class C> class PVector: public BaseView<T>
{
public:
  typedef C<T>  CC;
  typedef T Sub_t;

  Shape shape() const
  {
    return BaseView<T>::shape();
  }

  int getN() const
  {
    return shape().extent( static_cast<const CC*>(this)->axis() );
  }

  PVector( Shape shape , typename WordType<T>::Type_t *base_ ): BaseView<T>(shape,base_) {}


  PVector operator=(const PVector& rhs)
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


  


template<class T, template<class> class C>  
inline
istream& operator>>(istream& s, PVector<T,C>& d)
{
  for(int i=0; i < d.getN(); ++i)
    s >> d.elem(i);

  return s;
}

template<class T, template<class> class C>  
inline
StandardInputStream& operator>>(StandardInputStream& s, PVector<T,C>& d)
{
  for(int i=0; i < d.getN(); ++i)
    s >> d.elem(i);

  return s;
}


template<class T, template<class> class C>  
inline
ostream& operator<<(ostream& s, const PVector<T,C>& d)
{
  for(int i=0; i < d.getN(); ++i)
    s << d.elem(i);

  return s;
}


template<class T, template<class> class C>  
inline
StandardOutputStream& operator<<(StandardOutputStream& s, const PVector<T,C>& d)
{
  for(int i=0; i < d.getN(); ++i)
    s << d.elem(i);

  return s;
}



template<class T, template<class> class C>  
inline
TextReader& operator>>(TextReader& txt, PVector<T,C>& d)
{
  for(int i=0; i < d.getN(); ++i)
    txt >> d.elem(i);

  return txt;
}


template<class T, template<class> class C>  
inline
TextWriter& operator<<(TextWriter& txt, const PVector<T,C>& d)
{
  for(int i=0; i < d.getN(); ++i)
    txt << d.elem(i);

  return txt;
}

#ifndef QDP_NO_LIBXML2
//! XML output
template<class T, template<class> class C> 
inline
XMLWriter& operator<<(XMLWriter& xml, const PVector<T,C>& d)
{
  xml.openTag("Vector");

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


  

template<class T1, template<class> class C>
struct WordType<PVector<T1,C> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};


template<class T1, template<class> class C> 
struct ScalarType< PVector<T1,C> >
{
  typedef PVector< typename ScalarType<T1>::Type_t, C> Type_t;
};


template<class T1, template<class> class C> 
struct SinglePrecType< PVector<T1,C> >
{
  typedef PVector< typename SinglePrecType<T1>::Type_t, C> Type_t;
};

template<class T1, template<class> class C> 
struct DoublePrecType< PVector<T1,C> >
{
  typedef PVector< typename DoublePrecType<T1>::Type_t, C> Type_t;
};


template<class T, template<class> class C>
struct InternalScalar<PVector<T,C> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};


template<class T, template<class> class C>
struct PrimitiveScalar<PVector<T,C> > {
  typedef PScalar<typename PrimitiveScalar<T>::Type_t>  Type_t;
};


template<class T, template<class> class C>
struct LatticeScalar<PVector<T,C> > {
  typedef C<typename LatticeScalar<T>::Type_t>  Type_t;
};




template<class T1, template<class> class C, class Op>
struct UnaryReturn<PVector<T1,C>, Op> {
  typedef C<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C, class Op>
struct BinaryReturn<PScalar<T1>, PVector<T2,C>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C1, 
  template<class> class C2, class Op>
struct BinaryReturn<PMatrix<T1,C1>, PVector<T2,C2>, Op> {
  typedef C2<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C, class Op>
struct BinaryReturn<PVector<T1,C>, PScalar<T2>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C, class Op>
struct BinaryReturn<PVector<T1,C>, PVector<T2,C>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};




template<class T, template<class> class C>
struct UnaryReturn<PVector<T,C>, FnNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, template<class> class C>
struct UnaryReturn<PVector<T,C>, FnLocalNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
struct BinaryReturn<PVector<T1,C>, PVector<T2,C>, FnInnerProduct > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
struct BinaryReturn<PVector<T1,C>, PVector<T2,C>, FnLocalInnerProduct > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PVector<T1,C>, PVector<T2,C>, FnInnerProductReal > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
struct BinaryReturn<PVector<T1,C>, PVector<T2,C>, FnLocalInnerProductReal > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, class T3, template<class> class C>
struct TrinaryReturn<PScalar<T1>, PVector<T2,C>, PVector<T3,C>, FnWhere> {
  typedef C<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};




} // namespace QDP

#endif
