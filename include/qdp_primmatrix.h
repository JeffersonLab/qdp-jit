#ifndef QDP_PRIMMATRIX_H
#define QDP_PRIMMATRIX_H


namespace QDP {

template <class T, template<class> class C> class PMatrix: public BaseView<T>
{
public:
  typedef C<T>  CC;
  typedef T Sub_t;

  PMatrix( Shape shape , typename WordType<T>::Type_t* f): BaseView<T>(shape,f) {}

  Shape shape() const { return BaseView<T>::shape(); }

  int getN() const
  {
    return static_cast<const CC*>(this)->getN();
  }


  PMatrix operator=(const PMatrix& rhs)
  {
    for (int i = 0 ; i < getN() ; ++i )
      for (int j = 0 ; j < getN() ; ++j )
	elem(i,j) = rhs.elem(i,j);
    return *this;
  }


  T elem(int i, int j) const {return BaseView<T>::elem(j + getN() * i);}
};






template<class T, template<class> class C>  
inline
ostream& operator<<(ostream& s, PMatrix<T,C> d)
{
  for(int j=0; j < d.getN(); ++j) {
    for(int i=0; i < d.getN(); ++i)
      s << d.elem(i,j);
    s << "\n";
  }
  return s;
}


template<class T, template<class> class C>  
inline
StandardOutputStream& operator<<(StandardOutputStream& s, PMatrix<T,C> d)
{
  for(int j=0; j < d.getN(); ++j)
    for(int i=0; i < d.getN(); ++i)
      s << d.elem(i,j);
  return s;
}




template<class T, template<class> class C>  
inline
TextReader& operator>>(TextReader& txt, PMatrix<T,C> d)
{
  for(int j=0; j < d.getN(); ++j)
    for(int i=0; i < d.getN(); ++i)
      txt >> d.elem(i,j);

  return txt;
}


template<class T, template<class> class C>  
inline
TextWriter& operator<<(TextWriter& txt, PMatrix<T,C> d)
{
  for(int j=0; j < d.getN(); ++j)
    for(int i=0; i < d.getN(); ++i)
      txt << d.elem(i,j);

  return txt;
}

#ifndef QDP_NO_LIBXML2
template<class T, template<class> class C>  
inline
XMLWriter& operator<<(XMLWriter& xml, PMatrix<T,C> d)
{
  xml.openTag("Matrix");

  XMLWriterAPI::AttributeList alist;

  for(int i=0; i < d.getN(); ++i)
  {
    for(int j=0; j < d.getN(); ++j)
    {
      alist.clear();
      alist.push_back(XMLWriterAPI::Attribute("row", i));
      alist.push_back(XMLWriterAPI::Attribute("col", j));

      xml.openTag("elem", alist);
      xml << d.elem(i,j);
      xml.closeTag();
    }
  }

  xml.closeTag(); // Matrix
  return xml;
}
#endif




// Underlying word type
template<class T1, template<class> class C>
struct WordType<PMatrix<T1,C> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

// Fixed Precision
template<class T1, template<class> class C>
struct SinglePrecType< PMatrix<T1, C> >
{
  typedef PMatrix< typename SinglePrecType<T1>::Type_t, C > Type_t;
};


// Fixed Precision
template<class T1, template<class> class C>
struct DoublePrecType< PMatrix<T1, C> >
{
  typedef PMatrix< typename DoublePrecType<T1>::Type_t, C > Type_t;
};

// Internally used scalars
template<class T, template<class> class C>
struct InternalScalar<PMatrix<T,C> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive scalar leaving grid alone
template<class T, template<class> class C>
struct PrimitiveScalar<PMatrix<T,C> > {
  typedef PScalar<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T, template<class> class C>
struct LatticeScalar<PMatrix<T,C> > {
  typedef C<typename LatticeScalar<T>::Type_t>  Type_t;
};



template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, OpIdentity> {
  typedef C<typename UnaryReturn<T, OpIdentity>::Type_t>  Type_t;
};


template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, OpUnaryPlus> {
  typedef C<typename UnaryReturn<T, OpUnaryPlus>::Type_t>  Type_t;
};



template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, OpUnaryMinus> {
  typedef C<typename UnaryReturn<T, OpUnaryMinus>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PMatrix<T2,C>, OpAdd> {
  typedef C<typename BinaryReturn<T1, T2, OpAdd>::Type_t>  Type_t;
};




template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PScalar<T2>, OpAdd> {
  typedef C<typename BinaryReturn<T1, T2, OpAdd>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalar<T1>, PMatrix<T2,C>, OpAdd> {
  typedef C<typename BinaryReturn<T1, T2, OpAdd>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PMatrix<T2,C>, OpSubtract> {
  typedef C<typename BinaryReturn<T1, T2, OpSubtract>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PScalar<T2>, OpSubtract> {
  typedef C<typename BinaryReturn<T1, T2, OpSubtract>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalar<T1>, PMatrix<T2,C>, OpSubtract> {
  typedef C<typename BinaryReturn<T1, T2, OpSubtract>::Type_t>  Type_t;
};



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PScalar<T2>, OpMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiply>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PScalar<T2>, OpAdjMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiply>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PScalar<T2>, OpMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiplyAdj>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PScalar<T2>, OpAdjMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiplyAdj>::Type_t>  Type_t;
};



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalar<T1>, PMatrix<T2,C>, OpMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiply>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalar<T1>, PMatrix<T2,C>, OpAdjMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiply>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalar<T1>, PMatrix<T2,C>, OpMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiplyAdj>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalar<T1>, PMatrix<T2,C>, OpAdjMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiplyAdj>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PMatrix<T2,C>, OpMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiply>::Type_t>  Type_t;
};



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PMatrix<T2,C>, OpAdjMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiply>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PMatrix<T2,C>, OpMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiplyAdj>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PMatrix<T2,C>, OpAdjMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiplyAdj>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PScalar<T2>, OpDivide> {
  typedef C<typename BinaryReturn<T1, T2, OpDivide>::Type_t>  Type_t;
};



template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnAdjoint> {
  typedef C<typename UnaryReturn<T, FnAdjoint>::Type_t>  Type_t;
};



template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnConjugate> {
  typedef C<typename UnaryReturn<T, FnConjugate>::Type_t>  Type_t;
};


template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnTranspose> {
  typedef C<typename UnaryReturn<T, FnTranspose>::Type_t>  Type_t;
};



template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnTrace> {
  typedef PScalar<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};



template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnRealTrace> {
  typedef PScalar<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};



template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnImagTrace> {
  typedef PScalar<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};



template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnTraceColor> {
  typedef C<typename UnaryReturn<T, FnTraceColor>::Type_t>  Type_t;
};



template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnTraceSpin> {
  typedef C<typename UnaryReturn<T, FnTraceSpin>::Type_t>  Type_t;
};



template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnTransposeColor> {
  typedef C<typename UnaryReturn<T, FnTransposeColor>::Type_t> Type_t;
};



template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnTransposeSpin> {
  typedef C<typename UnaryReturn<T, FnTransposeSpin>::Type_t> Type_t;
};



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PMatrix<T2,C>, FnTraceMultiply> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PScalar<T2>, FnTraceMultiply> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalar<T1>, PMatrix<T2,C>, FnTraceMultiply> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PMatrix<T2,C>, FnTraceColorMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PScalar<T2>, FnTraceColorMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalar<T1>, PMatrix<T2,C>, FnTraceColorMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PMatrix<T2,C>, FnTraceSpinMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PScalar<T2>, FnTraceSpinMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalar<T1>, PMatrix<T2,C>, FnTraceSpinMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t>  Type_t;
};



template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnReal> {
  typedef C<typename UnaryReturn<T, FnReal>::Type_t>  Type_t;
};



template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnImag> {
  typedef C<typename UnaryReturn<T, FnImag>::Type_t>  Type_t;
};


template<class T1, template<class> class C>
struct UnaryReturn<PMatrix<T1,C>, FnIsFinite> {
  typedef PScalar< typename UnaryReturn<T1, FnIsFinite >::Type_t > Type_t;
};



template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnTimesI> {
  typedef C<typename UnaryReturn<T, FnTimesI>::Type_t>  Type_t;
};


template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnTimesMinusI> {
  typedef C<typename UnaryReturn<T, FnTimesMinusI>::Type_t>  Type_t;
};



template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnNorm2> {
  typedef PScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};


template<class T, template<class> class C>
struct UnaryReturn<PMatrix<T,C>, FnLocalNorm2> {
  typedef PScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PMatrix<T2,C>, FnInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PMatrix<T2,C>, FnLocalInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PScalar<T2>, FnLocalInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalar<T1>, PMatrix<T2,C>, FnLocalInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PMatrix<T2,C>, FnInnerProductReal > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PMatrix<T2,C>, FnLocalInnerProductReal > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrix<T1,C>, PScalar<T2>, FnLocalInnerProductReal > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalar<T1>, PMatrix<T2,C>, FnLocalInnerProductReal > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};



template<class T1, class T2, class T3, template<class> class C>
struct TrinaryReturn<PScalar<T1>, PMatrix<T2,C>, PMatrix<T3,C>, FnWhere> {
  typedef C<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};


  

  

} // namespace QDP

#endif
