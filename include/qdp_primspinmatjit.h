#ifndef QDP_PRIMSPINMATJIT_H
#define QDP_PRIMSPINMATJIT_H


namespace QDP {


template <class T> class PSpinMatrixJIT : public PMatrixJIT<T, PSpinMatrixJIT>
{
public:

  PSpinMatrixJIT( llvm::Value * base , JitDataFormat fmt , NestedIndex args , Shape shape ) : PMatrixJIT<T, PSpinMatrixJIT>(base,fmt,args,shape) {}

  // for eval Shape
  PSpinMatrixJIT( Shape shape ) : PMatrixJIT<T, PSpinMatrixJIT>(shape) {}
  
  Shape::Axis axis() const { return Shape::Axis::Spin; }

  int getN() const
  {
    return shape().extent(axis());
  }

  Shape shape() const { return PMatrixJIT<T, PSpinMatrixJIT>::shape(); }

  
  template<class T1>
  PSpinMatrixJIT(const PSpinMatrixJIT<T1>& a): PMatrixJIT<T, PSpinMatrixJIT>(a)
  {
    this->assign(a);
  }

  
  template<class T1>
  PSpinMatrixJIT operator=(PSpinMatrixJIT<T1> rhs) 
  {
    this->assign(rhs);
    return *this;
  }


  PSpinMatrixJIT operator=(PSpinMatrixJIT rhs) 
  {
    this->assign(rhs);
    return *this;
  }

  
  template<class T1>
  PSpinMatrixJIT& operator=(const PScalarJIT<T1>& rhs)
  {
    this->assign(rhs);
    return *this;
  }
};



template<class T>
class WordSize< PSpinMatrixJIT<T> >
{
public:
  static int value( const Shape& ret )
  {
    return ret.ns() * ret.ns() * WordSize<T>::value( ret );
  }
};

  

template<class T1>
struct JITType<PSpinMatrixJIT<T1> > 
{
  typedef PSpinMatrixJIT<typename JITType<T1>::Type_t>  Type_t;
};


template<class T1>
struct WordType<PSpinMatrixJIT<T1> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};


template<class T1>
struct SinglePrecType<PSpinMatrixJIT<T1> >
{
  typedef PSpinMatrixJIT< typename SinglePrecType<T1>::Type_t  > Type_t;
};


template<class T1>
struct DoublePrecType<PSpinMatrixJIT<T1> >
{
  typedef PSpinMatrixJIT< typename DoublePrecType<T1>::Type_t  > Type_t;
};


template<class T>
struct InternalScalar<PSpinMatrixJIT<T> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};


template<class T>
struct PrimitiveScalar<PSpinMatrixJIT<T> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};


template<class T>
struct LatticeScalar<PSpinMatrixJIT<T> > {
  typedef PSpinMatrixJIT<typename LatticeScalar<T>::Type_t>  Type_t;
};




template<class T1, class Op>
struct UnaryReturn<PSpinMatrixJIT<T1>, Op> {
  typedef PSpinMatrixJIT<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2>, Op> {
  typedef PSpinMatrixJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, Op> {
  typedef PSpinMatrixJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<PSpinMatrixJIT<T1>, PScalarJIT<T2>, Op> {
  typedef PSpinMatrixJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};



template<class T>
struct UnaryReturn<PSpinMatrixJIT<T>, FnTrace > {
  typedef PScalarJIT<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PSpinMatrixJIT<T>, FnRealTrace > {
  typedef PScalarJIT<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PSpinMatrixJIT<T>, FnImagTrace > {
  typedef PScalarJIT<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PSpinMatrixJIT<T>, FnNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PSpinMatrixJIT<T>, FnLocalNorm2 > {
  typedef PScalarJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrixJIT<T1>, PScalarJIT<T2>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrixJIT<T1>, PScalarJIT<T2>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};



template<class T1, class T2>
typename BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnLocalColorInnerProduct>::Type_t
localColorInnerProduct(PSpinMatrixJIT<T1> l, PSpinMatrixJIT<T2> r)
{
  auto d = binary_return<FnLocalColorInnerProduct>(l,r);
  int N = l.getN();
  
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      {
	d.elem(i,j) = localColorInnerProduct(l.elem(0,i), r.elem(0,j));
	for(int k=1; k < N; ++k)
	  d.elem(i,j) += localColorInnerProduct(l.elem(k,i), r.elem(k,j));
      }
    
  return d;
}


  
template<class T1, class T2>
struct BinaryReturn<PSpinMatrixJIT<T1>, PScalarJIT<T2>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrixJIT<T1>, PScalarJIT<T2>, FnInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2>, FnInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnLocalInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PSpinMatrixJIT<T1>, PScalarJIT<T2>, FnLocalInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2>, FnLocalInnerProductReal> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};




template<class T2>
struct BinaryReturn<GammaType, PSpinMatrixJIT<T2>, OpGammaTypeMultiply> {
  typedef PSpinMatrixJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2>
struct BinaryReturn<PSpinMatrixJIT<T2>, GammaType, OpMultiplyGammaType> {
  typedef PSpinMatrixJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};


template<class T2>
struct BinaryReturn<GammaTypeDP, PSpinMatrixJIT<T2>, OpGammaTypeMultiply> {
  typedef PSpinMatrixJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2>
struct BinaryReturn<PSpinMatrixJIT<T2>, GammaTypeDP, OpMultiplyGammaType> {
  typedef PSpinMatrixJIT<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};



template<class T>
struct UnaryReturn<PSpinMatrixJIT<T>, FnTraceSpin > {
  typedef PScalarJIT<typename UnaryReturn<T, FnTraceSpin>::Type_t>  Type_t;
};

template<class T>
typename UnaryReturn<PSpinMatrixJIT<T>, FnTraceSpin>::Type_t
traceSpin(PSpinMatrixJIT<T> s1)
{
  auto d = unary_return<FnTraceSpin>(s1);
  int N=s1.getN();
  
  d.elem() = s1.elem(0,0);
  JitForLoop i(1,N);
  {
    d.elem() += s1.elem(i,i);
  }
  i.end();
  
  return d;
}


template<class T1, class T2>
struct BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnTraceSpinMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t>  Type_t;
};

template<class T1, class T2>
typename BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(PSpinMatrixJIT<T1> l, PSpinMatrixJIT<T2> r)
{
  assert(l.getN() == r.getN());

  auto d = binary_return<FnTraceSpinMultiply>(l,r);
  int N=l.getN();
  
  d.elem() = l.elem(0,0) * r.elem(0,0);
  JitForLoop k(1,N);
  {
    d.elem() += l.elem(0,k) * r.elem(k,0);
  }
  k.end();

  JitForLoop j(1,N);
  {
    JitForLoop k(0,N);
    {
      d.elem() += l.elem(j,k) * r.elem(k,j);
    }
    k.end();
  }
  j.end();
  
  return d;
}


template<class T1, class T2>
struct BinaryReturn<PSpinMatrixJIT<T1>, PScalarJIT<T2>, FnTraceSpinMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class,int> class C>
typename BinaryReturn<PSpinMatrixJIT<T1>, PScalarJIT<T2>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(PSpinMatrixJIT<T1> l, const PScalarJIT<T2>& r)
{
  auto d = binary_return<FnTraceSpinMultiply>(l,r);
  int N=l.getN();

  d.elem() = l.elem(0,0) * r.elem();
  JitForLoop k(1,N);
  {
    d.elem() += l.elem(k,k) * r.elem();
  }
  k.end();
  
  return d;
}


template<class T1, class T2>
struct BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2>, FnTraceSpinMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t>  Type_t;
};


template<class T1, class T2>
typename BinaryReturn<PScalarJIT<T1>, PSpinMatrixJIT<T2>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(const PScalarJIT<T1>& l, PSpinMatrixJIT<T2> r)
{
  auto d = binary_return<FnTraceSpinMultiply>(l,r);
  int N=r.getN();

  d.elem() = l.elem() * r.elem(0,0);
  JitForLoop k(1,N);
  {
    d.elem() += l.elem() * r.elem(k,k);
  }
  k.end();
  
  return d;
}



template <class T>
struct UnaryReturn<PSpinMatrixJIT<T>, FnTransposeSpin > {
  typedef PSpinMatrixJIT<typename UnaryReturn<T, FnTransposeSpin>::Type_t> Type_t;
};



template<class T>
typename UnaryReturn<PSpinMatrixJIT<T>, FnTransposeSpin >::Type_t
transposeSpin(PSpinMatrixJIT<T> s1)
{
  auto d = unary_return<FnTransposeSpin>(s1);
  int N=s1.getN();

  JitForLoop i(0,N);
  {
    JitForLoop j(0,N);
    {
      d.elem(i,j) = s1.elem(j,i);
    }
    j.end();
  }
  i.end();
  
  return d;
}




template<class T1, class T2>
struct BinaryReturn<PSpinVectorJIT<T1>, PSpinVectorJIT<T2>, FnOuterProduct> {
  typedef PSpinMatrixJIT<typename BinaryReturn<T1, T2, FnOuterProduct>::Type_t>  Type_t;
};


template<class T1, class T2>
typename BinaryReturn<PSpinVectorJIT<T1>, PSpinVectorJIT<T2>, FnOuterProduct>::Type_t
outerProduct(const PSpinVectorJIT<T1>& l, const PSpinVectorJIT<T2>& r)
{
  assert(l.getN() == r.getN());

  auto d = binary_return<FnOuterProduct>(l,r);
  int N=l.getN();
  
  JitForLoop i(0,N);
  {
    JitForLoop j(0,N);
    {
      d.elem(i,j) = outerProduct(l.elem(i),r.elem(j));
    }
    j.end();
  }
  i.end();
  
  return d;
}



template<class T1, class T2>
struct BinaryReturn<PSpinVectorJIT<T1>, PSpinVectorJIT<T2>, FnTraceSpinOuterProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnOuterProduct>::Type_t>  Type_t;
};


template<class T1, class T2>
typename BinaryReturn<PSpinVectorJIT<T1>, PSpinVectorJIT<T2>, FnTraceSpinOuterProduct>::Type_t
traceSpinOuterProduct(const PSpinVectorJIT<T1>& l, const PSpinVectorJIT<T2>& r)
{
  assert(l.getN() == r.getN());

  auto d = binary_return<FnTraceSpinOuterProduct>(l,r);
  int N=l.getN();

  d.elem() = outerProduct(l.elem(0),r.elem(0));
  JitForLoop i(1,N);
  {
    d.elem() += outerProduct(l.elem(i),r.elem(i));
  }
  i.end();
  
  return d;
}



template<class T>
struct UnaryReturn<PSpinMatrixJIT<T>, FnPeekSpinMatrixJIT > {
  typedef PScalarJIT<T>  Type_t;
};


template<class T>
struct UnaryReturn<PSpinMatrixJIT<T>, FnPeekSpinMatrix > {
  typedef PScalarJIT<T>  Type_t;
};



template<class T>
typename UnaryReturn<PSpinMatrixJIT<T>, FnPeekSpinMatrixJIT >::Type_t
peekSpin(PSpinMatrixJIT<T> l, llvm::Value* row, llvm::Value* col)
{
  auto d = unary_return<FnPeekSpinMatrixJIT>(l);
  d.elem() = l.elem(row,col);
  return d;
}



template<class T>
struct UnaryReturn<PSpinMatrixJIT<T>, FnPeekColorMatrixJIT > {
  typedef PSpinMatrixJIT<typename UnaryReturn<T, FnPeekColorMatrixJIT >::Type_t>  Type_t;
};


template<class T>
struct UnaryReturn<PSpinMatrixJIT<T>, FnPeekColorMatrix > {
  typedef PSpinMatrixJIT<typename UnaryReturn<T, FnPeekColorMatrix >::Type_t>  Type_t;
};



template<class T>
typename UnaryReturn<PSpinMatrixJIT<T>, FnPeekColorMatrixJIT >::Type_t
peekColor(PSpinMatrixJIT<T> l, llvm::Value* row, llvm::Value* col)
{
  auto d = unary_return<FnPeekColorMatrixJIT>(l);

  int N=l.getN();
  
  JitForLoop i(0,N);
  {
    JitForLoop j(0,N);
    {
      d.elem(i,j) = peekColor( l.elem(i,j) , row, col );
    }
    j.end();
  }
  i.end();

  return d;
}


template<class T1, class T2>
void
pokeSpin(PSpinMatrixJIT<T1> l, const PScalarJIT<T2>& r, llvm::Value* row, llvm::Value* col)
{
  l.elem(row,col) = r.elem();
}


template<class T1, class T2>
void
pokeColor(PSpinMatrixJIT<T1> l, PSpinMatrixJIT<T2> r, llvm::Value* row, llvm::Value* col)
{
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      pokeColor( l.elem(i,j) , r.elem(i,j) , row , col );
    }
    j.end();
  }
  i.end();
}


template<class T2>
typename BinaryReturn<GammaType, PSpinMatrixJIT<T2>, OpGammaTypeMultiply>::Type_t
operator*(const GammaType& g, PSpinMatrixJIT<T2> r)
{
  auto d = binary_return<OpGammaTypeMultiply>(g,r);
  
  switch ( g.elem() ) {
  case 0:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = r.elem(0,i);
	d.elem(1,i) = r.elem(1,i);
	d.elem(2,i) = r.elem(2,i);
	d.elem(3,i) = r.elem(3,i);
      }
      i.end();
    }
    break;
  case 1:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesI(r.elem(3,i));
	d.elem(1,i) = timesI(r.elem(2,i));
	d.elem(2,i) = timesMinusI(r.elem(1,i));
	d.elem(3,i) = timesMinusI(r.elem(0,i));
      }
      i.end();
    }
    break;
  case 2:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = -r.elem(3,i);
	d.elem(1,i) = r.elem(2,i);
	d.elem(2,i) = r.elem(1,i);
	d.elem(3,i) = -r.elem(0,i);
      }
      i.end();
    }
    break;
  case 3:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesMinusI(r.elem(0,i));
	d.elem(1,i) = timesI(r.elem(1,i));
	d.elem(2,i) = timesMinusI(r.elem(2,i));
	d.elem(3,i) = timesI(r.elem(3,i));
      }
      i.end();
    }
    break;
  case 4:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesI(r.elem(2,i));
	d.elem(1,i) = timesMinusI(r.elem(3,i));
	d.elem(2,i) = timesMinusI(r.elem(0,i));
	d.elem(3,i) = timesI(r.elem(1,i));
      }
      i.end();
    }
    break;
  case 5:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = -r.elem(1,i);
	d.elem(1,i) = r.elem(0,i);
	d.elem(2,i) = -r.elem(3,i);
	d.elem(3,i) = r.elem(2,i);
      }
      i.end();
    }
    break;
  case 6:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesMinusI(r.elem(1,i));
	d.elem(1,i) = timesMinusI(r.elem(0,i));
	d.elem(2,i) = timesMinusI(r.elem(3,i));
	d.elem(3,i) = timesMinusI(r.elem(2,i));
      }
      i.end();
    }
    break;
  case 7:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = r.elem(2,i);
	d.elem(1,i) = r.elem(3,i);
	d.elem(2,i) = -r.elem(0,i);
	d.elem(3,i) = -r.elem(1,i);
      }
      i.end();
    }
    break;
  case 8:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = r.elem(2,i);
	d.elem(1,i) = r.elem(3,i);
	d.elem(2,i) = r.elem(0,i);
	d.elem(3,i) = r.elem(1,i);
      }
      i.end();
    }
    break;
  case 9:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesI(r.elem(1,i));
	d.elem(1,i) = timesI(r.elem(0,i));
	d.elem(2,i) = timesMinusI(r.elem(3,i));
	d.elem(3,i) = timesMinusI(r.elem(2,i));
      }
      i.end();
    }
    break;
  case 10:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = -r.elem(1,i);
	d.elem(1,i) = r.elem(0,i);
	d.elem(2,i) = r.elem(3,i);
	d.elem(3,i) = -r.elem(2,i);
      }
      i.end();
    }
    break;
  case 11:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesMinusI(r.elem(2,i));
	d.elem(1,i) = timesI(r.elem(3,i));
	d.elem(2,i) = timesMinusI(r.elem(0,i));
	d.elem(3,i) = timesI(r.elem(1,i));
      }
      i.end();
    }
    break;
  case 12:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesI(r.elem(0,i));
	d.elem(1,i) = timesMinusI(r.elem(1,i));
	d.elem(2,i) = timesMinusI(r.elem(2,i));
	d.elem(3,i) = timesI(r.elem(3,i));
      }
      i.end();
    }
    break;
  case 13:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = -r.elem(3,i);
	d.elem(1,i) = r.elem(2,i);
	d.elem(2,i) = -r.elem(1,i);
	d.elem(3,i) = r.elem(0,i);
      }
      i.end();
    }
    break;
  case 14:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesMinusI(r.elem(3,i));
	d.elem(1,i) = timesMinusI(r.elem(2,i));
	d.elem(2,i) = timesMinusI(r.elem(1,i));
	d.elem(3,i) = timesMinusI(r.elem(0,i));
      }
      i.end();
    }
    break;
  case 15:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = r.elem(0,i);
	d.elem(1,i) = r.elem(1,i);
	d.elem(2,i) = -r.elem(2,i);
	d.elem(3,i) = -r.elem(3,i);
      }
      i.end();
    }
    break;
  default:
    QDPIO::cout << "Invalid gamma matrix number: " << g.elem() << std::endl;
    QDP_abort(1);
  }
  
  return d;
}




template<class T2>
typename BinaryReturn<GammaTypeDP, PSpinMatrixJIT<T2>, OpGammaTypeMultiply>::Type_t
operator*(const GammaTypeDP& g, PSpinMatrixJIT<T2> r)
{
  auto d = binary_return<OpGammaTypeMultiply>(g,r);

  switch ( g.elem() ) {
  case 0:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = r.elem(0,i);
	d.elem(1,i) = r.elem(1,i);
	d.elem(2,i) = r.elem(2,i);
	d.elem(3,i) = r.elem(3,i);
      }
      i.end();
    }
    break;
  case 1:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesMinusI(r.elem(3,i));
	d.elem(1,i) = timesMinusI(r.elem(2,i));
	d.elem(2,i) = timesI(r.elem(1,i));
	d.elem(3,i) = timesI(r.elem(0,i));
      }
      i.end();
    }
    break;
  case 2:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = -r.elem(3,i);
	d.elem(1,i) = r.elem(2,i);
	d.elem(2,i) = r.elem(1,i);
	d.elem(3,i) = -r.elem(0,i);
      }
      i.end();
    }
    break;
  case 3:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesI(r.elem(0,i));
	d.elem(1,i) = timesMinusI(r.elem(1,i));
	d.elem(2,i) = timesI(r.elem(2,i));
	d.elem(3,i) = timesMinusI(r.elem(3,i));
      }
      i.end();
    }
    break;
  case 4:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesMinusI(r.elem(2,i));
	d.elem(1,i) = timesI(r.elem(3,i));
	d.elem(2,i) = timesI(r.elem(0,i));
	d.elem(3,i) = timesMinusI(r.elem(1,i));
      }
      i.end();
    }
    break;
  case 5:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = -r.elem(1,i);
	d.elem(1,i) = r.elem(0,i);
	d.elem(2,i) = -r.elem(3,i);
	d.elem(3,i) = r.elem(2,i);
      }
      i.end();
    }
    break;
  case 6:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesI(r.elem(1,i));
	d.elem(1,i) = timesI(r.elem(0,i));
	d.elem(2,i) = timesI(r.elem(3,i));
	d.elem(3,i) = timesI(r.elem(2,i));
      }
      i.end();
    }
    break;
  case 7:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = r.elem(2,i);
	d.elem(1,i) = r.elem(3,i);
	d.elem(2,i) = -r.elem(0,i);
	d.elem(3,i) = -r.elem(1,i);
      }
      i.end();
    }
    break;
  case 8:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = r.elem(0,i);
	d.elem(1,i) = r.elem(1,i);
	d.elem(2,i) = -r.elem(2,i);
	d.elem(3,i) = -r.elem(3,i);
      }
      i.end();
    }
    break;
  case 9:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesI(r.elem(3,i));
	d.elem(1,i) = timesI(r.elem(2,i));
	d.elem(2,i) = timesI(r.elem(1,i));
	d.elem(3,i) = timesI(r.elem(0,i));
      }
      i.end();
    }
    break;
  case 10:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) =  r.elem(3,i);
	d.elem(1,i) = -r.elem(2,i);
	d.elem(2,i) =  r.elem(1,i);
	d.elem(3,i) = -r.elem(0,i);
      }
      i.end();
    }
    break;
  case 11:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesI(r.elem(0,i));
	d.elem(1,i) = timesMinusI(r.elem(1,i));
	d.elem(2,i) = timesMinusI(r.elem(2,i));
	d.elem(3,i) = timesI(r.elem(3,i));
      }
      i.end();
    }
    break;
  case 12:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesI(r.elem(2,i));
	d.elem(1,i) = timesMinusI(r.elem(3,i));
	d.elem(2,i) = timesI(r.elem(0,i));
	d.elem(3,i) = timesMinusI(r.elem(1,i));
      }
      i.end();
    }
    break;
  case 13:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = -r.elem(1,i);
	d.elem(1,i) = r.elem(0,i);
	d.elem(2,i) = r.elem(3,i);
	d.elem(3,i) = -r.elem(2,i);
      }
      i.end();
    }
    break;
  case 14:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = timesI(r.elem(1,i));
	d.elem(1,i) = timesI(r.elem(0,i));
	d.elem(2,i) = timesMinusI(r.elem(3,i));
	d.elem(3,i) = timesMinusI(r.elem(2,i));
      }
      i.end();
    }
    break;
  case 15:
    {
      JitForLoop i(0,4);
      {
	d.elem(0,i) = -r.elem(2,i);
	d.elem(1,i) = -r.elem(3,i);
	d.elem(2,i) = -r.elem(0,i);
	d.elem(3,i) = -r.elem(1,i);
      }
      i.end();
    }
    break;
  default:
    QDPIO::cout << "Invalid gamma matrix number: " << g.elem() << std::endl;
    QDP_abort(1);
  }

  
  return d;
}



template<class T2>
typename BinaryReturn<PSpinMatrixJIT<T2>, GammaType, OpMultiplyGammaType>::Type_t
operator*(PSpinMatrixJIT<T2> l, const GammaType& g)
{
  auto d = binary_return<OpMultiplyGammaType>(l,g);
  
  switch ( g.elem() ) {
  case 0:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) =  l.elem(i,0);
	d.elem(i,1) =  l.elem(i,1);
	d.elem(i,2) =  l.elem(i,2);
	d.elem(i,3) =  l.elem(i,3);
      }
      i.end();
    }
    break;  
  case 1:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesMinusI(l.elem(i,3));
	d.elem(i,1) = timesMinusI(l.elem(i,2));
	d.elem(i,2) = timesI(l.elem(i,1));
	d.elem(i,3) = timesI(l.elem(i,0));
      }
      i.end();
    }
    break;  
  case 2:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = -l.elem(i,3);
	d.elem(i,1) =  l.elem(i,2);
	d.elem(i,2) =  l.elem(i,1);
	d.elem(i,3) = -l.elem(i,0);
      }
      i.end();
    }
    break;  
  case 3:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesMinusI(l.elem(i,0));
	d.elem(i,1) = timesI(l.elem(i,1));
	d.elem(i,2) = timesMinusI(l.elem(i,2));
	d.elem(i,3) = timesI(l.elem(i,3));
      }
      i.end();
    }
    break;  
  case 4:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesMinusI(l.elem(i,2));
	d.elem(i,1) = timesI(l.elem(i,3));
	d.elem(i,2) = timesI(l.elem(i,0));
	d.elem(i,3) = timesMinusI(l.elem(i,1));
      }
      i.end();
    }
    break;  
  case 5:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) =  l.elem(i,1);
	d.elem(i,1) = -l.elem(i,0);
	d.elem(i,2) =  l.elem(i,3);
	d.elem(i,3) = -l.elem(i,2);
      }
      i.end();
    }
    break;  
  case 6:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesMinusI(l.elem(i,1));
	d.elem(i,1) = timesMinusI(l.elem(i,0));
	d.elem(i,2) = timesMinusI(l.elem(i,3));
	d.elem(i,3) = timesMinusI(l.elem(i,2));
      }
      i.end();
    }
    break;  
  case 7:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = -l.elem(i,2);
	d.elem(i,1) = -l.elem(i,3);
	d.elem(i,2) =  l.elem(i,0);
	d.elem(i,3) =  l.elem(i,1);
      }
      i.end();
    }
    break;  
  case 8:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) =  l.elem(i,2);
	d.elem(i,1) =  l.elem(i,3);
	d.elem(i,2) =  l.elem(i,0);
	d.elem(i,3) =  l.elem(i,1);
      }
      i.end();
    }
    break;  
  case 9:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesI(l.elem(i,1));
	d.elem(i,1) = timesI(l.elem(i,0));
	d.elem(i,2) = timesMinusI(l.elem(i,3));
	d.elem(i,3) = timesMinusI(l.elem(i,2));
      }
      i.end();
    }
    break;  
  case 10:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) =  l.elem(i,1);
	d.elem(i,1) = -l.elem(i,0);
	d.elem(i,2) = -l.elem(i,3);
	d.elem(i,3) =  l.elem(i,2);
      }
      i.end();
    }
    break;  
  case 11:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesMinusI(l.elem(i,2));
	d.elem(i,1) = timesI(l.elem(i,3));
	d.elem(i,2) = timesMinusI(l.elem(i,0));
	d.elem(i,3) = timesI(l.elem(i,1));
      }
      i.end();
    }
    break;  
  case 12:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesI(l.elem(i,0));
	d.elem(i,1) = timesMinusI(l.elem(i,1));
	d.elem(i,2) = timesMinusI(l.elem(i,2));
	d.elem(i,3) = timesI(l.elem(i,3));
      }
      i.end();
    }
    break;  
  case 13:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) =  l.elem(i,3);
	d.elem(i,1) = -l.elem(i,2);
	d.elem(i,2) =  l.elem(i,1);
	d.elem(i,3) = -l.elem(i,0);
      }
      i.end();
    }
    break;  
  case 14:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesMinusI(l.elem(i,3));
	d.elem(i,1) = timesMinusI(l.elem(i,2));
	d.elem(i,2) = timesMinusI(l.elem(i,1));
	d.elem(i,3) = timesMinusI(l.elem(i,0));
      }
      i.end();
    }
    break;  
  case 15:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) =  l.elem(i,0);
	d.elem(i,1) =  l.elem(i,1);
	d.elem(i,2) = -l.elem(i,2);
	d.elem(i,3) = -l.elem(i,3);
      }
      i.end();
    }
    break;  
  default:
    QDPIO::cout << "Invalid gamma matrix number: " << g.elem() << std::endl;
    QDP_abort(1);
  }
 
  return d;
}



template<class T2>
typename BinaryReturn<PSpinMatrixJIT<T2>, GammaTypeDP, OpMultiplyGammaType>::Type_t
operator*(PSpinMatrixJIT<T2> l, const GammaTypeDP& g)
{
  auto d = binary_return<OpMultiplyGammaType>(l,g);

  switch ( g.elem() ) {
  case 0:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) =  l.elem(i,0);
	d.elem(i,1) =  l.elem(i,1);
	d.elem(i,2) =  l.elem(i,2);
	d.elem(i,3) =  l.elem(i,3);
      }
      i.end();
    }
    break;  
  case 1:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesI(l.elem(i,3));
	d.elem(i,1) = timesI(l.elem(i,2));
	d.elem(i,2) = timesMinusI(l.elem(i,1));
	d.elem(i,3) = timesMinusI(l.elem(i,0));
      }
      i.end();
    }
    break;  
  case 2:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = -l.elem(i,3);
	d.elem(i,1) =  l.elem(i,2);
	d.elem(i,2) =  l.elem(i,1);
	d.elem(i,3) = -l.elem(i,0);
      }
      i.end();
    }
    break;  
  case 3:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesI(l.elem(i,0));
	d.elem(i,1) = timesMinusI(l.elem(i,1));
	d.elem(i,2) = timesI(l.elem(i,2));
	d.elem(i,3) = timesMinusI(l.elem(i,3));
      }
      i.end();
    }
    break;  
  case 4:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesI(l.elem(i,2));
	d.elem(i,1) = timesMinusI(l.elem(i,3));
	d.elem(i,2) = timesMinusI(l.elem(i,0));
	d.elem(i,3) = timesI(l.elem(i,1));
      }
      i.end();
    }
    break;  
  case 5:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) =  l.elem(i,1);
	d.elem(i,1) = -l.elem(i,0);
	d.elem(i,2) =  l.elem(i,3);
	d.elem(i,3) = -l.elem(i,2);
      }
      i.end();
    }
    break;  
  case 6:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesI(l.elem(i,1));
	d.elem(i,1) = timesI(l.elem(i,0));
	d.elem(i,2) = timesI(l.elem(i,3));
	d.elem(i,3) = timesI(l.elem(i,2));
      }
      i.end();
    }
    break;  
  case 7:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = -l.elem(i,2);
	d.elem(i,1) = -l.elem(i,3);
	d.elem(i,2) =  l.elem(i,0);
	d.elem(i,3) =  l.elem(i,1);
      }
      i.end();
    }
    break;  
  case 8:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) =  l.elem(i,0);
	d.elem(i,1) =  l.elem(i,1);
	d.elem(i,2) = -l.elem(i,2);
	d.elem(i,3) = -l.elem(i,3);
      }
      i.end();
    }
    break;  
  case 9:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesI(l.elem(i,3));
	d.elem(i,1) = timesI(l.elem(i,2));
	d.elem(i,2) = timesI(l.elem(i,1));
	d.elem(i,3) = timesI(l.elem(i,0));
      }
      i.end();
    }
    break;  
  case 10:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = -l.elem(i,3);
	d.elem(i,1) =  l.elem(i,2);
	d.elem(i,2) = -l.elem(i,1);
	d.elem(i,3) =  l.elem(i,0);
      }
      i.end();
    }
    break;  
  case 11:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesI(l.elem(i,0));
	d.elem(i,1) = timesMinusI(l.elem(i,1));
	d.elem(i,2) = timesMinusI(l.elem(i,2));
	d.elem(i,3) = timesI(l.elem(i,3));
      }
      i.end();
    }
    break;  
  case 12:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesI(l.elem(i,2));
	d.elem(i,1) = timesMinusI(l.elem(i,3));
	d.elem(i,2) = timesI(l.elem(i,0));
	d.elem(i,3) = timesMinusI(l.elem(i,1));
      }
      i.end();
    }
    break;  
  case 13:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) =  l.elem(i,1);
	d.elem(i,1) = -l.elem(i,0);
	d.elem(i,2) = -l.elem(i,3);
	d.elem(i,3) =  l.elem(i,2);
      }
      i.end();
    }
    break;  
  case 14:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = timesI(l.elem(i,1));
	d.elem(i,1) = timesI(l.elem(i,0));
	d.elem(i,2) = timesMinusI(l.elem(i,3));
	d.elem(i,3) = timesMinusI(l.elem(i,2));
      }
      i.end();
    }
    break;  
  case 15:
    {
      JitForLoop i(0,4);
      {
	d.elem(i,0) = -l.elem(i,2);
	d.elem(i,1) = -l.elem(i,3);
	d.elem(i,2) = -l.elem(i,0);
	d.elem(i,3) = -l.elem(i,1);
      }
      i.end();
    }
    break;  
  default:
    QDPIO::cout << "Invalid gamma matrix number: " << g.elem() << std::endl;
    QDP_abort(1);
  }
 
  return d;
}




template<class T>
typename UnaryReturn<PSpinMatrixJIT<T>, FnChiralProjectPlus>::Type_t
chiralProjectPlus(PSpinMatrixJIT<T> s1)
{
  assert(s1.getN() == 4);
  auto d = unary_return<FnChiralProjectPlus>(s1);
  
  JitForLoop i(0,4);
  {
    d.elem(0,i) = s1.elem(0,i);
    d.elem(1,i) = s1.elem(1,i);
    zero_rep(d.elem(2,i));
    zero_rep(d.elem(3,i));
  }
  i.end();

  return d;
}


template<class T>
typename UnaryReturn<PSpinMatrixJIT<T>, FnChiralProjectMinus>::Type_t
chiralProjectMinus(PSpinMatrixJIT<T> s1)
{
  assert(s1.getN() == 4);
  auto d = unary_return<FnChiralProjectMinus>(s1);

  JitForLoop i(0,4);
  {
    zero_rep(d.elem(0,i));
    zero_rep(d.elem(1,i));
    d.elem(2,i) = s1.elem(2,i);
    d.elem(3,i) = s1.elem(3,i);
  }
  i.end();

  return d;
}


template<class T1, class T2>
struct BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnTraceSpinQuarkContract13> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceSpinQuarkContract13>::Type_t>  Type_t;
};


template<class T1, class T2>
typename BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnTraceSpinQuarkContract13>::Type_t
traceSpinQuarkContract13(PSpinMatrixJIT<T1> l, PSpinMatrixJIT<T2> r)
{
  assert(l.getN() == 4);

  auto d = binary_return<FnTraceSpinQuarkContract13>(l,r);

  d.elem() = quarkContractXX(l.elem(0,0), r.elem(0,0));
  JitForLoop k(1,4);
  {
    d.elem() += quarkContractXX(l.elem(k,0), r.elem(k,0));
  }
  k.end();
  
  JitForLoop j(1,4);
  {
    JitForLoop k(0,4);
    {
      d.elem() += quarkContractXX(l.elem(k,j), r.elem(k,j));
    }
    k.end();
  }
  j.end();

  return d;
}



template<class T1, class T2>
typename BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnQuarkContract13>::Type_t
quarkContract13(PSpinMatrixJIT<T1> s1, PSpinMatrixJIT<T2> s2)
{
  assert(s1.getN() == 4);

  auto d = binary_return<FnQuarkContract13>(s1,s2);

  JitForLoop j(0,4);
  {
    JitForLoop i(0,4);
    {
      d.elem(i,j) = quarkContractXX(s1.elem(0,i), s2.elem(0,j));
      JitForLoop k(1,4);
      {
       	d.elem(i,j) += quarkContractXX(s1.elem(k,i), s2.elem(k,j));
      }
      k.end();
    }
    i.end();
  }
  j.end();
    
  return d;
}
  

template<class T1, class T2>
typename BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnQuarkContract14>::Type_t
quarkContract14(PSpinMatrixJIT<T1> s1, PSpinMatrixJIT<T2> s2)
{
  assert(s1.getN() == 4);

  auto d = binary_return<FnQuarkContract14>(s1,s2);

  JitForLoop j(0,4);
  {
    JitForLoop i(0,4);
    {
      d.elem(i,j) = quarkContractXX(s1.elem(0,i), s2.elem(j,0));
      JitForLoop k(1,4);
      {
	d.elem(i,j) += quarkContractXX(s1.elem(k,i), s2.elem(j,k));
      }
      k.end();
    }
    i.end();
  }
  j.end();
    
  return d;
}

  
template<class T1, class T2>
typename BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnQuarkContract23>::Type_t
quarkContract23(PSpinMatrixJIT<T1> s1, PSpinMatrixJIT<T2> s2)
{
  assert(s1.getN() == 4);

  auto d = binary_return<FnQuarkContract23>(s1,s2);

  JitForLoop j(0,4);
  {
    JitForLoop i(0,4);
    {
      d.elem(i,j) = quarkContractXX(s1.elem(i,0), s2.elem(0,j));
      JitForLoop k(1,4);
      {
	d.elem(i,j) += quarkContractXX(s1.elem(i,k), s2.elem(k,j));
      }
      k.end();
    }
    i.end();
  }
  j.end();
    
  return d;
}

  
template<class T1, class T2>
typename BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnQuarkContract24>::Type_t
quarkContract24(PSpinMatrixJIT<T1> s1, PSpinMatrixJIT<T2> s2)
{
  assert(s1.getN() == 4);

  auto d = binary_return<FnQuarkContract24>(s1,s2);

  JitForLoop j(0,4);
  {
    JitForLoop i(0,4);
    {
      d.elem(i,j) = quarkContractXX(s1.elem(i,0), s2.elem(j,0));
      JitForLoop k(1,4);
      {
	d.elem(i,j) += quarkContractXX(s1.elem(i,k), s2.elem(j,k));
      }
      k.end();
    }
    i.end();
  }
  j.end();

  return d;
}

  
template<class T1, class T2>
typename BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnQuarkContract12>::Type_t
quarkContract12(PSpinMatrixJIT<T1> s1, PSpinMatrixJIT<T2> s2)
{
  assert(s1.getN() == 4);

  auto d = binary_return<FnQuarkContract12>(s1,s2);

  JitForLoop j(0,4);
  {
    JitForLoop i(0,4);
    {
      d.elem(i,j) = quarkContractXX(s1.elem(0,0), s2.elem(i,j));
      JitForLoop k(1,4);
      {
	d.elem(i,j) += quarkContractXX(s1.elem(k,k), s2.elem(i,j));
      }
      k.end();
    }
    i.end();
  }
  j.end();
    
  return d;
}

  
template<class T1, class T2>
typename BinaryReturn<PSpinMatrixJIT<T1>, PSpinMatrixJIT<T2>, FnQuarkContract34>::Type_t
quarkContract34(PSpinMatrixJIT<T1> s1, PSpinMatrixJIT<T2> s2)
{
  assert(s1.getN() == 4);
  
  auto d = binary_return<FnQuarkContract34>(s1,s2);

  JitForLoop j(0,4);
  {
    JitForLoop i(0,4);
    {
      d.elem(i,j) = quarkContractXX(s1.elem(i,j), s2.elem(0,0));
      JitForLoop k(1,4);
      {
	d.elem(i,j) += quarkContractXX(s1.elem(i,j), s2.elem(k,k));
      }
      k.end();
    }
    i.end();
  }
  j.end();
    

  return d;
}



} // namespace QDP

#endif
