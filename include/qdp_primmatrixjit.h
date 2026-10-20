#ifndef QDP_PRIMMATRIXJIT_H
#define QDP_PRIMMATRIXJIT_H

namespace QDP {


template <class T, template<class> class C> class PMatrixJIT : public BaseJIT< T >
{
public:
  typedef C<T>  CC;
  typedef T  Sub_t;

  PMatrixJIT( llvm::Value * base , JitDataFormat fmt , NestedIndex args , Shape shape ) : BaseJIT< T >(base,fmt,args,shape) {}

  // for eval Shape
  PMatrixJIT( Shape shape ) : BaseJIT< T >(shape) {}

  PMatrixJIT(const PMatrixJIT& rhs) : BaseJIT< T >(rhs) {}

  int getN() const
  {
    return shape().extent( axis() );
  }

  Shape::Axis axis() const
  {
    return static_cast<const CC*>(this)->axis();
  }


  int this_size() const
  {
    return getN() * getN();
  }

  
  Shape shape() const { return BaseJIT< T >::shape(); }


  template<class T1>
  CC assign(PScalarJIT<T1> rhs)
    {
      JitForLoop i( 0 , getN() );
      {
	JitForLoop j( 0 , getN() );
	{
	  JitIf eq( llvm_eq( i.index() , j.index() ) );
	  {
	    elem(i,j) = rhs.elem();
	  }
	  eq.els();
	  {
	    zero_rep(elem(i,j));
	  }
	  eq.end();
	}
	j.end();
      }
      i.end();
      return *static_cast<CC*>(this);
    }


  template<class T1>
  CC assign(C<T1> rhs)
    {
      JitForLoop i( 0 , getN() );
      {
	JitForLoop j( 0 , getN() );
	{
	  elem(i,j) = rhs.elem(i,j);
	}
	j.end();
      }
      i.end();
      
      return *static_cast<CC*>(this);
    }



  template<class T1>
  CC operator+=(C<T1> rhs) 
    {
      JitForLoop i( 0 , getN() );
      {
	JitForLoop j( 0 , getN() );
	{
	  elem(i,j) += rhs.elem(i,j);
	}
	j.end();
      }
      i.end();
      
      return *static_cast<CC*>(this);
    }


  template<class T1>
  CC operator-=(C<T1> rhs) 
    {
      JitForLoop i( 0 , getN() );
      {
	JitForLoop j( 0 , getN() );
	{
	  elem(i,j) -= rhs.elem(i,j);
	}
	j.end();
      }
      i.end();
      
      return *static_cast<CC*>(this);
    }


  template<class T1>
  CC operator+=(PScalarJIT<T1> rhs) 
    {
      JitForLoop i( 0 , getN() );
      {
	elem(i,i) += rhs.elem();
      }
      i.end();
      
      return *static_cast<CC*>(this);
    }


  template<class T1>
  CC operator-=(PScalarJIT<T1> rhs) 
    {
      JitForLoop i( 0 , getN() );
      {
	elem(i,i) -= rhs.elem();
      }
      i.end();
      
      return *static_cast<CC*>(this);
    }


  template<class T1>
  CC operator*=(PScalarJIT<T1> rhs) 
    {
      JitForLoop i( 0 , getN() );
      {
	JitForLoop j( 0 , getN() );
	{
	  elem(i,j) *= rhs.elem();
	}
	j.end();
      }
      i.end();
      
      return *static_cast<CC*>(this);
    }


  template<class T1>
  CC operator/=(PScalarJIT<T1> rhs) 
    {
      JitForLoop i( 0 , getN() );
      {
	JitForLoop j( 0 , getN() );
	{
	  elem(i,j) /= rhs.elem();
	}
	j.end();
      }
      i.end();
      return *static_cast<CC*>(this);
    }


  T elem(int          i, int          j) const {return BaseJIT< T >::elem( i , j );}
  T elem(llvm::Value* i, int          j) const {return BaseJIT< T >::elem( i , j );}
  T elem(int          i, llvm::Value* j) const {return BaseJIT< T >::elem( i , j );}
  T elem(llvm::Value* i, llvm::Value* j) const {return BaseJIT< T >::elem( i , j );}
  T elem( const JitForLoop& i , const JitForLoop& j ) const {return BaseJIT< T >::elem( i , j );}
  T elem( const JitForLoop& i , int j ) const {return BaseJIT< T >::elem( i , j );}
  T elem( int i , const JitForLoop& j ) const {return BaseJIT< T >::elem( i , j );}
};



template<class T1, template<class> class C>
struct WordType<PMatrixJIT<T1,C> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};


template<class T1, template<class> class C>
struct SinglePrecType< PMatrixJIT<T1, C> >
{
  typedef PMatrixJIT< typename SinglePrecType<T1>::Type_t, C > Type_t;
};



template<class T1, template<class> class C>
struct DoublePrecType< PMatrixJIT<T1, C> >
{
  typedef PMatrixJIT< typename DoublePrecType<T1>::Type_t, C > Type_t;
};


template<class T, template<class> class C>
struct InternalScalar<PMatrixJIT<T,C> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};


template<class T, template<class> class C>
struct PrimitiveScalar<PMatrixJIT<T,C> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};


template<class T, template<class> class C>
struct LatticeScalar<PMatrixJIT<T,C> > {
  typedef C<typename LatticeScalar<T>::Type_t>  Type_t;
};





template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, OpIdentity> {
  typedef C<typename UnaryReturn<T, OpIdentity>::Type_t>  Type_t;
};




template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, OpUnaryPlus> {
  typedef C<typename UnaryReturn<T, OpUnaryPlus>::Type_t>  Type_t;
};

  
template<class T1, template<class> class C>
typename UnaryReturn<PMatrixJIT<T1,C>, OpUnaryPlus>::Type_t
operator+(PMatrixJIT<T1,C> l)
{
  auto d = unary_return<OpUnaryPlus>(l);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = +l.elem(i,j);
    }
    j.end();
  }
  i.end();
  
  return d;
}



template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, OpUnaryMinus> {
  typedef C<typename UnaryReturn<T, OpUnaryMinus>::Type_t>  Type_t;
};

template<class T1, template<class> class C>
typename UnaryReturn<PMatrixJIT<T1,C>, OpUnaryMinus>::Type_t
operator-(PMatrixJIT<T1,C> l)
{
  auto d = unary_return<OpUnaryMinus>(l);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = -l.elem(i,j);
    }
    j.end();
  }
  i.end();

  return d;
}


#if 0
template<class T1, class T2, template<class> class C, class Op>
struct BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};
#endif



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, OpAdd> {
  typedef C<typename BinaryReturn<T1, T2, OpAdd>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, OpAdd>::Type_t
operator+(PMatrixJIT<T1,C> l, PMatrixJIT<T2,C> r)
{
  assert(l.getN() == r.getN());
  
  auto d = binary_return<OpAdd>(l,r);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = l.elem(i,j) + r.elem(i,j);
    }
    j.end();
  }
  i.end();

  return d;
}



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, OpAdd> {
  typedef C<typename BinaryReturn<T1, T2, OpAdd>::Type_t>  Type_t;
};

  
template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, OpAdd>::Type_t
operator+(PMatrixJIT<T1,C> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpAdd>(l,r);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      JitIf eq( llvm_eq( i.index() , j.index() ) );
      {
	d.elem(i,j) = l.elem(i,i) + r.elem();
      }
      eq.els();
      {
	d.elem(i,j) = l.elem(i,j);
      }
      eq.end();
    }
    j.end();
  }
  i.end();
  
  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, OpAdd> {
  typedef C<typename BinaryReturn<T1, T2, OpAdd>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, OpAdd>::Type_t
operator+(PScalarJIT<T1> l, PMatrixJIT<T2,C> r)
{
  auto d = binary_return<OpAdd>(l,r);
  
  JitForLoop i( 0 , r.getN() );
  {
    JitForLoop j( 0 , r.getN() );
    {
      JitIf eq( llvm_eq( i.index() , j.index() ) );
      {
	d.elem(i,j) = l.elem() + r.elem(i,i);
      }
      eq.els();
      {
	d.elem(i,j) = r.elem(i,j);
      }
      eq.end();
    }
    j.end();
  }
  i.end();
  
  return d;
}



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, OpSubtract> {
  typedef C<typename BinaryReturn<T1, T2, OpSubtract>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, OpSubtract>::Type_t
operator-(PMatrixJIT<T1,C> l, PMatrixJIT<T2,C> r)
{
  assert(l.getN() == r.getN());

  auto d = binary_return<OpSubtract>(l,r);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = l.elem(i,j) - r.elem(i,j);
    }
    j.end();
  }
  i.end();

  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, OpSubtract> {
  typedef C<typename BinaryReturn<T1, T2, OpSubtract>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, OpSubtract>::Type_t
operator-(PMatrixJIT<T1,C> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpSubtract>(l,r);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      JitIf eq( llvm_eq( i.index() , j.index() ) );
      {
	d.elem(i,j) = l.elem(i,i) - r.elem();
      }
      eq.els();
      {
	d.elem(i,j) = l.elem(i,j);
      }
      eq.end();
    }
    j.end();
  }
  i.end();
  
  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, OpSubtract> {
  typedef C<typename BinaryReturn<T1, T2, OpSubtract>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, OpSubtract>::Type_t
operator-(PScalarJIT<T1> l, PMatrixJIT<T2,C> r)
{
  auto d = binary_return<OpSubtract>(l,r);
  
  JitForLoop i( 0 , r.getN() );
  {
    JitForLoop j( 0 , r.getN() );
    {
      JitIf eq( llvm_eq( i.index() , j.index() ) );
      {
	d.elem(i,j) = l.elem() - r.elem(i,i);
      }
      eq.els();
      {
	d.elem(i,j) = l.elem(i,j);
      }
      eq.end();
    }
    j.end();
  }
  i.end();
  
  return d;
}



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, OpMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, OpMultiply>::Type_t
operator*(PMatrixJIT<T1,C> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpMultiply>(l,r);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = l.elem(i,j) * r.elem();
    }
    j.end();
  }
  i.end();
  
  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, OpAdjMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, OpAdjMultiply>::Type_t
adjMultiply(PMatrixJIT<T1,C> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpAdjMultiply>(l,r);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = adjMultiply(l.elem(j,i), r.elem());
    }
    j.end();
  }
  i.end();
  
  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, OpMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiplyAdj>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(PMatrixJIT<T1,C> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpMultiplyAdj>(l,r);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = multiplyAdj(l.elem(i,j), r.elem());
    }
    j.end();
  }
  i.end();
  
  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, OpAdjMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiplyAdj>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(PMatrixJIT<T1,C> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpAdjMultiplyAdj>(l,r);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = adjMultiplyAdj(l.elem(j,i), r.elem());
    }
    j.end();
  }
  i.end();

  return d;
}




template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, OpMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, OpMultiply>::Type_t
operator*(PScalarJIT<T1> l, PMatrixJIT<T2,C> r)
{
  auto d = binary_return<OpMultiply>(l,r);
  
  JitForLoop i( 0 , r.getN() );
  {
    JitForLoop j( 0 , r.getN() );
    {
      d.elem(i,j) = l.elem() * r.elem(i,j);
    }
    j.end();
  }
  i.end();

  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, OpAdjMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, OpAdjMultiply>::Type_t
adjMultiply(PScalarJIT<T1> l, PMatrixJIT<T2,C> r)
{
  auto d = binary_return<OpAdjMultiply>(l,r);
  
  JitForLoop i( 0 , r.getN() );
  {
    JitForLoop j( 0 , r.getN() );
    {
      d.elem(i,j) = adjMultiply(l.elem(), r.elem(i,j));
    }
    j.end();
  }
  i.end();

  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, OpMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiplyAdj>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, OpMultiplyAdj>::Type_t
multiplyAdj(PScalarJIT<T1> l, PMatrixJIT<T2,C> r)
{
  auto d = binary_return<OpMultiplyAdj>(l,r);
  
  JitForLoop i( 0 , r.getN() );
  {
    JitForLoop j( 0 , r.getN() );
    {
      d.elem(i,j) = multiplyAdj(l.elem(), r.elem(j,i));
    }
    j.end();
  }
  i.end();

  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, OpAdjMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiplyAdj>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(PScalarJIT<T1> l, PMatrixJIT<T2,C> r)
{
  auto d = binary_return<OpAdjMultiplyAdj>(l,r);
  
  JitForLoop i( 0 , r.getN() );
  {
    JitForLoop j( 0 , r.getN() );
    {
      d.elem(i,j) = adjMultiplyAdj(l.elem(), r.elem(j,i));
    }
    j.end();
  }
  i.end();

  return d;
}



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, OpMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiply>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, OpMultiply>::Type_t
operator*(PMatrixJIT<T1,C> l, PMatrixJIT<T2,C> r)
{
  assert(l.getN() == r.getN());

  auto d = binary_return<OpMultiply>(l,r);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = l.elem(i,0) * r.elem(0,j);
      JitForLoop k( 1 , l.getN() );
      {
	d.elem(i,j) += l.elem(i,k) * r.elem(k,j);	
      }
      k.end();      
    }
    j.end();
  }
  i.end();

  return d;
}




template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, OpAdjMultiply> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, OpAdjMultiply>::Type_t
adjMultiply(PMatrixJIT<T1,C> l, PMatrixJIT<T2,C> r)
{
  assert(l.getN() == r.getN());

  auto d = binary_return<OpAdjMultiply>(l,r);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = adjMultiply(l.elem(0,i), r.elem(0,j));
      JitForLoop k( 1 , l.getN() );
      {
	d.elem(i,j) += adjMultiply(l.elem(k,i), r.elem(k,j));
      }
      k.end();      
    }
    j.end();
  }
  i.end();
  
  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, OpMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpMultiplyAdj>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, OpMultiplyAdj>::Type_t
multiplyAdj(PMatrixJIT<T1,C> l, PMatrixJIT<T2,C> r)
{
  assert(l.getN() == r.getN());

  auto d = binary_return<OpMultiplyAdj>(l,r);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = multiplyAdj(l.elem(i,0), r.elem(j,0));
      JitForLoop k( 1 , l.getN() );
      {
	d.elem(i,j) += multiplyAdj(l.elem(i,k), r.elem(j,k));
      }
      k.end();      
    }
    j.end();
  }
  i.end();

  return d;
}



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, OpAdjMultiplyAdj> {
  typedef C<typename BinaryReturn<T1, T2, OpAdjMultiplyAdj>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(PMatrixJIT<T1,C> l, PMatrixJIT<T2,C> r)
{
  assert(l.getN() == r.getN());

  auto d = binary_return<OpAdjMultiplyAdj>(l,r);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = adjMultiplyAdj(l.elem(0,i), r.elem(j,0));
      JitForLoop k( 1 , l.getN() );
      {
	d.elem(i,j) += adjMultiplyAdj(l.elem(k,i), r.elem(j,k));
      }
      k.end();      
    }
    j.end();
  }
  i.end();

  return d;
}



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, OpDivide> {
  typedef C<typename BinaryReturn<T1, T2, OpDivide>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, OpDivide>::Type_t
operator/(PMatrixJIT<T1,C> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpDivide>(l,r);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = l.elem(i,j) / r.elem();
    }
    j.end();
  }
  i.end();

  return d;
}



template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnAdjoint> {
  typedef C<typename UnaryReturn<T, FnAdjoint>::Type_t>  Type_t;
};

template<class T1, template<class> class C>
typename UnaryReturn<PMatrixJIT<T1,C>, FnAdjoint>::Type_t
adj(PMatrixJIT<T1,C> l)
{
  auto d = unary_return<FnAdjoint>(l);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = adj(l.elem(j,i));
    }
    j.end();
  }
  i.end();
  
  return d;
}



template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnConjugate> {
  typedef C<typename UnaryReturn<T, FnConjugate>::Type_t>  Type_t;
};

template<class T1, template<class> class C>
typename UnaryReturn<PMatrixJIT<T1,C>, FnConjugate>::Type_t
conj(PMatrixJIT<T1,C> l)
{
  auto d = unary_return<FnConjugate>(l);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = conj(l.elem(i,j));
    }
    j.end();
  }
  i.end();

  return d;
}



template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnTranspose> {
  typedef C<typename UnaryReturn<T, FnTranspose>::Type_t>  Type_t;
};

template<class T1, template<class> class C>
typename UnaryReturn<PMatrixJIT<T1,C>, FnTranspose>::Type_t
transpose(PMatrixJIT<T1,C> l)
{
  auto d = unary_return<FnTranspose>(l);
  
  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = transpose(l.elem(j,i));
    }
    j.end();
  }
  i.end();

  return d;
}




template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnTrace> {
  typedef PScalarJIT<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T, template<class> class C>
typename UnaryReturn<PMatrixJIT<T,C>, FnTrace>::Type_t
trace(PMatrixJIT<T,C> s1)
{
  auto d = unary_return<FnTrace>(s1);
  
  d.elem() = trace(s1.elem(0,0));
  JitForLoop i( 1 , s1.getN() );
  {
    d.elem() += trace(s1.elem(i,i));
  }
  i.end();
  
  return d;
}



template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnRealTrace> {
  typedef PScalarJIT<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T1, template<class> class C>
typename UnaryReturn<PMatrixJIT<T1,C>, FnRealTrace>::Type_t
realTrace(PMatrixJIT<T1,C> s1)
{
  auto d = unary_return<FnRealTrace>(s1);
  
  d.elem() = realTrace(s1.elem(0,0));
  JitForLoop i( 1 , s1.getN() );
  {
    d.elem() += realTrace(s1.elem(i,i));
  }
  i.end();

  return d;
}



template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnImagTrace> {
  typedef PScalarJIT<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T1, template<class> class C>
typename UnaryReturn<PMatrixJIT<T1,C>, FnImagTrace>::Type_t
imagTrace(PMatrixJIT<T1,C> s1)
{
  auto d = unary_return<FnImagTrace>(s1);
  
  d.elem() = imagTrace(s1.elem(0,0));
  JitForLoop i( 1 , s1.getN() );
  {
    d.elem() += imagTrace(s1.elem(i,i));
  }
  i.end();

  return d;
}



template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnTraceColor> {
  typedef C<typename UnaryReturn<T, FnTraceColor>::Type_t>  Type_t;
};

template<class T, template<class> class C>
typename UnaryReturn<PMatrixJIT<T,C>, FnTraceColor>::Type_t
traceColor(PMatrixJIT<T,C> s1)
{
  auto d = unary_return<FnTraceColor>(s1);
  
  JitForLoop i( 0 , s1.getN() );
  {
    JitForLoop j( 0 , s1.getN() );
    {
      d.elem(i,j) = traceColor(s1.elem(i,j));
    }
    j.end();
  }
  i.end();
  
  return d;
}



template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnTraceSpin> {
  typedef C<typename UnaryReturn<T, FnTraceSpin>::Type_t>  Type_t;
};

template<class T, template<class> class C>
typename UnaryReturn<PMatrixJIT<T,C>, FnTraceSpin>::Type_t
traceSpin(PMatrixJIT<T,C> s1)
{
  auto d = unary_return<FnTraceSpin>(s1);
  
  JitForLoop i( 0 , s1.getN() );
  {
    JitForLoop j( 0 , s1.getN() );
    {
      d.elem(i,j) = traceSpin(s1.elem(i,j));
    }
    j.end();
  }
  i.end();

  return d;
}




template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnTransposeColor> {
  typedef C<typename UnaryReturn<T, FnTransposeColor>::Type_t> Type_t;
};


template<class T, template<class> class C>
typename UnaryReturn<PMatrixJIT<T,C>, FnTransposeColor>::Type_t
transposeColor(PMatrixJIT<T,C> s1)
{
  auto d = unary_return<FnTransposeColor>(s1);
  
  JitForLoop i( 0 , s1.getN() );
  {
    JitForLoop j( 0 , s1.getN() );
    {
      d.elem(i,j) = transposeColor(s1.elem(i,j));
    }
    j.end();
  }
  i.end();

  return d;
}




template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnTransposeSpin> {
  typedef C<typename UnaryReturn<T, FnTransposeSpin>::Type_t> Type_t;
};


template<class T, template<class> class C>
typename UnaryReturn<PMatrixJIT<T,C>, FnTransposeSpin>::Type_t
transposeSpin(PMatrixJIT<T,C> s1)
{ 
  auto d = unary_return<FnTransposeSpin>(s1);
  
  JitForLoop i( 0 , s1.getN() );
  {
    JitForLoop j( 0 , s1.getN() );
    {
      d.elem(i,j) = transposeSpin(s1.elem(i,j));
    }
    j.end();
  }
  i.end();

  return d;
}



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, FnTraceMultiply>::Type_t
traceMultiply(PMatrixJIT<T1,C> l, PMatrixJIT<T2,C> r)
{
  assert(l.getN() == r.getN());

  auto d = binary_return<FnTraceMultiply>(l,r);

  d.elem() = traceMultiply(l.elem(0,0), r.elem(0,0));
  JitForLoop k( 1 , l.getN() );
  {
    d.elem() += traceMultiply(l.elem(0,k), r.elem(k,0));
  }
  k.end();
  
  JitForLoop j( 1 , l.getN() );
  {
    JitForLoop k( 0 , l.getN() );
    {
      d.elem() += traceMultiply(l.elem(j,k), r.elem(k,j));
    }
    k.end();
  }
  j.end();
  
  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, FnTraceMultiply>::Type_t
traceMultiply(PMatrixJIT<T1,C> l, PScalarJIT<T2> r)
{
  auto d = binary_return<FnTraceMultiply>(l,r);

  d.elem() = traceMultiply(l.elem(0,0), r.elem());
  JitForLoop k( 1 , l.getN() );
  {
    d.elem() += traceMultiply(l.elem(k,k), r.elem());
  }
  k.end();
  
  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, FnTraceMultiply> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnTraceMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, FnTraceMultiply>::Type_t
traceMultiply(PScalarJIT<T1> l, PMatrixJIT<T2,C> r)
{
  auto d = binary_return<FnTraceMultiply>(l,r);

  d.elem() = traceMultiply(l.elem(), r.elem(0,0));
  JitForLoop k( 1 , r.getN() );
  {
    d.elem() += traceMultiply(l.elem(), r.elem(k,k));
  }
  k.end();
  
  return d;
}




template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, FnTraceColorMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, FnTraceColorMultiply>::Type_t
traceColorMultiply(PMatrixJIT<T1,C> l, PMatrixJIT<T2,C> r)
{
  assert(l.getN() == r.getN());

  auto d = binary_return<FnTraceColorMultiply>(l,r);

  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = traceColorMultiply(l.elem(i,0), r.elem(0,j));
      JitForLoop k( 1 , l.getN() );
      {
	d.elem(i,j) += traceColorMultiply(l.elem(i,k), r.elem(k,j));
      }
      k.end();
    }
    j.end();
  }
  i.end();
  
  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, FnTraceColorMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, FnTraceColorMultiply>::Type_t
traceColorMultiply(PMatrixJIT<T1,C> l, PScalarJIT<T2> r)
{
  auto d = binary_return<FnTraceColorMultiply>(l,r);

  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = traceColorMultiply(l.elem(i,j), r.elem());
    }
    j.end();
  }
  i.end();

  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, FnTraceColorMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceColorMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, FnTraceColorMultiply>::Type_t
traceColorMultiply(PScalarJIT<T1> l, PMatrixJIT<T2,C> r)
{
  auto d = binary_return<FnTraceColorMultiply>(l,r);

  JitForLoop i( 0 , r.getN() );
  {
    JitForLoop j( 0 , r.getN() );
    {
      d.elem(i,j) = traceColorMultiply(l.elem(), r.elem(i,j));
    }
    j.end();
  }
  i.end();

  return d;
}



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, FnTraceSpinMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(PMatrixJIT<T1,C> l, PMatrixJIT<T2,C> r)
{
  assert(l.getN() == r.getN());

  auto d = binary_return<FnTraceSpinMultiply>(l,r);

  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = traceSpinMultiply(l.elem(i,0), r.elem(0,j));
      JitForLoop k( 1 , l.getN() );
      {
	d.elem(i,j) += traceSpinMultiply(l.elem(i,k), r.elem(k,j));
      }
      k.end();
    }
    j.end();
  }
  i.end();
  
  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, FnTraceSpinMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(PMatrixJIT<T1,C> l, PScalarJIT<T2> r)
{
  auto d = binary_return<FnTraceSpinMultiply>(l,r);

  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem(i,j) = traceSpinMultiply(l.elem(i,j), r.elem());
    }
    j.end();
  }
  i.end();
  
  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, FnTraceSpinMultiply> {
  typedef C<typename BinaryReturn<T1, T2, FnTraceSpinMultiply>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(PScalarJIT<T1> l, PMatrixJIT<T2,C> r)
{
  auto d = binary_return<FnTraceSpinMultiply>(l,r);

  JitForLoop i( 0 , r.getN() );
  {
    JitForLoop j( 0 , r.getN() );
    {
      d.elem(i,j) = traceSpinMultiply(l.elem(), r.elem(i,j));
    }
    j.end();
  }
  i.end();

  return d;
}



template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnReal> {
  typedef C<typename UnaryReturn<T, FnReal>::Type_t>  Type_t;
};

template<class T, template<class> class C>
typename UnaryReturn<PMatrixJIT<T,C>, FnReal>::Type_t
real(PMatrixJIT<T,C> s1)
{
  auto d = unary_return<FnReal>(s1);

  JitForLoop i( 0 , s1.getN() );
  {
    JitForLoop j( 0 , s1.getN() );
    {
      d.elem(i,j) = real(s1.elem(i,j));
    }
    j.end();
  }
  i.end();

  return d;
}



template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnImag> {
  typedef C<typename UnaryReturn<T, FnImag>::Type_t>  Type_t;
};

template<class T, template<class> class C>
typename UnaryReturn<PMatrixJIT<T,C>, FnImag>::Type_t
imag(PMatrixJIT<T,C> s1)
{
  auto d = unary_return<FnImag>(s1);

  JitForLoop i( 0 , s1.getN() );
  {
    JitForLoop j( 0 , s1.getN() );
    {
      d.elem(i,j) = imag(s1.elem(i,j));
    }
    j.end();
  }
  i.end();

  return d;
}



template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, FnCmplx>::Type_t
cmplx(PMatrixJIT<T1,C> s1, PMatrixJIT<T2,C> s2)
{
  auto d = binary_return<FnImag>(s1,s2);

  JitForLoop i( 0 , s1.getN() );
  {
    JitForLoop j( 0 , s1.getN() );
    {
      d.elem(i,j) = cmplx(s1.elem(i,j), s2.elem(i,j));
    }
    j.end();
  }
  i.end();

  return d;
}





template<class T1, template<class> class C>
struct UnaryReturn<PMatrixJIT<T1,C>, FnIsFinite> {
  typedef PScalarJIT< typename UnaryReturn<T1, FnIsFinite >::Type_t > Type_t;
};

template<class T1, template<class> class C>
typename UnaryReturn<PMatrixJIT<T1,C>, FnIsFinite>::Type_t
isfinite(PMatrixJIT<T1,C> l)
{
  auto d = unary_return<FnIsFinite>(l);

  d.elem() = isfinite(l.elem(0,0));

  JitForLoop i( 0 , l.getN() );
  {
    JitForLoop j( 0 , l.getN() );
    {
      d.elem() &= isfinite(l.elem(i,j));
    }
    j.end();
  }
  i.end();

  return d;
}





template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnTimesI> {
  typedef C<typename UnaryReturn<T, FnTimesI>::Type_t>  Type_t;
};

template<class T, template<class> class C>
typename UnaryReturn<PMatrixJIT<T,C>, FnTimesI>::Type_t
timesI(PMatrixJIT<T,C> s1)
{
  auto d = unary_return<FnTimesI>(s1);

  JitForLoop i( 0 , s1.getN() );
  {
    JitForLoop j( 0 , s1.getN() );
    {
      d.elem(i,j) = timesI(s1.elem(i,j));
    }
    j.end();
  }
  i.end();

  return d;
}


template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnTimesMinusI> {
  typedef C<typename UnaryReturn<T, FnTimesMinusI>::Type_t>  Type_t;
};

template<class T, template<class> class C>
typename UnaryReturn<PMatrixJIT<T,C>, FnTimesMinusI>::Type_t
timesMinusI(PMatrixJIT<T,C> s1)
{
  auto d = unary_return<FnTimesMinusI>(s1);

  JitForLoop i( 0 , s1.getN() );
  {
    JitForLoop j( 0 , s1.getN() );
    {
      d.elem(i,j) = timesMinusI(s1.elem(i,j));
    }
    j.end();
  }
  i.end();

  return d;
}



template<class T, template<class> class C> 
void zero_rep(PMatrixJIT<T,C> dest)
{
  JitForLoop i( 0 , dest.getN() );
  {
    JitForLoop j( 0 , dest.getN() );
    {
      zero_rep(dest.elem(i,j));
    }
    j.end();
  }
  i.end();
}



template<class T, template<class> class C, class T1, class T2, class T3>
void
fill_random_jit(PMatrixJIT<T,C> d, T1 seed, T2 skewed_seed, const T3& seed_mult)
{
  JitForLoop i( 0 , d.getN() );
  {
    JitForLoop j( 0 , d.getN() );
    {
      fill_random_jit(d.elem(i,j), seed, skewed_seed, seed_mult);
    }
    j.end();
  }
  i.end();
}



template<class T, template<class> class C>
void
fill_gaussian_jit(PMatrixJIT<T,C> d, PMatrixJIT<T,C> r1, PMatrixJIT<T,C> r2)
{
  JitForLoop i( 0 , d.getN() );
  {
    JitForLoop j( 0 , d.getN() );
    {
      fill_gaussian_jit(d.elem(i,j), r1.elem(i,j), r2.elem(i,j));
    }
    j.end();
  }
  i.end();
}




template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnNorm2> {
  typedef PScalarJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, template<class> class C>
struct UnaryReturn<PMatrixJIT<T,C>, FnLocalNorm2> {
  typedef PScalarJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T, template<class> class C>
typename UnaryReturn<PMatrixJIT<T,C>, FnLocalNorm2>::Type_t
localNorm2(PMatrixJIT<T,C> s1)
{
  auto d = unary_return<FnLocalNorm2>(s1);

  d.elem() = localNorm2(s1.elem(0,0));
  JitForLoop j( 1 , s1.getN() );
  {
    d.elem() += localNorm2(s1.elem(0,j));
  }
  j.end();
  
  JitForLoop i( 1 , s1.getN() );
  {
    JitForLoop j( 0 , s1.getN() );
    {
      d.elem() += localNorm2(s1.elem(i,j));
    }
    j.end();
  }
  i.end();
  
  return d;
}



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, FnInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, FnLocalInnerProduct>::Type_t
localInnerProduct(PMatrixJIT<T1,C> s1, PMatrixJIT<T2,C> s2)
{
  assert(s1.getN() == s2.getN());

  auto d = binary_return<FnLocalInnerProduct>(s1,s2);

  d.elem() = localInnerProduct(s1.elem(0,0), s2.elem(0,0));
  JitForLoop k( 1 , s1.getN() );
  {
    d.elem() += localInnerProduct(s1.elem(k,0), s2.elem(k,0));
  }
  k.end();
  
  JitForLoop j( 1 , s1.getN() );
  {
    JitForLoop k( 0 , s1.getN() );
    {
      d.elem() += localInnerProduct(s1.elem(k,j), s2.elem(k,j));
    }
    k.end();
  }
  j.end();
  
  return d;
}



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(PMatrixJIT<T1,C> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnLocalInnerProduct>(s1,s2);

  d.elem() = localInnerProduct(s1.elem(0,0), s2.elem());
  JitForLoop k( 1 , s1.getN() );
  {
    d.elem() += localInnerProduct(s1.elem(k,k), s2.elem());
  }
  k.end();

  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, FnLocalInnerProduct> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};



template<class T1, class T2, template<class> class C>
typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, FnLocalInnerProduct>::Type_t
localInnerProduct(PScalarJIT<T1> s1, PMatrixJIT<T2,C> s2)
{
  auto d = binary_return<FnLocalInnerProduct>(s1,s2);

  d.elem() = localInnerProduct(s1.elem(), s2.elem(0,0));
  JitForLoop k( 1 , s2.getN() );
  {
    d.elem() += localInnerProduct(s1.elem(), s2.elem(k,k));
  }
  k.end();

  return d;
}



template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, FnInnerProductReal > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, FnLocalInnerProductReal > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PMatrixJIT<T2,C>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(PMatrixJIT<T1,C> s1, PMatrixJIT<T2,C> s2)
{
  auto d = binary_return<FnLocalInnerProductReal>(s1,s2);

  d.elem() = localInnerProductReal(s1.elem(0,0), s2.elem(0,0));
  JitForLoop k( 1 , s1.getN() );
  {
    d.elem() += localInnerProductReal(s1.elem(k,0), s2.elem(k,0));
  }
  k.end();
  
  JitForLoop j( 1 , s1.getN() );
  {
    JitForLoop k( 0 , s1.getN() );
    {
      d.elem() += localInnerProductReal(s1.elem(k,j), s2.elem(k,j));
    }
    k.end();
  }
  j.end();
  
  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, FnLocalInnerProductReal > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
typename BinaryReturn<PMatrixJIT<T1,C>, PScalarJIT<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(PMatrixJIT<T1,C> s1, PScalarJIT<T2> s2)
{
  auto d = binary_return<FnLocalInnerProductReal>(s1,s2);

  d.elem() = localInnerProductReal(s1.elem(0,0), s2.elem());
  JitForLoop k( 1 , s1.getN() );
  {
    d.elem() += localInnerProductReal(s1.elem(k,0), s2.elem());
  }
  k.end();

  return d;
}


template<class T1, class T2, template<class> class C>
struct BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, FnLocalInnerProductReal > {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};


template<class T1, class T2, template<class> class C>
typename BinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(PScalarJIT<T1> s1, PMatrixJIT<T2,C> s2)
{
  auto d = binary_return<FnLocalInnerProductReal>(s1,s2);

  d.elem() = localInnerProductReal(s1.elem(), s2.elem(0,0));
  JitForLoop k( 1 , s2.getN() );
  {
    d.elem() += localInnerProductReal(s1.elem(), s2.elem(k,k));
  }
  k.end();

  return d;
}



template<class T1, class T2, class T3, template<class> class C>
struct TrinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, PMatrixJIT<T3,C>, FnWhere> {
  typedef C<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3, template<class> class C>
typename TrinaryReturn<PScalarJIT<T1>, PMatrixJIT<T2,C>, PMatrixJIT<T3,C>, FnWhere>::Type_t
where(PScalarJIT<T1> a, PMatrixJIT<T2,C> b, PMatrixJIT<T3,C> c)
{
  assert(b.getN() == c.getN());

  auto d = trinary_return<FnWhere>(a,b,c);

  JitForLoop i( 0 , b.getN() );
  {
    JitForLoop j( 0 , b.getN() );
    {
      d.elem(i,j) = where(a.elem(), b.elem(i,j), c.elem(i,j));
    }
    j.end();
  }
  i.end();
  
  return d;
}



} // namespace QDP

#endif
