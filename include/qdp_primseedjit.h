#ifndef QDP_PRIMSEEDJIT_H
#define QDP_PRIMSEEDJIT_H


namespace QDP {



template <class T> class PSeedJIT : public BaseJIT< T >
{
public:
  typedef T SubType_t;

  int getN() const 
  {
    return 4;
  }

  int this_size() const { return getN(); }

  PSeedJIT() {}

  PSeedJIT( llvm::Value * base , JitDataFormat fmt , NestedIndex args , Shape shape ) : BaseJIT< T >(base,fmt,args,shape) {}

  // for eval Shape
  PSeedJIT( Shape shape ) : BaseJIT< T >(shape) {}
  

  template<class T1>
  PSeedJIT assign(const PScalarJIT<T1>& rhs) 
    {
      typedef typename InternalScalar<T1>::Type_t  S;

      auto s4095 = stack_alloc_jit<S>(); 
      auto s12 = stack_alloc_jit<S>(); 
      auto s24 = stack_alloc_jit<S>(); 

      s4095 = 4095;
      s12 = 12;
      s24 = 24;
      
      elem(0) = rhs.elem() & s4095;
      elem(1) = (rhs.elem() >> s12) & s4095;
      elem(2) = (rhs.elem() >> s24) & s4095;
//      elem(3) = (rhs.elem() >> S(36)) & S(2047);  // This probably will never be nonzero
      zero_rep(elem(3));    // assumes 32 bit integers

      return *this;
    }


  template<class T1>
  PSeedJIT operator=(PScalarJIT<T1> rhs) 
    {
      return assign(rhs);
    }


  template<class T1>
  PSeedJIT operator=(PSeedJIT<T1> rhs) 
    {
      JitForLoop i( 0 , 4 );
      {
	elem(i) = rhs.elem(i);
      }
      i.end();
      return *this;
    }


  PSeedJIT operator=(PSeedJIT rhs) 
    {
      JitForLoop i( 0 , 4 );
      {
	elem(i) = rhs.elem(i);
      }
      i.end();

      return *this;
    }

  

public:
  T elem(int i) const { return BaseJIT< T >::elem(i); }
  T elem(const JitForLoop& loop) const { return BaseJIT< T >::elem(loop); }

};





template<class T>
class WordSize< PSeedJIT<T> >
{
public:
  static int value( const Shape& ret )
  {
    return 4 * WordSize<T>::value( ret );
  }
};

  

  
template<class T> 
struct JITType< PSeedJIT<T> >
{
  typedef PSeedJIT<typename JITType<T>::Type_t>  Type_t;
};



template<class T1>
struct WordType<PSeedJIT<T1> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};



template<class T1>
struct SinglePrecType<PSeedJIT<T1> >
{
  typedef PSeedJIT< typename SinglePrecType<T1>::Type_t > Type_t;
};

template<class T1>
struct DoublePrecType<PSeedJIT<T1> >
{
  typedef PSeedJIT< typename DoublePrecType<T1>::Type_t > Type_t;
};



template<class T>
struct InternalScalar<PSeedJIT<T> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};


template<class T>
struct PrimitiveScalar<PSeedJIT<T> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};


template<class T>
struct LatticeScalar<PSeedJIT<T> > {
  typedef PSeedJIT<typename LatticeScalar<T>::Type_t>  Type_t;
};



template<class T1, class T2, class Op>
struct BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, Op> {
  typedef PSeedJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


  


template<class T1, class T2>
struct BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpEQ> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2>
typename BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpEQ>::Type_t
operator==(PSeedJIT<T1> l, PSeedJIT<T2> r)
{
  auto d = binary_return<OpEQ>(l,r);
  
  d.elem() =  
    (l.elem(0) == r.elem(0)) && 
    (l.elem(1) == r.elem(1)) && 
    (l.elem(2) == r.elem(2)) && 
    (l.elem(3) == r.elem(3));

  return d;
}



template<class T1, class T2>
struct BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpNE> {
  typedef PScalarJIT<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

  
template<class T1, class T2>
typename BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpNE>::Type_t
operator!=(PSeedJIT<T1> l, PSeedJIT<T2> r)
{
  auto d = binary_return<OpNE>(l,r);
  
  d.elem() =  
    (l.elem(0) != r.elem(0)) ||
    (l.elem(1) != r.elem(1)) || 
    (l.elem(2) != r.elem(2)) || 
    (l.elem(3) != r.elem(3));

  return d;
}



//! PSeedJIT<T> = PSeedJIT<T> * PSeedJIT<T>
/*!
 * A 47 bit seed multiplication is represented as the multiplication
 * of three 12bit and one 11bit integer
 *
 * i3 = s1(3)*s2(0) + s1(2)*s2(1)
 *    + s1(1)*s2(2) + s1(0)*s2(3);
 * i2 = s1(2)*s2(0) + s1(1)*s2(1)
 *    + s1(0)*s2(2);
 * i1 = s1(1)*s2(0) + s1(0)*s2(1);
 * i0 = s1(0)*s2(0);
 *
 * dest(0) = mod(i0, 4096);
 * i1      = i1 + i0/4096;
 * dest(1) = mod(i1, 4096);
 * i2      = i2 + i1/4096;
 * dest(2) = mod(i2, 4096);
 * i3      = i3 + i2/4096
 * dest(3) = mod(i3, 2048);
 */

// template<class T1, class T2>
// struct BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpMultiply> {
//   typedef PSeedJIT<typename BinaryReturn<T1, T2, OpMultiply>::Type_t>  Type_t;
// };

template<class T1, class T2>
typename BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpMultiply>::Type_t
operator*(PSeedJIT<T1> s1, PSeedJIT<T2> s2)
{
  typedef typename BinaryReturn<T1, T2, OpMultiply>::Type_t  T;
  typedef typename InternalScalar<T>::Type_t  S;

  auto d = binary_return<OpMultiply>(s1,s2);

  auto i0 = stack_alloc_jit<T>(); 
  auto i1 = stack_alloc_jit<T>(); 
  auto i2 = stack_alloc_jit<T>(); 
  auto i3 = stack_alloc_jit<T>(); 
  
  auto s4095 = stack_alloc_jit<S>(); 
  auto s2047 = stack_alloc_jit<S>(); 
  auto s12 = stack_alloc_jit<S>(); 
  auto s24 = stack_alloc_jit<S>(); 
  
  s4095 = 4095;
  s2047 = 2047;
  s12 = 12;
  s24 = 24;

  
  /* i3 = s1(3)*s2(0) + s1(2)*s2(1) + s1(1)*s2(2) + s1(0)*s2(3) */
  i3  = s1.elem(3) * s2.elem(0);
  i3 += s1.elem(2) * s2.elem(1);
  i3 += s1.elem(1) * s2.elem(2);
  i3 += s1.elem(0) * s2.elem(3);

  /* i2 = s1(2)*s2(0) + s1(1)*s2(1) + s1(0)*s2(2) */
  i2  = s1.elem(2) * s2.elem(0);
  i2 += s1.elem(1) * s2.elem(1);
  i2 += s1.elem(0) * s2.elem(2);

  /* i1 = s1(1)*s2(0) + s1(0)*s2(1) */
  i1  = s1.elem(1) * s2.elem(0);
  i1 += s1.elem(0) * s2.elem(1);

  /* i0 = s1(0)*s2(0) */
  i0 = s1.elem(0) * s2.elem(0);

  /* dest(0) = mod(i0, 4096) */
  d.elem(0) = i0 & s4095;

  /* i1 = i1 + i0/4096 */
  i1 += i0 >> s12;

  /* dest(1) = mod(i1, 4096) */
  d.elem(1) = i1 & s4095;
  
  /* i2 = i2 + i1/4096 */
  i2 += i1 >> s12;

  /* dest(2) = mod(i2, 4096) */
  d.elem(2) = i2 & s4095;

  /* i3 = i3 + i2/4096 */
  i3 += i2 >> s12;

  /* dest(3) = mod(i3, 2048) */
  d.elem(3) = i3 & s2047;
  
  return d;
}


template<class T1, class T2>
struct BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpBitwiseOr> {
  typedef PSeedJIT<typename BinaryReturn<T1, T2, OpBitwiseOr>::Type_t>  Type_t;
};

  
template<class T1, class T2>
typename BinaryReturn<PSeedJIT<T1>, PSeedJIT<T2>, OpBitwiseOr>::Type_t
operator|(PSeedJIT<T1> l, PSeedJIT<T2> r)
{
  auto d = binary_return<OpBitwiseOr>(l,r);

  d.elem(0) = l.elem(0) | r.elem(0);
  d.elem(1) = l.elem(1) | r.elem(1);
  d.elem(2) = l.elem(2) | r.elem(2);
  d.elem(3) = l.elem(3) | r.elem(3);

  return d;
}




template<class T1, class T2>
struct BinaryReturn<PSeedJIT<T1>, PScalarJIT<T2>, OpBitwiseOr> {
  typedef PSeedJIT<typename BinaryReturn<T1, T2, OpBitwiseOr>::Type_t>  Type_t;
};


template<class T1, class T2>
typename BinaryReturn<PSeedJIT<T1>, PScalarJIT<T2>, OpBitwiseOr>::Type_t
operator|(PSeedJIT<T1> l, PScalarJIT<T2> r)
{
  auto d = binary_return<OpBitwiseOr>(l,r);
  auto rs = binary_return<OpBitwiseOr>(l,r);

  rs = r;
  d = l | rs;
  
  return d;
}




template<class T1, class T2>
struct BinaryReturn<PSeedJIT<T1>, PScalarJIT<T2>, OpLeftShift> {
  typedef PSeedJIT<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};


template<class T1, class T2>
typename BinaryReturn<PSeedJIT<T1>, PScalarJIT<T2>, OpLeftShift>::Type_t
operator<<(PSeedJIT<T1> s1, PScalarJIT<T2> s2)
{
  typedef typename BinaryReturn<T1, T2, OpLeftShift>::Type_t  T;
  typedef typename InternalScalar<T>::Type_t  S;

  auto d = binary_return<OpLeftShift>(s1,s2);

  auto s12   = stack_alloc_jit<S>();
  auto s4095 = stack_alloc_jit<S>();
  auto s2047 = stack_alloc_jit<S>();
  s12   = 12;
  s4095 = 4095;
  s2047 = 2047;
  
  auto i0 = stack_alloc_jit<T>();
  auto i1 = stack_alloc_jit<T>();
  auto i2 = stack_alloc_jit<T>();
  auto i3 = stack_alloc_jit<T>();

  
  i0 = s1.elem(0) << s2.elem();
  i1 = s1.elem(1) << s2.elem();
  i2 = s1.elem(2) << s2.elem();
  i3 = s1.elem(3) << s2.elem();

  d.elem(0) = i0 & s4095;
  i0 >>= s12;
  i1 |= i0 & s4095;
  d.elem(1) = i1 & s4095;
  i1 >>= s12;
  i2 |= i1 & s4095;
  d.elem(2) = i2 & s4095;
  i2 >>= s12;
  i3 |= i2 & s4095;
  d.elem(3) = i3 & s2047;

  return d;
}



template<class T>
struct UnaryReturn<PSeedJIT<T>, FnSeedToFloat> {
  typedef PScalarJIT<typename UnaryReturn<T, FnSeedToFloat>::Type_t>  Type_t;
};



template<class T>
typename UnaryReturn<PSeedJIT<T>, FnSeedToFloat>::Type_t
seedToFloat( PSeedJIT<T> s1)
{
  auto d = unary_return<FnSeedToFloat>(s1);
  
  llvm::Value *twom11 = llvm_create_value(1.0 / 2048.0);
  llvm::Value *twom12 = llvm_create_value(1.0 / 4096.0);
  llvm::Value *fs1;
  llvm::Value *fs2;
  llvm::Value *dval;

  fs1 = s1.elem(0).elem().get_val();
  dval = llvm_mul( twom12 , s1.elem(0).elem().get_val() );

  fs1 = s1.elem(1).elem().get_val();
  fs2 = llvm_add( fs1 , dval );
  dval = llvm_mul( twom12 , fs2 );

  fs1 = s1.elem(2).elem().get_val();
  fs2 = llvm_add( fs1 , dval );
  dval = llvm_mul( twom12 , fs2 );

  fs1 = s1.elem(3).elem().get_val();
  fs2 = llvm_add( fs1 , dval );
  dval = llvm_mul( twom11 , fs2 );

  d.elem().elem().store( dval );

  return d;
}




template<class T> 
void zero_rep(PSeedJIT<T>& dest) 
{
  JitForLoop i( 0 , 4 );
  {
    zero_rep(dest.elem(i));
  }
  i.end();
}


} // namespace QDP

#endif
