#ifndef QDP_QDPTYPE_H
#define QDP_QDPTYPE_H


namespace QDP {


template<class T, class C> 
class QDPType
{
public:
  typedef T Subtype_t;
  typedef C Container_t;

  void print() const {}
  
  Shape shape() const { return static_cast<const C*>(this)->shape(); }

  
  JitDataFormat data_format() const { return static_cast<const C*>(this)->data_format(); }

  
  QDPType()  {}

  QDPType(const QDPType& rhs) = delete;


  C& assign(const typename WordType<T>::Type_t& rhs)
    {
      C* me = static_cast<C*>(this);
      typedef typename SimpleScalar<typename WordType<T>::Type_t>::Type_t  Scalar_t;
      evaluate(*me,OpAssign(),PETE_identity(Scalar_t(rhs)),all);
      return *me;
    }


  C& assign(const Zero&)
    {
      C* me = static_cast<C*>(this);
      zero_rep(*me);
      return *me;
    }

  
  template<class T1,class C1>
  C& assign(const QDPType<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);

      evaluate(*me,OpAssign(),PETE_identity(rhs),all);
      return *me;
    }

  template<class T1,class C1>
  C& assign(const QDPExpr<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);

      evaluate(*me,OpAssign(),rhs,all);
      return *me;
    }


  C& operator=(const QDPType& rhs)
    {
      return assign(rhs);
    }


  
  template<class Op, class RHS>
  C& compound_assign(Op op, const RHS& rhs)
  {
    C& self = static_cast<C&>(*this);
    self.check_shape(rhs);
    evaluate(self, op, rhs, all);
    return self;
  }


  
  C& operator+=(const typename WordType<T>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<T>::Type_t>::Type_t  Scalar_t;
      return compound_assign(OpAddAssign{}, PETE_identity(Scalar_t(rhs)));
    }


  template<class T1,class C1>
  C& operator+=(const QDPType<T1,C1>& rhs)
    {
      return compound_assign(OpAddAssign{}, PETE_identity(rhs));
    }


  template<class T1,class C1>
  C& operator+=(const QDPExpr<T1,C1>& rhs)
    {
      return compound_assign(OpAddAssign{}, rhs);
    }

  
  C& operator-=(const typename WordType<T>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<T>::Type_t>::Type_t  Scalar_t;
      return compound_assign(OpSubtractAssign{}, PETE_identity(Scalar_t(rhs)));
    }


  template<class T1,class C1>
  C& operator-=(const QDPType<T1,C1>& rhs)
    {
      return compound_assign(OpSubtractAssign{}, PETE_identity(rhs));
    }


  template<class T1,class C1>
  C& operator-=(const QDPExpr<T1,C1>& rhs)
    {
      return compound_assign(OpSubtractAssign{}, rhs);
    }


  C& operator*=(const typename WordType<T>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<T>::Type_t>::Type_t  Scalar_t;
      return compound_assign(OpMultiplyAssign{}, PETE_identity(Scalar_t(rhs)));
    }


  template<class T1,class C1>
  C& operator*=(const QDPType<T1,C1>& rhs)
    {
      return compound_assign(OpMultiplyAssign{}, PETE_identity(rhs));
    }


  template<class T1,class C1>
  C& operator*=(const QDPExpr<T1,C1>& rhs)
    {
      return compound_assign(OpMultiplyAssign{}, rhs);
    }


  
  C& operator/=(const typename WordType<T>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<T>::Type_t>::Type_t  Scalar_t;
      return compound_assign(OpDivideAssign{}, PETE_identity(Scalar_t(rhs)));
    }


  template<class T1,class C1>
  C& operator/=(const QDPType<T1,C1>& rhs)
    {
      return compound_assign(OpDivideAssign{}, PETE_identity(rhs));
    }


  template<class T1,class C1>
  C& operator/=(const QDPExpr<T1,C1>& rhs)
    {
      return compound_assign(OpDivideAssign{}, rhs);
    }


  C& operator%=(const typename WordType<T>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<T>::Type_t>::Type_t  Scalar_t;
      return compound_assign(OpModAssign{}, PETE_identity(Scalar_t(rhs)));
    }


  template<class T1,class C1>
  C& operator%=(const QDPType<T1,C1>& rhs)
    {
      return compound_assign(OpModAssign{}, PETE_identity(rhs));
    }


  template<class T1,class C1>
  C& operator%=(const QDPExpr<T1,C1>& rhs)
    {
      return compound_assign(OpModAssign{}, rhs);
    }


  C& operator|=(const typename WordType<T>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<T>::Type_t>::Type_t  Scalar_t;
      return compound_assign(OpBitwiseOrAssign{}, PETE_identity(Scalar_t(rhs)));
    }


  template<class T1,class C1>
  C& operator|=(const QDPType<T1,C1>& rhs)
    {
      return compound_assign(OpBitwiseOrAssign{}, PETE_identity(rhs));
    }


  template<class T1,class C1>
  C& operator|=(const QDPExpr<T1,C1>& rhs)
    {
      return compound_assign(OpBitwiseOrAssign{}, rhs);
    }

  
  C& operator&=(const typename WordType<T>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<T>::Type_t>::Type_t  Scalar_t;
      return compound_assign(OpBitwiseAndAssign{}, PETE_identity(Scalar_t(rhs)));
    }


  template<class T1,class C1>
  C& operator&=(const QDPType<T1,C1>& rhs)
    {
      return compound_assign(OpBitwiseAndAssign{}, PETE_identity(rhs));
    }


  template<class T1,class C1>
  C& operator&=(const QDPExpr<T1,C1>& rhs)
    {
      return compound_assign(OpBitwiseAndAssign{}, rhs);
    }

  
  C& operator^=(const typename WordType<T>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<T>::Type_t>::Type_t  Scalar_t;
      return compound_assign(OpBitwiseXorAssign{}, PETE_identity(Scalar_t(rhs)));
    }


  template<class T1,class C1>
  C& operator^=(const QDPType<T1,C1>& rhs)
    {
      return compound_assign(OpBitwiseXorAssign{}, PETE_identity(rhs));
    }


  template<class T1,class C1>
  C& operator^=(const QDPExpr<T1,C1>& rhs)
    {
      return compound_assign(OpBitwiseXorAssign{}, rhs);
    }


  C& operator<<=(const typename WordType<T>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<T>::Type_t>::Type_t  Scalar_t;
      return compound_assign(OpLeftShiftAssign{}, PETE_identity(Scalar_t(rhs)));
    }


  template<class T1,class C1>
  C& operator<<=(const QDPType<T1,C1>& rhs)
    {
      return compound_assign(OpLeftShiftAssign{}, PETE_identity(rhs));
    }


  template<class T1,class C1>
  C& operator<<=(const QDPExpr<T1,C1>& rhs)
    {
      return compound_assign(OpLeftShiftAssign{}, rhs);
    }


  C& operator>>=(const typename WordType<T>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<T>::Type_t>::Type_t  Scalar_t;
      return compound_assign(OpRightShiftAssign{}, PETE_identity(Scalar_t(rhs)));
    }


  template<class T1,class C1>
  C& operator>>=(const QDPType<T1,C1>& rhs)
    {
      return compound_assign(OpRightShiftAssign{}, PETE_identity(rhs));
    }


  template<class T1,class C1>
  C& operator>>=(const QDPExpr<T1,C1>& rhs)
    {
      return compound_assign(OpRightShiftAssign{}, rhs);
    }


  LatticeLayout layout() const { return static_cast<const C*>(this)->layout(); }
  

  T elem(int i) const {return static_cast<const C*>(this)->elem(i);}
  T elem()      const {return static_cast<const C*>(this)->elem();}
};


  template<class T>
  int jit_words()
  {
    T tmp(nullptr);
    return tmp.words();
  }







template<class T1, class C1, class Op>
struct UnaryReturn<QDPType<T1,C1>, Op> {
  typedef QDPType<typename UnaryReturn<T1, Op>::Type_t,
		  typename UnaryReturn<C1, Op>::Type_t>  Type_t;
};


template<class T1, class C1, class T2, class C2, class Op>
struct BinaryReturn<QDPType<T1,C1>, QDPType<T2,C2>, Op> {
  typedef QDPType<typename BinaryReturn<T1, T2, Op>::Type_t,
		  typename BinaryReturn<C1, C2, Op>::Type_t>  Type_t;
};


template<class T1, class C1, class T2, class C2, class T3, class C3, class Op>
struct TrinaryReturn<QDPType<T1,C1>, QDPType<T2,C2>, QDPType<T3,C3>, Op> {
  typedef QDPType<typename BinaryReturn<T2, T3, Op>::Type_t,
		  typename BinaryReturn<C2, C3, Op>::Type_t>  Type_t;
};



template<class T, class C>
struct CreateLeaf<QDPType<T,C> >
{
  typedef QDPType<T,C> Inp_t;
  typedef Reference<Inp_t> Leaf_t;
  //typedef Inp_t Leaf_t;
   static
  Leaf_t make(const Inp_t &a)
  {
    return Leaf_t(a);
  }
};




template<class T, class C>
struct LeafFunctor<QDPType<T,C>, ElemLeaf>
{
  //  typedef Reference<T> Type_t;
  typedef T Type_t;
   static Type_t apply(const QDPType<T,C> &a, ElemLeaf &f)
    { 
      return Type_t(a.elem());
    }
};

template<class T, class C>
struct LeafFunctor<QDPType<T,C>, EvalLeaf1>
{
  //  typedef Reference<T> Type_t;
  typedef T Type_t;
   static Type_t apply(const QDPType<T,C> &a, EvalLeaf1 &f)
    { 
      return Type_t(a.elem(f.val1()));
    }
};


template<class T>
struct LeafFunctor<QDPType<T,OLattice<T> >, MGParamLeaf>
{
  typedef typename JITType< OLattice<T> >::Type_t  Type_t;
  static Type_t apply(const QDPType<T,OLattice<T> > &a, MGParamLeaf& p)
  {
    // 1. base addr
    // 2. local lattice size (optional)
    // 3. translation table (optional)
    const PlanEntry& e = p.next( PlanEntry::Kind::Lattice );
    ParamRef    base_addr = p.lattice_base< typename WordType<T>::Type_t >();
    return Type_t( base_addr , p.getLatSizeParamRef( e.layout_slot ) , e.shape , e.layout_slot , e.data_format );
  }
};


template<class T>
struct LeafFunctor<QDPType<T,OScalar<T> >, MGParamLeaf>
{
  typedef typename JITType< OScalar<T> >::Type_t  Type_t;
   static Type_t apply(const QDPType<T,OScalar<T> > &a, MGParamLeaf& p)
  {
    const PlanEntry& e = p.next( PlanEntry::Kind::Scalar );
    return Type_t( llvm_add_param< typename WordType<T>::Type_t * >() , e.shape );
  }
};


template<class T>
struct LeafFunctor<QDPType<PScalar< PScalar < RScalar< Word< T > > > >,OScalar<PScalar< PScalar < RScalar< Word< T > > > > > >, MGParamLeaf>
{
  typedef typename JITType< OScalar<PScalar< PScalar < RScalar< Word< T > > > > > >::Type_t  Type_t;
   static Type_t apply(const QDPType<PScalar< PScalar < RScalar< Word< T > > > >,OScalar<PScalar< PScalar < RScalar< Word< T > > > > > > &a, MGParamLeaf& p)
  {
    p.next( PlanEntry::Kind::Literal );
    return Type_t( llvm_add_param< T >() , true );
  }
};




template<class T>
struct LeafFunctor<QDPType<T,OLattice<T> >, DummyJitLeaf>
{
  //typedef typename JITType< OLattice<T> >::Type_t  Type_t;
  typedef typename JITType< T >::Type_t  Type_t;
  static Type_t apply(const QDPType<T,OLattice<T> > &a, DummyJitLeaf& p)
  {
    return Type_t( a.shape() );
  }
};


template<class T>
struct LeafFunctor<QDPType<T,OScalar<T> >, DummyJitLeaf>
{
  //typedef typename JITType< OScalar<T> >::Type_t  Type_t;
  typedef typename JITType< T >::Type_t  Type_t;
  static Type_t apply(const QDPType<T,OScalar<T> > &a, DummyJitLeaf& p)
  {
    return Type_t( a.shape() );
  }
};


template<class T, class C>
struct LeafFunctor<QDPType<T,C>, ShiftPhase1>
{
  typedef int Type_t;
  static int apply(const QDPType<T,C> &s, ShiftPhase1 &f) {
    return LeafFunctor<C, ShiftPhase1>::apply(static_cast<const C&>(s), f);
  }
};



template<class T, class C>
struct LeafFunctor<QDPType<T,C>, ShiftPhase2>
{
  typedef int Type_t;
  static int apply(const QDPType<T,C> &s, ShiftPhase2 &f) {
    return 0;
  }
};




template<class T>
struct LeafFunctor<QDPType<T,OLattice<T> >, SelfAssignTag>
{
  typedef int Type_t;
   static
  Type_t apply(const QDPType<T,OLattice<T> >& s, SelfAssignTag& p) 
  {
    return ( p.shift_depth > 0 && static_cast<const OLattice<T>&>(s).getId() == p.id ) ? 1:0;
  }
};



template<class T>
struct LeafFunctor<QDPType<T,OLattice<T> >, CollectLayouts>
{
  typedef int Type_t;
   static
  Type_t apply(const QDPType<T,OLattice<T> >& s, CollectLayouts& p) 
  {
    p.layouts.push_back( s.layout().index() );
    return 0;
  }
};


} // namespace QDP

#endif

