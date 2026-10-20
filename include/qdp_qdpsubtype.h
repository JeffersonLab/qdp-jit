#ifndef QDP_QDPSUBTYPE_H
#define QDP_QDPSUBTYPE_H


namespace QDP {

template<class T, class C> 
class QDPSubType
{
  typedef typename QDPSubTypeTrait<C>::Type_t CC;

public:
  typedef T Subtype_t;

  typedef C Container_t;

  QDPSubType() {}

  QDPSubType(const QDPSubType&) {}

  ~QDPSubType() {}

  
  C& me()
  {
    return static_cast<CC*>(this)->me();
  }


  void assign(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
    evaluate(me(),OpAssign(),PETE_identity(Scalar_t(rhs)),subset());
  }

  

  void assign(const Zero&)
  {
    zero_rep(me(),subset());
  }

  
  
  template<class T1,class C1>
  void assign(const QDPType<T1,C1>& rhs)
  {
    evaluate(me(),OpAssign(),PETE_identity(rhs),subset());
  }


  template<class T1>
  void assign(const OSubLattice<T1>& rhs)
  {
    QDP_error_exit("assignment of two view subtypes is not supported");
  }


  template<class T1,class C1>
  void assign(const QDPExpr<T1,C1>& rhs)
  {
    evaluate( this->me() , OpAssign() , rhs , subset() );
  }


  template<class Op, class RHS>
  C& compound_assign(Op op, const RHS& rhs)
  {
    C& self = me();
    self.check_shape(rhs);
    evaluate(self, op, rhs, subset());
    return me();
  }


  void operator+=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t Scalar_t;
    compound_assign(OpAddAssign(), PETE_identity(Scalar_t(rhs)));
  }


  template<class T1, class C1>
  void operator+=(const QDPType<T1,C1>& rhs)
  {
    compound_assign(OpAddAssign(), PETE_identity(rhs));
  }


  template<class T1, class C1>
  void operator+=(const QDPExpr<T1,C1>& rhs)
  {
    compound_assign(OpAddAssign(), rhs);
  }


  void operator-=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t Scalar_t;
    compound_assign(OpSubtractAssign(), PETE_identity(Scalar_t(rhs)));
  }


  template<class T1, class C1>
  void operator-=(const QDPType<T1,C1>& rhs)
  {
    compound_assign(OpSubtractAssign(), PETE_identity(rhs));
  }


  template<class T1, class C1>
  void operator-=(const QDPExpr<T1,C1>& rhs)
  {
    compound_assign(OpSubtractAssign(), rhs);
  }


  void operator*=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t Scalar_t;
    compound_assign(OpMultiplyAssign(), PETE_identity(Scalar_t(rhs)));
  }


  template<class T1, class C1>
  void operator*=(const QDPType<T1,C1>& rhs)
  {
    compound_assign(OpMultiplyAssign(), PETE_identity(rhs));
  }


  template<class T1, class C1>
  void operator*=(const QDPExpr<T1,C1>& rhs)
  {
    compound_assign(OpMultiplyAssign(), rhs);
  }


  void operator/=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t Scalar_t;
    compound_assign(OpDivideAssign(), PETE_identity(Scalar_t(rhs)));
  }


  template<class T1, class C1>
  void operator/=(const QDPType<T1,C1>& rhs)
  {
    compound_assign(OpDivideAssign(), PETE_identity(rhs));
  }


  template<class T1, class C1>
  void operator/=(const QDPExpr<T1,C1>& rhs)
  {
    compound_assign(OpDivideAssign(), rhs);
  }


  void operator%=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t Scalar_t;
    compound_assign(OpModAssign(), PETE_identity(Scalar_t(rhs)));
  }


  template<class T1, class C1>
  void operator%=(const QDPType<T1,C1>& rhs)
  {
    compound_assign(OpModAssign(), PETE_identity(rhs));
  }


  template<class T1, class C1>
  void operator%=(const QDPExpr<T1,C1>& rhs)
  {
    compound_assign(OpModAssign(), rhs);
  }


  void operator|=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t Scalar_t;
    compound_assign(OpBitwiseOrAssign(), PETE_identity(Scalar_t(rhs)));
  }


  template<class T1, class C1>
  void operator|=(const QDPType<T1,C1>& rhs)
  {
    compound_assign(OpBitwiseOrAssign(), PETE_identity(rhs));
  }


  template<class T1, class C1>
  void operator|=(const QDPExpr<T1,C1>& rhs)
  {
    compound_assign(OpBitwiseOrAssign(), rhs);
  }


  void operator&=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t Scalar_t;
    compound_assign(OpBitwiseAndAssign(), PETE_identity(Scalar_t(rhs)));
  }


  template<class T1, class C1>
  void operator&=(const QDPType<T1,C1>& rhs)
  {
    compound_assign(OpBitwiseAndAssign(), PETE_identity(rhs));
  }


  template<class T1, class C1>
  void operator&=(const QDPExpr<T1,C1>& rhs)
  {
    compound_assign(OpBitwiseAndAssign(), rhs);
  }


  void operator^=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t Scalar_t;
    compound_assign(OpBitwiseXorAssign(), PETE_identity(Scalar_t(rhs)));
  }


  template<class T1, class C1>
  void operator^=(const QDPType<T1,C1>& rhs)
  {
    compound_assign(OpBitwiseXorAssign(), PETE_identity(rhs));
  }


  template<class T1, class C1>
  void operator^=(const QDPExpr<T1,C1>& rhs)
  {
    compound_assign(OpBitwiseXorAssign(), rhs);
  }


  void operator<<=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t Scalar_t;
    compound_assign(OpLeftShiftAssign(), PETE_identity(Scalar_t(rhs)));
  }


  template<class T1, class C1>
  void operator<<=(const QDPType<T1,C1>& rhs)
  {
    compound_assign(OpLeftShiftAssign(), PETE_identity(rhs));
  }


  template<class T1, class C1>
  void operator<<=(const QDPExpr<T1,C1>& rhs)
  {
    compound_assign(OpLeftShiftAssign(), rhs);
  }


  void operator>>=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t Scalar_t;
    compound_assign(OpRightShiftAssign(), PETE_identity(Scalar_t(rhs)));
  }


  template<class T1, class C1>
  void operator>>=(const QDPType<T1,C1>& rhs)
  {
    compound_assign(OpRightShiftAssign(), PETE_identity(rhs));
  }


  template<class T1, class C1>
  void operator>>=(const QDPExpr<T1,C1>& rhs)
  {
    compound_assign(OpRightShiftAssign(), rhs);
  }


private:
  C& operator=(const QDPSubType& rhs) {}

public:
  int getId() const   { return static_cast<const CC*>(this)->getId();}
  LatticeLayout layout() const { return static_cast<const CC*>(this)->layout();}
  Shape  shape() const   { return static_cast<const CC*>(this)->shape();}
  JitDataFormat data_format() const { return static_cast<const CC*>(this)->data_format();}

  
  typename ScalarType<T>::Type_t* getF() {return static_cast<CC*>(this)->getF();}
  typename ScalarType<T>::Type_t* getF() const {return static_cast<CC const *>(this)->getF();}
  const Subset& subset() const {return static_cast<const CC*>(this)->subset();}

};

  


template<class T>
struct LeafFunctor<QDPSubType<T,OLattice<T> >, DummyJitLeaf>
{
  typedef typename JITType< OSubLattice<T> >::Type_t  Type_t;
  inline static Type_t apply(const QDPSubType<T,OLattice<T> > &a, DummyJitLeaf& p)
  {
    return Type_t();
  }
};

  

} // namespace QDP

#endif



