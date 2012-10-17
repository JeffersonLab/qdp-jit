// -*- C++ -*-

#ifndef QDP_QDPTYPEJIT_H
#define QDP_QDPTYPEJIT_H

namespace QDP {


template<class T, class C> 
class QDPTypeJIT
{
public:
  //! Type of the first argument
  typedef T Subtype_t;

  //! Type of the container class
  typedef C Container_t;

#if 0
  // This must go !!!
  QDPTypeJIT(Jit& func_) : function(func_) {}
#endif

  QDPTypeJIT(Jit& func_, int addr_) : function(func_), r_addr(addr_) {}

  //! Copy constructor
  QDPTypeJIT(const QDPTypeJIT& a) : function(a.function), r_addr(a.r_addr) {}

  //! Destructor
  ~QDPTypeJIT(){}


public:
  T elem(int i) {return static_cast<const C*>(this)->elem(i);}

  const T elem(int i) const {return static_cast<const C*>(this)->elem(i);}

  T elem() {return static_cast<const C*>(this)->elem();}

  const T elem() const {return static_cast<const C*>(this)->elem();}

  Jit& getFunc() const { return function; }
  int getAddr() const { return r_addr; }

  private:
    Jit& function;
    int  r_addr;

};




//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

#if 0
// Default unary(QDPTypeJIT) -> QDPTypeJIT
template<class T1, class C1, class Op>
struct UnaryReturn<QDPTypeJIT<T1,C1>, Op> {
  typedef QDPTypeJIT<typename UnaryReturn<T1, Op>::Type_t,
		  typename UnaryReturn<C1, Op>::Type_t>  Type_t;
};

// Default binary(QDPTypeJIT,QDPTypeJIT) -> QDPTypeJIT
template<class T1, class C1, class T2, class C2, class Op>
struct BinaryReturn<QDPTypeJIT<T1,C1>, QDPTypeJIT<T2,C2>, Op> {
  typedef QDPTypeJIT<typename BinaryReturn<T1, T2, Op>::Type_t,
		  typename BinaryReturn<C1, C2, Op>::Type_t>  Type_t;
};

// Currently, the only trinary operator is ``where'', so return 
// based on T2 and T3
// Default trinary(QDPTypeJIT,QDPTypeJIT,QDPTypeJIT) -> QDPTypeJIT
template<class T1, class C1, class T2, class C2, class T3, class C3, class Op>
struct TrinaryReturn<QDPTypeJIT<T1,C1>, QDPTypeJIT<T2,C2>, QDPTypeJIT<T3,C3>, Op> {
  typedef QDPTypeJIT<typename BinaryReturn<T2, T3, Op>::Type_t,
		  typename BinaryReturn<C2, C3, Op>::Type_t>  Type_t;
};


//-----------------------------------------------------------------------------
// We need to specialize CreateLeaf<T> for our class, so that operators
// know what to stick in the leaves of the expression tree.
//-----------------------------------------------------------------------------

template<class T, class C>
struct CreateLeaf<QDPTypeJIT<T,C> >
{
  typedef QDPTypeJIT<T,C> Inp_t;
  typedef Reference<Inp_t> Leaf_t;
//  typedef Inp_t Leaf_t;
  PETE_DEVICE
  inline static
  Leaf_t make(const Inp_t &a) { return Leaf_t(a); }
};


//-----------------------------------------------------------------------------
// Specialization of LeafFunctor class for applying the EvalLeaf1
// tag to a QDPTypeJIT. The apply method simply returns the array
// evaluated at the point.
//-----------------------------------------------------------------------------

template<class T, class C>
struct LeafFunctor<QDPTypeJIT<T,C>, ElemLeaf>
{
  typedef Reference<T> Type_t;
//  typedef T Type_t;
  PETE_DEVICE
  inline static Type_t apply(const QDPTypeJIT<T,C> &a, const ElemLeaf &f)
    { 
      return Type_t(a.elem());
    }
};

template<class T, class C>
struct LeafFunctor<QDPTypeJIT<T,C>, EvalLeaf1>
{
  typedef Reference<T> Type_t;
//  typedef T Type_t;
  PETE_DEVICE
  inline static Type_t apply(const QDPTypeJIT<T,C> &a, const EvalLeaf1 &f)
    { 
      return Type_t(a.elem(f.val1()));
    }
};


#endif



template<class T>
struct LeafFunctor<QDPTypeJIT<T,OLattice<T> >, ParamLeaf>
{
  typedef QDPTypeJIT<T,typename JITContainerType<OLattice<T> >::Type_t>  TypeA_t;
  //typedef typename JITContainerType< OLattice<T> >::Type_t  TypeA_t;
  typedef TypeA_t  Type_t;
  inline static Type_t apply(const QDPTypeJIT<T,OLattice<T> > &a, const ParamLeaf& p)
  {
    return Type_t( p.getFunc() , p.getParamLattice( WordSize<T>::Size ) );
  }
};

template<class T>
struct LeafFunctor<QDPTypeJIT<T,OScalar<T> >, ParamLeaf>
{
  typedef QDPTypeJIT<T,typename JITContainerType<OScalar<T> >::Type_t>  TypeA_t;
  //typedef typename JITContainerType< OScalar<T> >::Type_t  TypeA_t;
  typedef TypeA_t  Type_t;
  inline static Type_t apply(const QDPTypeJIT<T,OScalar<T> > &a, const ParamLeaf& p)
  {
    return Type_t( p.getFunc() , p.getParamScalar() );
  }
};



template<class T, class C>
struct LeafFunctor<QDPTypeJIT<T,C>, ViewLeaf>
{
  typedef T Type_t;
  inline static
  Type_t apply(const QDPTypeJIT<T,C>& s, const ViewLeaf& v)
  { 
    return s.elem(v.val1());
  }
};



} // namespace QDP

#endif

