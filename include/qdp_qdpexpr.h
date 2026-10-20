#ifndef QDP_QDPEXPR_H
#define QDP_QDPEXPR_H

namespace QDP {


template<class T, class C>
class QDPExpr
{
public:

  typedef T Subtype_t;


  typedef T Expression_t;


  typedef C Container_t;


  QDPExpr(const T& expr) : expr_m(expr)
  {    
  }


  const Expression_t& expression() const
  {
    return expr_m;
  }

private:
  T expr_m;
};



template<class T, class C>
struct MakeReturn
{
  typedef QDPExpr<T, C>  Expression_t;
  inline static
  Expression_t make(const T &a) {
    return Expression_t(a); }
};


template<class T, class C>
struct CreateLeaf<QDPExpr<T,C> >
{
  typedef QDPExpr<T,C> Input_t;
  typedef typename Input_t::Expression_t Leaf_t;

  inline static
  const Leaf_t &make(const Input_t& a)
  {
    return a.expression();
  }
};


template<class T, class FTag, class CTag, class DTag>
struct ForEach<QDPExpr<T,DTag>, FTag, CTag>
{
  typedef typename ForEach<T, FTag, CTag>::Type_t Type_t;
  inline static
  Type_t apply(const QDPExpr<T,DTag>& expr, FTag &f, 
	       const CTag &c) 
  {
    return ForEach<T, FTag, CTag>::apply(expr.expression(), f, c);
  }
};




template<class A,class Op>
struct Combine1<Reference<A>, Op, OpCombine>
{
  typedef typename UnaryReturn<A, Op>::Type_t Type_t;
  inline static
  Type_t combine(Reference<A> a, const Op& op, OpCombine) { return op(a.reference()); }
};


template<class A,class B,class Op>
struct Combine2<Reference<A>, Reference<B>, Op, OpCombine>
{
  typedef typename BinaryReturn<A, B, Op>::Type_t Type_t;
  inline static
  Type_t combine(Reference<A> a, Reference<B> b, const Op& op, OpCombine)
  {
    return op(a.reference(), b.reference());
  }
};


template<class A,class B,class Op>
struct Combine2<Reference<A>, B, Op, OpCombine>
{
  typedef typename BinaryReturn<A, B, Op>::Type_t Type_t;
  inline static
  Type_t combine(Reference<A> a, const B& b, const Op& op, OpCombine)
  {
    return op(a.reference(), b);
  }
};


template<class A,class B,class Op>
struct Combine2<A, Reference<B>, Op, OpCombine>
{
  typedef typename BinaryReturn<A, B, Op>::Type_t Type_t;
  inline static
  Type_t combine(const A& a, Reference<B> b, const Op& op, OpCombine)
  {
    return op(a, b.reference());
  }
};


template<class A,class B,class C,class Op>
struct Combine3<Reference<A>, B, C, Op, OpCombine>
{
  typedef typename TrinaryReturn<A, B, C, Op>::Type_t Type_t;
  inline static
  Type_t combine(Reference<A> a, const B& b, const C& c, const Op& op, OpCombine)
  {
    return op(a.reference(), b, c);
  }
};


template<class A,class B,class C,class Op>
struct Combine3<A, Reference<B>, C, Op, OpCombine>
{
  typedef typename TrinaryReturn<A, B, C, Op>::Type_t Type_t;
  inline static
  Type_t combine(const A& a, Reference<B> b, const C& c, const Op& op, OpCombine)
  {
    return op(a, b.reference(), c);
  }
};


template<class A,class B,class C,class Op>
struct Combine3<A, B, Reference<C>, Op, OpCombine>
{
  typedef typename TrinaryReturn<A, B, C, Op>::Type_t Type_t;
  inline static
  Type_t combine(const A& a, const B& b, Reference<C> c, const Op& op, OpCombine)
  {
    return op(a, b, c.reference());
  }
};


template<class A,class B,class C,class Op>
struct Combine3<Reference<A>, Reference<B>, C, Op, OpCombine>
{
  typedef typename TrinaryReturn<A, B, C, Op>::Type_t Type_t;
  inline static
  Type_t combine(Reference<A> a, Reference<B> b, const C& c, const Op& op, OpCombine)
  {
    return op(a.reference(), b.reference(), c);
  }
};


template<class A,class B,class C,class Op>
struct Combine3<A, Reference<B>, Reference<C>, Op, OpCombine>
{
  typedef typename TrinaryReturn<A, B, C, Op>::Type_t Type_t;
  inline static
  Type_t combine(const A& a, Reference<B> b, Reference<C> c, const Op& op, OpCombine)
  {
    return op(a, b.reference(), c.reference());
  }
};


template<class A,class B,class C,class Op>
struct Combine3<Reference<A>, B, Reference<C>, Op, OpCombine>
{
  typedef typename TrinaryReturn<A, B, C, Op>::Type_t Type_t;
  inline static
  Type_t combine(Reference<A> a, const B& b, Reference<C> c, const Op& op, OpCombine)
  {
    return op(a.reference(), b, c.reference());
  }
};


template<class A,class B,class C,class Op>
struct Combine3<Reference<A>, Reference<B>, Reference<C>, Op, OpCombine>
{
  typedef typename TrinaryReturn<A, B, C, Op>::Type_t Type_t;
  inline static
  Type_t combine(Reference<A> a, Reference<B> b, Reference<C> c, const Op& op, OpCombine)
  {
    return op(a.reference(), b.reference(), c.reference());
  }
};


} // namespace QDP

#endif
