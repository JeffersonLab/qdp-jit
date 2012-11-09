#ifndef QDP_NEWOPTSJIT_H
#define QDP_NEWOPTSJIT_H


namespace QDP {

struct FnPeekColorMatrixJIT
{
  //PETE_EMPTY_CONSTRUCTORS(FnPeekColorMatrixJIT)

  FnPeekColorMatrixJIT(int _row, int _col): row(_row), col(_col) {}
  
  template<class T>
  inline typename UnaryReturn<T, FnPeekColorMatrixJIT>::Type_t
  operator()(const T &a) const
  {
    int r_addr = a.func().addGlobalMemory( T::Size_t * sizeof(typename WordType<T>::Type_t) * Layout::sitesOnNode() ,
					   a.func().getRegIdx() , sizeof(typename WordType<T>::Type_t) );

    T tmp(curry_t(a.func(),r_addr,Layout::sitesOnNode(),0));
    tmp = a;

    return (peekColor(tmp,row,col));

    typename UnaryReturn<T, FnPeekColorMatrixJIT>::Type_t d(a.func());
    return d;    
  }

private:
  int row, col;
};




template<class A>
struct ForEach<UnaryNode<FnPeekColorMatrix, A>, ParamLeaf, TreeCombine>
{
  typedef typename ForEach<A, ParamLeaf, TreeCombine>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnPeekColorMatrixJIT, TreeCombine>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnPeekColorMatrix, A> &expr, const ParamLeaf &p, const TreeCombine &c)
  {
    return Combine1<TypeA_t, FnPeekColorMatrixJIT, TreeCombine>::
      combine(ForEach<A, ParamLeaf, TreeCombine>::apply(expr.child(), p, c),
              FnPeekColorMatrixJIT( p.getFunc().addParam( Jit::s32 ) , p.getFunc().addParam( Jit::s32 ) ) , c);
  }
};


template<class A>
struct ForEach<UnaryNode<FnPeekColorMatrix, A>, AddressLeaf, NullCombine>
{
    typedef typename ForEach< A , AddressLeaf, NullCombine>::Type_t TypeA_t;
    typedef TypeA_t Type_t;
    inline
    static Type_t apply(const UnaryNode<FnPeekColorMatrix, A>& expr, const AddressLeaf &a, const NullCombine &n)
    {
      int row = expr.operation().getRow();
      int col = expr.operation().getCol();
      std::cout << "set peek color matrix row,col = " << row << " " << col << "\n";
      a.setLit( row );
      a.setLit( col );
      return Type_t( ForEach<A, AddressLeaf, NullCombine>::apply( expr.child() , a , n ) );
    }
};


}

#endif
