#ifndef QDP_NEWOPTSJIT_H
#define QDP_NEWOPTSJIT_H


namespace QDP {

struct FnPeekColorMatrixJIT
{
  FnPeekColorMatrixJIT(int _row, int _col): row(_row), col(_col) {}

#if 1
  template<class T>
  inline typename UnaryReturn<T, FnPeekColorMatrixJIT>::Type_t
  operator()(const T &a) const
  {
    return (peekColor(a,row,col));
  }
#else
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
#endif

private:
  int row, col;
};




#if 0
template<class T>
struct ForEach<UnaryNode<FnPeekColorMatrix, Reference<QDPType< T, OLattice<T> > > >, ParamLeaf, TreeCombine>
{
  typedef typename ForEach<Reference<QDPType< T, OLattice<T> > >, ParamLeaf, TreeCombine>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnPeekColorMatrixJIT, TreeCombine>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnPeekColorMatrix, Reference<QDPType< T, OLattice<T> > > > &expr, const ParamLeaf &p, const TreeCombine &c)
  {
    std::cout << __PRETTY_FUNCTION__ << "\n";

    int r_addr = p.getFunc().addGlobalMemory( sizeof(T) * Layout::sitesOnNode() ,
					      p.getFunc().getRegIdx() , sizeof(typename WordType<T>::Type_t) );
    std::cout << "r_addr = " << r_addr << " " << p.getFunc().getName(r_addr) << "\n";

    OLatticeJIT< typename JITContainerType<T>::Type_t > tmp(p.getFunc(),r_addr,Jit::LatticeLayout::COAL);
    std::cout << "000 \n";
    typedef ForEach<Reference<QDPType< T, OLattice<T> > > , ParamLeaf, TreeCombine> AJit_t;
    auto ejit = AJit_t::apply(expr.child(), p, c);
    std::cout << "001 \n";

    tmp.elem(0) = ejit.elem(0);

    printme<decltype(ejit)>();
    printme<TypeA_t>();

    //const_cast<decltype(ejit)>(tmp),
    return Combine1<TypeA_t, FnPeekColorMatrixJIT, TreeCombine>::
      combine( std::move(tmp) ,
              FnPeekColorMatrixJIT( p.getFunc().addParam( Jit::s32 ) , p.getFunc().addParam( Jit::s32 ) ) , c);
  }
};
#else
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
#endif


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
