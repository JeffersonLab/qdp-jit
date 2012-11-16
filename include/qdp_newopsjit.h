#ifndef QDP_NEWOPTSJIT_H
#define QDP_NEWOPTSJIT_H


namespace QDP {

struct FnPeekColorMatrixJIT
{
  FnPeekColorMatrixJIT(int _row, int _col): row(_row), col(_col) {}

  template<class T>
  inline typename UnaryReturn<T, FnPeekColorMatrixJIT>::Type_t
  operator()(const T &a) const
  {
    return (peekColor(a,row,col));
  }

private:
  int row, col;   // these are registers
};


struct FnPokeColorMatrixJIT
{
  FnPokeColorMatrixJIT(int _row, int _col): row(_row), col(_col) {}
  
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, FnPokeColorMatrixJIT>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeColor(const_cast<T1&>(a),b,row,col);
    return const_cast<T1&>(a);
  }

private:
  int row, col;   // these are registers
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
  
  
  
  template<>  
  struct AddOpParam< FnPokeColorMatrix, ParamLeaf> {
    static FnPokeColorMatrixJIT apply( const FnPokeColorMatrix& a, const ParamLeaf& p) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      return FnPokeColorMatrixJIT( p.getFunc().addParam( Jit::s32 ) , 
				   p.getFunc().addParam( Jit::s32 ) );
    }
  };

  template<>  
  struct AddOpAddress< FnPokeColorMatrix, AddressLeaf> {
    static void apply( const FnPokeColorMatrix& p, const AddressLeaf& a) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      int row = p.getRow();
      int col = p.getCol();
      //std::cout << "set poke color matrix row,col = " << row << " " << col << "\n";
      a.setLit( row );
      a.setLit( col );
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
      //std::cout << "set peek color matrix row,col = " << row << " " << col << "\n";
      a.setLit( row );
      a.setLit( col );
      return Type_t( ForEach<A, AddressLeaf, NullCombine>::apply( expr.child() , a , n ) );
    }
};


  //////////////////////////////

struct FnPeekSpinMatrixJIT
{
  FnPeekSpinMatrixJIT(int _row, int _col): row(_row), col(_col) {}

  template<class T>
  inline typename UnaryReturn<T, FnPeekSpinMatrixJIT>::Type_t
  operator()(const T &a) const
  {
    return (peekSpin(a,row,col));
  }

private:
  int row, col;   // these are registers
};


struct FnPokeSpinMatrixJIT
{
  FnPokeSpinMatrixJIT(int _row, int _col): row(_row), col(_col) {}
  
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, FnPokeSpinMatrixJIT>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeSpin(const_cast<T1&>(a),b,row,col);
    return const_cast<T1&>(a);
  }

private:
  int row, col;   // these are registers
};





template<class A>
struct ForEach<UnaryNode<FnPeekSpinMatrix, A>, ParamLeaf, TreeCombine>
{
  typedef typename ForEach<A, ParamLeaf, TreeCombine>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnPeekSpinMatrixJIT, TreeCombine>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnPeekSpinMatrix, A> &expr, const ParamLeaf &p, const TreeCombine &c)
  {
    return Combine1<TypeA_t, FnPeekSpinMatrixJIT, TreeCombine>::
      combine(ForEach<A, ParamLeaf, TreeCombine>::apply(expr.child(), p, c),
              FnPeekSpinMatrixJIT( p.getFunc().addParam( Jit::s32 ) , p.getFunc().addParam( Jit::s32 ) ) , c);
  }
};
  
  
  
  template<>  
  struct AddOpParam< FnPokeSpinMatrix, ParamLeaf> {
    static FnPokeSpinMatrixJIT apply( const FnPokeSpinMatrix& a, const ParamLeaf& p) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      return FnPokeSpinMatrixJIT( p.getFunc().addParam( Jit::s32 ) , 
				   p.getFunc().addParam( Jit::s32 ) );
    }
  };

  template<>  
  struct AddOpAddress< FnPokeSpinMatrix, AddressLeaf> {
    static void apply( const FnPokeSpinMatrix& p, const AddressLeaf& a) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      int row = p.getRow();
      int col = p.getCol();
      //std::cout << "set poke spin matrix row,col = " << row << " " << col << "\n";
      a.setLit( row );
      a.setLit( col );
    }
  };



template<class A>
struct ForEach<UnaryNode<FnPeekSpinMatrix, A>, AddressLeaf, NullCombine>
{
    typedef typename ForEach< A , AddressLeaf, NullCombine>::Type_t TypeA_t;
    typedef TypeA_t Type_t;
    inline
    static Type_t apply(const UnaryNode<FnPeekSpinMatrix, A>& expr, const AddressLeaf &a, const NullCombine &n)
    {
      int row = expr.operation().getRow();
      int col = expr.operation().getCol();
      //std::cout << "set peek spin matrix row,col = " << row << " " << col << "\n";
      a.setLit( row );
      a.setLit( col );
      return Type_t( ForEach<A, AddressLeaf, NullCombine>::apply( expr.child() , a , n ) );
    }
};


  /////////////////////////////////////////////////
  ///
  ///
  /////////////////////////////////////////////////


struct FnPeekColorVectorJIT
{
  FnPeekColorVectorJIT(int _row): row(_row) {}

  template<class T>
  inline typename UnaryReturn<T, FnPeekColorVectorJIT>::Type_t
  operator()(const T &a) const
  {
    return (peekColor(a,row));
  }

private:
  int row;   // these are registers
};


struct FnPokeColorVectorJIT
{
  FnPokeColorVectorJIT(int _row): row(_row) {}
  
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, FnPokeColorVectorJIT>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeColor(const_cast<T1&>(a),b,row);
    return const_cast<T1&>(a);
  }

private:
  int row;   // these are registers
};





template<class A>
struct ForEach<UnaryNode<FnPeekColorVector, A>, ParamLeaf, TreeCombine>
{
  typedef typename ForEach<A, ParamLeaf, TreeCombine>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnPeekColorVectorJIT, TreeCombine>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnPeekColorVector, A> &expr, const ParamLeaf &p, const TreeCombine &c)
  {
    return Combine1<TypeA_t, FnPeekColorVectorJIT, TreeCombine>::
      combine(ForEach<A, ParamLeaf, TreeCombine>::apply(expr.child(), p, c),
              FnPeekColorVectorJIT( p.getFunc().addParam( Jit::s32 ) ) , c);
  }
};
  
  
  
  template<>  
  struct AddOpParam< FnPokeColorVector, ParamLeaf> {
    static FnPokeColorVectorJIT apply( const FnPokeColorVector& a, const ParamLeaf& p) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      return FnPokeColorVectorJIT( p.getFunc().addParam( Jit::s32 ) );
    }
  };

  template<>  
  struct AddOpAddress< FnPokeColorVector, AddressLeaf> {
    static void apply( const FnPokeColorVector& p, const AddressLeaf& a) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      int row = p.getRow();
      //std::cout << "set poke color vector row,col = " << row << "\n";
      a.setLit( row );
    }
  };



template<class A>
struct ForEach<UnaryNode<FnPeekColorVector, A>, AddressLeaf, NullCombine>
{
    typedef typename ForEach< A , AddressLeaf, NullCombine>::Type_t TypeA_t;
    typedef TypeA_t Type_t;
    inline
    static Type_t apply(const UnaryNode<FnPeekColorVector, A>& expr, const AddressLeaf &a, const NullCombine &n)
    {
      int row = expr.operation().getRow();
      //std::cout << "set peek color vector row = " << row << "\n";
      a.setLit( row );
      return Type_t( ForEach<A, AddressLeaf, NullCombine>::apply( expr.child() , a , n ) );
    }
};


  //////////////////////////////

struct FnPeekSpinVectorJIT
{
  FnPeekSpinVectorJIT(int _row): row(_row) {}

  template<class T>
  inline typename UnaryReturn<T, FnPeekSpinVectorJIT>::Type_t
  operator()(const T &a) const
  {
    return (peekSpin(a,row));
  }

private:
  int row;   // these are registers
};


struct FnPokeSpinVectorJIT
{
  FnPokeSpinVectorJIT(int _row): row(_row) {}
  
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, FnPokeSpinVectorJIT>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeSpin(const_cast<T1&>(a),b,row);
    return const_cast<T1&>(a);
  }

private:
  int row;   // these are registers
};





template<class A>
struct ForEach<UnaryNode<FnPeekSpinVector, A>, ParamLeaf, TreeCombine>
{
  typedef typename ForEach<A, ParamLeaf, TreeCombine>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnPeekSpinVectorJIT, TreeCombine>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnPeekSpinVector, A> &expr, const ParamLeaf &p, const TreeCombine &c)
  {
    return Combine1<TypeA_t, FnPeekSpinVectorJIT, TreeCombine>::
      combine(ForEach<A, ParamLeaf, TreeCombine>::apply(expr.child(), p, c),
              FnPeekSpinVectorJIT( p.getFunc().addParam( Jit::s32 ) ) , c);
  }
};
  
  
  
  template<>  
  struct AddOpParam< FnPokeSpinVector, ParamLeaf> {
    static FnPokeSpinVectorJIT apply( const FnPokeSpinVector& a, const ParamLeaf& p) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      return FnPokeSpinVectorJIT( p.getFunc().addParam( Jit::s32 ) );
    }
  };

  template<>  
  struct AddOpAddress< FnPokeSpinVector, AddressLeaf> {
    static void apply( const FnPokeSpinVector& p, const AddressLeaf& a) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      int row = p.getRow();
      //std::cout << "set poke spin vector row = " << row << "\n";
      a.setLit( row );
    }
  };



template<class A>
struct ForEach<UnaryNode<FnPeekSpinVector, A>, AddressLeaf, NullCombine>
{
    typedef typename ForEach< A , AddressLeaf, NullCombine>::Type_t TypeA_t;
    typedef TypeA_t Type_t;
    inline
    static Type_t apply(const UnaryNode<FnPeekSpinVector, A>& expr, const AddressLeaf &a, const NullCombine &n)
    {
      int row = expr.operation().getRow();
      //std::cout << "set peek spin vector row = " << row << "\n";
      a.setLit( row );
      return Type_t( ForEach<A, AddressLeaf, NullCombine>::apply( expr.child() , a , n ) );
    }
};


}

#endif
