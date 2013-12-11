#ifndef QDP_NEWOPTSREG_H
#define QDP_NEWOPTSREG_H


namespace QDP {

struct FnPeekColorMatrixREG
{
  FnPeekColorMatrixREG(ParamRef  _row, ParamRef  _col): row(_row), col(_col) {}

  template<class T>
  inline typename UnaryReturn<T, FnPeekColorMatrixREG>::Type_t
  operator()(const T &a) const
  {
    return (peekColor(a,llvm_derefParam(row),llvm_derefParam(col)));
  }

private:
  ParamRef row;
  ParamRef col;
};


struct FnPokeColorMatrixREG
{
  FnPokeColorMatrixREG(ParamRef  _row, ParamRef  _col): row(_row), col(_col) {}
  
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, FnPokeColorMatrixREG>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeColor(const_cast<T1&>(a),b,llvm_derefParam(row),llvm_derefParam(col));
    return const_cast<T1&>(a);
  }

private:
  ParamRef  row;
  ParamRef col;
};




template<class A>
struct ForEach<UnaryNode<FnPeekColorMatrix, A>, ParamLeaf, TreeCombine>
{
  typedef typename ForEach<A, ParamLeaf, TreeCombine>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnPeekColorMatrixREG, TreeCombine>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnPeekColorMatrix, A> &expr, const ParamLeaf &p, const TreeCombine &c)
  {
    return Combine1<TypeA_t, FnPeekColorMatrixREG, TreeCombine>::
      combine(ForEach<A, ParamLeaf, TreeCombine>::apply(expr.child(), p, c),
              FnPeekColorMatrixREG( llvm_add_param<int>() , llvm_add_param<int>() ) , c);
  }
};
  
  
  
  template<>  
  struct AddOpParam< FnPokeColorMatrix, ParamLeaf> {
    typedef FnPokeColorMatrixREG Type_t;
    static FnPokeColorMatrixREG apply( const FnPokeColorMatrix& a, const ParamLeaf& p) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      return FnPokeColorMatrixREG( llvm_add_param<int>() , 
				   llvm_add_param<int>() );
    }
  };

  template<>  
  struct AddOpAddress< FnPokeColorMatrix, AddressLeaf> {
    static void apply( const FnPokeColorMatrix& p, const AddressLeaf& a) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      int row = p.getRow();
      int col = p.getCol();
      //std::cout << "set poke color matrix row,llvm_derefParam(col) = " << row << " " << col << "\n";
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
      //std::cout << "set peek color matrix row,llvm_derefParam(col) = " << row << " " << col << "\n";
      a.setLit( row );
      a.setLit( col );
      return Type_t( ForEach<A, AddressLeaf, NullCombine>::apply( expr.child() , a , n ) );
    }
};


  //////////////////////////////

struct FnPeekSpinMatrixREG
{
  FnPeekSpinMatrixREG(ParamRef  _row, ParamRef  _col): row(_row), col(_col) {}

  template<class T>
  inline typename UnaryReturn<T, FnPeekSpinMatrixREG>::Type_t
  operator()(const T &a) const
  {
    return (peekSpin(a,llvm_derefParam(row),llvm_derefParam(col)));
  }

private:
  ParamRef  row;
  ParamRef  col;
};


struct FnPokeSpinMatrixREG
{
  FnPokeSpinMatrixREG(ParamRef  _row, ParamRef  _col): row(_row), col(_col) {}
  
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, FnPokeSpinMatrixREG>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeSpin(const_cast<T1&>(a),b,llvm_derefParam(row),llvm_derefParam(col));
    return const_cast<T1&>(a);
  }

private:
  ParamRef  row;
  ParamRef  col;
};





template<class A>
struct ForEach<UnaryNode<FnPeekSpinMatrix, A>, ParamLeaf, TreeCombine>
{
  typedef typename ForEach<A, ParamLeaf, TreeCombine>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnPeekSpinMatrixREG, TreeCombine>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnPeekSpinMatrix, A> &expr, const ParamLeaf &p, const TreeCombine &c)
  {
    return Combine1<TypeA_t, FnPeekSpinMatrixREG, TreeCombine>::
      combine(ForEach<A, ParamLeaf, TreeCombine>::apply(expr.child(), p, c),
              FnPeekSpinMatrixREG( llvm_add_param<int>() , llvm_add_param<int>() ) , c);
  }
};
  
  
  
  template<>  
  struct AddOpParam< FnPokeSpinMatrix, ParamLeaf> {
    typedef FnPokeSpinMatrixREG Type_t;
    static FnPokeSpinMatrixREG apply( const FnPokeSpinMatrix& a, const ParamLeaf& p) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      return FnPokeSpinMatrixREG( llvm_add_param<int>() , 
				   llvm_add_param<int>() );
    }
  };

  template<>  
  struct AddOpAddress< FnPokeSpinMatrix, AddressLeaf> {
    static void apply( const FnPokeSpinMatrix& p, const AddressLeaf& a) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      int row = p.getRow();
      int col = p.getCol();
      //std::cout << "set poke spin matrix row,llvm_derefParam(col) = " << row << " " << col << "\n";
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
      //std::cout << "set peek spin matrix row,llvm_derefParam(col) = " << row << " " << col << "\n";
      a.setLit( row );
      a.setLit( col );
      return Type_t( ForEach<A, AddressLeaf, NullCombine>::apply( expr.child() , a , n ) );
    }
};


  /////////////////////////////////////////////////
  ///
  ///
  /////////////////////////////////////////////////


struct FnPeekColorVectorREG
{
  FnPeekColorVectorREG(ParamRef  _row): row(_row) {}

  template<class T>
  inline typename UnaryReturn<T, FnPeekColorVectorREG>::Type_t
  operator()(const T &a) const
  {
    return (peekColor(a,llvm_derefParam(row)));
  }

private:
  ParamRef  row;   // these are registers
};


struct FnPokeColorVectorREG
{
  FnPokeColorVectorREG(ParamRef  _row): row(_row) {}
  
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, FnPokeColorVectorREG>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeColor(const_cast<T1&>(a),b,llvm_derefParam(row));
    return const_cast<T1&>(a);
  }

private:
  ParamRef  row;   // these are registers
};





template<class A>
struct ForEach<UnaryNode<FnPeekColorVector, A>, ParamLeaf, TreeCombine>
{
  typedef typename ForEach<A, ParamLeaf, TreeCombine>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnPeekColorVectorREG, TreeCombine>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnPeekColorVector, A> &expr, const ParamLeaf &p, const TreeCombine &c)
  {
    return Combine1<TypeA_t, FnPeekColorVectorREG, TreeCombine>::
      combine(ForEach<A, ParamLeaf, TreeCombine>::apply(expr.child(), p, c),
              FnPeekColorVectorREG( llvm_add_param<int>() ) , c);
  }
};
  
  
  
  template<>  
  struct AddOpParam< FnPokeColorVector, ParamLeaf> {
    typedef FnPokeColorVectorREG Type_t;
    static FnPokeColorVectorREG apply( const FnPokeColorVector& a, const ParamLeaf& p) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      return FnPokeColorVectorREG( llvm_add_param<int>() );
    }
  };

  template<>  
  struct AddOpAddress< FnPokeColorVector, AddressLeaf> {
    static void apply( const FnPokeColorVector& p, const AddressLeaf& a) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      int row = p.getRow();
      //std::cout << "set poke color vector row,llvm_derefParam(col) = " << row << "\n";
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

struct FnPeekSpinVectorREG
{
  FnPeekSpinVectorREG(ParamRef  _row): row(_row) {}

  template<class T>
  inline typename UnaryReturn<T, FnPeekSpinVectorREG>::Type_t
  operator()(const T &a) const
  {
    return (peekSpin(a,llvm_derefParam(row)));
  }

private:
  ParamRef  row;   // these are registers
};


struct FnPokeSpinVectorREG
{
  FnPokeSpinVectorREG(ParamRef  _row): row(_row) {}
  
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, FnPokeSpinVectorREG>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeSpin(const_cast<T1&>(a),b,llvm_derefParam(row));
    return const_cast<T1&>(a);
  }

private:
  ParamRef  row;   // these are registers
};





template<class A>
struct ForEach<UnaryNode<FnPeekSpinVector, A>, ParamLeaf, TreeCombine>
{
  typedef typename ForEach<A, ParamLeaf, TreeCombine>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnPeekSpinVectorREG, TreeCombine>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnPeekSpinVector, A> &expr, const ParamLeaf &p, const TreeCombine &c)
  {
    return Combine1<TypeA_t, FnPeekSpinVectorREG, TreeCombine>::
      combine(ForEach<A, ParamLeaf, TreeCombine>::apply(expr.child(), p, c),
              FnPeekSpinVectorREG( llvm_add_param<int>() ) , c);
  }
};
  
  
  
  template<>  
  struct AddOpParam< FnPokeSpinVector, ParamLeaf> {
    typedef FnPokeSpinVectorREG Type_t;
    static FnPokeSpinVectorREG apply( const FnPokeSpinVector& a, const ParamLeaf& p) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      return FnPokeSpinVectorREG( llvm_add_param<int>() );
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
