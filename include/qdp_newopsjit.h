#ifndef QDP_NEWOPSJIT_H
#define QDP_NEWOPSJIT_H


namespace QDP {


struct FnPeekColorMatrixJIT
{
  FnPeekColorMatrixJIT(ParamRef  _row, ParamRef  _col): row(_row), col(_col) {}
  
  template<class T>
  inline typename UnaryReturn<T, FnPeekColorMatrixJIT>::Type_t
  operator()(const T &a) const
  {
    return peekColor(a,llvm_derefParam(row),llvm_derefParam(col));
  }
  
  llvm::Value* get_row_jit() const { return llvm_derefParam(row); }
  llvm::Value* get_col_jit() const { return llvm_derefParam(col); }

private:
  ParamRef row;
  ParamRef col;
};


struct FnPokeColorMatrixJIT
{
  FnPokeColorMatrixJIT(ParamRef  _row, ParamRef  _col): row(_row), col(_col) {}
  
  template<class T1, class T2>
  void
  operator()(T1 a, const T2 &b) const
  {
    pokeColor(a,b,llvm_derefParam(row),llvm_derefParam(col));
  }

private:
  ParamRef  row;
  ParamRef col;
};






template<class A>
struct ForEach<UnaryNode<FnPeekColorMatrix, A>, MGParamLeaf, TreeCombine>
{
  typedef typename ForEach<A, MGParamLeaf, TreeCombine>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnPeekColorMatrixJIT, TreeCombine>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnPeekColorMatrix, A> &expr, MGParamLeaf &p, const TreeCombine &c)
  {
    p.next( PlanEntry::Kind::OpParam );
    ParamRef a = llvm_add_param<int>();
    p.next( PlanEntry::Kind::OpParam );
    ParamRef b = llvm_add_param<int>();
    return Combine1<TypeA_t, FnPeekColorMatrixJIT, TreeCombine>::
      combine(ForEach<A, MGParamLeaf, TreeCombine>::apply(expr.child(), p, c),
              FnPeekColorMatrixJIT( a , b ) , c);
  }
};



  
  


  template<>  
  struct AddOpParam< FnPokeColorMatrix, MGParamLeaf> {
    typedef FnPokeColorMatrixJIT Type_t;
    static FnPokeColorMatrixJIT apply( const FnPokeColorMatrix& a, MGParamLeaf& p) {
      p.next( PlanEntry::Kind::OpParam );
      ParamRef pa = llvm_add_param<int>();
      p.next( PlanEntry::Kind::OpParam );
      ParamRef pb = llvm_add_param<int>();
      return FnPokeColorMatrixJIT( pa , pb );
    }
  };

  




struct FnPeekSpinMatrixJIT
{
  FnPeekSpinMatrixJIT(ParamRef  _row, ParamRef  _col): row(_row), col(_col) {}

  template<class T>
  inline typename UnaryReturn<T, FnPeekSpinMatrixJIT>::Type_t
  operator()(const T &a) const
  {
    return (peekSpin(a,llvm_derefParam(row),llvm_derefParam(col)));
  }

  llvm::Value* get_row_jit() const { return llvm_derefParam(row); }
  llvm::Value* get_col_jit() const { return llvm_derefParam(col); }
  
private:
  ParamRef  row;
  ParamRef  col;
};


struct FnPokeSpinMatrixJIT
{
  FnPokeSpinMatrixJIT(ParamRef  _row, ParamRef  _col): row(_row), col(_col) {}
  
  template<class T1, class T2>
  void
  operator()(const T1 &a, const T2 &b) const
  {
    pokeSpin(const_cast<T1&>(a),b,llvm_derefParam(row),llvm_derefParam(col));
  }


  llvm::Value* jitRow() const { return llvm_derefParam(row); }
  llvm::Value* jitCol() const { return llvm_derefParam(col); }

private:
  ParamRef  row;
  ParamRef  col;
};






template<class A>
struct ForEach<UnaryNode<FnPeekSpinMatrix, A>, MGParamLeaf, TreeCombine>
{
  typedef typename ForEach<A, MGParamLeaf, TreeCombine>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnPeekSpinMatrixJIT, TreeCombine>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnPeekSpinMatrix, A> &expr, MGParamLeaf &p, const TreeCombine &c)
  {
    p.next( PlanEntry::Kind::OpParam );
    ParamRef a = llvm_add_param<int>();
    p.next( PlanEntry::Kind::OpParam );
    ParamRef b = llvm_add_param<int>();
    return Combine1<TypeA_t, FnPeekSpinMatrixJIT, TreeCombine>::
      combine(ForEach<A, MGParamLeaf, TreeCombine>::apply(expr.child(), p, c),
              FnPeekSpinMatrixJIT( a , b ) , c);
  }
};



  template<>  
  struct AddOpParam< FnPokeSpinMatrix, MGParamLeaf> {
    typedef FnPokeSpinMatrixJIT Type_t;
    static FnPokeSpinMatrixJIT apply( const FnPokeSpinMatrix& a, MGParamLeaf& p) {
      p.next( PlanEntry::Kind::OpParam );
      ParamRef pa = llvm_add_param<int>();
      p.next( PlanEntry::Kind::OpParam );
      ParamRef pb = llvm_add_param<int>();
      return FnPokeSpinMatrixJIT( pa , pb );
    }
  };

  



struct FnPeekColorVectorJIT
{
  FnPeekColorVectorJIT(ParamRef  _row): row(_row) {}

  template<class T>
  inline typename UnaryReturn<T, FnPeekColorVectorJIT>::Type_t
  operator()(const T &a) const
  {
    return (peekColor(a,llvm_derefParam(row)));
  }
  
  llvm::Value* get_row_jit() const { return llvm_derefParam(row); }

private:
  ParamRef  row;   // these are registers
};


struct FnPokeColorVectorJIT
{
  FnPokeColorVectorJIT(ParamRef  _row): row(_row) {}
  
  template<class T1, class T2>
  void
  operator()(const T1 &a, const T2 &b) const
  {
    pokeColor(const_cast<T1&>(a),b,llvm_derefParam(row));
  }

private:
  ParamRef  row;   // these are registers
};







template<class A>
struct ForEach<UnaryNode<FnPeekColorVector, A>, MGParamLeaf, TreeCombine>
{
  typedef typename ForEach<A, MGParamLeaf, TreeCombine>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnPeekColorVectorJIT, TreeCombine>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnPeekColorVector, A> &expr, MGParamLeaf &p, const TreeCombine &c)
  {
    p.next( PlanEntry::Kind::OpParam );
    ParamRef a = llvm_add_param<int>();
    return Combine1<TypeA_t, FnPeekColorVectorJIT, TreeCombine>::
      combine(ForEach<A, MGParamLeaf, TreeCombine>::apply(expr.child(), p, c),
              FnPeekColorVectorJIT( a ) , c);
  }
};

  
  


  template<>  
  struct AddOpParam< FnPokeColorVector, MGParamLeaf> {
    typedef FnPokeColorVectorJIT Type_t;
    static FnPokeColorVectorJIT apply( const FnPokeColorVector& a, MGParamLeaf& p) {
      p.next( PlanEntry::Kind::OpParam );
      return FnPokeColorVectorJIT( llvm_add_param<int>() );
    }
  };







struct FnPeekSpinVectorJIT
{
  FnPeekSpinVectorJIT(ParamRef  _row): row(_row) {}

  template<class T>
  inline typename UnaryReturn<T, FnPeekSpinVectorJIT>::Type_t
  operator()(const T &a) const
  {
    return (peekSpin(a,llvm_derefParam(row)));
  }

  llvm::Value* get_row_jit() const { return llvm_derefParam(row); }

private:
  ParamRef  row;   // these are registers
};


struct FnPokeSpinVectorJIT
{
  FnPokeSpinVectorJIT(ParamRef  _row): row(_row) {}
  
  template<class T1, class T2>
  void
  operator()(const T1 &a, const T2 &b) const
  {
    pokeSpin(const_cast<T1&>(a),b,llvm_derefParam(row));
  }

private:
  ParamRef  row;   // these are registers
};







template<class A>
struct ForEach<UnaryNode<FnPeekSpinVector, A>, MGParamLeaf, TreeCombine>
{
  typedef typename ForEach<A, MGParamLeaf, TreeCombine>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnPeekSpinVectorJIT, TreeCombine>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnPeekSpinVector, A> &expr, MGParamLeaf &p, const TreeCombine &c)
  {
    p.next( PlanEntry::Kind::OpParam );
    ParamRef a = llvm_add_param<int>();
    return Combine1<TypeA_t, FnPeekSpinVectorJIT, TreeCombine>::
      combine(ForEach<A, MGParamLeaf, TreeCombine>::apply(expr.child(), p, c),
              FnPeekSpinVectorJIT( a ) , c);
  }
};
  
  


    template<>  
  struct AddOpParam< FnPokeSpinVector, MGParamLeaf> {
    typedef FnPokeSpinVectorJIT Type_t;
    static FnPokeSpinVectorJIT apply( const FnPokeSpinVector& a, MGParamLeaf& p) {
      p.next( PlanEntry::Kind::OpParam );
      return FnPokeSpinVectorJIT( llvm_add_param<int>() );
    }
  };

}

#endif
