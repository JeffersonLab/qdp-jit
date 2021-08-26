// -*- C++ -*-

/*! @file
 * @brief Additional operations on QDPTypes
 */

#ifndef QDP_NEWOPS_H
#define QDP_NEWOPS_H

namespace QDP {

//-----------------------------------------------------------------------------
// Operator tags that are only used for type resolution
//-----------------------------------------------------------------------------

struct FnSpinProject
{
  PETE_EMPTY_CONSTRUCTORS(FnSpinProject)
};

struct FnSpinReconstruct
{
  PETE_EMPTY_CONSTRUCTORS(FnSpinReconstruct)
};

struct FnQuarkContractXX
{
  PETE_EMPTY_CONSTRUCTORS(FnQuarkContractXX)
};

struct FnSum
{
  PETE_EMPTY_CONSTRUCTORS(FnSum)
};

struct FnGlobalMax
{
  PETE_EMPTY_CONSTRUCTORS(FnGlobalMax)
};

struct FnGlobalMin
{
  PETE_EMPTY_CONSTRUCTORS(FnGlobalMin)
};

struct FnIsNan
{
  PETE_EMPTY_CONSTRUCTORS(FnIsNan)
};

struct FnIsInf
{
  PETE_EMPTY_CONSTRUCTORS(FnIsInf)
};

struct FnIsFinite
{
  PETE_EMPTY_CONSTRUCTORS(FnIsFinite)
};

struct FnIsNormal
{
  PETE_EMPTY_CONSTRUCTORS(FnIsNormal)
};

struct FnNorm2
{
  PETE_EMPTY_CONSTRUCTORS(FnNorm2)
};

struct FnInnerProduct
{
  PETE_EMPTY_CONSTRUCTORS(FnInnerProduct)
};

struct FnInnerProductReal
{
  PETE_EMPTY_CONSTRUCTORS(FnInnerProductReal)
};

struct FnSumMulti
{
  PETE_EMPTY_CONSTRUCTORS(FnSumMulti)
};

struct FnNorm2Multi
{
  PETE_EMPTY_CONSTRUCTORS(FnNorm2Multi)
};

struct FnInnerProductMulti
{
  PETE_EMPTY_CONSTRUCTORS(FnInnerProductMulti)
};

struct FnInnerProductRealMulti
{
  PETE_EMPTY_CONSTRUCTORS(FnInnerProductRealMulti)
};


//-----------------------------------------------------------------------------
// Operators and tags for accessing elements of a QDP object
//-----------------------------------------------------------------------------

struct FnGetSite
{
  PETE_EMPTY_CONSTRUCTORS(FnGetSite)
};

struct FnPeekSite
{
  PETE_EMPTY_CONSTRUCTORS(FnPeekSite)
};

struct FnPokeSite
{
  PETE_EMPTY_CONSTRUCTORS(FnPokeSite)
};


//! Structure for extracting color matrix components
struct FnPeekColorMatrix
{
  PETE_EMPTY_CONSTRUCTORS(FnPeekColorMatrix)

  FnPeekColorMatrix(int _row, int _col): row(_row), col(_col) {}
  
  template<class T>
  inline typename UnaryReturn<T, FnPeekColorMatrix>::Type_t
  operator()(const T &a) const
  {
    return (peekColor(a,row,col));
  }

  int getRow() const { return row; }
  int getCol() const { return col; }

private:
  int row, col;
};



//! Extract color matrix components
/*! @ingroup group1
  @relates QDPType */
template<class T1,class C1>
inline typename MakeReturn<UnaryNode<FnPeekColorMatrix,
  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>,
  typename UnaryReturn<C1,FnPeekColorMatrix >::Type_t >::Expression_t
peekColor(const QDPType<T1,C1> & l, int row, int col)
{
  typedef UnaryNode<FnPeekColorMatrix,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekColorMatrix >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekColorMatrix(row,col),
    CreateLeaf<QDPType<T1,C1> >::make(l)));
}


#if 0
template<class T1,class C1>
inline typename MakeReturn<UnaryNode<FnPeekColorMatrix,
  typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
peekColor(const QDPExpr<T1,C1> & l, int row, int col)
{
  typedef UnaryNode<FnPeekColorMatrix, 
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekColorMatrix >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekColorMatrix(row,col),
    CreateLeaf<QDPExpr<T1,C1> >::make(l)));
}
#endif
  

//! Structure for extracting color vector components
struct FnPeekColorVector
{
  PETE_EMPTY_CONSTRUCTORS(FnPeekColorVector)

  FnPeekColorVector(int _row): row(_row) {}
  
  template<class T>
  inline typename UnaryReturn<T, FnPeekColorVector>::Type_t
  operator()(const T &a) const
  {
    return (peekColor(a,row));
  }

  int getRow() const { return row; }
  //int getCol() const { return col; }

private:
  int row;
};


//! Extract color vector components
/*! @ingroup group1
  @relates QDPType */
template<class T1,class C1>
inline typename MakeReturn<UnaryNode<FnPeekColorVector,
  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>,
  typename UnaryReturn<C1,FnPeekColorVector >::Type_t >::Expression_t
peekColor(const QDPType<T1,C1> & l, int row)
{
  typedef UnaryNode<FnPeekColorVector,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekColorVector >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekColorVector(row),
    CreateLeaf<QDPType<T1,C1> >::make(l)));
}


#if 0
template<class T1,class C1>
inline typename MakeReturn<UnaryNode<FnPeekColorVector,
  typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
peekColor(const QDPExpr<T1,C1> & l, int row)
{
  typedef UnaryNode<FnPeekColorVector,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekColorVector >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekColorVector(row),
    CreateLeaf<QDPExpr<T1,C1> >::make(l)));
}
#endif

//! Structure for extracting spin matrix components
struct FnPeekSpinMatrix
{
  PETE_EMPTY_CONSTRUCTORS(FnPeekSpinMatrix)

  FnPeekSpinMatrix(int _row, int _col): row(_row), col(_col) {}
  
  template<class T>
  inline typename UnaryReturn<T, FnPeekSpinMatrix>::Type_t
  operator()(const T &a) const
  {
    return (peekSpin(a,row,col));
  }

  int getRow() const { return row; }
  int getCol() const { return col; }

private:
  int row, col;
};

//! Extract spin matrix components
/*! @ingroup group1
  @relates QDPType */
template<class T1,class C1>
inline typename MakeReturn<UnaryNode<FnPeekSpinMatrix,
  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>,
  typename UnaryReturn<C1,FnPeekSpinMatrix >::Type_t >::Expression_t
peekSpin(const QDPType<T1,C1> & l, int row, int col)
{
  typedef UnaryNode<FnPeekSpinMatrix,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekSpinMatrix >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekSpinMatrix(row,col),
    CreateLeaf<QDPType<T1,C1> >::make(l)));
}

#if 0
template<class T1,class C1>
inline typename MakeReturn<UnaryNode<FnPeekSpinMatrix,
  typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
peekSpin(const QDPExpr<T1,C1> & l, int row, int col)
{
  typedef UnaryNode<FnPeekSpinMatrix,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekSpinMatrix >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekSpinMatrix(row,col),
    CreateLeaf<QDPExpr<T1,C1> >::make(l)));
}
#endif

//! Structure for extracting spin vector components
struct FnPeekSpinVector
{
  PETE_EMPTY_CONSTRUCTORS(FnPeekSpinVector)

  FnPeekSpinVector(int _row): row(_row) {}
  
  template<class T>
  inline typename UnaryReturn<T, FnPeekSpinVector>::Type_t
  operator()(const T &a) const
  {
    return (peekSpin(a,row));
  }

  int getRow() const { return row; }
  //  int getCol() const { return col; }

private:
  int row;
};


//! Extract spin vector components
/*! @ingroup group1
  @relates QDPType */
template<class T1,class C1>
inline typename MakeReturn<UnaryNode<FnPeekSpinVector,
  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>,
  typename UnaryReturn<C1,FnPeekSpinVector >::Type_t >::Expression_t
peekSpin(const QDPType<T1,C1> & l, int row)
{
  typedef UnaryNode<FnPeekSpinVector,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekSpinVector >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekSpinVector(row),
    CreateLeaf<QDPType<T1,C1> >::make(l)));
}


#if 0
template<class T1,class C1>
inline typename MakeReturn<UnaryNode<FnPeekSpinVector,
  typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
peekSpin(const QDPExpr<T1,C1> & l, int row)
{
  typedef UnaryNode<FnPeekSpinVector,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekSpinVector >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekSpinVector(row),
    CreateLeaf<QDPExpr<T1,C1> >::make(l)));
}
#endif



//---------------------------------------
//! Structure for inserting color matrix components
struct FnPokeColorMatrix
{
  PETE_EMPTY_CONSTRUCTORS(FnPokeColorMatrix)

  FnPokeColorMatrix(int _row, int _col): row(_row), col(_col) {}
  
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, FnPokeColorMatrix>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeColor(const_cast<T1&>(a),b,row,col);
    return const_cast<T1&>(a);
  }

  int getRow() const { return row; }
  int getCol() const { return col; }

private:
  int row, col;
};


//! Insert color matrix components
/*! @ingroup group1
  @param l  target to update
  @param r  source
  @param row  row of color matrix
  @param col  column of color matrix
  @return updated target
  @ingroup group1
  @relates QDPType */
template<class T1,class C1,class T2,class C2>
inline C1& 
pokeColor(QDPType<T1,C1> & l, const QDPType<T2,C2>& r, int row, int col)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeColorMatrix(row,col),PETE_identity(r),all);
  return ll;
}


template<class T1,class C1,class T2,class C2>
inline C1& 
pokeColor(QDPType<T1,C1> & l, const QDPExpr<T2,C2>& r, int row, int col)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeColorMatrix(row,col),r,all);
  return ll;
}


template<class T1,class C1,class T2,class C2>
inline C1
pokeColor(const QDPSubType<T1,C1>& l, const QDPType<T2,C2>& r, int row, int col)
{
  //C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  C1 ll( l.getId() , 1.0 );
  const Subset& s = l.subset();

  evaluate(ll,FnPokeColorMatrix(row,col),PETE_identity(r),s);
  return ll;
}


template<class T1,class C1,class T2,class C2>
inline C1
pokeColor(const QDPSubType<T1,C1>& l, const QDPExpr<T2,C2>& r, int row, int col)
{
  //C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  C1 ll( l.getId() , 1.0 );
  const Subset& s = l.subset();

  evaluate(ll,FnPokeColorMatrix(row,col),r,s);
  return ll;
}


//! Structure for inserting color vector components
struct FnPokeColorVector
{
  PETE_EMPTY_CONSTRUCTORS(FnPokeColorVector)

  FnPokeColorVector(int _row): row(_row) {}
  
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, FnPokeColorVector>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeColor(const_cast<T1&>(a),b,row);
    return const_cast<T1&>(a);
  }

  int getRow() const { return row; }
  //  int getCol() const { return col; }

private:
  int row;
};



//! Insert color vector components
/*! @ingroup group1
  @param l  target to update
  @param r  source
  @param row  row of color vector
  @return updated target
  @ingroup group1
  @relates QDPType */
template<class T1,class C1,class T2,class C2>
inline C1& 
pokeColor(QDPType<T1,C1>& l, const QDPType<T2,C2>& r, int row)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeColorVector(row),PETE_identity(r),all);
  return ll;
}

template<class T1,class C1,class T2,class C2>
inline C1& 
pokeColor(QDPType<T1,C1>& l, const QDPExpr<T2,C2>& r, int row)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeColorVector(row),r,all);
  return ll;
}


template<class T1,class C1,class T2,class C2>
inline C1
pokeColor(const QDPSubType<T1,C1>& l, const QDPType<T2,C2>& r, int row)
{
  //C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  C1 ll( l.getId() , 1.0 );

  const Subset& s = l.subset();

  evaluate(ll,FnPokeColorVector(row),PETE_identity(r),s);
  return ll;
}


template<class T1,class C1,class T2,class C2>
inline C1
pokeColor(const QDPSubType<T1,C1>& l, const QDPExpr<T2,C2>& r, int row)
{
  //C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  C1 ll( l.getId() , 1.0 );

  const Subset& s = l.subset();

  evaluate(ll,FnPokeColorVector(row),r,s);
  return ll;
}


//! Structure for inserting spin matrix components
struct FnPokeSpinMatrix
{
  PETE_EMPTY_CONSTRUCTORS(FnPokeSpinMatrix)

  FnPokeSpinMatrix(int _row, int _col): row(_row), col(_col) {}
  
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, FnPokeSpinMatrix>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeSpin(const_cast<T1&>(a),b,row,col);
    return const_cast<T1&>(a);
  }

  int getRow() const { return row; }
  int getCol() const { return col; }

private:
  int row, col;
};

//! Insert spin matrix components
/*! @ingroup group1
  @param l  target to update
  @param r  source
  @param row  row of spin matrix
  @param col  column of spin matrix
  @return updated target
  @ingroup group1
  @relates QDPType */
template<class T1,class C1,class T2,class C2>
inline C1& 
pokeSpin(QDPType<T1,C1> & l, const QDPType<T2,C2>& r, int row, int col)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeSpinMatrix(row,col),PETE_identity(r),all);
  return ll;
}


template<class T1,class C1,class T2,class C2>
inline C1& 
pokeSpin(QDPType<T1,C1> & l, const QDPExpr<T2,C2>& r, int row, int col)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeSpinMatrix(row,col),r,all);
  return ll;
}


template<class T1,class C1,class T2,class C2>
inline C1
pokeSpin(const QDPSubType<T1,C1>& l, const QDPType<T2,C2>& r, int row, int col)
{
  //C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  C1 ll( l.getId() , 1.0 );

  const Subset& s = l.subset();

  evaluate(ll,FnPokeSpinMatrix(row,col),PETE_identity(r),s);
  return ll;
}


template<class T1,class C1,class T2,class C2>
inline C1
pokeSpin(const QDPSubType<T1,C1>& l, const QDPExpr<T2,C2>& r, int row, int col)
{
  //C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  C1 ll( l.getId() , 1.0 );

  const Subset& s = l.subset();

  evaluate(ll,FnPokeSpinMatrix(row,col),r,s);
  return ll;
}



//! Structure for inserting spin vector components
struct FnPokeSpinVector
{
  PETE_EMPTY_CONSTRUCTORS(FnPokeSpinVector)

  FnPokeSpinVector(int _row): row(_row) {}
  
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, FnPokeSpinVector>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeSpin(const_cast<T1&>(a),b,row);
    return const_cast<T1&>(a);
  }

  int getRow() const { return row; }
  //  int getCol() const { return col; }

private:
  int row;
};


//! Insert spin vector components
/*! @ingroup group1
  @param l  target to update
  @param r  source
  @param row  row of spin vector
  @return updated target
  @ingroup group1
  @relates QDPType */
template<class T1,class C1,class T2,class C2>
inline C1& 
pokeSpin(QDPType<T1,C1>& l, const QDPType<T2,C2>& r, int row)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeSpinVector(row),PETE_identity(r),all);
  return ll;
}

template<class T1,class C1,class T2,class C2>
inline C1& 
pokeSpin(QDPType<T1,C1>& l, const QDPExpr<T2,C2>& r, int row)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeSpinVector(row),r,all);
  return ll;
}


template<class T1,class C1,class T2,class C2>
inline C1
pokeSpin(const QDPSubType<T1,C1>& l, const QDPType<T2,C2>& r, int row)
{
  //C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  C1 ll( l.getId() , 1.0 );

  const Subset& s = l.subset();

  evaluate(ll,FnPokeSpinVector(row),PETE_identity(r),s);
  return ll;
}


template<class T1,class C1,class T2,class C2>
inline C1
pokeSpin(const QDPSubType<T1,C1>& l, const QDPExpr<T2,C2>& r, int row)
{
  //C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  C1 ll( l.getId() , 1.0 );

  const Subset& s = l.subset();

  evaluate(ll,FnPokeSpinVector(row),r,s);
  return ll;
}



//-----------------------------------------------------------------------------
// Additional operator tags 
//-----------------------------------------------------------------------------



#if 0
// Explicit casts
template<class T1,class T2,class C2>
inline typename MakeReturn<UnaryNode<OpCast<T1>,
  typename CreateLeaf<QDPType<T2,C2> >::Leaf_t>,
  typename UnaryReturn<C2,OpCast<T1> >::Type_t>::Expression_t
peteCast(const T1&, const QDPType<T2,C2>& l)
{
  typedef UnaryNode<OpCast<T1>,
    typename CreateLeaf<QDPType<T2,C2> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C2,OpCast<T1> >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPType<T2,C2> >::make(l)));
}
#endif

} // namespace QDP

#endif
