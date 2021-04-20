// -*- C++ -*-
/*! @file
 * @brief QDPType after a subset
 *
 * Subclass of QDPType used for subset operations
 */

#ifndef QDP_QDPSUBTYPE_H
#define QDP_QDPSUBTYPE_H

namespace QDP {


//! QDPSubType - type representing a field living on a subset
/*! 
 * This class is meant to be an auxilliary class used only for
 * things like lvalues - left hand side of expressions, arguments
 * to calls that modify the source (like RNG), etc.
 */
template<class T, class C> 
class QDPSubType
{
  //! This is a type name like OSubLattice<T> or OSubScalar<T>
  typedef typename QDPSubTypeTrait<C>::Type_t CC;

public:
  //! Type of the first argument
  typedef T Subtype_t;

  //! Type of the container class
  typedef C Container_t;

  //! Default constructor 
  QDPSubType() {}

  //! Copy constructor
  QDPSubType(const QDPSubType&) {}

  //! Destructor
  ~QDPSubType() {}


  //---------------------------------------------------------
  // Operators

  inline
  void assign(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpAssign(),PETE_identity(Scalar_t(rhs)),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }
  }

  inline
  void assign(const Zero&)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      zero_rep_subtype(*me,subset());
    } else {
      C tmp(getId(),1.0);
      zero_rep(tmp,subset());
    }
  }

  template<class T1,class C1>
  inline
  void assign(const QDPType<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpAssign(),PETE_identity(rhs),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpAssign(),PETE_identity(rhs),subset());
    }
  }


  template<class T1>
  inline
  //void assign(const QDPSubType<T1,C1>& rhs)
  void assign(const OSubLattice<T1>& rhs)
  {
    if (getOwnsMemory() && rhs.getOwnsMemory()) {
      if (subset().numSiteTable() != rhs.subset().numSiteTable())
	QDP_error_exit("assignment with incompatible subset sizes");
      CC* me = static_cast<CC*>(this);
      operator_subtype_subtype(*me,OpAssign(),rhs,subset());
    } else
#if 0
    if (!getOwnsMemory() && rhs.getOwnsMemory()) {
      //std::cout << "view = own\n";
      if (subset().numSiteTable() != rhs.subset().numSiteTable())
	QDP_error_exit("assignment with incompatible subset sizes");
      const int *tab = subset().siteTable().slice();
      for(int j=0; j < subset().numSiteTable(); ++j) {
	int i = tab[j];
	getId()[i] = rhs.getId()[j];
      }
    }
    if (getOwnsMemory() && !rhs.getOwnsMemory()) {
      //std::cout << "own = view\n";
      if (subset().numSiteTable() != rhs.subset().numSiteTable())
	QDP_error_exit("assignment with incompatible subset sizes");
      const int *tab = rhs.subset().siteTable().slice();
      for(int j=0; j < rhs.subset().numSiteTable(); ++j) {
	int i = tab[j];
	getId()[j] = rhs.getId()[i];
      }
    }
    if (!getOwnsMemory() && !rhs.getOwnsMemory())
#endif
      QDP_error_exit("assignment of two view subtypes is not supported");
  }


  template<class T1,class C1>
  inline
  void assign(const QDPExpr<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpAssign(),rhs,subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpAssign(),rhs,subset());
    }
  }

  inline
  void operator+=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpAddAssign(),PETE_identity(Scalar_t(rhs)),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpAddAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator+=(const QDPType<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpAddAssign(),PETE_identity(rhs),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpAddAssign(),PETE_identity(rhs),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator+=(const QDPExpr<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpAddAssign(),rhs,subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpAddAssign(),rhs,subset());
    }
  }

  inline
  void operator-=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpSubtractAssign(),PETE_identity(Scalar_t(rhs)),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpSubtractAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator-=(const QDPType<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpSubtractAssign(),PETE_identity(rhs),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpSubtractAssign(),PETE_identity(rhs),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator-=(const QDPExpr<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpSubtractAssign(),rhs,subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpSubtractAssign(),rhs,subset());
    }
  }

  inline
  void operator*=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpMultiplyAssign(),PETE_identity(Scalar_t(rhs)),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpMultiplyAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator*=(const QDPType<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpMultiplyAssign(),PETE_identity(rhs),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpMultiplyAssign(),PETE_identity(rhs),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator*=(const QDPExpr<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpMultiplyAssign(),rhs,subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpMultiplyAssign(),rhs,subset());
    }
  }

  inline
  void operator/=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpDivideAssign(),PETE_identity(Scalar_t(rhs)),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpDivideAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator/=(const QDPType<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpDivideAssign(),PETE_identity(rhs),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpDivideAssign(),PETE_identity(rhs),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator/=(const QDPExpr<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpDivideAssign(),rhs,subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpDivideAssign(),rhs,subset());
    }
  }

  inline
  void operator%=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpModAssign(),PETE_identity(Scalar_t(rhs)),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpModAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator%=(const QDPType<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpModAssign(),PETE_identity(rhs),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpModAssign(),PETE_identity(rhs),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator%=(const QDPExpr<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpModAssign(),rhs,subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpModAssign(),rhs,subset());
    }
  }

  inline
  void operator|=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpBitwiseOrAssign(),PETE_identity(Scalar_t(rhs)),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpBitwiseOrAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator|=(const QDPType<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpBitwiseOrAssign(),PETE_identity(rhs),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpBitwiseOrAssign(),PETE_identity(rhs),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator|=(const QDPExpr<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpBitwiseOrAssign(),PETE_identity(rhs),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpBitwiseOrAssign(),PETE_identity(rhs),subset());
    }
  }

  inline
  void operator&=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpBitwiseAndAssign(),PETE_identity(Scalar_t(rhs)),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpBitwiseAndAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator&=(const QDPType<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpBitwiseAndAssign(),PETE_identity(rhs),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpBitwiseAndAssign(),PETE_identity(rhs),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator&=(const QDPExpr<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpBitwiseAndAssign(),rhs,subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpBitwiseAndAssign(),rhs,subset());
    }
  }

  inline
  void operator^=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpBitwiseXorAssign(),PETE_identity(Scalar_t(rhs)),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpBitwiseXorAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator^=(const QDPType<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpBitwiseXorAssign(),PETE_identity(rhs),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpBitwiseXorAssign(),PETE_identity(rhs),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator^=(const QDPExpr<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpBitwiseXorAssign(),rhs,subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpBitwiseXorAssign(),rhs,subset());
    }
  }

  inline
  void operator<<=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpLeftShiftAssign(),PETE_identity(Scalar_t(rhs)),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpLeftShiftAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator<<=(const QDPType<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpLeftShiftAssign(),PETE_identity(rhs),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpLeftShiftAssign(),PETE_identity(rhs),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator<<=(const QDPExpr<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpLeftShiftAssign(),rhs,subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpLeftShiftAssign(),rhs,subset());
    }
  }


  inline
  void operator>>=(const typename WordType<C>::Type_t& rhs)
  {
    typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpRightShiftAssign(),PETE_identity(Scalar_t(rhs)),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpRightShiftAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator>>=(const QDPType<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpRightShiftAssign(),PETE_identity(rhs),subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpRightShiftAssign(),PETE_identity(rhs),subset());
    }
  }

  template<class T1,class C1>
  inline
  void operator>>=(const QDPExpr<T1,C1>& rhs)
  {
    if (getOwnsMemory()) {
      CC* me = static_cast<CC*>(this);
      evaluate_subtype_type(*me,OpRightShiftAssign(),rhs,subset());
    } else {
      C tmp(getId(),1.0);
      evaluate(tmp,OpRightShiftAssign(),rhs,subset());
    }
  }

private:
  //! Hide default operator=
  inline
  C& operator=(const QDPSubType& rhs) {}

public:
  //C& field() {return static_cast<CC*>(this)->field();}
  bool getOwnsMemory() { return static_cast<CC*>(this)->getOwnsMemory(); }
  bool getOwnsMemory() const { return static_cast<const CC*>(this)->getOwnsMemory(); }

  int getId() const   { return static_cast<const CC*>(this)->getId();}

  T* getF() {return static_cast<CC*>(this)->getF();}
  T* getF() const {return static_cast<CC const *>(this)->getF();}
  const Subset& subset() const {return static_cast<const CC*>(this)->subset();}

};



  
template<class T>
struct LeafFunctor<QDPSubType<T,OLattice<T> >, ParamLeaf>
{
  typedef QDPSubTypeJIT<typename JITType<T>::Type_t,typename JITType<OLattice<T> >::Type_t>  TypeA_t;
  //typedef typename JITType< OLattice<T> >::Type_t  TypeA_t;
  typedef TypeA_t  Type_t;
  inline static Type_t apply(const QDPSubType<T,OLattice<T> > &a, const ParamLeaf& p)
  {
    ParamRef    base_addr = llvm_add_param< typename WordType<T>::Type_t * >();
    return Type_t( base_addr );
  }
};


#if 0
template<class T>
struct LeafFunctor<QDPSubType<T,OScalar<T> >, ParamLeaf>
{
  typedef QDPSubTypeJIT<typename JITType<T>::Type_t,typename JITType<OScalar<T> >::Type_t>  TypeA_t;
  //typedef typename JITType< OScalar<T> >::Type_t  TypeA_t;
  typedef TypeA_t  Type_t;
  inline static Type_t apply(const QDPSubType<T,OScalar<T> > &a, const ParamLeaf& p)
  {
    ParamRef    base_addr = llvm_add_param< typename WordType<T>::Type_t * >();
    return Type_t( base_addr );
  }
};
#endif


template<class T, class C>
struct LeafFunctor<QDPSubType<T,C>, AddressLeaf>
{
  typedef int Type_t;
  inline static
  Type_t apply(const QDPSubType<T,C>& s, const AddressLeaf& p) 
  {
    //p.setAddr( QDP_get_global_cache().getDevicePtr( s.getId() ) );
    p.setId( s.getId() );
    return 0;
  }
};

  

} // namespace QDP

#endif



