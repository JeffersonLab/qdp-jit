// -*- C++ -*-
/*! \file
 * \brief Outer grid classes after a subset
 */


#ifndef QDP_OUTERSUBTYPE_H
#define QDP_OUTERSUBTYPE_H

#include "qdp_allocator.h"
namespace QDP {

//! OScalar class narrowed to a subset
/*! 
 * Only used for lvalues
 */

  


//-------------------------------------------------------------------------------------
//! OLattice class narrowed to a subset
/*! 
 * Only used for lvalues
 */
template<class T> 
class OSubLattice: public QDPSubType<T, OLattice<T> >
{
  typedef OLattice<T> C;
  
public:
  OSubLattice(OLattice<T>& a, const Subset& ss): myId(a.getId()), ownsMemory(false), s(&(const_cast<Subset&>(ss))) {}
  
  OSubLattice(const OSubLattice& a): QDPSubType<T, OLattice<T> >() {
    ownsMemory = a.ownsMemory;
    s = a.s;
    if (a.ownsMemory) {
      alloc_mem();
      operator_subtype_subtype(*this,OpAssign(),a,subset());
    } else {
      myId = a.getId();
    }
  }


  OSubLattice(): QDPSubType<T, OLattice<T> >(), ownsMemory(false), s(NULL)  {}


  void setSubset( const Subset& ss ) {
    if (myId >= 0 && !ownsMemory)
      QDP_error_exit("You try to set the subset on an OSubLattice that is a view of an OLattice!");
    if (myId >= 0) {
      free_mem();
    }
    s = &(const_cast<Subset&>(ss));
    alloc_mem();
    ownsMemory = true;
  }


  OSubLattice(const Subset& ss , OLattice<T>& a): ownsMemory(true), s(&(const_cast<Subset&>(ss)))  {
    alloc_mem();
    if (QDP_get_global_cache().isOnDevice( a.getId() ))
      {
	evaluate_subtype_type(*this,OpAssign(),PETE_identity(a),subset());
      }
    else
      {
	// here we have an optimization opportunity

	if (ss.numSiteTable())
	  {
	    typename ScalarType<T>::Type_t* aF = a.getF();
	    typename ScalarType<T>::Type_t* thisF = this->getF();

	    const int *tab = ss.siteTable().slice();
	    for(int n=0; n < ss.numSiteTable(); ++n)
	      {
		thisF[ n ] = aF[ tab[n] ];
	      }
	  }
      }
  }


  void free_mem()  { 
    if (myId >= 0)
      QDP_get_global_cache().signoff( myId ); 
  }


  void alloc_mem() {
    if (s->numSiteTable() > 0)
      myId = QDP_get_global_cache().add( sizeof(T) * s->numSiteTable() ); 
  }

  ~OSubLattice() {
    if (ownsMemory) {
      free_mem();
    }
  }


  //---------------------------------------------------------
  // Operators
  // NOTE: all assignment-like operators except operator= are
  // inherited from QDPType

  inline
  void operator=(const typename WordType<T>::Type_t& rhs)
  {
    this->assign(rhs);
  }

  inline
  void operator=(const Zero& rhs)
  {
    this->assign(rhs);
  }

  template<class T1,class C1>
  inline
  void operator=(const QDPType<T1,C1>& rhs)
  {
    this->assign(rhs);
  }

  template<class T1,class C1>
  inline
  void operator=(const QDPExpr<T1,C1>& rhs)
  {
    this->assign(rhs);
  }


  inline
  void operator=(const OSubLattice& rhs)
  {
    this->assign(rhs);
  }

  inline void assert_on_host() const {
    // Here or somewhere we sould make sure that 
    // if the pointer is still valid, we do not too much
    QDP_get_global_cache().getHostPtr( (void**)&F_private , myId );
  }

  inline typename ScalarType<T>::Type_t* getF() const { 
    assert_on_host(); 
    return F_private; 
  }

  inline T& elem(int i) { 
    assert_on_host(); 
    return F_private[i]; 
  }
  
  inline const T& elem(int i) const { 
    assert_on_host(); 
    return F_private[i]; 
  }


public:
  bool getOwnsMemory() const { return ownsMemory; }
  bool getOwnsMemory() { return ownsMemory; }
  const Subset& subset() const {return *s;}

  int getId() const {
    return myId;
  }

private:
  mutable typename ScalarType<T>::Type_t* F_private;
  int myId = -2;
  bool ownsMemory;
  Subset* s;
};


//-----------------------------------------------------------------------------
// Traits class for returning the subset-ted class name of a outer grid class
//-----------------------------------------------------------------------------

template<class T>
struct QDPSubTypeTrait<OScalar<T> > 
{
  typedef OSubScalar<T>  Type_t;
};


template<class T>
struct QDPSubTypeTrait<OLattice<T> > 
{
  typedef OSubLattice<T>  Type_t;
};


//-----------------------------------------------------------------------------
// Traits classes to support operations of simple scalars (floating constants, 
// etc.) on QDPTypes
//-----------------------------------------------------------------------------

template<class T>
struct WordType<OSubScalar<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};


template<class T>
struct WordType<OSubLattice<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};


// ------------------------------------------------------------
// Get Single Precision Types of OuterSubType templates
// ------------------------------------------------------------
template<class T>
struct SinglePrecType<OSubScalar<T> > 
{
  typedef OSubScalar<typename SinglePrecType<T>::Type_t>  Type_t;
};


template<class T>
struct SinglePrecType<OSubLattice<T> > 
{
  typedef OSubLattice<typename SinglePrecType<T>::Type_t>  Type_t;
};


// ------------------------------------------------------------
// Get Single Precision Types of OuterSubType templates
// ------------------------------------------------------------
template<class T>
struct DoublePrecType<OSubScalar<T> > 
{
  typedef OSubScalar<typename DoublePrecType<T>::Type_t>  Type_t;
};

template<class T>
struct DoublePrecType<OSubLattice<T> > 
{
  typedef OSubLattice<typename DoublePrecType<T>::Type_t>  Type_t;
};



  

template<class T> 
struct JITType<OSubLattice<T> >
{
  typedef OSubLatticeJIT<typename JITType<T>::Type_t>  Type_t;
};




template<class T>
struct LeafFunctor<OSubLattice<T>, ParamLeaf>
{
  typedef typename JITType< OSubLattice<T> >::Type_t  Type_t;
  inline static
  Type_t apply(const OSubLattice<T>& do_not_use, const ParamLeaf& p) 
  {
    ParamRef    base_addr = llvm_add_param< typename WordType<T>::Type_t * >();
    return Type_t( base_addr );
  }
};

template<class T>
struct LeafFunctor<OSubLattice<T>, ParamLeafScalar>
{
  typedef typename JITType< OSubLattice< typename ScalarType<T>::Type_t > >::Type_t  Type_t;
  inline static
  Type_t apply(const OSubLattice<T>& do_not_use, const ParamLeafScalar& p) 
  {
    ParamRef    base_addr = llvm_add_param< typename WordType<T>::Type_t * >();
    return Type_t( base_addr );
  }
};





template<class T>
struct LeafFunctor<OSubLattice<T>, AddressLeaf>
{
  typedef int Type_t;
  inline static
  Type_t apply(const OSubLattice<T>& s, const AddressLeaf& p) 
  {
    p.setId( s.getId() );
    return 0;
  }
};


  

//-----------------------------------------------------------------------------
// Scalar Operations
//-----------------------------------------------------------------------------

//! dest = 0
template<class T> 
void zero_rep(OScalar<T>& dest, const Subset& s) 
{
  zero_rep(dest.field().elem());
}

//! dest = 0
template<class T>
void zero_rep(OSubScalar<T> dest) 
{
  zero_rep(dest.field().elem());
}



//-----------------------------------------------------------------------------
// Random numbers
//! dest  = random  
/*! Implementation is in the specific files */
template<class T>
void random(OSubScalar<T> d);

//! dest  = gaussian
template<class T>
void gaussian(OSubScalar<T> dd)
{
  OLattice<T>& d = dd.field();
  const Subset& s = dd.subset();

  OScalar<T>  r1, r2;

  random(r1(s));
  random(r2(s));

  fill_gaussian(d.elem(), r1.elem(), r2.elem());
}

} // namespace QDP

#endif
