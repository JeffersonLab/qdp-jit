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

#if 0
template<class T>
class OSubScalar: public QDPSubType<T, OScalar<T> >
{
  typedef OScalar<T> C;

public:
  //OSubScalar(OScalar<T>& a, const Subset& ss): F(a.getF()), s(ss) {}
  OSubScalar(OScalar<T>& a, const Subset& ss): F(a.getF()), s(&(const_cast<Subset&>(ss))) {}
  OSubScalar(const OSubScalar& a): F(a.F), s(a.s) {}
  ~OSubScalar() {}

  //---------------------------------------------------------
  // Operators
  // NOTE: all assignment-like operators except operator= are
  // inherited from QDPSubType

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
  void operator=(const OSubScalar& rhs)
    {
      this->assign(rhs);
    }


private:
  // Hide default constructor
  OSubScalar() {}

public:
  T* getF() {return F;}
  T* getF() const {return F;}
  const Subset& subset() const {return *s;}
  bool getOwnsMemory() const { return ownsMemory; }
  bool getOwnsMemory() { return ownsMemory; }

private:
  T* F;
  bool ownsMemory;
  Subset* s;

  //const Subset& s;
};
#endif
  


// not sure about this
#if 0
template<class T, class T1, class Op, class RHS>
void evaluate(OSubScalar<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs, const Subset& s)
{
  // Subset is not used at this level. It may be needed, though, within an inner operation
  op(dest.elem(), forEach(rhs, ElemLeaf(), OpCombine()));
}
#endif


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
  //OSubLattice(OLattice<T>& a, const Subset& ss): F(a.getF()), s(ss), ownsMemory(false) {}

  OSubLattice(OLattice<T>& a, const Subset& ss): myId(a.getId()), ownsMemory(false), s(&(const_cast<Subset&>(ss))) {}
  OSubLattice(const OSubLattice& a): QDPSubType<T, OLattice<T> >() {
    ownsMemory = a.ownsMemory; 
    s = a.s;
    if (a.ownsMemory) {
      QDPIO::cout << "OSubLattice copy ctor, must copy   subset size = " << s->numSiteTable() << "\n";
      alloc_mem();
      T* Fh = getF();
      T* Fa = a.getF();
      for( int i = 0 ; i < s->numSiteTable() ; ++i )
	Fh[i] = Fa[i];
    } else {
      myId = a.getId();
    }
  }


#if 1
  OSubLattice(): QDPSubType<T, OLattice<T> >(), ownsMemory(false), s(NULL)  {}


  void setSubset( const Subset& ss ) {
    //std::cout << "OSubLattice::setSubset was id = " << myId << "   subset size = " << ss.numSiteTable() << "\n";
    if (myId >= 0 && !ownsMemory)
      QDP_error_exit("You try to set the subset on an OSubLattice that is a view of an OLattice!");
    if (myId >= 0) {
      free_mem();
    }
    s = &(const_cast<Subset&>(ss));
    alloc_mem();
    ownsMemory = true;
    //QDPIO::cout << "subset set on OSubLattice\n";
    //std::cout << "OSubLattice::setSubset now id = " << myId << "\n";
  }


  OSubLattice(const Subset& ss , OLattice<T>& a): ownsMemory(true), s(&(const_cast<Subset&>(ss)))  {
    alloc_mem();
    const int *tab = s->siteTable().slice();
    T* Fh = getF();
    QDPIO::cout << "OSubLattice ctor(subset,lat), must copy (on CPU)\n";
    for( int j = 0 ; j < s->numSiteTable() ; ++j ) {
      int i = tab[j];
      Fh[j] = a.elem(i);
    }
  }


  void free_mem()  { 
    if (myId >= 0)
      QDP_get_global_cache().signoff( myId ); 
  }


  void alloc_mem() {
    //QDP_info("OSubLattice alloc for %d sites",s->numSiteTable());
    if (s->numSiteTable() > 0)
      myId = QDP_get_global_cache().registrate( sizeof(T) * s->numSiteTable() , 1 , NULL ); 
  }

  ~OSubLattice() {
    if (ownsMemory) {
      free_mem();
    }
  }

  // void free_mem() {
  // 	  QDP::Allocator::theQDPAllocator::Instance().free(F);
  // }
#endif




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

  inline T* getF() const { 
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


private:
  // Hide default constructor
  //OSubLattice() {}

public:
  bool getOwnsMemory() const { return ownsMemory; }
  bool getOwnsMemory() { return ownsMemory; }
  // T* getF() {return F;}
  // T* getF() const {return F;}
  //const Subset& subset() const {return s;}
  const Subset& subset() const {return *s;}

  int getId() const {
    return myId;
  }

private:
  mutable T* F_private;
  int myId = -2;
  bool ownsMemory;
  Subset* s;

  //C&      F;
  //const Subset& s;
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
struct JITType<OSubScalar<T> >
{
  typedef OSubScalarJIT<typename JITType<T>::Type_t>  Type_t;
};




  template<class T>
  struct LeafFunctor<OSubLattice<T>, ParamLeaf>
{
  typedef typename JITType< OSubLattice<T> >::Type_t  TypeA_t;
  typedef TypeA_t  Type_t;
  inline static
  Type_t apply(const OSubLattice<T>& do_not_use, const ParamLeaf& p) 
  {
    ParamRef    base_addr = llvm_add_param< typename WordType<T>::Type_t * >();
    return Type_t( base_addr );
  }
};

#if 0  
template<class T>
struct LeafFunctor<OScalar<T>, ParamLeaf>
{
  typedef typename JITType< OScalar<T> >::Type_t  TypeA_t;
  typedef TypeA_t  Type_t;
  inline static
  Type_t apply(const OScalar<T>& do_not_use, const ParamLeaf& p) 
  {
    ParamRef    base_addr = llvm_add_param< typename WordType<T>::Type_t * >();
    return Type_t( base_addr );
  }
};
#endif


template<class T>
struct LeafFunctor<OSubLattice<T>, AddressLeaf>
{
  typedef int Type_t;
  inline static
  Type_t apply(const OSubLattice<T>& s, const AddressLeaf& p) 
  {
    p.setAddr( QDP_get_global_cache().getDevicePtr( s.getId() ) );
    return 0;
  }
};

#if 0
template<class T>
struct LeafFunctor<OScalar<T>, AddressLeaf>
{
  typedef int Type_t;
  inline static
  Type_t apply(const OScalar<T>& s, const AddressLeaf& p) 
  {
    p.setAddr( QDP_get_global_cache().getDevicePtr( s.getId() ) );
    return 0;
  }
};
#endif


  

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

//! dest = (mask) ? s1 : dest
template<class T1, class T2> 
void copymask(OSubScalar<T2> dest, const OScalar<T1>& mask, 
	      const OScalar<T2>& s1) 
{
  copymask(dest.field().elem(), mask.elem(), s1.elem());
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
