// -*- C++ -*-

#ifndef QDP_OUTER_H
#define QDP_OUTER_H

#include "qdp_config.h"

/*! \file
 * \brief Outer grid classes
 */

namespace QDP {

/*! \defgroup fiberbundle Fiberbundle types and operations
 *
 * A fiberbundle is a base space (here a lattice or scalar) with a fiber
 * at each site. Mathematically, we represent a fiber as a tensor product
 * over various vector spaces that have some indices.
 * QDP constructs types via a nested composition of lattice and fiber types.
 */


/*! \addtogroup oscalar Outer grid scalar
 * \ingroup fiberbundle
 *
 * Outer grid scalar means sites are not the slowest varying index. There can
 * still be an Inner grid.
 *
 * @{
 */

  
  
//! Outer grid Scalar class */
/*! All outer lattices are of OScalar or OLattice type */
  template<class T>
  class OScalar: public QDPType<T, OScalar<T> >
  {
  public:
    typedef T SubType_t;
    
    OScalar(): QDPType<T, OScalar<T> >()
    {
#ifdef QDP_DEEP_LOG
      zero_rep( elem() );
#endif
    }

    virtual ~OScalar() {
      free_mem();
    }

    OScalar(const typename WordType<T>::Type_t& rhs)//: QDPType<T, OScalar<T> >()
    {
      typedef typename InternalScalar<T>::Type_t  Scalar_t;
      elem() = Scalar_t(rhs);
    }

    

    OScalar(const Zero& rhs)//: QDPType<T, OScalar<T> >()
    {
      this->assign(rhs);
    }


    template<class T1>
    OScalar(const OScalar<T1>& rhs)//: QDPType<T, OScalar<T> >()
    {
      this->assign(rhs);
    }


    template<class RHS, class T1>
    OScalar(const QDPExpr<RHS, OScalar<T1> >& rhs)//: QDPType<T, OScalar<T> >()
    {
      this->assign(rhs);
    }


    inline
    OScalar& operator=(const typename WordType<T>::Type_t& rhs)
    {
      return this->assign(rhs);
    }

    inline
    OScalar& operator=(const Zero& rhs)
    {
      return this->assign(rhs);
    }

    template<class T1,class C1>
    inline
    OScalar& operator=(const QDPType<T1,C1>& rhs)
    {
      return this->assign(rhs);
    }

    template<class T1,class C1>
    inline
    OScalar& operator=(const QDPExpr<T1,C1>& rhs)
    {
      return this->assign(rhs);
    }


    inline
    OScalar& operator=(const OScalar& rhs)
    {
      return this->assign(rhs);
    }

    OScalar(const OScalar& rhs)//: QDPType<T, OScalar<T> >()
    {
      this->assign(rhs);
    }

    
  public:

    inline T* get_raw_F() const {
      return &F;
    }
    
    inline T* getF() const {
      assert_on_host();
      return &F;
    };


    inline T& elem() {
      //std::cout << (void*)this << " elem() myId = " << myId << std::endl;
      assert_on_host();
      return F;
    }
    
    inline const T& elem() const {
      //std::cout << (void*)this << " elem() myId = " << myId << std::endl;
      assert_on_host();
      return F;
    }



    int getId() const {
      alloc_mem(); 
      return myId; 
    }

    int getElemNum() const {       
      return elem_num;
    }

    void setId(int id) {
      //std::cout << (void*)this << "set id to " << id << std::endl; 
      myId = id;
    }

    void setElemNum(int elemnum) {
      //std::cout << (void*)this << "set elem to " << elemnum << std::endl; 
      elem_num = elemnum;
    }

    bool accessed() const { return accessed_on_host; }


    typename WordType<T>::Type_t get_word_value() const
    {
      typename WordType<T>::Type_t tmp = FirstWord<T>::get(F);
      return tmp;
    }

  private:
    inline void alloc_mem() const {
      if ( myId >= 0 )
	return;

      QDPCache::Status status = accessed_on_host ? QDPCache::Status::host : QDPCache::Status::undef;

      myId = QDP_get_global_cache().add( sizeof(T) , QDPCache::Flags::OwnHostMemory , status , &F , NULL , NULL );
    }
    
    inline void free_mem() {
      if ( myId >= 0  &&  elem_num < 0 )
	QDP_get_global_cache().signoff( myId );
    }
    
    inline void assert_on_host() const {
      //std::cout << "Oscalar assert on host\n";
      
      accessed_on_host = true;
      
      if (myId < 0)
	return;

      if ( elem_num < 0 )
	QDP_get_global_cache().assureOnHost( myId );
      else
	QDP_get_global_cache().assureOnHost( myId , elem_num );
    }

    mutable int myId = -1;
    mutable bool accessed_on_host = false;
    mutable int elem_num = -1;
    mutable T  F;
  };


//! Ascii input
/*! Treat all istreams here like all nodes can read. To use specialized ones
 *  that can broadcast, use TextReader */
  
template<class T>
istream& operator>>(istream& s, OScalar<T>& d)
{
  return s >> d.elem();
}

  
//! Ascii output
/*! Treat all ostreams here like all nodes can write. To use specialized ones
 *  that can broadcast, use TextReader */
template<class T>
inline
ostream& operator<<(ostream& s, const OScalar<T>& d)
{
  return s << d.elem();
}

//! Text output
template<class RHS, class T1>
ostream& operator<<(ostream& s, const QDPExpr<RHS, OScalar<T1> >& l)
{
  typedef OScalar<T1> C1;
  return s << C1(l);
}

//! Text input
template<class T>
TextReader& operator>>(TextReader& txt, OScalar<T>& d)
{
  return txt >> d.elem();
}

//! Text input
template<class T>
StandardInputStream& operator>>(StandardInputStream& is, OScalar<T>& d)
{
  return is >> d.elem();
}

//! Text output
template<class T>
inline
TextWriter& operator<<(TextWriter& txt, const OScalar<T>& d)
{
  return txt << d.elem();
}

//! Text output
template<class T>
StandardOutputStream& operator<<(StandardOutputStream& s, const OScalar<T>& d)
{
  return s << d.elem();
}

//! Text output
template<class RHS, class T1>
StandardOutputStream& operator<<(StandardOutputStream& s, const QDPExpr<RHS, OScalar<T1> >& l)
{
  typedef OScalar<T1> C1;
  return s << C1(l);
}

#ifndef QDP_NO_LIBXML2
//! XML output
/*! Supports also having an inner grid */
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, const OScalar<T>& d)
{
  return xml << d.elem();
}

//! XML input
/*! Supports also having an inner grid */
template<class T>
inline
void read(XMLReader& xml, const string& path, OScalar<T>& d)
{
  read(xml, path, d.elem());
}
#endif

/*! @} */  // end of group oscalar


// Empty leaf functor tag
struct ElemLeaf
{
  inline ElemLeaf() { }
};





//! OScalar Op OScalar(Expression(source))
/*! 
 * OScalar Op Expression, where Op is some kind of binary operation 
 * involving the destination 
 */
#if 1
template<class T, class T1, class Op, class RHS>
inline
void evaluate(OScalar<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs,
	      const Subset& s)
{
  // Subset is not used at this level. It may be needed, though, within an inner operation
  op(dest.elem(), forEach(rhs, ElemLeaf(), OpCombine()));

#ifdef QDP_DEEP_LOG
  if (jit_config_deep_log())
    {
      gpu_deep_logger( dest.getF() , typeid(typename WordType<T>::Type_t).name() , sizeof(T) , 0 , 1 , __PRETTY_FUNCTION__ );
    }
#endif
  
}
#endif


//-------------------------------------------------------------------------------------
/*! \addtogroup olattice Lattice outer grid 
 * \ingroup fiberbundle
 *
 * Outer grid lattice means sites are slowest varying index. There can
 * still be an Inner grid.
 *
 * @{
 */

//! Outer grid Lattice type
/*! All outer lattices are of OScalar or OLattice type */
  template<class T> 
  class OLattice: public QDPType<T, OLattice<T> >
  {
  public:
    //typedef T Subtype;
    typedef T SubType_t;

    OLattice()
    {
      alloc_mem("create");
    }


    virtual ~OLattice()
    {
      free_mem();
    }

    
    OLattice( int Id , float f ): myId(Id), mem(false)
    {
    }


    template<class T1>
    OLattice(const OScalar<T1>& rhs)//: QDPType<T, OLattice<T> >()
    {
      alloc_mem("construct from OScalar");
      this->assign(rhs);
    }


    template<class T1>
    OLattice(const OLattice<T1>& rhs)//: QDPType<T, OLattice<T> >()
    {
      alloc_mem("construct from OLattice");
      this->assign(rhs);
    }


    template<class RHS, class T1>
    OLattice(const QDPExpr<RHS, OLattice<T1> >& rhs)//: QDPType<T, OLattice<T> >()
    {
      alloc_mem("construct from expr");
      this->assign(rhs);
    }


    OLattice(const typename WordType<T>::Type_t& rhs)//: QDPType<T, OLattice<T> >()
    {
      alloc_mem("construct from const");

      typedef OScalar<typename InternalScalar<T>::Type_t>  Scalar_t;
      this->assign(Scalar_t(rhs));
    }


    OLattice(const Zero& rhs)//: QDPType<T, OLattice<T> >()
    {
      alloc_mem("construct from zero");
      this->assign(rhs);
    }

  //---------------------------------------------------------
  // Operators
  // NOTE: all assignment-like operators except operator= are
  // inherited from QDPType

    inline
    OLattice& operator=(const typename WordType<T>::Type_t& rhs)
    {
      return this->assign(rhs);
    }

    inline
    OLattice& operator=(const Zero& rhs)
    {
      return this->assign(rhs);
    }

    template<class T1,class C1>
    inline
    OLattice& operator=(const QDPType<T1,C1>& rhs)
    {
      return this->assign(rhs);
    }

    template<class T1,class C1>
    inline
    OLattice& operator=(const QDPExpr<T1,C1>& rhs)
    {
      return this->assign(rhs);
    }

    inline
    OLattice& operator=(const OLattice& rhs)
    {
      return this->assign(rhs);
    }


    template<class T1>
    inline
    void operator=(const OSubLattice<T1>& rhs)
    {
      this->assign(rhs);
    }


    OSubLattice<T> operator[](const Subset& s)
    {return OSubLattice<T>(*this,const_cast<Subset&>(s));}


    OLattice(const OLattice& rhs)//: QDPType<T, OLattice<T> >()
    {
      alloc_mem("copy");
      this->assign(rhs);
    }


    inline T* getF() const { 
      assert_on_host(); 
      return F; 
    }

    inline T& elem(int i) { 
      assert_on_host(); 
      return F[i]; 
    }
    inline const T& elem(int i) const { 
      assert_on_host(); 
      return F[i]; 
    }


  inline void moveToFastMemoryHint(bool copy=false) {}
  inline void revertFromFastMemoryHint(bool copy=false) {}


    void static changeLayout(bool toDev,void * outPtr,void * inPtr)
    {
      //QDP_info_primary("changing lattice data layout to %s format" , toDev? "device" : "host");

      if (toDev)
	jit_stats_lattice2dev();
      else
	jit_stats_lattice2host();



      typename WordType<T>::Type_t * in_data  = (typename WordType<T>::Type_t *)inPtr;
      typename WordType<T>::Type_t * out_data = (typename WordType<T>::Type_t *)outPtr;

      size_t lim_rea = GetLimit<T,2>::Limit_v; //T::ThisSize;
      size_t lim_col = GetLimit<T,1>::Limit_v; //T::ThisSize;
      size_t lim_spi = GetLimit<T,0>::Limit_v; //T::ThisSize;

      // QDP_info_primary("lim_rea = %d" , lim_rea );
      // QDP_info_primary("lim_col = %d" , lim_col );
      // QDP_info_primary("lim_spi = %d" , lim_spi );

      for ( int site = 0 ; site < Layout::sitesOnNode() ; site++ ) {
	for ( size_t reality = 0 ; reality < lim_rea ; reality++ ) {
	  for ( size_t color = 0 ; color < lim_col ; color++ ) {
	    for ( size_t spin = 0 ; spin < lim_spi ; spin++ ) {
	      size_t hst_idx = 
		reality + 
		lim_rea * color +
		lim_rea * lim_col * spin +
		lim_rea * lim_col * lim_spi * site;
	      size_t dev_idx = 
		site + 
		Layout::sitesOnNode() * spin +
		Layout::sitesOnNode() * lim_spi * color +
		Layout::sitesOnNode() * lim_spi * lim_col * reality;
	      if (toDev)
		out_data[dev_idx] = in_data[hst_idx];
	      else {
		//std::cout << hst_idx  << " <= " << dev_idx << "\n";
		out_data[hst_idx] = in_data[dev_idx];
	      }
	    }
	  }
	}
      }
    }

    int getId() const {
      return myId;
    }

    int getElemNum() const {       
      return -1;
    }


  private:


    inline void alloc_mem(const char* msg) const {
      myId = QDP_get_global_cache().registrate( Layout::sitesOnNode() * sizeof(T) , 1 , &changeLayout );
      mem = true;
    }
    inline void free_mem()  { 
      if (myId >= 0 && mem)
	QDP_get_global_cache().signoff( myId );
      mem = false;
    }



    inline void assert_on_host() const {
      // Here or somewhere we sould make sure that 
      // if the pointer is still valid, we do not too much
      QDP_get_global_cache().getHostPtr( (void**)&F , myId );
    }

  private:

    mutable T *F;
    mutable int myId = -2;
    mutable bool mem = false;       // did this class register the memory?

  };


template<class T>
struct LeafFunctor<OScalar<T>, ShiftPhase1>
{
  typedef int Type_t;
  inline static Type_t apply(const OScalar<T> &a, const ShiftPhase1 &f) {
    return 0;
  }
};

template<class T>
struct LeafFunctor<OLattice<T>, ShiftPhase1>
{
  typedef int Type_t;
  inline static Type_t apply(const OLattice<T> &a, const ShiftPhase1 &f) {
    return 0;
  }
};


template<class T>
struct LeafFunctor<OScalar<T>, ShiftPhase2>
{
  typedef int Type_t;
  inline static Type_t apply(const OScalar<T> &a, const ShiftPhase2 &f) {
    return 0;
  }
};

template<class T>
struct LeafFunctor<OLattice<T>, ShiftPhase2>
{
  typedef int Type_t;
  inline static Type_t apply(const OLattice<T> &a, const ShiftPhase2 &f) {
    return 0;
  }
};




template<class T> 
struct JITType<OLattice<T> >
{
  typedef OLatticeJIT<typename JITType<T>::Type_t>  Type_t;
};


template<class T> 
struct JITType<OScalar<T> >
{
  typedef OScalarJIT<typename JITType<T>::Type_t>  Type_t;
};




  // ------------------------------------------------------
  
template<>
struct LeafFunctor<QDPType< PScalar< PScalar < RScalar< Word< float > > > > ,OScalar<PScalar< PScalar < RScalar< Word< float > > > > > >, ParamLeaf>
{
  //typedef QDPTypeJIT<typename JITType<T>::Type_t,typename JITType<OScalar<PScalar< PScalar < RScalar< Word< float > > > > > > >::Type_t>  Type_t;
  typedef typename JITType< OScalar<PScalar< PScalar < RScalar< Word< float > > > > > >::Type_t  Type_t;
  inline static Type_t apply(const QDPType< PScalar< PScalar < RScalar< Word< float > > > > , OScalar<PScalar< PScalar < RScalar< Word< float > > > > > > &a, const ParamLeaf& p)
  {
    return Type_t( llvm_add_param< float >() );
  }
};

template<>
struct LeafFunctor<QDPType< PScalar< PScalar < RScalar< Word< double > > > > ,OScalar<PScalar< PScalar < RScalar< Word< double > > > > > >, ParamLeaf>
{
  //typedef QDPTypeJIT<typename JITType<T>::Type_t,typename JITType<OScalar<PScalar< PScalar < RScalar< Word< double > > > > > > >::Type_t>  Type_t;
  typedef typename JITType< OScalar<PScalar< PScalar < RScalar< Word< double > > > > > >::Type_t  Type_t;
  inline static Type_t apply(const QDPType< PScalar< PScalar < RScalar< Word< double > > > > , OScalar<PScalar< PScalar < RScalar< Word< double > > > > > > &a, const ParamLeaf& p)
  {
    return Type_t( llvm_add_param< double >() );
  }
};

template<>
struct LeafFunctor<QDPType< PScalar< PScalar < RScalar< Word< int > > > > ,OScalar<PScalar< PScalar < RScalar< Word< int > > > > > >, ParamLeaf>
{
  //typedef QDPTypeJIT<typename JITType<T>::Type_t,typename JITType<OScalar<PScalar< PScalar < RScalar< Word< int > > > > > > >::Type_t>  Type_t;
  typedef typename JITType< OScalar<PScalar< PScalar < RScalar< Word< int > > > > > >::Type_t  Type_t;
  inline static Type_t apply(const QDPType< PScalar< PScalar < RScalar< Word< int > > > > , OScalar<PScalar< PScalar < RScalar< Word< int > > > > > > &a, const ParamLeaf& p)
  {
    return Type_t( llvm_add_param< int >() );
  }
};

template<>
struct LeafFunctor<QDPType< PScalar< PScalar < RScalar< Word< bool > > > > , OScalar<PScalar< PScalar < RScalar< Word< bool > > > > > >, ParamLeaf>
{
  //typedef QDPTypeJIT<typename JITType<T>::Type_t,typename JITType<OScalar<PScalar< PScalar < RScalar< Word< bool > > > > > > >::Type_t>  Type_t;
  typedef typename JITType< OScalar<PScalar< PScalar < RScalar< Word< bool > > > > > >::Type_t  Type_t;
  inline static Type_t apply(const QDPType< PScalar< PScalar < RScalar< Word< bool > > > > , OScalar<PScalar< PScalar < RScalar< Word< bool > > > > > > &a, const ParamLeaf& p)
  {
    return Type_t( llvm_add_param< bool >() );
  }
};


template<>
struct LeafFunctor<QDPType<PScalar< PScalar < RScalar< Word< float > > > >  , OScalar< PScalar< PScalar < RScalar< Word< float > > > > > >, AddressLeaf>
{
  typedef int Type_t;
  inline static
  Type_t apply(const QDPType<PScalar< PScalar < RScalar< Word< float > > > > , OScalar< PScalar< PScalar < RScalar< Word< float > > > > > >& s, const AddressLeaf& p) 
  {
    p.setLit< float >( s.get_word_value() );	
    return 0;
  }
};

template<>
struct LeafFunctor<QDPType<PScalar< PScalar < RScalar< Word< double > > > >  , OScalar< PScalar< PScalar < RScalar< Word< double > > > > > >, AddressLeaf>
{
  typedef int Type_t;
  inline static
  Type_t apply(const QDPType<PScalar< PScalar < RScalar< Word< double > > > > , OScalar< PScalar< PScalar < RScalar< Word< double > > > > > >& s, const AddressLeaf& p) 
  {
    p.setLit< double >( s.get_word_value() );	
    return 0;
  }
};

template<>
struct LeafFunctor<QDPType<PScalar< PScalar < RScalar< Word< int > > > >  , OScalar< PScalar< PScalar < RScalar< Word< int > > > > > >, AddressLeaf>
{
  typedef int Type_t;
  inline static
  Type_t apply(const QDPType<PScalar< PScalar < RScalar< Word< int > > > > , OScalar< PScalar< PScalar < RScalar< Word< int > > > > > >& s, const AddressLeaf& p) 
  {
    p.setLit< int >( s.get_word_value() );	
    return 0;
  }
};

template<>
struct LeafFunctor<QDPType<PScalar< PScalar < RScalar< Word< bool > > > >  , OScalar< PScalar< PScalar < RScalar< Word< bool > > > > > >, AddressLeaf>
{
  typedef int Type_t;
  inline static
  Type_t apply(const QDPType<PScalar< PScalar < RScalar< Word< bool > > > > , OScalar< PScalar< PScalar < RScalar< Word< bool > > > > > >& s, const AddressLeaf& p) 
  {
    p.setLit< bool >( s.get_word_value() );	
    return 0;
  }
};


  // ----------------------------------------------


  

/*! @} */  // end of group olattice


template<class T>
struct LeafFunctor<OLattice<T>, ParamLeaf>
{
  typedef typename JITType< OLattice<T> >::Type_t  TypeA_t;
  typedef TypeA_t  Type_t;
  inline static
  Type_t apply(const OLattice<T>& do_not_use, const ParamLeaf& p) 
  {
    ParamRef    base_addr = llvm_add_param< typename WordType<T>::Type_t * >();
    return Type_t( base_addr );
  }
};

template<class T>
struct LeafFunctor<OScalar<T>, ParamLeaf>
{
  typedef typename JITType< OScalar<T> >::Type_t  TypeA_t;
  typedef TypeA_t  Type_t;
  inline static
  Type_t apply(const OScalar<T>& a, const ParamLeaf& p)
  {
    return Type_t( llvm_add_param< typename WordType<T>::Type_t * >() );
  }
};


template<>
struct LeafFunctor<OScalar< PScalar< PScalar < RScalar< Word< float > > > > >, ParamLeaf>
{
  typedef typename JITType< OScalar< PScalar< PScalar < RScalar< Word< float > > > > > >::Type_t  Type_t;
  inline static
  Type_t apply(const OScalar< PScalar< PScalar < RScalar< Word< float > > > > >& a, const ParamLeaf& p)
  {
    return Type_t( llvm_add_param< float >() );
  }
};

template<>
struct LeafFunctor<OScalar< PScalar< PScalar < RScalar< Word< double > > > > >, ParamLeaf>
{
  typedef typename JITType< OScalar< PScalar< PScalar < RScalar< Word< double > > > > > >::Type_t  Type_t;
  inline static
  Type_t apply(const OScalar< PScalar< PScalar < RScalar< Word< double > > > > >& a, const ParamLeaf& p)
  {
    return Type_t( llvm_add_param< double >() );
  }
};

template<>
struct LeafFunctor<OScalar< PScalar< PScalar < RScalar< Word< int > > > > >, ParamLeaf>
{
  typedef typename JITType< OScalar< PScalar< PScalar < RScalar< Word< int > > > > > >::Type_t  Type_t;
  inline static
  Type_t apply(const OScalar< PScalar< PScalar < RScalar< Word< int > > > > >& a, const ParamLeaf& p)
  {
    return Type_t( llvm_add_param< int >() );
  }
};

template<>
struct LeafFunctor<OScalar< PScalar< PScalar < RScalar< Word< bool > > > > >, ParamLeaf>
{
  typedef typename JITType< OScalar< PScalar< PScalar < RScalar< Word< bool > > > > > >::Type_t  Type_t;
  inline static
  Type_t apply(const OScalar< PScalar< PScalar < RScalar< Word< bool > > > > >& a, const ParamLeaf& p)
  {
    return Type_t( llvm_add_param< bool >() );
  }
};






template<class T>
struct LeafFunctor<OLattice<T>, AddressLeaf>
{
  typedef int Type_t;
  inline static
  Type_t apply(const OLattice<T>& s, const AddressLeaf& p) 
  {
    p.setIdElem( s.getId() , s.getElemNum() );
    return 0;
  }
};

template<class T>
struct LeafFunctor<OScalar<T>, AddressLeaf>
{
  typedef int Type_t;
  inline static
  Type_t apply(const OScalar<T>& s, const AddressLeaf& p)
  {
    p.setIdElem( s.getId() , s.getElemNum() );
    return 0;
  }
};


template<>
struct LeafFunctor<OScalar< PScalar< PScalar < RScalar< Word< float > > > > >, AddressLeaf>
{
  typedef int  Type_t;
  inline static
  Type_t apply(const OScalar< PScalar< PScalar < RScalar< Word< float > > > > >& a, const AddressLeaf& p)
  {
    //std::cout << "AddressLeaf: Real\n";
    p.setLit< float >( a.get_word_value() );
    return 0;
  }
};

template<>
struct LeafFunctor<OScalar< PScalar< PScalar < RScalar< Word< double > > > > >, AddressLeaf>
{
  typedef int  Type_t;
  inline static
  Type_t apply(const OScalar< PScalar< PScalar < RScalar< Word< double > > > > >& a, const AddressLeaf& p)
  {
    //std::cout << "AddressLeaf: Double\n";
    p.setLit< double >( a.get_word_value() );
    return 0;
  }
};

template<>
struct LeafFunctor<OScalar< PScalar< PScalar < RScalar< Word< int > > > > >, AddressLeaf>
{
  typedef int  Type_t;
  inline static
  Type_t apply(const OScalar< PScalar< PScalar < RScalar< Word< int > > > > >& a, const AddressLeaf& p)
  {
    //std::cout << "AddressLeaf: Int\n";
    p.setLit< int >( a.get_word_value() );
    return 0;
  }
};

template<>
struct LeafFunctor<OScalar< PScalar< PScalar < RScalar< Word< bool > > > > >, AddressLeaf>
{
  typedef int  Type_t;
  inline static
  Type_t apply(const OScalar< PScalar< PScalar < RScalar< Word< bool > > > > >& a, const AddressLeaf& p)
  {
    //std::cout << "AddressLeaf: Boolean\n";
    p.setLit< bool >( a.get_word_value() );
    return 0;
  }
};


  


//-----------------------------------------------------------------------------
// We need to specialize CreateLeaf<T> for our class, so that operators
// know what to stick in the leaves of the expression tree.
//-----------------------------------------------------------------------------

template<class T>
struct CreateLeaf<OScalar<T> >
{
  //typedef OScalar<T> Leaf_t;
  typedef Reference<OScalar<T> > Leaf_t;
  inline static
  Leaf_t make(const OScalar<T> &a) { return Leaf_t(a); }
};

template<class T>
struct CreateLeaf<OLattice<T> >
{
//  typedef OLattice<T> Leaf_t;
  typedef Reference<OLattice<T> > Leaf_t;
  inline static
  Leaf_t make(const OLattice<T> &a) { return Leaf_t(a); }
};

//-----------------------------------------------------------------------------
// Specialization of LeafFunctor class for applying the EvalLeaf1
// tag to a OScalar and OLattice. The apply method simply returns the array
// evaluated at the point.
//-----------------------------------------------------------------------------

template<class T>
struct LeafFunctor<OScalar<T>, ElemLeaf>
{
//  typedef T Type_t;
  typedef Reference<T> Type_t;
  inline static Type_t apply(const OScalar<T> &a, const ElemLeaf &f)
    {return Type_t(a.elem());}
};


template<class T>
struct LeafFunctor<OScalar<T>, EvalLeaf1>
{
//  typedef T Type_t;
  typedef Reference<T> Type_t;
  inline static Type_t apply(const OScalar<T> &a, const EvalLeaf1 &f)
    {return Type_t(a.elem());}
};

template<class T>
struct LeafFunctor<OLattice<T>, EvalLeaf1>
{
//  typedef T Type_t;
  typedef Reference<T> Type_t;
  inline static Type_t apply(const OLattice<T> &a, const EvalLeaf1 &f)
    {return Type_t(a.elem(f.val1()));}
};


//-----------------------------------------------------------------------------
// Traits classes to support operations of simple scalars (floating constants, 
// etc.) on QDPTypes
//-----------------------------------------------------------------------------

template<class T>
struct WordType<OScalar<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

template<class T>
struct WordType<OLattice<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

template<class T> 
struct SinglePrecType<OScalar<T> >
{
  typedef OScalar<typename SinglePrecType<T>::Type_t> Type_t;
};

template<class T>
struct DoublePrecType<OScalar<T> >
{
  typedef OScalar<typename DoublePrecType<T>::Type_t> Type_t;
};

template<class T> 
struct SinglePrecType<OLattice<T> >
{
  typedef OLattice<typename SinglePrecType<T>::Type_t> Type_t;
};

template<class T>
struct DoublePrecType<OLattice<T> >
{
  typedef OLattice<typename DoublePrecType<T>::Type_t> Type_t;
};

// Internally used scalars
template<class T>
struct InternalScalar<OScalar<T> > {
  typedef OScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

template<class T>
struct InternalScalar<OLattice<T> > {
  typedef OScalar<typename InternalScalar<T>::Type_t>  Type_t;
};


// Trait to make a primitive scalar leaving grid along
template<class T>
struct PrimitiveScalar<OScalar<T> > {
  typedef OScalar<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

template<class T>
struct PrimitiveScalar<OLattice<T> > {
  typedef OLattice<typename PrimitiveScalar<T>::Type_t>  Type_t;
};


// Trait to make a lattice scalar leaving primitive indices alone
template<class T>
struct LatticeScalar<OScalar<T> > {
  typedef OScalar<typename LatticeScalar<T>::Type_t>  Type_t;
};

template<class T>
struct LatticeScalar<OLattice<T> > {
  typedef OScalar<typename LatticeScalar<T>::Type_t>  Type_t;
};


// Internally used real scalars
template<class T>
struct RealScalar<OScalar<T> > {
  typedef OScalar<typename RealScalar<T>::Type_t>  Type_t;
};

template<class T>
struct RealScalar<OLattice<T> > {
  typedef OScalar<typename RealScalar<T>::Type_t>  Type_t;
};



//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(OScalar) -> OScalar
template<class T1, class Op>
struct UnaryReturn<OScalar<T1>, Op> {
  typedef OScalar<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

// Default unary(OLattice) -> OLattice
template<class T1, class Op>
struct UnaryReturn<OLattice<T1>, Op> {
  typedef OLattice<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

// Default binary(OScalar,OScalar) -> OScalar
template<class T1, class T2, class Op>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, Op> {
  typedef OScalar<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Currently, the only trinary operator is ``where'', so return 
// based on T2 and T3
// Default trinary(OScalar,OScalar,OScalar) -> OScalar
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OScalar<T1>, OScalar<T2>, OScalar<T3>, Op> {
  typedef OScalar<typename BinaryReturn<T2, T3, Op>::Type_t>  Type_t;
};

// Default binary(OLattice,OLattice) -> OLattice
template<class T1, class T2, class Op>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, Op> {
  typedef OLattice<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Default binary(OScalar,OLattice) -> OLattice
template<class T1, class T2, class Op>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, Op> {
  typedef OLattice<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Default binary(OLattice,OScalar) -> OLattice
template<class T1, class T2, class Op>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, Op> {
  typedef OLattice<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


// Default trinary(OLattice,OLattice,OLattice) -> OLattice
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OLattice<T1>, OLattice<T2>, OLattice<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};


// Default trinary(OLattice,OScalar,OLattice) -> OLattice
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OLattice<T1>, OScalar<T2>, OLattice<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};

// Default trinary(OLattice,OLattice,OScalar) -> OLattice
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OLattice<T1>, OLattice<T2>, OScalar<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};

// Default trinary(OScalar,OLattice,OLattice) -> OLattice
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OScalar<T1>, OLattice<T2>, OLattice<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};


// Default trinary(OScalar,OScalar,OLattice) -> OLattice
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OScalar<T1>, OScalar<T2>, OLattice<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};

// Default trinary(OSscalar,OLattice,OScalar) -> OLattice
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OScalar<T1>, OLattice<T2>, OScalar<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};

// Default trinary(OLattice,OScalar,OScalar) -> OLattice
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OLattice<T1>, OScalar<T2>, OScalar<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};




// Specific OScalar cases
// Global operations
template<class T>
struct UnaryReturn<OScalar<T>, FnPeekSite> {
  typedef OScalar<typename UnaryReturn<T, FnPeekSite>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnSum> {
  typedef OScalar<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnSumMulti > {
  typedef multi1d<OScalar<typename UnaryReturn<T, FnSumMulti>::Type_t> >  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnNorm2 > {
  typedef OScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnGlobalMax> {
  typedef OScalar<typename UnaryReturn<T, FnGlobalMax>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnGlobalMin> {
  typedef OScalar<typename UnaryReturn<T, FnGlobalMin>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnLocalNorm2 > {
  typedef OScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, FnLocalInnerProduct > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, FnLocalColorInnerProduct > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnLocalColorInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, FnLocalInnerProductReal > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};


// Gamma algebra
template<int N, int m, class T2, class OpGammaConstMultiply>
struct BinaryReturn<GammaConst<N,m>, OScalar<T2>, OpGammaConstMultiply> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConst>
struct BinaryReturn<OScalar<T2>, GammaConst<N,m>, OpMultiplyGammaConst> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeMultiply>
struct BinaryReturn<GammaType<N>, OScalar<T2>, OpGammaTypeMultiply> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaType>
struct BinaryReturn<OScalar<T2>, GammaType<N>, OpMultiplyGammaType> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

// Gamma algebra
template<int N, int m, class T2, class OpGammaConstDPMultiply>
struct BinaryReturn<GammaConstDP<N,m>, OScalar<T2>, OpGammaConstDPMultiply> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConstDP>
struct BinaryReturn<OScalar<T2>, GammaConstDP<N,m>, OpMultiplyGammaConstDP> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeDPMultiply>
struct BinaryReturn<GammaTypeDP<N>, OScalar<T2>, OpGammaTypeDPMultiply> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaTypeDP>
struct BinaryReturn<OScalar<T2>, GammaTypeDP<N>, OpMultiplyGammaTypeDP> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};



// Local operations
template<class T>
struct UnaryReturn<OScalar<T>, OpNot > {
  typedef OScalar<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
};


#if 0
template<class T1, class T2>
struct UnaryReturn<OScalar<T2>, OpCast<T1> > {
  typedef OScalar<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpLT > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpLE > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpGT > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpGE > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpEQ > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpNE > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpAnd > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpOr > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpLeftShift > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpRightShift > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpAddAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpSubtractAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpMultiplyAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpDivideAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpModAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpBitwiseOrAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpBitwiseAndAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpBitwiseXorAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpLeftShiftAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpRightShiftAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2, class T3>
struct TrinaryReturn<OScalar<T1>, OScalar<T2>, OScalar<T3>, FnColorContract> {
  typedef OScalar<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};

// Specific OLattice cases
// Global operations
template<class T>
struct UnaryReturn<OLattice<T>, FnGetSite> {
  typedef OScalar<typename UnaryReturn<T, FnGetSite>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnPeekSite> {
  typedef OScalar<typename UnaryReturn<T, FnPeekSite>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnSum > {
  typedef OScalar<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnGlobalMax> {
  typedef OScalar<typename UnaryReturn<T, FnGlobalMax>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnGlobalMin> {
  typedef OScalar<typename UnaryReturn<T, FnGlobalMin>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnSumMulti > {
  typedef multi1d<OScalar<typename UnaryReturn<T, FnSumMulti>::Type_t> >  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnNorm2 > {
  typedef OScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnLocalNorm2 > {
  typedef OLattice<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, FnLocalInnerProduct > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, FnLocalColorInnerProduct > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalColorInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, FnLocalInnerProductReal > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};


// Gamma algebra
template<int N, int m, class T2, class OpGammaConstMultiply>
struct BinaryReturn<GammaConst<N,m>, OLattice<T2>, OpGammaConstMultiply> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConst>
struct BinaryReturn<OLattice<T2>, GammaConst<N,m>, OpMultiplyGammaConst> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeMultiply>
struct BinaryReturn<GammaType<N>, OLattice<T2>, OpGammaTypeMultiply> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaType>
struct BinaryReturn<OLattice<T2>, GammaType<N>, OpMultiplyGammaType> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};


// Gamma algebra
template<int N, int m, class T2, class OpGammaConstDPMultiply>
struct BinaryReturn<GammaConstDP<N,m>, OLattice<T2>, OpGammaConstDPMultiply> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConstDP>
struct BinaryReturn<OLattice<T2>, GammaConstDP<N,m>, OpMultiplyGammaConstDP> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeDPMultiply>
struct BinaryReturn<GammaTypeDP<N>, OLattice<T2>, OpGammaTypeDPMultiply> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaTypeDP>
struct BinaryReturn<OLattice<T2>, GammaTypeDP<N>, OpMultiplyGammaTypeDP> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};



// Local operations
template<class T>
struct UnaryReturn<OLattice<T>, OpNot > {
  typedef OLattice<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
};


#if 0
template<class T1, class T2>
struct UnaryReturn<OLattice<T2>, OpCast<T1> > {
  typedef OLattice<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpLT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpLE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpGT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpGE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpEQ > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpNE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpAnd > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpOr > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpLeftShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpRightShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpAddAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpSubtractAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpMultiplyAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpDivideAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpModAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpBitwiseOrAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpBitwiseAndAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpBitwiseXorAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpLeftShiftAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpRightShiftAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2, class T3>
struct TrinaryReturn<OLattice<T1>, OLattice<T2>, OLattice<T3>, FnColorContract> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};


// Mixed OLattice & OScalar cases
// Global operations
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, FnInnerProduct > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, FnInnerProductReal > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, FnInnerProduct > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, FnInnerProductReal > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, FnLocalInnerProduct > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, FnLocalColorInnerProduct > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalColorInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, FnLocalInnerProductReal > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, FnLocalInnerProduct > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, FnLocalColorInnerProduct > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalColorInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, FnLocalInnerProductReal > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};


// Local operations
template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpLT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpLE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpGT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpGE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpEQ > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpNE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpAnd > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpOr > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpLeftShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpRightShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpAddAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpSubtractAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpMultiplyAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpDivideAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpModAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpBitwiseOrAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpBitwiseAndAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpBitwiseXorAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpLeftShiftAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpRightShiftAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  &Type_t;
};
 

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpLT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpLE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpGT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpGE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpEQ > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpNE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpAnd > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpOr > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpLeftShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpRightShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 




//-----------------------------------------------------------------------------
// Scalar Operations
//-----------------------------------------------------------------------------

/*! \addtogroup oscalar */
/*! @{ */

//! QDP Wordtype to primitive wordtype
template<class T> 
inline typename WordType< OScalar<T> >::Type_t
toWordType(const OScalar<T>& s) 
{
  return toWordType(s.elem());
}

//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T>
struct UnaryReturn<OScalar<T>, FnGetSite> {
  typedef OScalar<typename UnaryReturn<T, FnGetSite>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<OScalar<T>, FnGetSite>::Type_t
getSite(const OScalar<T>& s1, int innersite)
{
  typename UnaryReturn<OScalar<T>, FnGetSite>::Type_t  d;

  d.elem() = getSite(s1.elem(), innersite);
  return d;
}


//! dest = 0
template<class T> 
void zero_rep(OScalar<T>& dest) 
{
  zero_rep(dest.elem());
}

//! dest = (mask) ? s1 : dest
template<class T1, class T2> 
void copymask(OScalar<T2>& dest, const OScalar<T1>& mask, const OScalar<T2>& s1) 
{
  copymask(dest.elem(), mask.elem(), s1.elem());
}

//! dest [some type] = source [some type]
template<class T, class T1>
void cast_rep(T& d, const OScalar<T1>& s1)
{
  cast_rep(d, s1.elem());
}

//! dest [some type] = source [some type]
template<class T, class T1>
void recast_rep(OScalar<T>& d, const OScalar<T1>& s1)
{
  cast_rep(d.elem(), s1.elem());
}


//-----------------------------------------------------------------------------
// Random numbers
//! dest  = random  
/*! Implementation is in the specific files */
template<class T>
void random(OScalar<T>& d);


//! dest  = gaussian
template<class T>
void gaussian(OSubScalar<T>& d)
{
  OScalar<T>  r1, r2;

  random(OSubScalar<T>(r1,d.subset()));
  random(OSubScalar<T>(r2,d.subset()));

  fill_gaussian(d.elem(), r1.elem(), r2.elem());
}



/*! @} */  // end of group oscalar

} // namespace QDP

#endif
