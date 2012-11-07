#ifndef QDP_OUTERJIT_H
#define QDP_OUTERJIT_H


namespace QDP {

  template<class T>
  class OLatticeJIT: public QDPTypeJIT<T, OLatticeJIT<T> >
  {
  public:
    //! Create a view of the lattice object
    // NOTE: This is the address of the original lattice object. No additional offset. 

    //OLatticeJIT(Jit& func_, int addr_) : function(func_), r_addr(addr_) {}

#if 0
    OLatticeJIT(const OLatticeJIT& a): 
      QDPTypeJIT<T, OLatticeJIT<T> >( a.getFunc() , a.getAddr() ),
      view( a.getFunc() , a.getAddr() , Layout::sitesOnNode() , 0 ) 
    {
      std::cout << "OLatJIT copy ctor " << (void*)this <<"\n";
    }
#endif


    OLatticeJIT(Jit& func_, int addr_, Jit::LatticeLayout lay ) : 
      QDPTypeJIT<T, OLatticeJIT<T> >(func_,addr_,lay)
    {
      //std::cout << "OLatJIT() constr " << (void*)this << " " << (void*)&func_ << "\n";
    }


  private:
    // These are not working yet
    void operator=(const OLatticeJIT& a);


  public:
    // inline T elem(unsigned site) {
    //   //std::cout << "OLatJIT elem() \n";
    //   return T( curry_t( QDPTypeJIT<T, OLatticeJIT<T> >::getFunc() , 
    // 			 QDPTypeJIT<T, OLatticeJIT<T> >::getAddr(), 
    // 			 QDPTypeJIT<T, OLatticeJIT<T> >::innerSites(), 
    // 			 site ) );
    // }
    // inline const T elem(unsigned site) const {
    //   //std::cout << "OLatJIT elem() \n";
    //   return T( curry_t( QDPTypeJIT<T, OLatticeJIT<T> >::getFunc() , 
    // 			 QDPTypeJIT<T, OLatticeJIT<T> >::getAddr(), 
    // 			 QDPTypeJIT<T, OLatticeJIT<T> >::innerSites(), 
    // 			 site ) );
    // }
    
  };





  template<class T>
  class OScalarJIT: public QDPTypeJIT<T, OScalarJIT<T> >
  {
  public:
    typedef T Subtype_t;

    OScalarJIT(Jit& func_, int addr_) : 
      QDPTypeJIT<T, OScalarJIT<T> >(func_,addr_,Jit::LatticeLayout::SCAL) {}

  private:
    // These are not working yet
    void operator=(const OScalarJIT& a) {}

  public:
    // inline T elem(unsigned site) {
    //   std::cout << "OScaJIT elem(int) \n";
    //   //return T(QDPTypeJIT<T, OScalarJIT<T> >::getFunc(), QDPTypeJIT<T, OScalarJIT<T> >::getAddr(), 1,0 );
    //   return T( curry_t( QDPTypeJIT<T, OScalarJIT<T> >::getFunc(), 
    // 			 QDPTypeJIT<T, OScalarJIT<T> >::getAddr(), 
    // 			 QDPTypeJIT<T, OScalarJIT<T> >::innerSites() , 0 ) );
    // }
    // inline const T elem(unsigned site) const {
    //   std::cout << "OScaJIT elem(int) \n";
    //   //return T(QDPTypeJIT<T, OScalarJIT<T> >::getFunc(), QDPTypeJIT<T, OScalarJIT<T> >::getAddr(), 1,0 );
    //   return T( curry_t( QDPTypeJIT<T, OScalarJIT<T> >::getFunc(), 
    // 			 QDPTypeJIT<T, OScalarJIT<T> >::getAddr(), 
    // 			 QDPTypeJIT<T, OScalarJIT<T> >::innerSites() , 0 ) );
    // }
    // inline T elem() {
    //   std::cout << "OScaJIT elem() \n";
    //   //return T(QDPTypeJIT<T, OScalarJIT<T> >::getFunc(), QDPTypeJIT<T, OScalarJIT<T> >::getAddr(), 1,0 );
    //   return T( curry_t( QDPTypeJIT<T, OScalarJIT<T> >::getFunc(), 
    // 			 QDPTypeJIT<T, OScalarJIT<T> >::getAddr(), 
    // 			 QDPTypeJIT<T, OScalarJIT<T> >::innerSites() , 0 ) );
    // }
    // inline const T elem() const {
    //   std::cout << "OScaJIT elem() \n";
    //   return T( curry_t( QDPTypeJIT<T, OScalarJIT<T> >::getFunc(), 
    // 			 QDPTypeJIT<T, OScalarJIT<T> >::getAddr(), 
    // 			 QDPTypeJIT<T, OScalarJIT<T> >::innerSites() , 0 ) );
    // }


  };

  template<class T>
  struct LeafFunctor<OScalarJIT<T>, ViewLeaf>
  {
    typedef T Type_t;
    inline static
    Type_t apply(const OScalarJIT<T>& s, const ViewLeaf& v)
    { 
      return s.elem();
    }
  };


  template<class T>
  struct LeafFunctor<OLatticeJIT<T>, ViewLeaf>
  {
    typedef T Type_t;
    inline static
    Type_t apply(const OLatticeJIT<T>& s, const ViewLeaf& v)
    { 
      return s.elem( v.val1() );
    }
  };





  template<class T>
  struct WordType<OLatticeJIT<T> >
  {
    typedef typename WordType<T>::Type_t  Type_t;
  };
  
  template<class T>
  struct WordType<OScalarJIT<T> >
  {
    typedef typename WordType<T>::Type_t  Type_t;
  };



  // Default binary(OLattice,OLattice) -> OLattice
  template<class T1, class T2, class Op>
  struct BinaryReturn<OLatticeJIT<T1>, OLatticeJIT<T2>, Op> {
    typedef OLatticeJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };

  template<class T1, class T2, class Op>
  struct BinaryReturn<OScalarJIT<T1>, OScalarJIT<T2>, Op> {
    typedef OScalarJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };


}

#endif
