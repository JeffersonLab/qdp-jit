#ifndef QDP_OUTERJIT_H
#define QDP_OUTERJIT_H


namespace QDP {

  template<class T>
  class OLatticeJIT: public QDPTypeJIT<T, OLatticeJIT<T> >
  {
  public:
    OLatticeJIT( ParamRef base_ ) : QDPTypeJIT<T, OLatticeJIT<T> >(base_) {}
    OLatticeJIT( const OLatticeJIT& rhs ) : QDPTypeJIT<T, OLatticeJIT<T> >(rhs) {}

  private:
    void operator=(const OLatticeJIT& a);
  };





  template<class T>
  class OScalarJIT: public QDPTypeJIT<T, OScalarJIT<T> >
  {
  public:
    OScalarJIT( ParamRef base_ ) : QDPTypeJIT<T, OScalarJIT<T> >(base_) {}

    OScalarJIT(const OScalarJIT& rhs) : QDPTypeJIT<T, OScalarJIT<T> >(rhs) {}

  private:
    void operator=(const OScalarJIT& a) {}
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
