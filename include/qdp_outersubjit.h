#ifndef QDP_OUTERSUBJIT_H
#define QDP_OUTERSUBJIT_H


namespace QDP {

  template<class T>
  class OSubLatticeJIT: public QDPTypeJIT<T, OSubLatticeJIT<T> >
  {
  public:
    OSubLatticeJIT( ParamRef base_ ) : QDPTypeJIT<T, OSubLatticeJIT<T> >(base_) {}
    OSubLatticeJIT( const OSubLatticeJIT& rhs ) : QDPTypeJIT<T, OSubLatticeJIT<T> >(rhs) {}

  private:
    void operator=(const OSubLatticeJIT& a);
  };





  template<class T>
  class OSubScalarJIT: public QDPTypeJIT<T, OSubScalarJIT<T> >
  {
  public:
    OSubScalarJIT( ParamRef base_ ) : QDPTypeJIT<T, OSubScalarJIT<T> >(base_) {}

    OSubScalarJIT(const OSubScalarJIT& rhs) : QDPTypeJIT<T, OSubScalarJIT<T> >(rhs) {}

  private:
    void operator=(const OSubScalarJIT& a) {}
  };




  template<class T>
  struct WordType<OSubLatticeJIT<T> >
  {
    typedef typename WordType<T>::Type_t  Type_t;
  };
  
  template<class T>
  struct WordType<OSubScalarJIT<T> >
  {
    typedef typename WordType<T>::Type_t  Type_t;
  };



  // Default binary(OSubLattice,OSubLattice) -> OSubLattice
  template<class T1, class T2, class Op>
  struct BinaryReturn<OSubLatticeJIT<T1>, OSubLatticeJIT<T2>, Op> {
    typedef OSubLatticeJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };

  template<class T1, class T2, class Op>
  struct BinaryReturn<OSubScalarJIT<T1>, OSubScalarJIT<T2>, Op> {
    typedef OSubScalarJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };


}

#endif
