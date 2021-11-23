#ifndef QDP_OUTERSUBJIT_H
#define QDP_OUTERSUBJIT_H


namespace QDP {

  template<class T>
  class OSubLatticeJIT//: public QDPTypeJIT<T, OSubLatticeJIT<T> >
  {
  public:
    OSubLatticeJIT( ParamRef base_ ) : base_m(base_) {}
    OSubLatticeJIT( const OSubLatticeJIT& rhs ) : base_m(rhs.base_m) {}

    T elem( JitDeviceLayout lay , llvm::Value * index ) const
    {
      T F;
      IndexDomainVector args;
      args.push_back( make_pair( Layout::sitesOnNode() , index ) );
      F.setup( llvm_derefParam(base_m) , lay , args );
      return F;
    }

    typename ScalarType<T>::Type_t elemScalar( JitDeviceLayout lay , llvm::Value * index ) const
    {
      typename ScalarType<T>::Type_t F;
      IndexDomainVector args;
      args.push_back( make_pair( Layout::sitesOnNode() , index ) );
      F.setup( llvm_derefParam(base_m) , lay , args );
      return F;
    }


    void set_base( ParamRef p ) const
    {
      base_m = p;
    }

  private:
    mutable ParamRef    base_m;
  };



  

}

#endif
