#ifndef QDP_OUTERSUBJIT_H
#define QDP_OUTERSUBJIT_H


namespace QDP {

  template<class T>
  class OSubLatticeJIT
  {
  public:
    OSubLatticeJIT( ParamRef base_ , ParamRef latsize_ , Shape shape , JitDataFormat fmt ): base_m(base_), latsize_m(latsize_), shape_(shape), data_format_(fmt) {}
    OSubLatticeJIT( const OSubLatticeJIT& rhs ) : base_m(rhs.base_m), latsize_m(rhs.latsize_m), shape_(rhs.shape_), data_format_(rhs.data_format_) {}

    

    T elem( llvm::Value * index ) const
    {     
      NestedIndex idv;
      idv.push_back( latsize_m , index );

      T F( llvm_derefParam(base_m) , data_format_ , idv , shape_ );
      
      return F;
    }


    void set_base( ParamRef p ) 
    {
      base_m = p;
    }

  private:
    void operator=(const OSubLatticeJIT& a);

    ParamRef    base_m;
    ParamRef latsize_m;
    Shape shape_;
    JitDataFormat data_format_ = JitDataOrder::SiteFast;

  };



  

}

#endif
