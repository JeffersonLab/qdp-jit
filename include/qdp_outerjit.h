#ifndef QDP_OUTERJIT_H
#define QDP_OUTERJIT_H


namespace QDP {

  template<class T>
  class OLatticeJIT//: public QDPTypeJIT<T, OLatticeJIT<T> >
  {
  public:
    //! Type of the first argument
    typedef T Subtype_t;


    OLatticeJIT( ParamRef base_ )
    {
      base_m = base_;
    }

    OLatticeJIT( const OLatticeJIT& rhs )
    {
      base_m = rhs.base_m;
    }


    typename REGType<T>::Type_t elemREG( JitDeviceLayout lay , llvm::Value * index ) const
    {
      typename REGType<T>::Type_t ret;
      ret.setup( elem( lay , index ) );
      return ret;
    }

    

    T elem( JitDeviceLayout lay , llvm::Value * index ) const
    {
      T F;
      IndexDomainVector args;
      args.push_back( make_pair( Layout::sitesOnNode() , index ) );
      F.setup( llvm_derefParam(base_m) , lay , args );
      return F;
    }



    T elem( JitDeviceLayout lay , llvm::Value * index , llvm::Value * multi_index ) const
    {
      T F;
      IndexDomainVector args;
      args.push_back( make_pair( Layout::sitesOnNode() , index ) );
      F.setup( llvm_array_type_indirection( base_m , multi_index ) , lay , args );
      return F;
    }


    void set_base( ParamRef p ) const
    {     
      base_m = p;
    }

  private:
    void operator=(const OLatticeJIT& a);

    mutable ParamRef base_m;
  };





  template<class T>
  class OScalarJIT//: public QDPTypeJIT<T, OScalarJIT<T> >
  {
  public:
    //! Type of the first argument
    typedef T Subtype_t;


    OScalarJIT( ParamRef base_ ) : base_m(base_)
    {
    }
    


    OScalarJIT(const OScalarJIT& rhs)
    {
      base_m = rhs.base_m;
    }

    

    llvm::Value* get_word_value() const
    {
      return llvm_derefParam( base_m );
    }

    T elem() const {
      T F;
      IndexDomainVector args;
      args.push_back( make_pair( 1 , llvm_create_value(0) ) );
      F.setup( llvm_derefParam(base_m) , JitDeviceLayout::Scalar , args );
      return F;
    }


    typename REGType<T>::Type_t elemRegValue() const
    {
      typename REGType<T>::Type_t reg;
      reg.setup_value( this->elem() );
      return reg;
    }

    
    void set_base( ParamRef p ) const
    {
      base_m = p;
    }

  private:
    void operator=(const OScalarJIT& a);

    mutable ParamRef    base_m;
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
  struct BinaryReturn<OLatticeJIT<T1>, OScalarJIT<T2>, Op> {
    typedef OLatticeJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };

  template<class T1, class T2, class Op>
  struct BinaryReturn<OScalarJIT<T1>, OLatticeJIT<T2>, Op> {
    typedef OLatticeJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };
  
  template<class T1, class T2, class Op>
  struct BinaryReturn<OScalarJIT<T1>, OScalarJIT<T2>, Op> {
    typedef OScalarJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };


}

#endif
