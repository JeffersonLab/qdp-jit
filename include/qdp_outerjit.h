#ifndef QDP_OUTERJIT_H
#define QDP_OUTERJIT_H


namespace QDP {

  template<class T>
  class OLatticeJIT
  {
  public:
    typedef T Subtype_t;

    // put in for EvalQV
    OLatticeJIT() {
      layout_slot_ = 0;
      data_format_ = JitDataOrder::SiteFast;
    }

    OLatticeJIT( Shape shape ): shape_(shape) {
      layout_slot_ = 0;
      data_format_ = JitDataOrder::SiteFast;
    }
    
    
    OLatticeJIT( ParamRef base_ , ParamRef latsize_ , Shape shape , JitDataFormat fmt ): shape_(shape)
    {
      layout_slot_ = 0;
      base_m = base_;
      latsize_m = latsize_;
      data_format_ = fmt;
    }


    OLatticeJIT( ParamRef base_ , ParamRef latsize_ , Shape shape , int slot , JitDataFormat fmt ): shape_(shape), layout_slot_(slot)
    {
      base_m = base_;
      latsize_m = latsize_;
      data_format_ = fmt;
    }

    
    OLatticeJIT( const OLatticeJIT& rhs ): shape_(rhs.shape_), layout_slot_(rhs.layout_slot_)
    {
      base_m     = rhs.base_m;
      latsize_m  = rhs.latsize_m;
      data_format_ = rhs.data_format_;
    }

    T elem( llvm::Value * index ) const
    {
      NestedIndex idv;
      idv.push_back( latsize_m , index );

      T F( llvm_derefParam(base_m) , data_format_ , idv , shape_ );
      
      return F;
    }

    T elem( llvm::Value * index , llvm::Value * multi_index ) const
    {
      NestedIndex args;
      args.push_back( latsize_m , index );
      T F( llvm_array_type_indirection< typename WordType<T>::Type_t* >( base_m , multi_index ) , data_format_ , args , shape_ );
      return F;
    }

    void set_base( ParamRef p ) 
    {     
      base_m = p;
    }

    Shape shape() const { return shape_; }


    int layout_slot() const { return layout_slot_; }

    JitDataFormat data_format() const { return data_format_; }
    
  private:
    void operator=(const OLatticeJIT& a);

    ParamRef base_m;
    ParamRef latsize_m;
    Shape  shape_;
    int layout_slot_;
    JitDataFormat data_format_ = JitDataOrder::SiteFast;
  };



  template<class T>
  class OScalarJIT
  {
  public:
    typedef T Subtype_t;

    // put in for EvalQV
    OScalarJIT() {}
    OScalarJIT( Shape shape ): shape_(shape) {}

    
    OScalarJIT( ParamRef base_ , Shape shape , bool isLiteral_ = false ) : base_m(base_), isLiteral(isLiteral_), shape_(shape)
    {
    }
    
    OScalarJIT( ParamRef base_ , bool isLiteral_ = false ) : base_m(base_), isLiteral(isLiteral_), shape_{Shape()}
    {
    }

    OScalarJIT(const OScalarJIT& rhs): base_m(rhs.base_m), isLiteral(rhs.isLiteral), shape_(rhs.shape_)
    {
    }
    
    T elem() const
    {
      typedef typename WordType<T>::Type_t WT;

      if (isLiteral)
	{
	  llvm::Value* ptr = llvm_alloca( llvm_get_type<WT>() , 1 );
	  llvm::Value* lit = llvm_derefParam(base_m);
	  
	  llvm_store_ptr_idx<WT>( llvm_cast( llvm_get_type<WT>() , lit ) , ptr , llvm_create_value(0) );

	  NestedIndex args;
	  args.push_back( llvm_create_value(1) , llvm_create_value(0) );
	  T F( ptr , JitDataOrder::SiteSlow , args , shape_ );
	  return F;
	}
      else
	{
	  NestedIndex args;
	  args.push_back( llvm_create_value(1) , llvm_create_value(0) );
	  T F( llvm_derefParam(base_m) , JitDataOrder::SiteSlow , args , shape_ );
	  return F;
	}
    }
    
    void set_base( ParamRef p )
    {
      base_m = p;
    }

    Shape shape() const { return shape_; }


    
  private:
    void operator=(const OScalarJIT& a);

    ParamRef    base_m;
    bool        isLiteral;
    Shape          shape_;
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

  
  template<class T>
  struct LeafFunctor<OLatticeJIT<T>, MGViewLeaf>
  {
    typedef T Type_t;
    inline static
    Type_t apply( OLatticeJIT<T> s, MGViewLeaf& v)
    {
      return s.elem( v.getIndex( s.layout_slot() ) );
    }
  };


  template<class T>
  struct LeafFunctor<OLatticeJIT<T>, DummyJitLeaf>
  {
    typedef T Type_t;
    inline static
    Type_t apply(const OLatticeJIT<T>& a, DummyJitLeaf& p) 
    {
      return Type_t( a.shape() );
    }
  };


  template<class T>
  struct LeafFunctor<OScalarJIT<T>, DummyJitLeaf>
  {
    typedef T Type_t;
    inline static
    Type_t apply(const OScalarJIT<T>& a, DummyJitLeaf& p) 
    {
      return Type_t( a.shape() );
    }
  };


}

#endif
