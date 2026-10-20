#ifndef QDP_JIT_UTIL_H
#define QDP_JIT_UTIL_H

namespace QDP {

  class JitValue
  {
    llvm::Value* value;
  public:
    JitValue( llvm::Value* v ): value(v) {}
    JitValue& operator++() { value = llvm_add( value , llvm_create_value(1) ); return *this; }
    operator llvm::Value*() const { return value; }
  };

  
  class RingBuffer
  {
    std::vector<int> ringBufferOScalar;
    int ringBufferNext;
  public:
    RingBuffer();
    int  allocate( size_t size , const void *hstPtr );
    void done();
  };
    
  RingBuffer& QDP_get_global_ring_buffer();



    
  void jit_get_function(JitFunction&);

  void jit_util_sync_init();
  void jit_util_sync_done();
  void jit_util_sync_copy();

      
  

  void jit_launch       ( JitFunction& function , int th_count , std::vector<QDPCache::ArgKey>& ids , bool set_l0_event = false );
  bool jit_launch_explicit_geom( JitFunction& function , std::vector<QDPCache::ArgKey>& ids , const kernel_geom_t& geom , unsigned int shared );

#ifdef QDP_DEEP_LOG
  //void jit_deep_log(JitFunction& f);
#endif



  

  template<class T>
  typename JITType<T>::Type_t stack_alloc()
  {
    int type_size = JITType<T>::Type_t::ScalarSize_t;
    llvm::Value * ptr = llvm_alloca( llvm_get_type< typename WordType< T >::Type_t >() , type_size );
      
    typename JITType<T>::Type_t t_jit_stack;
    t_jit_stack.setup( ptr , JitDataOrder::SiteSlow );

    return t_jit_stack;
  }


  template<class T>
  class StackAllocJit
  {
  public:
    static T alloc( const Shape& shape )
    {
      const int words = WordSize<T>::value(shape);
      if (words == 0)
	QDP_error_exit("StackAllocJit: zero words for Ns=%d, Nc=%d", shape.ns(), shape.nc());
      llvm::Value * ptr = llvm_alloca( llvm_get_type< typename WordType< T >::Type_t >() , words );
      T t( ptr , JitDataOrder::SiteSlow , NestedIndex() , shape );
      return t;
    }
    static T alloc()
    {
      return alloc( Shape() );
    }
  };    
  template<class T>
  class StackAllocJit< WordJIT<T> >
  {
  public:
    static WordJIT<T> alloc( const Shape& shape )
    {
      llvm::Value * ptr = llvm_alloca( llvm_get_type< typename WordType< T >::Type_t >() , 1 );
      //QDPIO::cout << "stack alloc jit: words = " << WordSize<T>::value(shape) << std::endl;
      WordJIT<T> t( ptr , JitDataOrder::SiteSlow , NestedIndex() );
      return t;
    }
  };    

  template<class T>
  T stack_alloc_jit( const Shape& shape )
  {
    return StackAllocJit< T >::alloc( shape );
  }
  template<class T>
  T stack_alloc_jit()
  {
    Shape shape;
    return StackAllocJit< T >::alloc( shape );
  }



  
  template<class T>
  std::vector< T > stack_alloc_array_jit(int n)
  {
    std::vector<T> ret;
    for (int i = 0 ; i < n ; ++i )
      {
	ret.push_back( StackAllocJit<T>::alloc() );
      }
    return ret;
  }
  
  
  
  template<class T>
  WordJIT<T> create_word(T val)
  {
    WordJIT<T> t( llvm_alloca( llvm_get_type< T >() , 1 ) , JitDataOrder::SiteSlow );
    
    t = llvm_create_value(val);
    
    return t;
  }


  template<class T>
  WordJIT<T> create_word()
  {
    WordJIT<T> t( llvm_alloca( llvm_get_type< T >() , 1 ) , JitDataOrder::SiteSlow );
    return t;
  }






  template<class Result, class Op, class A>
  inline Shape unary_result_shape(const A& a)
  {
    // Inherit the already-evaluated child metadata.
    Shape shape = a.shape();

    // Apply exceptional changes such as Ns: 4 -> 2.
    UnaryShapeTransform<Op,A>::apply( shape , a );

    // Remove dimensions absent from Result.
    shape = fit_shape_jit<Result>(shape);

    return shape;
  }

  
  template<class Op, class T1>
  typename UnaryReturn<T1, Op>::Type_t
  unary_return(T1 a)
  {
    using Result =
      typename UnaryReturn<T1, Op>::Type_t;

    const Shape shape =
      unary_result_shape<Result,Op>(a);

    return StackAllocJit<Result>::alloc(shape);
  }


  

  template<class Result, class Op, class A, class B>
  inline Shape binary_result_shape(const A& a,
			     const B& b)
  {
    // QDPIO::cout << "binary_result_shape, a: " << a.shape() << "\n";
    // QDPIO::cout << "binary_result_shape, b: " << b.shape() << "\n";

    // Combine the already-evaluated child metadata.
    Shape shape = merge_operand_shapes(a, b);

    //QDPIO::cout << "binary_result_shape, after merge: " << shape << "\n";

    // Apply exceptional operation-specific changes.
    BinaryShapeTransform<Op,A,B>::apply( shape, a, b);

    //QDPIO::cout << "binary_result_shape, after transform: " << shape << "\n";

    // Remove dimensions absent from Result.
    shape = fit_shape_jit<Result>(shape);

    //QDPIO::cout << "binary_result_shape, after restrict: " << shape << "\n";

    return shape;
  }


  
  template<class Op, class T1, class T2>
  typename BinaryReturn<T1, T2, Op>::Type_t
  binary_return(T1 lhs, T2 rhs)
  {
    using Result =
      typename BinaryReturn<T1, T2, Op>::Type_t;

    const Shape shape =
      binary_result_shape<Result,Op>(lhs, rhs);

    return StackAllocJit<Result>::alloc(shape);
  }



  template<class Result, class Op, class A, class B, class C>
  inline Shape trinary_result_shape(const A& a,
			      const B& b,
			      const C& c)
  {
    // Combine the already-evaluated child metadata.
    Shape shape = merge_operand_shapes(a, b, c);

    // Apply exceptional operation-specific changes.
    TrinaryShapeTransform<Op,A,B,C>::apply( shape, a, b, c);

    // Remove dimensions absent from Result.
    shape = fit_shape_jit<Result>(shape);

    return shape;
  }

  

  template< class Op , class T1 , class T2 , class T3 >
  typename TrinaryReturn< T1 , T2 , T3 , Op >::Type_t trinary_return( T1 a , T2 b , T3 c )
  {
    using Result =
      typename TrinaryReturn< T1, T2, T3, Op >::Type_t;

    const Shape shape = trinary_result_shape<Result,Op>(a,b,c);

    return StackAllocJit< Result >::alloc( shape );
  }

  
  


  
  template< class Op , class T1 >
  typename UnaryReturn< WordJIT<T1> , Op >::Type_t unary_return( WordJIT<T1> l )
  {
    typedef typename UnaryReturn< WordJIT<T1> , Op >::Type_t  T_;

    Shape shape;
    T_ d = StackAllocJit< T_ >::alloc(shape);

    return d;
  }

  

  

  class JitForLoop
  {
  public:
    ~JitForLoop() { if (!closed) { QDPIO::cout << "jit loop not closed" << std::endl; QDP_abort(1);} }
    JitForLoop( int start          , int end ):           JitForLoop( llvm_create_value(start) , llvm_create_value(end) ) {}
    JitForLoop( int start          , llvm::Value*  end ): JitForLoop( llvm_create_value(start) , end ) {}
    JitForLoop( llvm::Value* start , int  end ):          JitForLoop( start , llvm_create_value(end) ) {}
    JitForLoop( llvm::Value* start , llvm::Value*  end );
    llvm::Value * index() const;
    JitValue jit() const;
    void unroll_full() { full_unroll = true; }
    void end();
    operator llvm::Value*() const { return index(); }
  private:
    llvm::BasicBlock * block_outer;
    llvm::BasicBlock * block_loop_cond;
    llvm::BasicBlock * block_loop_body;
    llvm::BasicBlock * block_loop_exit;
    llvm::Value * r_i;
    llvm::Value * r_start;
    llvm::Value * r_end;
    bool closed = false;
    bool full_unroll = false;
  };


  llvm::Value* operator+( const JitForLoop& loop , int i );
  llvm::Value* operator-( const JitForLoop& loop , int i );
  llvm::Value* operator*( const JitForLoop& loop , int i );

  llvm::Value* operator+( int i , const JitForLoop& loop );
  llvm::Value* operator-( int i , const JitForLoop& loop );
  llvm::Value* operator*( int i , const JitForLoop& loop );


  

  class JitForLoopPower
  {
  public:
    JitForLoopPower( llvm::Value* start );
    llvm::Value * index();
    void end();
  private:
    llvm::BasicBlock * block_outer;
    llvm::BasicBlock * block_loop_cond;
    llvm::BasicBlock * block_loop_body;
    llvm::BasicBlock * block_loop_exit;
    llvm::Value * r_i;
  };


  
  class JitIf
  {
    llvm::BasicBlock * block_exit;
    llvm::BasicBlock * block_true;
    llvm::BasicBlock * block_false;
    bool else_called = false;
  public:
    JitIf( llvm::Value* cond )
    {
      block_exit  = llvm_new_basic_block();
      block_true  = llvm_new_basic_block();
      block_false = llvm_new_basic_block();

      llvm_cond_branch( cond , block_true , block_false );

      llvm_set_insert_point(block_true);
    }

    template<class T>
    JitIf( const WordJIT<T>& cond ): JitIf( cond.elem() ) {}

    
    
    void els()
    {
      else_called=true;
      llvm_branch( block_exit );
      llvm_set_insert_point(block_false);
    }


    void end()
    {
      if (!else_called)
	els();
      llvm_branch( block_exit );
      llvm_set_insert_point(block_exit);
    }

    llvm::BasicBlock* get_block_true() { return block_true; }
    llvm::BasicBlock* get_block_false() { return block_false; }
  };

  


  class JitSwitch
  {
    std::vector<llvm::BasicBlock*> BB;
    llvm::BasicBlock* BBcont;
    llvm::SwitchInst *SI;

  public:
    JitSwitch( llvm::Value* value )
    {
      BB.resize(1);
      BB[0]  = llvm_new_basic_block();
      BBcont = llvm_new_basic_block();
      
      SI = llvm_switch_create(value, BB[0]);
      llvm_set_insert_point( BBcont );
    }

    void case_begin( int val )
    {
      BB.push_back( llvm_new_basic_block() );

      //SI->addCase( llvm_create_const_int(val) , BB.back() );
      llvm_switch_add_case( SI , val , BB.back() );
      
      llvm_set_insert_point( BB.back() );
    }

    void case_end()
    {
      llvm_branch( BBcont );
      llvm_set_insert_point( BBcont );
    }

    void case_default()
    {
      llvm_set_insert_point(BB[0]);
    }

  };





  

  

  

  llvm::Value* jit_ternary( llvm::Value* cond , llvm::Value* val_true , llvm::Value* val_false );
  llvm::Value* llvm_epsilon_1st( int p1 , llvm::Value* j );
  llvm::Value* llvm_epsilon_2nd( int p2 , llvm::Value* i );



  class DeviceMulti
  {
    multi1d<void*> dev_ptr;
    JitParam param;
    
  public:
    DeviceMulti( const multi1d<QDPCache::ArgKey>& ids ):
      dev_ptr( QDP_get_global_cache().get_dev_ptrs( ids ) ),
      param( QDP_get_global_cache().addOwnHostMemNoPage( dev_ptr.size() * sizeof(void*) , dev_ptr.slice() ) )
    {      
    }

    
    QDPCache::ArgKey get_id()
    {
      return param.get_id();
    }

  };


  template<class T, class C> 
  Shape shape_of(const QDPType< T , C >& rhs)
  {
    return rhs.shape();
  }


  
  template<class RHS, class T1>
  Shape shape_of(const QDPExpr<RHS, OLattice<T1> >& rhs)
  {
    DummyJitLeaf jitleaf;

    Shape shape = forEach( rhs , jitleaf , OpEvalShape() ).shape();

    return shape;
  }

  
  template<class RHS, class T1>
  Shape shape_of(const QDPExpr<RHS, OScalar<T1> >& rhs)
  {
    DummyJitLeaf jitleaf;

    Shape shape = forEach( rhs , jitleaf , OpEvalShape() ).shape();

    return shape;
  }



  template<class T, class C>
  bool shape_accepts_rhs(const Shape& lhs, const QDPType< T , C >& rhs)
  {
    return shape_accepts_jit<typename JITType<T>::Type_t>(lhs, rhs.shape());
  }

  template<class RHS, class T1>
  bool shape_accepts_rhs(const Shape& lhs, const QDPExpr<RHS, OLattice<T1> >& rhs)
  {
    return shape_accepts_jit<typename JITType<T1>::Type_t>(lhs, shape_of(rhs));
  }

  template<class RHS, class T1>
  bool shape_accepts_rhs(const Shape& lhs, const QDPExpr<RHS, OScalar<T1> >& rhs)
  {
    return shape_accepts_jit<typename JITType<T1>::Type_t>(lhs, shape_of(rhs));
  }

  template<class RHS>
  void check_shape_accepts_rhs(const char* who, const Shape& lhs, const RHS& rhs)
  {
    if (!shape_accepts_rhs(lhs, rhs))
      incompatible_shapes_exit(who, lhs, shape_of(rhs));
  }



  

  JitValue operator+( const JitValue& l , const JitValue& r );
  JitValue operator-( const JitValue& l , const JitValue& r );
  JitValue operator*( const JitValue& l , const JitValue& r );
  JitValue operator/( const JitValue& l , const JitValue& r );
  JitValue operator%( const JitValue& l , const JitValue& r );

  
  template<Arithmetic A>
  JitValue operator+( A l , const JitValue& r )
  {
    return JitValue(llvm_create_value(l)) + r;
  }

  template<Arithmetic A>
  JitValue operator+( const JitValue& l , A r )
  {
    return l + JitValue(llvm_create_value(r));
  }

  template<Arithmetic A>
  JitValue operator-( A l , const JitValue& r )
  {
    return JitValue(llvm_create_value(l)) - r;
  }

  template<Arithmetic A>
  JitValue operator-( const JitValue& l , A r )
  {
    return l - JitValue(llvm_create_value(r));
  }

  template<Arithmetic A>
  JitValue operator*( A l , const JitValue& r )
  {
    return JitValue(llvm_create_value(l)) * r;
  }

  template<Arithmetic A>
  JitValue operator*( const JitValue& l , A r )
  {
    return l * JitValue(llvm_create_value(r));
  }

  template<Arithmetic A>
  JitValue operator/( A l , const JitValue& r )
  {
    return JitValue(llvm_create_value(l)) / r;
  }

  template<Arithmetic A>
  JitValue operator/( const JitValue& l , A r )
  {
    return l / JitValue(llvm_create_value(r));
  }
  
  template<Arithmetic A>
  JitValue operator%( A l , const JitValue& r )
  {
    return JitValue(llvm_create_value(l)) % r;
  }

  template<Arithmetic A>
  JitValue operator%( const JitValue& l , A r )
  {
    return l % JitValue(llvm_create_value(r));
  }
  

} // namespace

#endif
