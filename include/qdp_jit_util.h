#ifndef QDP_JITFUNCUTIL_H
#define QDP_JITFUNCUTIL_H

namespace QDP {

  llvm::Value *jit_function_preamble_get_idx( const std::vector<ParamRef>& vec );
  std::vector<ParamRef> jit_function_preamble_param( const char* ftype , const char* pretty );

  int jit_util_get_tune_count();

  void jit_get_function(JitFunction&);

  void jit_build_seedToFloat();
  void jit_build_seedMultiply();


  void jit_util_sync_init();
  void jit_util_sync_done();
  void jit_util_sync_copy();

      
  void jit_stats_lattice2dev();
  void jit_stats_lattice2host();
  void jit_stats_jitted();
  void jit_stats_special(int i);

  long get_jit_stats_lattice2dev();
  long get_jit_stats_lattice2host();
  long get_jit_stats_jitted();
  long get_jit_stats_special(int i);
  std::map<int,std::string>& get_jit_stats_special_names();
  
  std::vector<llvm::Value *> llvm_seedMultiply( llvm::Value* a0 , llvm::Value* a1 , llvm::Value* a2 , llvm::Value* a3 , 
						llvm::Value* a4 , llvm::Value* a5 , llvm::Value* a6 , llvm::Value* a7 );

  llvm::Value * llvm_seedToFloat( llvm::Value* a0,llvm::Value* a1,llvm::Value* a2,llvm::Value* a3);


  void jit_launch(JitFunction& function,int th_count,std::vector<QDPCache::ArgKey>& ids);
  void jit_launch_explicit_geom( JitFunction& function , std::vector<QDPCache::ArgKey>& ids , const kernel_geom_t& geom , unsigned int shared );

  std::string jit_util_get_static_dynamic_string( const std::string& pretty );

  void db_tune_write( std::string filename );
  void db_tune_read( std::string filename );


  template<class T>
  typename JITType<T>::Type_t stack_alloc()
  {
    int type_size = JITType<T>::Type_t::Size_t;
    llvm::Value * ptr = llvm_alloca( llvm_get_type< typename WordType< T >::Type_t >() , type_size );
      
    typename JITType<T>::Type_t t_jit_stack;
    t_jit_stack.setup( ptr , JitDeviceLayout::Scalar );

    return t_jit_stack;
  }



  class JitForLoop
  {
  public:
    JitForLoop( int start          , int end ):           JitForLoop( llvm_create_value(start) , llvm_create_value(end) ) {}
    JitForLoop( int start          , llvm::Value*  end ): JitForLoop( llvm_create_value(start) , end ) {}
    JitForLoop( llvm::Value* start , int  end ):          JitForLoop( start , llvm_create_value(end) ) {}
    JitForLoop( llvm::Value* start , llvm::Value*  end );
    llvm::Value * index();
    void end();
  private:
    llvm::BasicBlock * block_outer;
    llvm::BasicBlock * block_loop_cond;
    llvm::BasicBlock * block_loop_body;
    llvm::BasicBlock * block_loop_exit;
    llvm::PHINode * r_i;
  };


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
    llvm::PHINode * r_i;
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
      //block_outer = llvm_get_insert_point();
      block_exit  = llvm_new_basic_block();
      block_true  = llvm_new_basic_block();
      block_false = llvm_new_basic_block();

      llvm_cond_branch( cond , block_true , block_false );

      llvm_set_insert_point(block_true);
    }

    
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
  };

  


  class JitDefer
  {
  public:
    virtual llvm::Value* val() const = 0;
  };


  class JitDeferValue: public JitDefer
  {
    llvm::Value* r;
  public:
    JitDeferValue( llvm::Value* r ): r(r) {}
    virtual llvm::Value* val() const
    {
      return r;
    }
  };


  class JitDeferAdd: public JitDefer
  {
    llvm::Value* r;
    llvm::Value* l;
  public:
    JitDeferAdd( llvm::Value* l , llvm::Value* r ): l(l), r(r) {}
    virtual llvm::Value* val() const
    {
      return llvm_add( l , r );
    }
  };


  class JitDeferArrayTypeIndirection: public JitDefer
  {
    const ParamRef& p;
    llvm::Value* r;
  public:
    JitDeferArrayTypeIndirection( const ParamRef& p , llvm::Value* r ): p(p), r(r) {}
    virtual llvm::Value* val() const
    {
      return llvm_array_type_indirection( p , r );
    }
  };





  llvm::Value* jit_ternary( llvm::Value* cond , llvm::Value*    val_true , llvm::Value*    val_false );
  llvm::Value* jit_ternary( llvm::Value* cond , const JitDefer& val_true , llvm::Value*    val_false );
  llvm::Value* jit_ternary( llvm::Value* cond , llvm::Value*    val_true , const JitDefer& val_false );
  llvm::Value* jit_ternary( llvm::Value* cond , const JitDefer& val_true , const JitDefer& val_false );



  template<class T, int N>
  class JitStackArray
  {
    typedef typename JITType<  T >::Type_t T_jit;
    typedef typename WordType< T >::Type_t W;
    BaseJIT<T_jit,N> array;
    llvm::Value * ptr;

  public:
    template<class C>
    JitStackArray( const C& c )
    {
      // QDPIO::cout << "Size = " << T_jit::Size_t << "\n";
      // QDPIO::cout << "N    = " << N << "\n";

      ptr = llvm_alloca( llvm_get_type<W>() , N * T_jit::Size_t );

      array.setup( ptr , JitDeviceLayout::Scalar );

      for( int i = 0 ; i < N ; ++i )
	array.arrayF(i) = c.elem( i );
      
    }

    JitStackArray()
    {
      // QDPIO::cout << "Size = " << T_jit::Size_t << "\n";
      // QDPIO::cout << "N    = " << N << "\n";

      ptr = llvm_alloca( llvm_get_type<W>() , N * T_jit::Size_t );

      array.setup( ptr , JitDeviceLayout::Scalar );
    }

    
    T_jit elemJIT(int i)
    {
      return array.arrayF(i);
    }

    T_jit elemJIT(llvm::Value * index)
    {
      return array.getJitElem(index);
    }

    T elemREG(llvm::Value * index)
    {
      return array.getRegElem(index);
    }
  };




  



  template<class T, int N>
  class JitStackMatrix
  {
    typedef typename JITType<  T >::Type_t T_jit;
    typedef typename WordType< T >::Type_t W;
    BaseJIT<T_jit,N*N> array;
    llvm::Value * ptr;

  public:
    template<class C>
    JitStackMatrix( const C& c )
    {
      // QDPIO::cout << "Size = " << T_jit::Size_t << "\n";
      // QDPIO::cout << "N    = " << N << "\n";

      ptr = llvm_alloca( llvm_get_type<W>() , N * N * T_jit::Size_t );

      array.setup( ptr , JitDeviceLayout::Scalar );

      for( int i = 0 ; i < N ; ++i )
	for( int j = 0 ; j < N ; ++j )
	  array.arrayF(j+N*i) = c.elem( i , j );
      
    }

    JitStackMatrix()
    {
      // QDPIO::cout << "Size = " << T_jit::Size_t << "\n";
      // QDPIO::cout << "N    = " << N << "\n";

      ptr = llvm_alloca( llvm_get_type<W>() , N * N * T_jit::Size_t );

      array.setup( ptr , JitDeviceLayout::Scalar );
    }

    T_jit elemJIT(int i,int j)
    {
      return array.arrayF( i * N + j );
    }

    T_jit elemJIT(llvm::Value * i,llvm::Value * j)
    {
      return array.getJitElem(llvm_add( llvm_mul( i , llvm_create_value( N ) ) , j ));
    }

    T elemREG(llvm::Value * i,llvm::Value * j)
    {
      return array.getRegElem(llvm_add( llvm_mul( i , llvm_create_value( N ) ) , j ));
    }
    
  };





  template<class T, int N>
  class JitSharedArray
  {
    typedef typename JITType<  T >::Type_t T_jit;
    typedef typename WordType< T >::Type_t W;
    BaseJIT<T_jit,N> array;

  public:
    template<class C>
    JitSharedArray( const C& c )
    {
      // QDPIO::cout << "Size = " << T_jit::Size_t << "\n";
      // QDPIO::cout << "N    = " << N << "\n";

#if 1
      llvm::Value * ptr_base = llvm_get_shared_ptr( llvm_get_type<W>() );
      llvm::Value * ptr_adv = llvm_createGEP( ptr_base ,
					      llvm_mul( llvm_call_special_tidx() ,
							llvm_create_value( N * T_jit::Size_t )
							)
					      );
      array.setup( ptr_adv , JitDeviceLayout::Scalar );
#else
      llvm::Value * ptr_base = llvm_get_shared_ptr( llvm_get_type<W>() );
      IndexDomainVector args;
      args.push_back( make_pair( 256 , llvm_call_special_tidx() ) );  // sitesOnNode irrelevant since Scalar access later
      array.setup( ptr_base , JitDeviceLayout::Scalar , args );
#endif

      for( int i = 0 ; i < N ; ++i )
	array.arrayF(i) = c.elem( i );
      
    }

    JitSharedArray()
    {
      // QDPIO::cout << "Size = " << T_jit::Size_t << "\n";
      // QDPIO::cout << "N    = " << N << "\n";

#if 1
      llvm::Value * ptr_base = llvm_get_shared_ptr( llvm_get_type<W>() );
      llvm::Value * ptr_adv = llvm_createGEP( ptr_base ,
					      llvm_mul( llvm_call_special_tidx() ,
							llvm_create_value( N * T_jit::Size_t )
							)
					      );
      array.setup( ptr_adv , JitDeviceLayout::Scalar );
#else
      llvm::Value * ptr_base = llvm_get_shared_ptr( llvm_get_type<W>() );
      IndexDomainVector args;
      args.push_back( make_pair( 256 , llvm_call_special_tidx() ) );  // sitesOnNode irrelevant since Scalar access later
      array.setup( ptr_base , JitDeviceLayout::Scalar , args );
#endif
    }

    
    T_jit elemJIT(int i)
    {
      return array.arrayF(i);
    }

    T_jit elemJIT(llvm::Value * index)
    {
      return array.getJitElem(index);
    }

    T elemREG(llvm::Value * index)
    {
      return array.getRegElem(index);
    }
  };

  


  template<class T, int N>
  class JitSharedMatrix
  {
    typedef typename JITType<  T >::Type_t T_jit;
    typedef typename WordType< T >::Type_t W;
    BaseJIT<T_jit,N*N> array;

  public:
    template<class C>
    JitSharedMatrix( const C& c )
    {
      // QDPIO::cout << "Size = " << T_jit::Size_t << "\n";
      // QDPIO::cout << "N    = " << N << "\n";

      llvm::Value * ptr_base = llvm_get_shared_ptr( llvm_get_type<W>() );

      llvm::Value * ptr_adv = llvm_createGEP( ptr_base ,
					      llvm_mul( llvm_call_special_tidx() ,
							llvm_create_value( N * N * T_jit::Size_t )
							)
					      );
      
      array.setup( ptr_adv , JitDeviceLayout::Scalar );

      for( int i = 0 ; i < N ; ++i )
	for( int j = 0 ; j < N ; ++j )
	  array.arrayF( j + N * i ) = c.elem( i , j );
      
    }

    JitSharedMatrix()
    {
      // QDPIO::cout << "Size = " << T_jit::Size_t << "\n";
      // QDPIO::cout << "N    = " << N << "\n";

      llvm::Value * ptr_base = llvm_get_shared_ptr( llvm_get_type<W>() );
      
      llvm::Value * ptr_adv = llvm_createGEP( ptr_base ,
					      llvm_mul( llvm_call_special_tidx() ,
							llvm_create_value( N * N * T_jit::Size_t )
							)
					      );
      
      array.setup( ptr_adv , JitDeviceLayout::Scalar );
    }

    
    T_jit elemJIT(int i,int j)
    {
      return array.arrayF( i * N + j );
    }

    T_jit elemJIT(llvm::Value * i,llvm::Value * j)
    {
      return array.getJitElem(llvm_add( llvm_mul( i , llvm_create_value( N ) ) , j ));
    }

    T elemREG(llvm::Value * i,llvm::Value * j)
    {
      return array.getRegElem(llvm_add( llvm_mul( i , llvm_create_value( N ) ) , j ));
    }

    
  };


  
  llvm::Value* llvm_epsilon_1st( int p1 , llvm::Value* j );
  llvm::Value* llvm_epsilon_2nd( int p2 , llvm::Value* i );




  

} // namespace

#endif
