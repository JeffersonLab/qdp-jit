#ifndef QDP_JITFUNCUTIL_H
#define QDP_JITFUNCUTIL_H

namespace QDP {


  //template<class T> ParamRef jit_add_param();

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



    
  int jit_util_get_tune_count();

  void jit_get_function(JitFunction&);

  void jit_util_sync_init();
  void jit_util_sync_done();
  void jit_util_sync_copy();

      
  

  void jit_launch       ( JitFunction& function , int th_count , std::vector<QDPCache::ArgKey>& ids );
  void jit_launch_explicit_geom( JitFunction& function , std::vector<QDPCache::ArgKey>& ids , const kernel_geom_t& geom , unsigned int shared );

#ifdef QDP_DEEP_LOG
  void jit_deep_log(JitFunction& f);
#endif

  void db_tune_write( std::string filename );
  void db_tune_read( std::string filename );


  template<class T>
  typename JITType<T>::Type_t stack_alloc()
  {
    int type_size = JITType<T>::Type_t::ScalarSize_t;
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
    llvm::Value * r_i;
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
      if (IsWordVec<T>::value)
	ptr = llvm_alloca( llvm_get_vectype<W>() , N * T_jit::ScalarSize_t );
      else
	ptr = llvm_alloca( llvm_get_type<W>() , N * T_jit::ScalarSize_t );
      
      array.setup( ptr , JitDeviceLayout::Scalar );

      for( int i = 0 ; i < N ; ++i )
	array.arrayF(i) = c.elem( i );
      
    }

    JitStackArray()
    {
      if (IsWordVec<T>::value)
	ptr = llvm_alloca( llvm_get_vectype<W>() , N * T_jit::ScalarSize_t );
      else
	ptr = llvm_alloca( llvm_get_type<W>() , N * T_jit::ScalarSize_t );

      array.setup( ptr , JitDeviceLayout::Scalar );
    }

    
    T_jit elemJITvalue(llvm::Value * index)
    {
      return array.getJitElem(index);
    }

    T_jit elemJITint(int i)
    {
      return array.arrayF(i);
    }

    T elemREGvalue(llvm::Value * index)
    {
      return array.getRegElem(index);
    }

    T elemREGint(int i)
    {
      return array.getRegElem(llvm_create_value(i));
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
      // QDPIO::cout << "Size = " << T_jit::ScalarSize_t << "\n";
      // QDPIO::cout << "N    = " << N << "\n";

      ptr = llvm_alloca( llvm_get_type<W>() , N * N * T_jit::ScalarSize_t );

      array.setup( ptr , JitDeviceLayout::Scalar );

      for( int i = 0 ; i < N ; ++i )
	for( int j = 0 ; j < N ; ++j )
	  array.arrayF(j+N*i) = c.elem( i , j );
      
    }

    JitStackMatrix()
    {
      // QDPIO::cout << "Size = " << T_jit::ScalarSize_t << "\n";
      // QDPIO::cout << "N    = " << N << "\n";

      ptr = llvm_alloca( llvm_get_type<W>() , N * N * T_jit::ScalarSize_t );

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




#if 0
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
      // QDPIO::cout << "Size = " << T_jit::ScalarSize_t << "\n";
      // QDPIO::cout << "N    = " << N << "\n";

      llvm::Value * ptr_base = llvm_get_shared_ptr( llvm_get_type<W>() );
      llvm::Value * ptr_adv = llvm_createGEP( ptr_base ,
					      llvm_mul( llvm_call_special_tidx() ,
							llvm_create_value( N * T_jit::ScalarSize_t )
							)
					      );
      array.setup( ptr_adv , JitDeviceLayout::Scalar );

      for( int i = 0 ; i < N ; ++i )
	array.arrayF(i) = c.elem( i );
      
    }

    JitSharedArray()
    {
      // QDPIO::cout << "Size = " << T_jit::ScalarSize_t << "\n";
      // QDPIO::cout << "N    = " << N << "\n";

      llvm::Value * ptr_base = llvm_get_shared_ptr( llvm_get_type<W>() );
      llvm::Value * ptr_adv = llvm_createGEP( ptr_base ,
					      llvm_mul( llvm_call_special_tidx() ,
							llvm_create_value( N * T_jit::ScalarSize_t )
							)
					      );
      array.setup( ptr_adv , JitDeviceLayout::Scalar );
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
      // QDPIO::cout << "Size = " << T_jit::ScalarSize_t << "\n";
      // QDPIO::cout << "N    = " << N << "\n";

      llvm::Value * ptr_base = llvm_get_shared_ptr( llvm_get_type<W>() );

      llvm::Value * ptr_adv = llvm_createGEP( ptr_base ,
					      llvm_mul( llvm_call_special_tidx() ,
							llvm_create_value( N * N * T_jit::ScalarSize_t )
							)
					      );
      
      array.setup( ptr_adv , JitDeviceLayout::Scalar );

      for( int i = 0 ; i < N ; ++i )
	for( int j = 0 ; j < N ; ++j )
	  array.arrayF( j + N * i ) = c.elem( i , j );
      
    }

    JitSharedMatrix()
    {
      // QDPIO::cout << "Size = " << T_jit::ScalarSize_t << "\n";
      // QDPIO::cout << "N    = " << N << "\n";

      llvm::Value * ptr_base = llvm_get_shared_ptr( llvm_get_type<W>() );
      
      llvm::Value * ptr_adv = llvm_createGEP( ptr_base ,
					      llvm_mul( llvm_call_special_tidx() ,
							llvm_create_value( N * N * T_jit::ScalarSize_t )
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
#endif
  

  class WorkgroupGuard
  {
    ParamRef p_th_count;
  public:
    WorkgroupGuard()
    {
#if defined (QDP_BACKEND_ROCM) || defined (QDP_BACKEND_CUDA) || defined (QDP_BACKEND_L0)
      p_th_count = llvm_add_param<int>();
#endif
    }
    void check( llvm::Value* r_idx )
    {
#if defined (QDP_BACKEND_ROCM) || defined (QDP_BACKEND_CUDA) || defined (QDP_BACKEND_L0)
      llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
      llvm_cond_exit( llvm_ge( r_idx , r_th_count ) );
#endif
    }
  };


  class WorkgroupGuardExec
  {
#if defined (QDP_BACKEND_ROCM) || defined (QDP_BACKEND_CUDA) || defined (QDP_BACKEND_L0)
    JitParam jit_th_count;
#endif
  public:
#if defined (QDP_BACKEND_ROCM) || defined (QDP_BACKEND_CUDA) || defined (QDP_BACKEND_L0)
    WorkgroupGuardExec( int th_count ): jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) )
    {
    }
#else
    WorkgroupGuardExec( int th_count )
    {
    }
#endif
    void check(std::vector<QDPCache::ArgKey>& ids)
    {
#if defined (QDP_BACKEND_ROCM) || defined (QDP_BACKEND_CUDA) || defined (QDP_BACKEND_L0)
      ids.push_back( jit_th_count.get_id() );
#endif
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

  

} // namespace

#endif
