#include "qdp.h"

#include <map>
#include <list>
#include <functional>

#include <iostream>
#include <fstream>



namespace QDP
{

  typedef QDPPoolAllocator<QDPCUDAAllocator> CUDAPOOL;

  namespace
  {
    static CUDAPOOL* __cache_pool_allocator;

    bool __cacheverbose = false;

    CUDAPOOL& get__cache_pool_allocator()
    {
      if (!__cache_pool_allocator) {
	__cache_pool_allocator = new CUDAPOOL();
      }
      return *__cache_pool_allocator;
    }
  }




  bool qdp_cache_get_cache_verbose() { return __cacheverbose; }
  void qdp_cache_set_cache_verbose(bool b) {
    std::cout << "Set cache to verbose\n";
    __cacheverbose = b;
  }

  
#if defined (QDP_ENABLE_MANAGED_MEMORY)
  bool QDPCache::gpu_allocate_base( void ** ptr , size_t n_bytes , int id )
  {
    return QDP_get_global_alloc_cache().allocate( ptr , n_bytes , id );
  }
  void QDPCache::gpu_free_base( void * ptr , size_t n_bytes )
  {
    QDP_get_global_alloc_cache().free( ptr );
  }
#else
  bool QDPCache::gpu_allocate_base( void ** ptr , size_t n_bytes , int id )
  {
    bool ret;
    if ( jit_config_get_max_allocation() < 0   ||  (int)n_bytes <= jit_config_get_max_allocation() )
      {
	ret = get__cache_pool_allocator().allocate( ptr , n_bytes , id );

#ifdef QDP_DEEP_LOG    
	if (ret)
	  {
	    gpu_memset( *ptr , 0 , n_bytes );
	  }
#endif
	return ret;
      }
    else
      {
	bool ret = gpu_malloc( ptr , n_bytes );
#ifdef QDP_DEEP_LOG    
	if (ret)
	  {
	    gpu_memset( *ptr , 0 , n_bytes );
	  }
#endif
	return ret;
      }
  }
  void QDPCache::gpu_free_base( void * ptr , size_t n_bytes )
  {
    if ( jit_config_get_max_allocation() < 0   ||  (int)n_bytes <= jit_config_get_max_allocation() )
      {
	get__cache_pool_allocator().free( ptr );
      }
    else
      {
	return gpu_free( ptr );
      }
  }
#endif

  

  
  
  std::map<size_t,size_t>& QDPCache::get_alloc_count()
  {
    return get__cache_pool_allocator().get_count();
  }



  void QDPCache::enableMemset(unsigned val)
  {
    get__cache_pool_allocator().enableMemset(val);
  }

  void QDPCache::setPoolSize(size_t s)
  {
    get__cache_pool_allocator().setPoolSize(s);
  }

  size_t QDPCache::getPoolSize()
  {
    return get__cache_pool_allocator().getPoolSize();
  }


  
  std::string QDPCache::stringStatus( Status s )
  {
    switch (s)
      {
      case Status::undef:
	return "-";
      case Status::host:
	return "host";
      case Status::device:
	return "device";
      default:
	return "unknown";
      }
  }

  
  void QDPCache::printInfo(int id)
  {
    if (id >= 0)
      {
	assert( vecEntry.size() > id );
	const Entry& e = vecEntry[id];
	printInfo(e);
      }
    else
      {
	QDPIO::cout << "id = -1 (NULL pointer)\n";
      }
  }

  
  void QDPCache::printInfo(const Entry& e)
  {
    QDPIO::cout << "id = " << e.Id
		<< ", size = " << e.size
		<< ", flags = ";
    
    if (e.flags & OwnHostMemory) QDPIO::cout << "own hst|";

    QDPIO::cout << ", ";

    QDPIO::cout << "status = ";
    switch (e.status)
      {
      case Status::undef:
	QDPIO::cout << "-,";
	break;
      case Status::host:
	QDPIO::cout << "host,";
	break;
      case Status::device:
	QDPIO::cout << "device,";
	break;
      default:
	QDPIO::cout << "unknown, \n";
      }

    if (e.location == Location::literal)
      {
	QDPIO::cout << "value = ";
	switch(e.param_type) {
	case LiteralType::float_: QDPIO::cout << (float)e.param.float_ << ", "; break;
	case LiteralType::double_: QDPIO::cout << (double)e.param.double_ << ", "; break;
	case LiteralType::int_: QDPIO::cout << (int)e.param.int_ << ", "; break;
	case LiteralType::int64_: QDPIO::cout << (int64_t)e.param.int64_ << ", "; break;
	case LiteralType::bool_:
	  if (e.param.bool_)
	    QDPIO::cout << "true, ";
	  else
	    QDPIO::cout << "false, ";
	  break;
	default:
	  QDPIO::cout << "(unkown jit param type)\n"; break;
	  assert(0);
	}
      }
    QDPIO::cout << "\n";
  }




  size_t QDPCache::getSize(int id) {
    assert( vecEntry.size() > id );
    const Entry& e = vecEntry[id];
    return e.size;
  }

    
  void QDPCache::growStack()
  {
    const int portion = 1024;
    QDPIO::cout << "growing stack to " << vecEntry.size() + portion << "\n";
    vecEntry.resize( vecEntry.size() + portion );
    for ( int i = 0 ; i < portion ; i++ ) {
      stackFree.push( vecEntry.size()-i-1 );
    }
  }

  int QDPCache::addOwnHostMemStatus( size_t size, const void* ptr , Status st )
  {
    if (size)
      return add_pool( size , Flags::OwnHostMemory , st , ptr , NULL , NULL );
    else
      return -1;    
  }

  int QDPCache::addOwnHostMemNoPage( size_t size, const void *ptr )
  {
    if (size)
      return add_pool( size , Flags::OwnHostMemory | Flags::NoPage , Status::host , ptr , NULL , NULL );
    else
      return -1;
  }

  int QDPCache::addOwnHostMem( size_t size, const void* ptr )
  {
    if (size)
      return add_pool( size , Flags::OwnHostMemory , Status::host , ptr , NULL , NULL );
    else
      return -1;
  }


  int QDPCache::addLayout( size_t size, QDPCache::LayoutFptr func )
  {
    if (size)
      return add_pool( size , Flags::Empty , Status::undef , NULL , NULL , func );
    else
      return -1;
  }



  int QDPCache::add( size_t size )
  {
    if (size)
      return add_pool( size , Flags::Empty , Status::undef , NULL , NULL , NULL );
    else
      return -1;
  }

  

  int QDPCache::getNewId()
  {
    if (stackFree.size() == 0) {
      growStack();
    }

    int Id = stackFree.top();
    assert( vecEntry.size() > Id );

    stackFree.pop();

    return Id;
  }


  

  int QDPCache::addJitParamFloat(float i)
  {
    int Id = getNewId();
    Entry& e = vecEntry[ Id ];
    e.Id           = Id;
    e.size         = 0;
    e.flags        = Flags::Empty;
    e.param.float_ = i;
    e.param_type   = LiteralType::float_;
    e.iterTrack    = lstTracker.insert( lstTracker.end() , Id );
    e.location     = Location::literal;
    return Id;
  }

  int QDPCache::addJitParamDouble(double i)
  {
    int Id = getNewId();
    Entry& e = vecEntry[ Id ];
    e.Id            = Id;
    e.size         = 0;
    e.flags         = Flags::Empty;
    e.param.double_ = i;
    e.param_type   = LiteralType::double_;
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
    e.location     = Location::literal;
    return Id;
  }

  int QDPCache::addJitParamInt(int i)
  {
    int Id = getNewId();
    Entry& e = vecEntry[ Id ];
    e.Id           = Id;
    e.size         = 0;
    e.flags        = Flags::Empty;
    e.param.int_   = i;
    e.param_type   = LiteralType::int_;
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
    e.location     = Location::literal;
    return Id;
  }

  int QDPCache::addJitParamInt64(int64_t i)
  {
    int Id = getNewId();
    Entry& e = vecEntry[ Id ];
    e.Id            = Id;
    e.size         = 0;
    e.flags         = Flags::Empty;
    e.param.int64_  = i;
    e.param_type    = LiteralType::int64_;
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
    e.location     = Location::literal;
    return Id;
  }

  int QDPCache::addJitParamBool(bool i)
  {
    int Id = getNewId();
    Entry& e = vecEntry[ Id ];
    e.Id           = Id;
    e.size         = 0;
    e.flags        = Flags::Empty;
    e.param.bool_  = i;
    e.param_type   = LiteralType::bool_;
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
    e.location     = Location::literal;
    return Id;
  }

  

  namespace QDP_JIT_CACHE
  {
    std::map<void*,int> map_ptr_id;
  }

  
  void QDPCache::signoffViaPtr( void* ptr )
  {
    auto search = QDP_JIT_CACHE::map_ptr_id.find(ptr);
    if(search != QDP_JIT_CACHE::map_ptr_id.end()) {
      signoff( search->second );
      QDP_JIT_CACHE::map_ptr_id.erase(search);
    } else {
      QDPIO::cerr << "QDP Cache: Ptr (sign off via ptr) not found\n";
      QDP_abort(1);
    }
  }


  size_t QDPCache::free_mem()
  {
    return get__cache_pool_allocator().free_mem();
  }

  void QDPCache::print_pool()
  {
    get__cache_pool_allocator().print_pool();
  }


  size_t QDPCache::get_max_allocated()
  {
    return get__cache_pool_allocator().get_max_allocated();
  }
  
  
  int QDPCache::addDeviceStatic( size_t n_bytes )
  {
    void* dummy;
    return addDeviceStatic( &dummy, n_bytes );
  }

  int QDPCache::addDeviceStatic( void** ptr, size_t n_bytes , bool track_ptr )
  {
    int Id = getNewId();
    Entry& e = vecEntry[ Id ];

    e.Id        = Id;
    e.size      = n_bytes;
    e.flags     = Flags::NoPage;
    e.devPtr    = nullptr;
    e.hstPtr    = nullptr;
    e.location  = Location::pool;
    e.fptr      = nullptr;
    
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );

    allocateDeviceMemory(e);

    e.status    = Status::device;

    // set the pointer
    *ptr = e.devPtr;
    
    if (track_ptr)
      {
	// Sanity: make sure the address is not already stored
	auto search = QDP_JIT_CACHE::map_ptr_id.find(e.devPtr);
	assert(search == QDP_JIT_CACHE::map_ptr_id.end());
	
	QDP_JIT_CACHE::map_ptr_id.insert( std::make_pair( e.devPtr , Id ) );
      }

    //gpu_prefetch( e.devPtr, e.size);
    
    return Id;
  }


  int QDPCache::add_pool( size_t size, Flags flags, Status status, const void* hstptr_, const void* devptr_, QDPCache::LayoutFptr func )
  {
    void * hstptr = const_cast<void*>(hstptr_);
    void * devptr = const_cast<void*>(devptr_);
    
    if (stackFree.size() == 0) {
      growStack();
    }

    int Id = stackFree.top();

    assert( vecEntry.size() > Id );
    Entry& e = vecEntry[ Id ];

    e.Id        = Id;
    e.size      = size;
    e.flags     = flags;
    e.hstPtr    = hstptr;
    e.devPtr    = devptr;
    e.fptr      = func;
    e.status    = status;
    e.location  = Location::pool;

    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );

    stackFree.pop();

    return Id;
  }


  


  
  void QDPCache::signoff(int id) {
    assert( vecEntry.size() > id );
    Entry& e = vecEntry[id];
    
    lstTracker.erase( e.iterTrack );

    freeHostMemory( e );
    freeDeviceMemory( e );
    
    stackFree.push( id );
  }
  

  

  void QDPCache::assureOnHost(int id) {
    assert( vecEntry.size() > id );
    Entry& e = vecEntry[id];
    assert(e.location == Location::pool);
    assureHost( e );
  }


  void QDPCache::getHostPtr(void ** ptr , int id) {
    assert( vecEntry.size() > id );
    Entry& e = vecEntry[id];

    assert(e.location == Location::pool);

    assureHost( e );

    *ptr = e.hstPtr;
  }





  void QDPCache::freeHostMemory(Entry& e)
  {
    if (e.location == Location::pool)
      if ( e.hstPtr )
	if ( ! ( e.flags & Flags::OwnHostMemory ) )
	  {
	    QDP::Allocator::theQDPAllocator::Instance().free( e.hstPtr );
	    e.hstPtr=nullptr;
	  }
  }


  
  void QDPCache::allocateHostMemory(Entry& e) {
    assert(e.location == Location::pool);
    
    if ( e.flags & Flags::OwnHostMemory )
      return;
    
    if (e.hstPtr)
      return;
    
    try {
      e.hstPtr = (void*)QDP::Allocator::theQDPAllocator::Instance().allocate( e.size , QDP::Allocator::DEFAULT );
    }
    catch(std::bad_alloc) {
      QDP_error_exit("cache allocateHostMemory: host memory allocator flags=1 failed");
    }
  }


  void QDPCache::allocateDeviceMemory(Entry& e)
  {
    if (e.location != Location::pool)
      return;

    if (e.devPtr)
      return;

#if defined (QDP_ENABLE_MANAGED_MEMORY)
    if (!gpu_allocate_base( &e.devPtr , e.size , e.Id ))
      {
	QDPIO::cerr << "Failed to allocate managed memory. size = " << e.size << std::endl;
	QDP_abort(1);
      }
#else
    while (!gpu_allocate_base( &e.devPtr , e.size , e.Id ))
      {
	if (!spill_lru())
	  {
	    QDPIO::cerr << "Failed to swap any object to host memory. Could not allocate device GPU memory." << std::endl;
	    QDP_abort(1);
	  }
      }
#endif
    //gpu_prefetch( e.devPtr, e.size);
  }



  void QDPCache::freeDeviceMemory(Entry& e)
  {
    if (e.location != Location::pool)
      return;

    if (e.devPtr == nullptr)
      return;

    // This is the same for managed/explicit
    gpu_free_base( e.devPtr , e.size );
    e.devPtr = nullptr;
  }




  void QDPCache::assureDevice(int id) {
    if (id < 0)
      return;
    assert( vecEntry.size() > id );
    Entry& e = vecEntry[id];
    assureDevice(e);
  }

  
  void QDPCache::assureDevice(Entry& e)
  {
    if (e.location != Location::pool)
      return;
	
    // new
    lstTracker.splice( lstTracker.end(), lstTracker , e.iterTrack );

    if (e.status == Status::device)
      {
	//gpu_prefetch( e.devPtr, e.size);
	return;
      }

    allocateDeviceMemory(e);
    
    if ( e.status == Status::host )
      {
	if (qdp_cache_get_cache_verbose())
	  {
	    if (e.size >= 1024*1024)
	      QDPIO::cerr << "copy host --> GPU " << e.size/1024/1024 << " MB\n";
	    else
	      QDPIO::cerr << "copy host --> GPU " << e.size << " bytes\n";
	  }

	if (e.fptr)
	  {
#if defined (QDP_ENABLE_MANAGED_MEMORY)
	    e.fptr( true , e.devPtr , e.hstPtr );
#else
	    char * tmp = new char[e.size];
	    e.fptr(true,tmp,e.hstPtr);
	    gpu_memcpy_h2d( e.devPtr , tmp , e.size );
	    delete[] tmp;
#endif
	  }
	else
	  {
#if defined (QDP_ENABLE_MANAGED_MEMORY)
	    gpu_memcpy( e.devPtr , e.hstPtr , e.size );
#else
	    gpu_memcpy_h2d( e.devPtr , e.hstPtr , e.size );
#endif
	  }
      }

    e.status = Status::device;
  }


  


  void QDPCache::assureHost(Entry& e)
  {
    assert(e.location == Location::pool);
    
    allocateHostMemory(e);

    if ( e.status == Status::device )
      {
	    
	if (qdp_cache_get_cache_verbose())
	  {
	    if (e.size >= 1024*1024)
	      QDPIO::cerr << "copy host <-- GPU " << e.size/1024/1024 << " MB\n";
	    else
	      QDPIO::cerr << "copy host <-- GPU " << e.size << " bytes\n";
	  }
	    
	if (e.fptr)
	  {
#if defined (QDP_ENABLE_MANAGED_MEMORY)
	    e.fptr( false , e.hstPtr , e.devPtr );
#else
	    char * tmp = new char[e.size];
	    gpu_memcpy_d2h( tmp , e.devPtr , e.size );
	    e.fptr(false,e.hstPtr,tmp);
	    delete[] tmp;
#endif
	  }
	else
	  {
#if defined (QDP_ENABLE_MANAGED_MEMORY)
	    gpu_memcpy( e.hstPtr , e.devPtr , e.size );
#else
	    gpu_memcpy_d2h( e.hstPtr , e.devPtr , e.size );
#endif
	  }
      }

    e.status = Status::host;

    freeDeviceMemory(e);
  }

  



  bool QDPCache::spill_lru()
  {
    if (lstTracker.size() < 1)
      return false;

    list<int>::iterator it_key = lstTracker.begin();
      
    bool found=false;
    Entry* e;

    while ( !found  &&  it_key != lstTracker.end() ) {
      e = &vecEntry[ *it_key ];

      found = ( ( e->devPtr != NULL) &&
		( e->location == Location::pool ) &&
		( ! ( e->flags & ( Flags::NoPage ) ) ) );
    
      if (!found)
	it_key++;
    }

    if (found)
      {
	//QDPIO::cout << "spill id = " << e->Id << "   size = " << e->size << "\n";
	assureHost( *e );
	return true;
      }
    else
      {
	//QDPIO::cout << "nothing found to spill " << std::endl;
	
	return false;
      }
  }







  QDPCache::QDPCache() : vecEntry(1024)
  {
    for ( int i = vecEntry.size()-1 ; i >= 0 ; --i )
      {
	stackFree.push(i);
      }
  }


  bool QDPCache::isOnDevice(int id)
  {
    if (id < 0)
      return true;
    
    assert( vecEntry.size() > id );
    Entry& e = vecEntry[id];

    if (e.location != Location::pool)
      return true;

    return e.status == QDPCache::Status::device;
  }



  
  void QDPCache::updateDevPtr(int id, void* ptr)
  {
    assert( vecEntry.size() > id );
    Entry& e = vecEntry[id];

    e.devPtr = ptr;
  }


  
  namespace {
    void* jit_param_null_ptr = NULL;
  }


  
  

#ifdef QDP_BACKEND_ROCM
  namespace 
  {
    void round_off( std::vector<unsigned char>& ret )
    {
      ret.resize( ret.size() + ret.size() % sizeof(void*) );
    }
    template<class T>
    void insert_ret( std::vector<unsigned char>& ret, T t )
    {
      ret.resize( ret.size() + ret.size() % sizeof(T) );
      ret.resize( ret.size() + sizeof(T) );
      *(T*)&ret[ ret.size() - sizeof(T) ] = t;
    }
    template<>
    void insert_ret<bool>( std::vector<unsigned char>& ret, bool t )
    {
      ret.resize( ret.size() + 4 );
      *(bool*)&ret[ ret.size() - 4 ] = t;
    }

  } // namespace
#endif



#ifdef QDP_BACKEND_L0
  namespace 
  {
    template<class T>
    void insert_l0( QDPCache::KernelArgs_t& ret, void* ptr )
    {
      ret.push_back( std::make_pair( (int)(sizeof(T)) , ptr ) );
    }
  }
#endif
  


  QDPCache::KernelArgs_t QDPCache::get_kernel_args(std::vector<int>& ids , bool for_kernel )
  {
#ifdef QDP_BACKEND_ROCM
    if ( for_kernel )
      {
	if ( ids.size() < 2 )
	  {
	    QDPIO::cerr << "get_kernel_args: Size less than 2\n";
	    QDP_abort(1);
	  }
	
	Entry& e0 = vecEntry[ ids[0] ];
	Entry& e1 = vecEntry[ ids[1] ];

	if ( ( e0.param_type != LiteralType::int_ ) ||
	     ( e1.param_type != LiteralType::int_ ) )
	  {
	    QDPIO::cerr << "get_kernel_args: For AMD expected for two parameters being integers\n";
	    QDP_abort(1);
	  }
      }
#endif
    
    // Here we do two cycles through the ids:
    // 1) cache all objects
    // 2) check all are cached
    // This should replace the old 'lock set'

    for ( auto i : ids )
      {
	assureDevice(i);
      }
    
    bool all = true;
    for ( auto i : ids )
      {
	all = all && isOnDevice(i);
      }
    
    if (!all) {
      QDPIO::cerr << "qdp-jit cache: Failed to provide all fields into device memory\n";
      for ( auto i : ids )
	{
	  if (i >= 0)
	    {
	      assert( vecEntry.size() > i );
	      Entry& e = vecEntry[i];
	      QDPIO::cout << "id = " << i << "  size = " << e.size << "  flags = " << e.flags << "  status = ";
	      switch (e.status)
		{
		case Status::undef:
		  QDPIO::cout << "undef,";
		  break;
		case Status::host:
		  QDPIO::cout << "host,";
		  break;
		case Status::device:
		  QDPIO::cout << "device,";
		  break;
		default:
		  QDPIO::cout << "unknown";
		}
	      QDPIO::cout << std::endl;
	    }
	  else
	    {
	      QDPIO::cout << "id = " << i << "\n";
	    }
	}
      QDP_abort(1);
    }

    
    const bool print_param = false;

    if (print_param)
      QDPIO::cout << "Jit function param: ";

    
    QDPCache::KernelArgs_t ret;

    for ( auto i : ids )
      {
	if (i >= 0)
	  {
	    Entry& e = vecEntry[i];
	    if (e.location == Location::literal)
	      {
		if (print_param)
		  {
		    switch(e.param_type) {
		    case LiteralType::float_: QDPIO::cout << (float)e.param.float_ << ", "; break;
		    case LiteralType::double_: QDPIO::cout << (double)e.param.double_ << ", "; break;
		    case LiteralType::int_: QDPIO::cout << (int)e.param.int_ << ", "; break;
		    case LiteralType::int64_: QDPIO::cout << (int64_t)e.param.int64_ << ", "; break;
		    case LiteralType::bool_:
		      if (e.param.bool_)
			QDPIO::cout << "true, ";
		      else
			QDPIO::cout << "false, ";
		      break;
		    default:
		      QDPIO::cout << "(unkown jit param type)\n"; break;
		      assert(0);
		    }
		  }
	  
		assert(for_kernel);
#if defined (QDP_BACKEND_ROCM)
		switch(e.param_type)
		  {
		  case LiteralType::float_ : insert_ret<float>  (ret, e.param.float_ ); break;
		  case LiteralType::double_: insert_ret<double> (ret, e.param.double_ ); break;
		  case LiteralType::int_   : insert_ret<int>    (ret, e.param.int_ );break;
		  case LiteralType::int64_ : insert_ret<int64_t>(ret, e.param.int64_ ); break;
		  case LiteralType::bool_  : insert_ret<bool>   (ret, e.param.bool_ ); ;break;
		  }
#elif defined (QDP_BACKEND_CUDA)
		ret.push_back( &e.param );
#elif defined (QDP_BACKEND_AVX)
		ArgTypes at;
		switch(e.param_type)
		  {
		  case LiteralType::float_ : at.f32 = e.param.float_; break;
		  case LiteralType::double_: at.f64 = e.param.double_; break;
		  case LiteralType::int_   : at.i32 = e.param.int_; break;
		  case LiteralType::int64_ : at.i64 = e.param.int64_; break;
		  case LiteralType::bool_  : at.i1  = e.param.bool_; ;break;
		  }
		ret.push_back( at );
#elif defined (QDP_BACKEND_L0)
		switch(e.param_type)
		  {
		  case LiteralType::float_ : insert_l0<float>  (ret, &e.param.float_ ); break;
		  case LiteralType::double_: insert_l0<double> (ret, &e.param.double_ ); break;
		  case LiteralType::int_   : insert_l0<int>    (ret, &e.param.int_ );break;
		  case LiteralType::int64_ : insert_l0<int64_t>(ret, &e.param.int64_ ); break;
		  case LiteralType::bool_  : insert_l0<bool>   (ret, &e.param.bool_ ); ;break;
		  }
#else
#error "No backend specified"
#endif
	      }
	    else
	      {
		if (print_param)
		  {
		    QDPIO::cout << (size_t)e.devPtr << ", ";
		  }

#if defined (QDP_BACKEND_ROCM)
		insert_ret<void*>( ret , e.devPtr );
#elif defined (QDP_BACKEND_CUDA)
		ret.push_back( for_kernel ? &e.devPtr : e.devPtr );
#elif defined (QDP_BACKEND_AVX)
		ArgTypes at;
		at.ptr  = e.devPtr;
		ret.push_back( at );
#elif defined (QDP_BACKEND_L0)
		insert_l0<void*>( ret , &e.devPtr );
#else
#error "No backend specified"
#endif
	      }
	  }
	else
	  {
	
	    if (print_param)
	      {
		QDPIO::cout << "NULL(id=" << i << "), ";
	      }

	    assert(for_kernel);
#if defined (QDP_BACKEND_ROCM)
	    insert_ret<void*>( ret , &jit_param_null_ptr );
#elif defined (QDP_BACKEND_CUDA)
	    ret.push_back( &jit_param_null_ptr );
#elif defined (QDP_BACKEND_AVX)
	    ArgTypes at;
	    at.ptr  = jit_param_null_ptr;
	    ret.push_back( at );
#elif defined (QDP_BACKEND_L0)
	    insert_l0<void*>( ret , jit_param_null_ptr );
#else
#error "No backend specified"
#endif
	
	  }
      }
    if (print_param)
      QDPIO::cout << "\n";

#if defined (QDP_BACKEND_ROCM)
    round_off(ret);
#endif
    
    return ret;
  }


  multi1d<void*> QDPCache::get_dev_ptrs( const multi1d<int>& ids )
  {
    // Here we do two cycles through the ids:
    // 1) cache all objects
    // 2) check all are cached
    // This should replace the old 'lock set'

    int num = ids.size();

    for ( int i = 0 ; i < num ; ++i )
      {
	assureDevice(ids[i]);
      }
    //QDPIO::cout << "done\n";
    
    bool all = true;
    for ( int i = 0 ; i < num ; ++i )
      {
	all = all && isOnDevice(ids[i]);
      }
    
    if (!all)
      {
	QDPIO::cerr << "qdp-jit cache: Failed to provide all objects in device memory\n";
	QDP_abort(1);
      }
    
    multi1d<void*> ret(ids.size());
    for ( int k = 0 ; k < ids.size() ; ++k )
      {
	int i = ids[k];

	if (i >= 0)
	  {
	    Entry& e = vecEntry[i];
	    if (e.location == Location::literal)
	      {
		QDPIO::cerr << __func__ << ": shouldnt be here\n";
		QDP_abort(1);
	      }
	    else
	      {
		ret[k] = e.devPtr;
	      }
	  }
	else
	  {
	    ret[k] = NULL;
	  }
      }

    return ret;
  }


  void* QDPCache::get_dev_ptr( int id )
  {
    multi1d<int> ids(1);
    ids[0] = id;
    multi1d<void*> tmp = get_dev_ptrs( ids );
    return tmp[0];
  }



  
  namespace {
    static QDPCache* __global_cache;
  }

  QDPCache& QDP_get_global_cache()
  {
    if (!__global_cache) {
      __global_cache = new QDPCache();
    }
    return *__global_cache;
  }



  QDPCache::Flags operator|(QDPCache::Flags a, QDPCache::Flags b)
  {
    return static_cast<QDPCache::Flags>(static_cast<int>(a) | static_cast<int>(b));
  }



} // QDP
