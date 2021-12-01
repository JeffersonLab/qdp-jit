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


  std::vector<QDPCache::Entry>& QDPCache::get__vec_backed() {
    return __vec_backed;
  }


  bool qdp_cache_get_cache_verbose() { return __cacheverbose; }
  void qdp_cache_set_cache_verbose(bool b) {
    std::cout << "Set cache to verbose\n";
    __cacheverbose = b;
  }


#if defined(QDP_BACKEND_CUDA) || defined(QDP_BACKEND_ROCM)
  bool QDPCache::gpu_allocate_base( void ** ptr , size_t n_bytes , int id )
  {
    if ( jit_config_get_max_allocation() < 0   ||  (int)n_bytes <= jit_config_get_max_allocation() )
      {
	bool ret = get__cache_pool_allocator().allocate( ptr , n_bytes , id );

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
	return get__cache_pool_allocator().free( ptr );
      }
    else
      {
	return gpu_free( ptr );
      }
  }
#else
  bool QDPCache::gpu_allocate_base( void ** ptr , size_t n_bytes , int id )
  {
    gpu_host_alloc( ptr , n_bytes );

#ifdef QDP_DEEP_LOG    
    gpu_memset( *ptr , 0 , n_bytes );
#endif

    return true;
  }
  void QDPCache::gpu_free_base( void * ptr , size_t n_bytes )
  {
    gpu_host_free( ptr );
  }
#endif

  

  
  
  std::map<size_t,size_t>& QDPCache::get_alloc_count()
  {
    return get__cache_pool_allocator().get_count();
  }


  int QDPCache::getPoolDefrags()
  {
    return get__cache_pool_allocator().defrag_occurrences();
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
    if (e.flags & OwnDeviceMemory) QDPIO::cout << "own dev|";
    if (e.flags & JitParam) QDPIO::cout << "param|";
    if (e.flags & Static) QDPIO::cout << "static|";
    if (e.flags & Multi) QDPIO::cout << "multi|";
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

    if (e.flags & QDPCache::Flags::JitParam)
      {
	QDPIO::cout << "value = ";
	switch(e.param_type) {
	case JitParamType::float_: QDPIO::cout << (float)e.param.float_ << ", "; break;
	case JitParamType::double_: QDPIO::cout << (double)e.param.double_ << ", "; break;
	case JitParamType::int_: QDPIO::cout << (int)e.param.int_ << ", "; break;
	case JitParamType::int64_: QDPIO::cout << (int64_t)e.param.int64_ << ", "; break;
	case JitParamType::bool_:
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
    vecEntry.resize( vecEntry.size() + portion );
    for ( int i = 0 ; i < portion ; i++ ) {
      stackFree.push( vecEntry.size()-i-1 );
    }
  }

  int QDPCache::registrateOwnHostMemStatus( size_t size, const void* ptr , Status st )
  {
    if (size)
      return add( size , Flags::OwnHostMemory , st , ptr , NULL , NULL );
    else
      return -1;    
  }

  int QDPCache::registrateOwnHostMemNoPage( size_t size, const void *ptr )
  {
    if (size)
      return add( size , Flags::OwnHostMemory | Flags::NoPage , Status::host , ptr , NULL , NULL );
    else
      return -1;
  }

  int QDPCache::registrateOwnHostMem( size_t size, const void* ptr , QDPCache::LayoutFptr func )
  {
    if (size)
      return add( size , Flags::OwnHostMemory , Status::host , ptr , NULL , func );
    else
      return -1;
  }

  int QDPCache::registrate( size_t size, unsigned flags, QDPCache::LayoutFptr func )
  {
    if (size)
      return add( size , Flags::Empty , Status::undef , NULL , NULL , func );
    else
      return -1;
  }

  int QDPCache::registrate_no_layout_conversion( size_t size )
  {
    if (size)
      return add( size , Flags::Empty , Status::undef , NULL , NULL , NULL );
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
    e.flags        = Flags::JitParam;
    e.param.float_ = i;
    e.param_type   = JitParamType::float_;
    e.multi.clear();
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
    return Id;
  }

  int QDPCache::addJitParamDouble(double i)
  {
    int Id = getNewId();
    Entry& e = vecEntry[ Id ];
    e.Id            = Id;
    e.size         = 0;
    e.flags         = Flags::JitParam;
    e.param.double_ = i;
    e.param_type   = JitParamType::double_;
    e.multi.clear();
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
    return Id;
  }

  int QDPCache::addJitParamInt(int i)
  {
    int Id = getNewId();
    Entry& e = vecEntry[ Id ];
    e.Id           = Id;
    e.size         = 0;
    e.flags        = Flags::JitParam;
    e.param.int_   = i;
    e.param_type   = JitParamType::int_;
    e.multi.clear();
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
    return Id;
  }

  int QDPCache::addJitParamInt64(int64_t i)
  {
    int Id = getNewId();
    Entry& e = vecEntry[ Id ];
    e.Id            = Id;
    e.size         = 0;
    e.flags         = Flags::JitParam;
    e.param.int64_  = i;
    e.param_type    = JitParamType::int64_;
    e.multi.clear();
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
    return Id;
  }

  int QDPCache::addJitParamBool(bool i)
  {
    int Id = getNewId();
    Entry& e = vecEntry[ Id ];
    e.Id           = Id;
    e.size         = 0;
    e.flags        = Flags::JitParam;
    e.param.bool_  = i;
    e.param_type   = JitParamType::bool_;
    e.multi.clear();
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
    return Id;
  }

  
  int QDPCache::addMulti( const multi1d<int>& ids )
  {
    int Id = getNewId();
    Entry& e = vecEntry[ Id ];
    e.Id        = Id;
    e.flags     = Flags::Multi;
    e.hstPtr    = nullptr;
    e.devPtr    = nullptr;
    e.fptr      = nullptr;
    e.size      = ids.size() * sizeof(void*);
    
    e.multi.clear();
    e.multi.resize(ids.size());
    for( int i = 0 ; i < ids.size() ; ++i )
      e.multi[i] = ids[i];

    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );

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
      QDPIO::cout << "QDP Cache: Ptr (sign off via ptr) not found\n";
      QDP_error_exit("giving up");
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

  void QDPCache::defrag()
  {
    get__cache_pool_allocator().defrag();
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

    if ( qdp_jit_config_defrag() )
      {
	bool allocated = get__cache_pool_allocator().allocate_fixed( ptr , n_bytes , Id );
	
	while (!allocated)
	  {
	    bool defrag = false;
	    
	    if (!spill_lru())
	      {
		get__cache_pool_allocator().defrag();

		defrag = true;
	      }

	    allocated = get__cache_pool_allocator().allocate_fixed( ptr , n_bytes , Id );
	    
	    if ( !allocated && defrag )
	      {
		QDPIO::cout << "Can't allocate static memory even after pool defragmentation." << std::endl;
		QDP_abort(1);
	      }
	  }
      }
    else
      {
	//while (!get__cache_pool_allocator().allocate( ptr , n_bytes , Id ))
	while (!gpu_allocate_base( ptr , n_bytes , Id ))
	  {
	    if (!spill_lru())
	      {
		QDP_error_exit("cache allocate_device_static: can't spill LRU object");
	      }
	  }
      }
    

    e.Id        = Id;
    e.size      = n_bytes;
    e.flags     = Flags::Static;
    e.devPtr    = *ptr;
    e.multi.clear();

    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );

    if (track_ptr)
      {
	// Sanity: make sure the address is not already stored
	auto search = QDP_JIT_CACHE::map_ptr_id.find(e.devPtr);
	assert(search == QDP_JIT_CACHE::map_ptr_id.end());
	
	QDP_JIT_CACHE::map_ptr_id.insert( std::make_pair( e.devPtr , Id ) );
      }
    gpu_prefetch( e.devPtr, e.size);
    
    return Id;
  }


  int QDPCache::add( size_t size, Flags flags, Status status, const void* hstptr_, const void* devptr_, QDPCache::LayoutFptr func )
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
    e.multi.clear();
    e.status    = status;

    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );

    stackFree.pop();

    return Id;
  }


  


  
  void QDPCache::signoff(int id) {
    assert( vecEntry.size() > id );
    Entry& e = vecEntry[id];
    
    lstTracker.erase( e.iterTrack );

    if ( !( e.flags & ( Flags::JitParam | Flags::Static | Flags::Multi ) ) )
      {
	freeHostMemory( e );
      }
    
    if ( !( e.flags & ( Flags::JitParam ) ) )
      {
	freeDeviceMemory( e );
      }

    if ( e.flags & ( Flags::Multi ) )
      {
	e.multi.clear();
      }
    
    stackFree.push( id );
  }
  

  

  void QDPCache::assureOnHost(int id) {
    assert( vecEntry.size() > id );
    Entry& e = vecEntry[id];
    assert(e.flags != Flags::JitParam);
    assert(e.flags != Flags::Static);
    assureHost( e );
  }


  void QDPCache::getHostPtr(void ** ptr , int id) {
    assert( vecEntry.size() > id );
    Entry& e = vecEntry[id];

    assert(e.flags != Flags::JitParam);
    assert(e.flags != Flags::Static);

    assureHost( e );

    *ptr = e.hstPtr;
  }





  void QDPCache::freeHostMemory(Entry& e)
  {
    if ( e.flags & ( Flags::OwnHostMemory ) )
      return;
    
    if (!e.hstPtr)
      return;

    assert(e.flags != Flags::JitParam);
    assert(e.flags != Flags::Static);

    QDP::Allocator::theQDPAllocator::Instance().free( e.hstPtr );
    e.hstPtr=nullptr;
  }


  
  void QDPCache::allocateHostMemory(Entry& e) {
    assert(e.flags != Flags::JitParam);
    assert(e.flags != Flags::Static);
    
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


  void QDPCache::allocateDeviceMemory(Entry& e) {
    assert(e.flags != Flags::JitParam);
    assert(e.flags != Flags::Static);
    
    if ( e.flags & Flags::OwnDeviceMemory )
      return;

    if (e.devPtr)
      return;

    //while (!get__cache_pool_allocator().allocate( &e.devPtr , e.size , e.Id )) {
    while (!gpu_allocate_base( &e.devPtr , e.size , e.Id )) {
      if (!spill_lru()) {
	QDP_info("Device pool:");
	//get__cache_pool_allocator().printListPool();
	//printLockSets();
	QDP_error_exit("cache assureDevice: can't spill LRU object. Out of GPU memory!");
      }
    }
    gpu_prefetch( e.devPtr, e.size);
  }



  void QDPCache::freeDeviceMemory(Entry& e) {
    //QDPIO::cout << "free size = " << e.size << "\n";
    assert(e.flags != Flags::JitParam);
    
    if ( e.flags & Flags::OwnDeviceMemory )
      return;

    if (!e.devPtr)
      return;

    //get__cache_pool_allocator().free( e.devPtr );
    gpu_free_base( e.devPtr , e.size );
    e.devPtr = NULL;
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
    if (e.flags & Flags::JitParam)
      return;
    if (e.flags & Flags::Static)
      return;

    // new
    lstTracker.splice( lstTracker.end(), lstTracker , e.iterTrack );

    //std::cout << "id = " << e.Id << "  flags = " << e.flags << "\n";

    if (e.flags & Flags::Multi)
      {
	allocateDeviceMemory(e);
      }
    else
      {
	if (e.status == Status::device){
    gpu_prefetch( e.devPtr, e.size);
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

	    if (e.fptr) {

	      char * tmp = new char[e.size];
	      e.fptr(true,tmp,e.hstPtr);
	      gpu_memcpy_h2d( e.devPtr , tmp , e.size );
	      delete[] tmp;

	    } else {
	      gpu_memcpy_h2d( e.devPtr , e.hstPtr , e.size );
	    }

	  }

	e.status = Status::device;
      }
  }


  


  void QDPCache::assureHost(Entry& e)
  {
    assert(e.flags != Flags::JitParam);
    assert(e.flags != Flags::Static);
    
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
	    
	if (e.fptr) {
	  char * tmp = new char[e.size];
	  gpu_memcpy_d2h( tmp , e.devPtr , e.size );
	  e.fptr(false,e.hstPtr,tmp);
	  delete[] tmp;
	} else {
	  gpu_memcpy_d2h( e.hstPtr , e.devPtr , e.size );
	}
      }

    e.status = Status::host;

    freeDeviceMemory(e);
  }

  



  bool QDPCache::spill_lru() {
    if (lstTracker.size() < 1)
      return false;

    list<int>::iterator it_key = lstTracker.begin();
      
    bool found=false;
    Entry* e;

    while ( !found  &&  it_key != lstTracker.end() ) {
      e = &vecEntry[ *it_key ];

      found = ( ( e->devPtr != NULL) &&
		( ! ( e->flags & ( Flags::JitParam | Flags::Static | Flags::Multi | Flags::OwnDeviceMemory | Flags::NoPage ) ) ) );
    
      if (!found)
	it_key++;
    }

    if (found) {
      //QDPIO::cout << "spill id = " << e->Id << "   size = " << e->size << "\n";
      assureHost( *e );
      return true;
    } else {
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
    // An id < 0 indicates a NULL pointer
    if (id < 0)
      return true;
    
    assert( vecEntry.size() > id );
    Entry& e = vecEntry[id];

    if (e.flags & QDPCache::JitParam)
      return true;

    if (e.flags & QDPCache::Static)
      return true;

    if (e.flags & QDPCache::Multi)
      {
	return e.devPtr != NULL;
      }

    return ( e.status == QDPCache::Status::device );
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
    template<class T>
    void insert_ret( std::vector<unsigned char>& ret, T t )
    {
      if (std::is_pointer<T>::value)
	{
	  //std::cout << "size is " << ret.size() << " resizing by " << ret.size() % sizeof(T) << " to meet padding requirements\n";
	  ret.resize( ret.size() + ret.size() % sizeof(T) );
	  ret.resize( ret.size() + sizeof(T) );
	  *(T*)&ret[ ret.size() - sizeof(T) ] = t;
	  //std::cout << "inserted pointer (size " << sizeof(T) << ") " << t << "\n";
	}
      else
	{
	  ret.resize( ret.size() + sizeof(T) );
	  *(T*)&ret[ ret.size() - sizeof(T) ] = t;
	  //std::cout << "inserted (size " << sizeof(T) << ") " << t << "\n";
	}
    }
    template<>
    void insert_ret<bool>( std::vector<unsigned char>& ret, bool t )
    {
      ret.resize( ret.size() + 4 );
      *(bool*)&ret[ ret.size() - 4 ] = t;
      //std::cout << "inserted bool (as size 4) " << t << "\n";
    }

  } // namespace
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

	if ( ( e0.param_type != JitParamType::int_ ) ||
	     ( e1.param_type != JitParamType::int_ ) )
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

    //QDPIO::cout << "ids: ";
    std::vector<int> allids;
    for ( auto i : ids )
      {
	allids.push_back(i);
	if (i >= 0)
	  {
	    //QDPIO::cout << i << " ";
	    assert( vecEntry.size() > i );
	    Entry& e = vecEntry[i];
	    if (e.flags & QDPCache::Flags::Multi)
	      {
		for ( auto ak : e.multi )
		  {
		    allids.push_back( ak );
		  }
	      }
	  }
      }
    //QDPIO::cout << "\n";

    //QDPIO::cout << "assureDevice:\n";
    for ( auto i : allids )
      {
	//QDPIO::cout << "id = " << i.id << ", elem = " << i.elem << "\n";
	assureDevice(i);
      }
    //QDPIO::cout << "done\n";
    
    bool all = true;
    for ( auto i : allids )
      {
	all = all && isOnDevice(i);
      }
    
    if (!all) {
      QDPIO::cout << "It was not possible to load all objects required by the kernel into device memory\n";
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
		  QDPIO::cout << "unkown\n";
		}
	    }
	  else
	    {
	      QDPIO::cout << "id = " << i << "\n";
	    }
	}
      QDP_error_exit("giving up");
    }


    // Handle multi-ids
    for ( auto i : ids )
      {
	if (i >= 0)
	  {
	    Entry& e = vecEntry[i];
	    if (e.flags & QDPCache::Flags::Multi)
	      {
		assert( isOnDevice(i) );
		multi1d<void*> dev_ptr(e.multi.size());

		int count=0;
		for( auto ak : e.multi )  // q == id
		  {
		    if ( ak >= 0 )
		      {
			assert( vecEntry.size() > ak );
			Entry& qe = vecEntry[ ak ];
			assert( ! (qe.flags & QDPCache::Flags::Multi) );
			assert( isOnDevice( ak ) );
			dev_ptr[count++] = qe.devPtr;
		      }
		    else
		      {
			dev_ptr[count++] = NULL;
		      }
		  }
		
		if (qdp_cache_get_cache_verbose())
		  {
		    QDPIO::cerr << "copy host --> GPU " << e.multi.size() * sizeof(void*) << " bytes (kernel args, multi)\n";
		  }
		
		gpu_memcpy_h2d( e.devPtr , dev_ptr.slice() , e.multi.size() * sizeof(void*) );
		//QDPIO::cout << "multi-ids: copied elements = " << e.multi.size() << "\n";
	      }
	  }
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
	    if (e.flags & QDPCache::Flags::JitParam)
	      {
		if (print_param)
		  {
		    switch(e.param_type) {
		    case JitParamType::float_: QDPIO::cout << (float)e.param.float_ << ", "; break;
		    case JitParamType::double_: QDPIO::cout << (double)e.param.double_ << ", "; break;
		    case JitParamType::int_: QDPIO::cout << (int)e.param.int_ << ", "; break;
		    case JitParamType::int64_: QDPIO::cout << (int64_t)e.param.int64_ << ", "; break;
		    case JitParamType::bool_:
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
		  case JitParamType::float_ : insert_ret<float>  (ret, e.param.float_ ); break;
		  case JitParamType::double_: insert_ret<double> (ret, e.param.double_ ); break;
		  case JitParamType::int_   : insert_ret<int>    (ret, e.param.int_ );break;
		  case JitParamType::int64_ : insert_ret<int64_t>(ret, e.param.int64_ ); break;
		  case JitParamType::bool_  : insert_ret<bool>   (ret, e.param.bool_ ); ;break;
		  }
#elif defined (QDP_BACKEND_CUDA)
		ret.push_back( &e.param );
#elif defined (QDP_BACKEND_AVX)
		ArgTypes at;
		switch(e.param_type)
		  {
		  case JitParamType::float_ : at.f32 = e.param.float_; break;
		  case JitParamType::double_: at.f64 = e.param.double_; break;
		  case JitParamType::int_   : at.i32 = e.param.int_; break;
		  case JitParamType::int64_ : at.i64 = e.param.int64_; break;
		  case JitParamType::bool_  : at.i1  = e.param.bool_; ;break;
		  }
		ret.push_back( at );
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
#else
#error "No backend specified"
#endif
	
	  }
      }
    if (print_param)
      QDPIO::cout << "\n";

    return ret;
  }


  std::vector<void*> QDPCache::get_dev_ptrs( const std::vector<int>& ids )
  {
    // Here we do two cycles through the ids:
    // 1) cache all objects
    // 2) check all are cached
    // This should replace the old 'lock set'

    //QDPIO::cout << "ids: ";
    std::vector<int> allids;
    for ( auto i : ids )
      {
	allids.push_back(i);
	if (i >= 0)
	  {
	    //QDPIO::cout << i << " ";
	    assert( vecEntry.size() > i );
	    Entry& e = vecEntry[i];
	    if (e.flags & QDPCache::Flags::Multi)
	      {
		for ( auto ak : e.multi )
		  {
		    allids.push_back( ak );
		  }
	      }
	  }
      }
    //QDPIO::cout << "\n";

    int run = 0;
    bool all;
      
    do
      {
	// This never happens
	// Can't defrag from pool's side as recv buffers are
	// allocated as device static.
	if (run == 1)
	  {
	    get__cache_pool_allocator().defrag();
	  }
	
	//QDPIO::cout << "assureDevice:\n";
	for ( auto i : allids ) {
	  //QDPIO::cout << "id = " << i << "\n";
	  assureDevice(i);
	}
	//QDPIO::cout << "done\n";
    
	all = true;
	for ( auto i : allids )
	  all = all && isOnDevice(i);
	run++;
      }
    while ( (run < 1) && (!all) );
    
    if (!all) {
      QDPIO::cerr << "It was not possible to load all objects required by the kernel into device memory\n";
      QDP_abort(1);
    }


    // Handle multi-ids
    for ( auto i : ids )
      {
	if (i >= 0)
	  {
	    Entry& e = vecEntry[i];
	    if (e.flags & QDPCache::Flags::Multi)
	      {
		assert( isOnDevice(i) );
		multi1d<void*> dev_ptr(e.multi.size());

		int count=0;
		for( auto ak : e.multi )  // q == id
		  {
		    if ( ak >= 0 )
		      {
			assert( vecEntry.size() > ak );
			Entry& qe = vecEntry[ ak ];
			assert( ! (qe.flags & QDPCache::Flags::Multi) );
			assert( isOnDevice( ak ) );
			dev_ptr[count++] = qe.devPtr;
		      }
		    else
		      {
			dev_ptr[count++] = NULL;
		      }
		  }
		
		if (qdp_cache_get_cache_verbose())
		  {
		    QDPIO::cerr << "copy host --> GPU " << e.multi.size() * sizeof(void*) << " bytes (kernel args, multi)\n";
		  }
		
		gpu_memcpy_h2d( e.devPtr , dev_ptr.slice() , e.multi.size() * sizeof(void*) );
		//QDPIO::cout << "multi-ids: copied elements = " << e.multi.size() << "\n";
	      }
	  }
      }

    std::vector<void*> ret;
    for ( auto i : ids )
      {
	if (i >= 0)
	  {
	    Entry& e = vecEntry[i];
	    if (e.flags & QDPCache::Flags::JitParam)
	      {
		QDPIO::cerr << __func__ << ": shouldnt be here\n";
		QDP_abort(1);
	      }
	    else
	      {
		ret.push_back( e.devPtr );
	      }
	  }
	else
	  {
	    QDPIO::cerr << __func__ << ": shouldnt be here neither\n";
	    QDP_abort(1);
	  }
      }

    return ret;
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



