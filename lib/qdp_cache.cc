#include "qdp.h"

#include <map>
#include <list>
#include <functional>

#include <iostream>
#include <fstream>



namespace QDP
{

  namespace STACK {
    bool stack_scalars = false;
    void* stack_ptr = NULL;
    void* current;
    size_t size;
    int id;
  }
  
  void qdp_stack_scalars_start( size_t size )
  {
    assert(!STACK::stack_scalars);
    
    STACK::id = QDP_get_global_cache().addDeviceStatic( &STACK::stack_ptr, size );
    STACK::current = STACK::stack_ptr;
    STACK::stack_scalars = true;
    STACK::size = size;
  }
  
  void qdp_stack_scalars_end()
  {
    assert(STACK::stack_scalars);
    STACK::stack_scalars = false;
  }

  void qdp_stack_scalars_free_stack()
  {
    assert(!STACK::stack_scalars);
    assert(STACK::stack_ptr);
    QDP_get_global_cache().signoff( STACK::id );
    STACK::stack_ptr = NULL;
  }

  bool qdp_stack_scalars_enabled()
  {
    return STACK::stack_scalars;
  }

  void* qdp_stack_scalars_alloc( size_t size )
  {
    if ( (size_t)STACK::current + size - (size_t)STACK::stack_ptr > STACK::size )
      QDP_error_exit("out of memory (stack scalars)");
      
    void* ret = STACK::current;
    STACK::current = (void*)((size_t)STACK::current + size);
    
    return ret;
  }


  



  struct QDPCache::Entry {
    union JitParamUnion {
      void *  ptr;
      float   float_;
      int     int_;
      int64_t int64_;
      double  double_;
      bool    bool_;
    };

    int    Id;
    size_t size;
    Flags  flags;
    void*  hstPtr;  // NULL if not allocated
    void*  devPtr;  // NULL if not allocated
    Status status;
    int    lockCount;
    list<int>::iterator iterTrack;
    LayoutFptr fptr;
    JitParamUnion param;
    QDPCache::JitParamType param_type;
    std::vector<int> multi;
  };




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



  int QDPCache::registrateOwnHostMem( size_t size, const void* ptr , QDPCache::LayoutFptr func )
  {
    return add( size , Flags::OwnHostMemory , Status::host , ptr , NULL , func );
  }

  int QDPCache::registrate( size_t size, unsigned flags, QDPCache::LayoutFptr func )
  {
    return add( size , Flags::Empty , Status::undef , NULL , NULL , func );
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
    e.status    = Status::undef;
    e.hstPtr    = NULL;
    e.devPtr    = NULL;
    e.fptr      = NULL;
    e.size      = ids.size() * sizeof(void*);
    
    e.multi.clear();
    e.multi.resize(ids.size());
    int count=0;
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
    }
  }

  
  int QDPCache::addDeviceStatic( void** ptr, size_t n_bytes , bool track_ptr )
  {
    int Id = getNewId();
    Entry& e = vecEntry[ Id ];

    while (!pool_allocator.allocate( ptr , n_bytes )) {
      if (!spill_lru()) {
	QDP_error_exit("cache allocate_device_static: can't spill LRU object");
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
    e.status    = status;
    e.hstPtr    = hstptr;
    e.devPtr    = devptr;
    e.fptr      = func;
    e.multi.clear();

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



  void QDPCache::freeHostMemory(Entry& e) {
    if ( e.flags & Flags::OwnHostMemory )
      return;
    
    if (!e.hstPtr)
      return;

    assert(e.flags != Flags::JitParam);
    assert(e.flags != Flags::Static);

    QDP::Allocator::theQDPAllocator::Instance().free( e.hstPtr );
    e.hstPtr=NULL;
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

    while (!pool_allocator.allocate( &e.devPtr , e.size )) {
      if (!spill_lru()) {
	QDP_info("Device pool:");
	pool_allocator.printListPool();
	//printLockSets();
	QDP_error_exit("cache assureDevice: can't spill LRU object. Out of GPU memory!");
      }
    }
  }

  void QDPCache::printLockSet() {
    QDPIO::cout << "lockset: ";
    for (auto a : vecLocked)
      QDPIO::cout << a << "(" << vecEntry[a].lockCount << "), ";
    QDPIO::cout << "\n";
  }


  void QDPCache::freeDeviceMemory(Entry& e) {
    //QDPIO::cout << "free size = " << e.size << "\n";
    assert(e.flags != Flags::JitParam);
    
    if ( e.flags & Flags::OwnDeviceMemory )
      return;
    
    if (!e.devPtr)
      return;

    pool_allocator.free( e.devPtr );
    e.devPtr = NULL;
  }


  void QDPCache::assureDevice(int id) {
    if (id < 0)
      return;
    assert( vecEntry.size() > id );
    Entry& e = vecEntry[id];
    assureDevice(e);
  }
  
  
  void QDPCache::assureDevice(Entry& e) {
    if (e.flags & Flags::JitParam)
      return;
    if (e.flags & Flags::Static)
      return;

    // new
    lstTracker.splice( lstTracker.end(), lstTracker , e.iterTrack );

    if (e.status == Status::device)
      return;
    
    allocateDeviceMemory(e);

    if ( e.status == Status::host )
      {
	if (e.fptr) {

	  char * tmp = new char[e.size];
	  e.fptr(true,tmp,e.hstPtr);
	  CudaMemcpyH2D( e.devPtr , tmp , e.size );
	  delete[] tmp;

	} else {
	  CudaMemcpyH2D( e.devPtr , e.hstPtr , e.size );
	}
	CudaSyncTransferStream();
      }

    e.status = Status::device;
  }


#if 0
  void QDPCache::lockId(int id) {
    Entry& e = vecEntry[id];
    vecLocked.push_back(e.Id);
    e.lockCount++;
  }
#endif



  void QDPCache::assureHost(Entry& e) {
    assert(e.flags != Flags::JitParam);
    assert(e.flags != Flags::Static);
    
    allocateHostMemory(e);

    if ( e.status == Status::device )
      {
	if (e.fptr) {
	  char * tmp = new char[e.size];
	  CudaMemcpyD2H( tmp , e.devPtr , e.size );
	  e.fptr(false,e.hstPtr,tmp);
	  delete[] tmp;
	} else {
	  CudaMemcpyD2H( e.hstPtr , e.devPtr , e.size );
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

      found = ( (e->devPtr != NULL) &&
		(e->flags != Flags::JitParam) &&
		(e->flags != Flags::Static) &&
		(e->flags != Flags::Multi) &&
		( ! (e->flags & Flags::OwnDeviceMemory) ) );
    
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







  QDPCache::QDPCache() : vecEntry(1024)  {
    for ( int i = vecEntry.size()-1 ; i >= 0 ; --i ) {
      stackFree.push(i);
    }
    vecLocked.reserve(1024);
  }


  bool QDPCache::isOnDevice(int id) {
    // An id < 0 indicates a NULL pointer
    if (id < 0)
      return true;
    
    assert( vecEntry.size() > id );
    Entry& e = vecEntry[id];

    if (e.flags & QDPCache::JitParam)
      return true;

    if (e.flags & QDPCache::Static)
      return true;

    return ( e.status == QDPCache::Status::device );
  }

  namespace {
    void* jit_param_null_ptr = NULL;
  }


#if 0
  std::vector<void*> QDPCache::get_kernel_args(std::vector<int>& ids)
  {
    // Here we do two cycles through the ids:
    // 1) cache all objects
    // 2) check all are cached
    // This should replace the old 'lock set'

    for ( auto i : ids )
      assureDevice(i);

    bool all = true;
    for ( auto i : ids ) 
      all = all && isOnDevice(i);

    if (!all) {
      QDPIO::cout << "It was not possible to put all required objects on the device memory cache\n";
      for ( auto i : ids ) {
	if (i >= 0) {
	  assert( vecEntry.size() > i );
	  Entry& e = vecEntry[i];
	  QDPIO::cout << "id = " << i << "  size = " << e.size << "  flags = " << e.flags << "  status = ";
	  switch (e.status) {
	  case Status::undef:
	    QDPIO::cout << "undef\n";
	    break;
	  case Status::host:
	    QDPIO::cout << "host\n";
	    break;
	  case Status::device:
	    QDPIO::cout << "device\n";
	    break;
	  default:
	    QDPIO::cout << "unkown\n";
	  }
	} else {
	  QDPIO::cout << "id = " << i << "\n";
	}
      }
      QDP_error_exit("giving up");
    }

    const bool print_param = false;

    if (print_param)
      QDPIO::cout << "Jit function param: ";

    std::vector<void*> ret;
    for ( auto i : ids ) {
      if (i >= 0) {
	Entry& e = vecEntry[i];
	if (e.flags & QDPCache::Flags::JitParam) {
	  
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
	  
	  ret.push_back( &e.param );
	  
	} else {
	  
	  if (print_param)
	    {
	      QDPIO::cout << (size_t)e.devPtr << ", ";
	    }
	  
	  ret.push_back( &e.devPtr );
	}
      } else {
	
	if (print_param)
	  {
	    QDPIO::cout << "NULL(id=" << i<< "), ";
	  }
	
	ret.push_back( &jit_param_null_ptr );
	
      }
    }
    if (print_param)
      QDPIO::cout << "\n";

    return ret;
  }
#else
  std::vector<void*> QDPCache::get_kernel_args(std::vector<int>& ids , bool for_kernel )
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
		for ( auto u : e.multi )
		  {
		    allids.push_back(u);
		  }
	      }
	  }
      }
    //QDPIO::cout << "\n";

    //QDPIO::cout << "allids: ";
    for ( auto i : allids ) {
      //QDPIO::cout << i << " ";
      assureDevice(i);
    }
    //QDPIO::cout << "\n";
    
    bool all = true;
    for ( auto i : allids )
      all = all && isOnDevice(i);

    if (!all) {
      QDPIO::cout << "It was not possible to load all objects required by the kernel into device memory\n";
      for ( auto i : ids ) {
	if (i >= 0) {
	  assert( vecEntry.size() > i );
	  Entry& e = vecEntry[i];
	  QDPIO::cout << "id = " << i << "  size = " << e.size << "  flags = " << e.flags << "  status = ";
	  switch (e.status) {
	  case Status::undef:
	    QDPIO::cout << "undef\n";
	    break;
	  case Status::host:
	    QDPIO::cout << "host\n";
	    break;
	  case Status::device:
	    QDPIO::cout << "device\n";
	    break;
	  default:
	    QDPIO::cout << "unkown\n";
	  }
	} else {
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
		for( auto q : e.multi )
		  {
		    assert( vecEntry.size() > q );
		    Entry& qe = vecEntry[q];
		    assert( ! (qe.flags & QDPCache::Flags::Multi) );
		    assert( isOnDevice(q) );
		    dev_ptr[count++] = qe.devPtr;
		  }
		
		CudaMemcpyH2D( e.devPtr , dev_ptr.slice() , e.multi.size() * sizeof(void*) );
		//QDPIO::cout << "multi-ids: copied elements = " << e.multi.size() << "\n";
	      }
	  }
      }

    
    const bool print_param = false;

    if (print_param)
      QDPIO::cout << "Jit function param: ";
    
    std::vector<void*> ret;
    for ( auto i : ids ) {
      if (i >= 0) {
	Entry& e = vecEntry[i];
	if (e.flags & QDPCache::Flags::JitParam) {
	  
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
	  ret.push_back( &e.param );
	  
	} else {
	  
	  if (print_param)
	    {
	      QDPIO::cout << (size_t)e.devPtr << ", ";
	    }
	  
	  ret.push_back( for_kernel ? &e.devPtr : e.devPtr );
	}
      } else {
	
	if (print_param)
	  {
	    QDPIO::cout << "NULL(id=" << i<< "), ";
	  }

	assert(for_kernel);
	ret.push_back( &jit_param_null_ptr );
	
      }
    }
    if (print_param)
      QDPIO::cout << "\n";

    return ret;
  }
#endif


  QDPCache& QDP_get_global_cache()
  {
    static QDPCache* global_cache;
    if (!global_cache) {
      global_cache = new QDPCache();
    }
    return *global_cache;
  }

  QDPCache::Flags operator|(QDPCache::Flags a, QDPCache::Flags b)
  {
    return static_cast<QDPCache::Flags>(static_cast<int>(a) | static_cast<int>(b));
  }



} // QDP



