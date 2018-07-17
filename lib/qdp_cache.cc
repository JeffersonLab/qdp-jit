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
  }
  
  void qdp_stack_scalars_start( size_t size )
  {
    assert(!STACK::stack_scalars);
    
    if (!QDP_get_global_cache().allocate_device_static( &STACK::stack_ptr, size ))
      QDP_error_exit("out of memory (GPU)");

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
    QDP_get_global_cache().free_device_static( STACK::stack_ptr );
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
    int    Id;
    size_t size;
    Flags  flags;
    void*  hstPtr;  // NULL if not allocated
    void*  devPtr;  // NULL if not allocated
    Status status;
    int    lockCount;
    list<int>::iterator iterTrack;
    LayoutFptr fptr;
  };



  void QDPCache::newLockSet() {
    
    //std::vector<int> tmp; // sanity
    
    while ( vecLocked.size() > 0 ) {
      assert( vecEntry.size() > vecLocked.back() );
      Entry& e = vecEntry[ vecLocked.back() ];
      e.lockCount--;
      //tmp.push_back(e.Id);
      vecLocked.pop_back();
    }

    // A sanity check can't be done this way
    // since the object Id might have already been re-assigned to a different object.
#if 0
    for (i : tmp)
      {
	if ( vecEntry[i].lockCount != 0 )
	  {
	    QDPIO::cout << "id = " << i << "  lockCount = " << vecEntry[i].lockCount << "\n";
	  }
      }
#endif
  }



  bool QDPCache::allocate_device_static( void** ptr, size_t n_bytes ) {
    while (!pool_allocator.allocate( ptr , n_bytes )) {
      if (!spill_lru()) {
	QDP_error_exit("cache allocate_device_static: can't spill LRU object");
      }
    }
    return true;
  }

  void QDPCache::free_device_static( void* ptr ) {
    pool_allocator.free( ptr );
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



  int QDPCache::registrateOwnHostMem( size_t size, const void* ptr , QDPCache::LayoutFptr func )
  {
    return add( size , Flags::OwnHostMemory , Status::host , ptr , NULL , func );
  }

  int QDPCache::registrate( size_t size, unsigned flags, QDPCache::LayoutFptr func )
  {
    return add( size , Flags::Empty , Status::undef , NULL , NULL , func );
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
    e.lockCount = 0;
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
    e.fptr      = func;

    stackFree.pop();

    return Id;
  }



  

  void QDPCache::signoff(int id) {
    assert( vecEntry.size() > id );

    lstTracker.erase( vecEntry[id].iterTrack );

    freeDeviceMemory( vecEntry[id] );
    freeHostMemory( vecEntry[id] );
    
    stackFree.push( id );
  }
  


  void * QDPCache::getDevicePtr(int id) {
    if (id < 0) return NULL;

    assert( vecEntry.size() > id );
    Entry& e = vecEntry[id];
    assureDevice( e );
    lstTracker.splice( lstTracker.end(), lstTracker , e.iterTrack );

    return e.devPtr;
  }


  void QDPCache::assureOnHost(int id) {
    Entry& e = vecEntry[id];
    assureHost( e );
  }


  void QDPCache::getHostPtr(void ** ptr , int id) {
    assert( vecEntry.size() > id );
    Entry& e = vecEntry[id];

    assureHost( e );

    *ptr = e.hstPtr;
  }








  void QDPCache::freeHostMemory(Entry& e) {
    if ( e.flags & Flags::OwnHostMemory )
      return;
    
    if (!e.hstPtr)
      return;
    
    QDP::Allocator::theQDPAllocator::Instance().free( e.hstPtr );
    e.hstPtr=NULL;
  }


  
  void QDPCache::allocateHostMemory(Entry& e) {
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
    
    if ( e.flags & Flags::OwnDeviceMemory )
      return;
    
    if (!e.devPtr)
      return;

    pool_allocator.free( e.devPtr );
    e.devPtr = NULL;
  }



  
  void QDPCache::assureDevice(Entry& e) {

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

    //lockId(e.Id);

    e.status = Status::device;
  }


  void QDPCache::lockId(int id) {
    Entry& e = vecEntry[id];
    vecLocked.push_back(e.Id);
    e.lockCount++;
  }




  void QDPCache::assureHost(Entry& e) {

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

      found = ( (e->lockCount == 0) &&
		(e->devPtr != NULL) &&
		( ! (e->flags & Flags::OwnDeviceMemory) ) );
    
      if (!found)
	it_key++;
    }

    if (found) {
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



