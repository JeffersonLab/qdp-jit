#include "qdp.h"

#include <map>
#include <list>
#include <functional>

#include <iostream>
#include <fstream>



namespace QDP
{

  enum class Status { undef , host , device };

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
    const QDPCached* object;
  };



  void QDPCache::newLockSet() {
    while ( vecLocked.size() > 0 ) {
      assert( vecEntry.size() > vecLocked.back() );
      Entry& e = vecEntry[ vecLocked.back() ];
      e.lockCount--;
      assert( e.lockCount == 0 );
      vecLocked.pop_back();
    }
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
    return add( size , Flags::OwnHostMemory , ptr , func , NULL );
  }

  int QDPCache::registrateOScalar( size_t size, void* ptr , QDPCache::LayoutFptr func , const QDPCached* object)
  {
    return add( size , Flags::OwnHostMemory | Flags::UpdateCachedFlag , ptr , func , object ); 
  }

  int QDPCache::registrate( size_t size, unsigned flags, QDPCache::LayoutFptr func )
  {
    return add( size , Flags::Empty , NULL , func , NULL );
  }
  


  int QDPCache::add( size_t size, Flags flags, const void* hstptr_, QDPCache::LayoutFptr func , const QDPCached* object)
  {
    void * hstptr = const_cast<void*>(hstptr_);
    
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
    e.devPtr    = NULL;
    e.lockCount = 0;
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
    e.fptr      = func;
    e.object    = object;

    if (flags & Flags::OwnHostMemory)
      e.status    = Status::host;
    else
      e.status    = Status::undef;

    stackFree.pop();

    return Id;
  }

  int QDPCache::add( size_t size, Flags flags, const void* hstptr, QDPCache::LayoutFptr func )
  {
    return add(  size,  flags, hstptr, func , NULL );
  }

  int QDPCache::add( size_t size, Flags flags, const void* hstptr )
  {
    return add( size, flags, hstptr, NULL , NULL );
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


  void QDPCache::freeDeviceMemory(Entry& e) {
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

    e.status = Status::device;

    if ( e.flags & Flags::UpdateCachedFlag )
      e.object->onHost=false;
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

    if ( e.flags & Flags::UpdateCachedFlag )
      e.object->onHost=true;

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



