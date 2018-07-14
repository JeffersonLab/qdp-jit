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
    // 1 - OLattice (fully managed)
    // 2 - own host memory
    // 3 - OScalar
    int    flags;
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


  int QDPCache::registrate( size_t size, unsigned flags, LayoutFptr func)
  {
    if (stackFree.size() == 0) {
      growStack();
    }

    int Id = stackFree.top();

    assert( vecEntry.size() > Id );
    Entry& e = vecEntry[ Id ];

    e.Id        = Id;
    e.size      = size;
    e.flags     = flags;
    e.hstPtr    = NULL;
    e.devPtr    = NULL;
    e.lockCount = 0;
    e.status    = Status::undef;
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
    e.fptr      = func;
      
    stackFree.pop();

    return Id;
  }



  int QDPCache::registrateOwnHostMem( size_t size, const void* ptr_, LayoutFptr func)
  {
    void * ptr = const_cast<void*>(ptr_);
    if (stackFree.size() == 0) {
      growStack();
    }

    int Id = stackFree.top();
    Entry& e = vecEntry[ Id ];

    e.Id        = Id;
    e.fptr      = func;
    e.size      = size;
    e.flags     = 2;
    e.hstPtr    = ptr;
    e.devPtr    = NULL;
    e.status    = Status::host;
    e.lockCount = 0;
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
      
    stackFree.pop();

    return Id;
  }
    

  int QDPCache::registrateOScalar( size_t size, void* ptr, LayoutFptr func, const QDPCached* object)
  {
    if (stackFree.size() == 0) {
      growStack();
    }

    int Id = stackFree.top();
    Entry& e = vecEntry[ Id ];

    e.Id        = Id;
    e.fptr      = func;
    e.size      = size;
    e.flags     = 3;
    e.hstPtr    = ptr;
    e.devPtr    = NULL;
    e.status    = Status::host;
    e.lockCount = 0;
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
    e.object    = object;
      
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
    switch(e.flags) {
    case 1:
      if (!e.hstPtr)
	return;
      QDP::Allocator::theQDPAllocator::Instance().free( e.hstPtr );
      e.hstPtr=NULL;
      break;
    case 2:
      break;
    case 3:
      break;
    default:
      QDP_error_exit("cache delete objects: unkown host memory allocator");
      break;
    }
  }


  
  void QDPCache::allocateHostMemory(Entry& e) {
    if (e.hstPtr)
      return;
    
    switch(e.flags) {
    case 1:
      try {
	e.hstPtr = (void*)QDP::Allocator::theQDPAllocator::Instance().allocate( e.size , QDP::Allocator::DEFAULT );
      }
      catch(std::bad_alloc) {
	QDP_error_exit("cache allocateHostMemory: host memory allocator flags=1 failed");
      }
      break;
    case 2:
      // has it's own host memory
      break;
    case 3:
      // it's an OScalar and has it's own host memory
      break;
    default:
      QDP_error_exit("cache allocateHostMemory: unkown host memory allocator");
      break;
    }
  }


  void QDPCache::allocateDeviceMemory(Entry& e) {
    if (e.devPtr)
      return;

    switch(e.flags) {
    case 1:
    case 2:
    case 3:
      while (!pool_allocator.allocate( &e.devPtr , e.size )) {
	if (!spill_lru()) {
	  QDP_info("Device pool:");
	  pool_allocator.printListPool();
	  //printLockSets();
	  QDP_error_exit("cache assureDevice: can't spill LRU object. Out of GPU memory!");
	}
      }
      break;
    case 4:
      // scratch OScalar manages it's own memory
      break;
    default:
      QDP_error_exit("cache allocateDeviceMemory: unkown flag");
      break;
    }
  }


  void QDPCache::freeDeviceMemory(Entry& e) {
    if (!e.devPtr)
      return;

    switch(e.flags) {
    case 1:
    case 2:
    case 3:
      pool_allocator.free( e.devPtr );
      e.devPtr = NULL;
      break;
    case 4:
      // scratch OScalar manages it's own memory
      break;
    default:
      QDP_error_exit("cache freeDeviceMemory: unkown flag");
      break;
    }
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
    if (e.flags == 3)
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
    if (e.flags == 3)
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

      switch (e->flags) {
      case 1:
      case 2:
      case 3:
	found = ( (e->lockCount == 0) && (e->devPtr != NULL) );
	break;
      default:
	found = false;
      }
      
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




  void QDPCache::printTracker() {
  }




  QDPCache::QDPCache() : vecEntry(1024)  {
    for ( int i = vecEntry.size()-1 ; i >= 0 ; --i ) {
      stackFree.push(i);
    }
    vecLocked.reserve(1024);
  }



  QDPCache::~QDPCache() {
  }



  QDPCache& QDP_get_global_cache()
  {
    static QDPCache* global_cache;
    if (!global_cache) {
      global_cache = new QDPCache();
    }
    return *global_cache;
  }


} // QDP



