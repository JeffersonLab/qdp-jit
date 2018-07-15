#ifndef QDP_CACHE
#define QDP_CACHE

#include <iostream>
#include <map>
#include <vector>
#include <stack>
#include <list>
#include "string.h"
#include "math.h"

//#define SANITY_CHECKS_CACHE

using namespace std;

namespace QDP 
{  
  class QDPJitArgs;
  class QDPCached;

  namespace {
    typedef QDPPoolAllocator<QDPCUDAAllocator>     CUDADevicePoolAllocator;
  }
  
  class QDPCache
  {
  public:
    enum Flags {
      Empty = 0,
      OwnHostMemory = 1,
      OwnDeviceMemory = 2,
      UpdateCachedFlag = 4
    };

    
    QDPCache();
    
    typedef void (* LayoutFptr)(bool toDev,void * outPtr,void * inPtr);


    int add( size_t size, Flags flags, const void* ptr  );
    int add( size_t size, Flags flags, const void* ptr, LayoutFptr func );
    int add( size_t size, Flags flags, const void* ptr, LayoutFptr func , const QDPCached* object);
    
    // Wrappers to the previous interface
    int registrate( size_t size, unsigned flags, LayoutFptr func );
    int registrateOwnHostMem( size_t size, const void* ptr , LayoutFptr func );
    int registrateOScalar( size_t size, void* ptr , LayoutFptr func , const QDPCached* object);
    
    void signoff(int id);
    void assureOnHost(int id);

    void * getDevicePtr(int id);
    void getHostPtr(void ** ptr , int id);

    size_t getSize(int id);
    void newLockSet();
    bool allocate_device_static( void** ptr, size_t n_bytes );
    void free_device_static( void* ptr );

    CUDADevicePoolAllocator& get_allocator() { return pool_allocator; }

  private:
    class Entry;
    void growStack();

    void lockId(int id);

    void freeHostMemory(Entry& e);
    void allocateHostMemory(Entry& e);
    void freeDeviceMemory(Entry& e);
    void allocateDeviceMemory(Entry& e);
    
    void assureDevice(Entry& e);
    void assureHost(Entry& e);
    
    bool spill_lru();
    void printTracker();
    
  private:
    vector<Entry>       vecEntry;
    stack<int>          stackFree;
    list<int>           lstTracker;
    vector<int>         vecLocked;   // with duplicate entries
    CUDADevicePoolAllocator pool_allocator;
  };

  QDPCache::Flags operator|(QDPCache::Flags a, QDPCache::Flags b);


  QDPCache& QDP_get_global_cache();

}


#endif

