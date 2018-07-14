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
    struct Entry;
  public:
    typedef void (* LayoutFptr)(bool toDev,void * outPtr,void * inPtr);

    size_t getSize(int id);
    void newLockSet();
    void printLockSets();
    bool allocate_device_static( void** ptr, size_t n_bytes );
    void free_device_static( void* ptr );
    void growStack();

    // flags
    //  - 0: unused. (former: use host pool allocator (for OScalar))
    //  - 1: use host malloc allocator (for OLattice)
    //  - 2: don't use host allocator (for objects that manage their own memory)
    //  - 3: OScalars
    int registrate( size_t size, unsigned flags, LayoutFptr func );

    int registrateOwnHostMem( size_t size, const void* ptr , LayoutFptr func );
    int registrateOScalar( size_t size, void* ptr , LayoutFptr func , const QDPCached* object);
    void signoff(int id);
    void lockId(int id);
    void * getDevicePtr(int id);
    void getHostPtr(void ** ptr , int id);
    void assureOnHost(int id);

    void freeHostMemory(Entry& e);
    void allocateHostMemory(Entry& e);
    void freeDeviceMemory(Entry& e);
    void allocateDeviceMemory(Entry& e);
    
    void assureDevice(Entry& e);
    void assureHost(Entry& e);
    bool spill_lru();
    void printTracker();
    void deleteObjects();
    QDPCache();
    ~QDPCache();

    CUDADevicePoolAllocator& get_allocator() { return pool_allocator; }
    
  private:
    vector<Entry>       vecEntry;
    stack<int>          stackFree;
    list<int>           lstTracker;
    vector<int>         vecLocked;   // with duplicate entries
    CUDADevicePoolAllocator pool_allocator;
  };

  QDPCache& QDP_get_global_cache();

}


#endif

