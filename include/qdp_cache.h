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


  class QDPCache
  {
    struct Entry;
  public:
    typedef void (* LayoutFptr)(bool toDev,void * outPtr,void * inPtr);
    static QDPCache& Instance();

    size_t getSize(int id);
    void beginNewLockSet();
    void releasePrevLockSet();
    void printLockSets();
    bool allocate_device_static( void** ptr, size_t n_bytes );
    void free_device_static( void* ptr );
#if 0
    void backupOnHost();
    void restoreFromHost();
#endif
    void sayHi();
    bool onDevice(int id) const;    
    void enlargeStack();

    // flags
    //  - 0: use host pool allocator (for OScalar)
    //  - 1: use host malloc allocator (for OLattice)
    //  - 2: don't use host allocator (for objects that manage their own memory)
    int registrate( size_t size, unsigned flags, LayoutFptr func );

    int registrateOwnHostMem( size_t size, void* ptr , LayoutFptr func );
    void signoff(int id);
    void lockId(int id);
    void * getDevicePtr(int id);
    void * getDevicePtrNoLock(int id);
    void getHostPtr(void ** ptr , int id);
    void freeHostMemory(Entry& e);
    void allocateHostMemory(Entry& e);
    void assureDevice(Entry& e);
    bool assureHost(Entry& e);
    bool spill_lru();
    void printTracker();
    void deleteObjects();
    QDPCache();
    ~QDPCache();

  private:
    list<void *>        lstStatic;

    vector<Entry>       vecEntry;
    stack<int>          stackFree;

    list<int>           lstTracker;

    list<int>           lstDel;
    vector<int>         vecLockSet[2];   // with duplicate entries
    int                 currLS;
    int                 prevLS;
    list<char*>         listBackup;

  };



}


#endif

