#ifndef QDP_CACHE
#define QDP_CACHE

#include <iostream>
#include <map>
#include <vector>
#include <stack>
#include <list>
//#include "string.h"
//#include "math.h"

//#define SANITY_CHECKS_CACHE

using namespace std;

namespace QDP 
{
  template<class T> class multi1d;

  class QDPJitArgs;

  namespace {
    typedef QDPPoolAllocator<QDPCUDAAllocator>     CUDADevicePoolAllocator;
  }

  bool   qdp_cache_get_pool_bisect();
  size_t qdp_cache_get_pool_bisect_max();
  
  void qdp_cache_set_pool_bisect(bool b);
  void qdp_cache_set_pool_bisect_max(size_t val);

  std::vector<void*> get_backed_kernel_args( CUDADevicePoolAllocator& pool_allocator );
    
  class QDPCache
  {
  public:
    class Entry;

    enum Flags {
      Empty = 0,
      OwnHostMemory = 1,
      OwnDeviceMemory = 2,
      JitParam = 4,
      Static = 8,
      Multi = 16,
      Array = 32
    };

    enum class Status { undef , host , device };

    enum class JitParamType { float_, int_, int64_, double_, bool_ };

    struct ArgKey {
      ArgKey(int id): id(id), elem(-1) {}
      ArgKey(int id,int elem): id(id), elem(elem) {}
      int id;
      int elem;
    };
    
    
    QDPCache();
    
    typedef void (* LayoutFptr)(bool toDev,void * outPtr,void * inPtr);

    std::vector<void*> get_kernel_args(std::vector<ArgKey>& ids , bool for_kernel = true );
    void backup_last_kernel_args();
    
    void printInfo(int id);

    int addJitParamFloat(float i);
    int addJitParamDouble(double i);
    int addJitParamInt(int i);
    int addJitParamInt64(int64_t i);
    int addJitParamBool(bool i);

    // track_ptr - this enables to sign off later via the pointer (needed for QUDA, where we hijack cudaMalloc)
    int addDeviceStatic( void** ptr, size_t n_bytes , bool track_ptr = false );
    void signoffViaPtr( void* ptr );
    int addDeviceStatic( size_t n_bytes );
    
    int add( size_t size, Flags flags, Status st, const void* ptr_host, const void* ptr_dev, LayoutFptr func );
    int addArray( size_t element_size , int num_elements );

    int addMulti( const multi1d<int>& ids );
    
    // Wrappers to the previous interface
    int registrate( size_t size, unsigned flags, LayoutFptr func );
    int registrateOwnHostMem( size_t size, const void* ptr , LayoutFptr func );
    
    void signoff(int id);
    void assureOnHost(int id);
    void assureOnHost(int id, int elem_num);

    //void * getDevicePtr(int id);
    void  getHostPtr(void ** ptr , int id);
    void* getHostArrayPtr( int id , int elem );
    
    size_t getSize(int id);
    //bool allocate_device_static( void** ptr, size_t n_bytes );
    //void free_device_static( void* ptr );
    void printLockSet();
    
    CUDADevicePoolAllocator& get_allocator() { return pool_allocator; }

    void suspend();
    void resume();
    
  private:
    void printInfo(const Entry& e);
    void growStack();

    void lockId(int id);
    int getNewId();
    
    void freeHostMemory(Entry& e);
    void allocateHostMemory(Entry& e);
    void freeDeviceMemory(Entry& e);
    void allocateDeviceMemory(Entry& e);
    
    void assureDevice(Entry& e);
    void assureDevice(Entry& e,int elem);

    void assureDevice(int id);
    void assureDevice(int id,int elem);
    
    void assureHost( Entry& e);
    void assureHost( Entry& e, int elem_num );

    bool isOnDevice(int id);
    bool isOnDevice(int id, int elem);
    
    bool spill_lru();
    void printTracker();
    
  private:
    vector<Entry>       vecEntry;
    stack<int>          stackFree;
    list<int>           lstTracker;
    CUDADevicePoolAllocator pool_allocator;
  };

  QDPCache& QDP_get_global_cache();


  class JitParam
  {
  private:
    int id;
  public:
    JitParam( int id ): id(id) {}
    ~JitParam() {
      QDP_get_global_cache().signoff(id);
    }
    int get_id() const { return id; }
  };


  QDPCache::Flags operator|(QDPCache::Flags a, QDPCache::Flags b);


}


#endif

