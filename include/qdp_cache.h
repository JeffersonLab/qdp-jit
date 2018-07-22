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

  void  qdp_stack_scalars_start( size_t size );
  void  qdp_stack_scalars_end();
  bool  qdp_stack_scalars_enabled();
  void* qdp_stack_scalars_alloc( size_t size );
  void  qdp_stack_scalars_free_stack();
  
  class QDPCache
  {
  public:
    enum Flags {
      Empty = 0,
      OwnHostMemory = 1,
      OwnDeviceMemory = 2,
      JitParam = 4,
      Static = 8,
      Multi = 16
    };

    enum class Status { undef , host , device };

    enum class JitParamType { float_, int_, int64_, double_, bool_ };


    
    QDPCache();
    
    typedef void (* LayoutFptr)(bool toDev,void * outPtr,void * inPtr);

    std::vector<void*> get_kernel_args(std::vector<int>& ids , bool for_kernel = true );
    
    int addJitParamFloat(float i);
    int addJitParamDouble(double i);
    int addJitParamInt(int i);
    int addJitParamInt64(int64_t i);
    int addJitParamBool(bool i);

    // track_ptr - this enables to free the memory later via the pointer (needed for QUDA, where we hijack cudaMalloc)
    int addDeviceStatic( void** ptr, size_t n_bytes , bool track_ptr = false );
    void signoffViaPtr( void* ptr );
    
    int add( size_t size, Flags flags, Status st, const void* ptr_host, const void* ptr_dev, LayoutFptr func );

    int addMulti( const multi1d<int>& ids );
    
    // Wrappers to the previous interface
    int registrate( size_t size, unsigned flags, LayoutFptr func );
    int registrateOwnHostMem( size_t size, const void* ptr , LayoutFptr func );
    
    void signoff(int id);
    void assureOnHost(int id);

    //void * getDevicePtr(int id);
    void getHostPtr(void ** ptr , int id);

    size_t getSize(int id);
    //bool allocate_device_static( void** ptr, size_t n_bytes );
    //void free_device_static( void* ptr );
    void printLockSet();
    
    CUDADevicePoolAllocator& get_allocator() { return pool_allocator; }

  private:
    class Entry;
    void growStack();

    void lockId(int id);
    int getNewId();
    
    void freeHostMemory(Entry& e);
    void allocateHostMemory(Entry& e);
    void freeDeviceMemory(Entry& e);
    void allocateDeviceMemory(Entry& e);
    
    void assureDevice(Entry& e);
    void assureDevice(int id);
    
    void assureHost(Entry& e);
    bool isOnDevice(int id);
    
    bool spill_lru();
    void printTracker();
    
  private:
    vector<Entry>       vecEntry;
    stack<int>          stackFree;
    list<int>           lstTracker;
    vector<int>         vecLocked;   // with duplicate entries
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

