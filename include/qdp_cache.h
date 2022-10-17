#ifndef QDP_CACHE
#define QDP_CACHE


#include <iostream>
#include <vector>
#include <stack>
#include <list>

#include <qdp_config.h>

namespace QDP 
{
  template<class T> class multi1d;

  void qdp_cache_set_cache_verbose(bool b);
  
  class QDPCache
  {
  public:
    typedef void (* LayoutFptr)(bool toDev,void * outPtr,void * inPtr);
    enum class Status      { undef , host , device };
    enum       Flags       { Empty = 0, OwnHostMemory = 1, NoPage = 2  };
    
  private:
    enum class Location    { pool , literal };
    enum class LiteralType { float_, int_ , int64_, double_, bool_ };

    struct Entry 
    {
      union LiteralUnion {
	void *  ptr;
	float   float_;
	int     int_;
	int64_t int64_;
	double  double_;
	bool    bool_;
      };

      int      Id;
      size_t   size;
      Location location;
      Flags    flags;
      void*    hstPtr;
      void*    devPtr;  // NULL if not allocated
      std::list<int>::iterator iterTrack;
      LayoutFptr fptr;
      LiteralUnion param;
      LiteralType param_type;

      Status status;
    };


  public:
    typedef int ArgKey;
    
#if defined QDP_BACKEND_ROCM
    typedef std::vector<unsigned char> KernelArgs_t;
#elif defined QDP_BACKEND_CUDA
    typedef std::vector<void*> KernelArgs_t;
#elif defined QDP_BACKEND_AVX
    union ArgTypes {
      void *  ptr;
      float   f32;
      int     i32;
      int64_t i64;
      double  f64;
      bool    i1;
    };
    typedef std::vector<ArgTypes> KernelArgs_t;
#elif defined QDP_BACKEND_L0
    typedef std::vector< std::pair< int , void* > > KernelArgs_t;
#else
#error "No backend specified"
#endif

    
    QDPCache();

    KernelArgs_t get_kernel_args(std::vector<int>& ids , bool for_kernel = true );
    
    multi1d<void*> get_dev_ptrs( const multi1d<QDPCache::ArgKey>& ids );
    void* get_dev_ptr( QDPCache::ArgKey id );
    
    void printInfo(int id);

    size_t free_mem();
    void print_pool();
    size_t get_max_allocated();

    std::map<size_t,size_t>& get_alloc_count();

    int addJitParamFloat(float i);
    int addJitParamDouble(double i);
    int addJitParamInt(int i);
    int addJitParamInt64(int64_t i);
    int addJitParamBool(bool i);

    // track_ptr - this enables to sign off via the pointer (needed for QUDA, where we hijack cudaMalloc)
    int addDeviceStatic( void** ptr, size_t n_bytes , bool track_ptr = false );
    void signoffViaPtr( void* ptr );
    int addDeviceStatic( size_t n_bytes );
    


    void zero_rep( int id );
    void copyD2H(int id);
    

    int add( size_t size );
    int addLayout( size_t size, LayoutFptr func );
    int addOwnHostMem( size_t size, const void* ptr );
    int addOwnHostMemStatus( size_t size, const void* ptr , Status st );
    int addOwnHostMemNoPage( size_t size, const void* ptr );

    void signoff(int id);
    void assureOnHost(int id);

    void  getHostPtr(void ** ptr , int id);
    
    size_t getSize(int id);

    void updateDevPtr(int id, void* ptr);
    
    void   setPoolSize(size_t s);
    size_t getPoolSize();
    void   enableMemset(unsigned val);

    bool isOnDevice(int id);

    
  private:
    int add_pool( size_t size, Flags flags, Status st, const void* ptr_host, const void* ptr_dev, LayoutFptr func );

    void printInfo(const Entry& e);
    std::string stringStatus( Status s );

    bool gpu_allocate_base( void ** ptr , size_t n_bytes , int id );
    void gpu_free_base( void * ptr , size_t n_bytes );

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

    bool isOnDevice(int id, int elem);
    
    bool spill_lru();
    void printTracker();

    std::vector<Entry>       vecEntry;
    std::stack<int>          stackFree;
    std::list<int>           lstTracker;

    std::vector<int> __ids_last;
    std::vector<QDPCache::Entry> __vec_backed;
    std::vector<void*>  __hst_ptr;
  };

  QDPCache& QDP_get_global_cache();

  QDPCache::Flags operator|(QDPCache::Flags a, QDPCache::Flags b);

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

}
#endif
