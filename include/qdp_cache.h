#ifndef QDP_CACHE_H
#define QDP_CACHE_H


#include <iostream>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
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
    typedef void (* ConvertFptr)(bool toDev,void * outPtr,void * inPtr, int layout_ref, Shape shape , JitDataFormat data_format );
    enum class Status { undef, host, device };

    enum class Flags : unsigned
    {
      Empty         = 0,
      OwnHostMemory = 1u << 0,
      NonEvictable  = 1u << 1,

      NoPage        = NonEvictable
    };
    
  private:
    enum class Location    { pool , direct_device , literal };
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

	LiteralUnion() : ptr(nullptr) {}
      };

      int      Id = -1;
      size_t   size = 0;
      Location location = Location::pool;
      Flags    flags = Flags::Empty;
      void*    hstPtr = nullptr;
      void*    devPtr = nullptr;
      std::list<int>::iterator iterTrack;
      ConvertFptr fptr = nullptr;
      int      layout_ref = -1;
      LiteralUnion param;
      LiteralType param_type = LiteralType::int_;

      Status status = Status::undef;
      Shape shape{};
      JitDataFormat data_format;
      size_t   host_size = 0;

      bool active = false;
      unsigned pin_count = 0;
      bool tracked_device_ptr = false;
    };

    class ScopedPinSet
    {
    public:
      ScopedPinSet(QDPCache& cache, const std::vector<int>& ids);
      ~ScopedPinSet();

      ScopedPinSet(const ScopedPinSet&) = delete;
      ScopedPinSet& operator=(const ScopedPinSet&) = delete;

    private:
      QDPCache& cache_;
      const std::vector<int>& ids_;
    };


  public:
    typedef int ArgKey;

    class KernelArgPin
    {
    public:
      KernelArgPin(KernelArgPin&& rhs) noexcept;
      KernelArgPin& operator=(KernelArgPin&& rhs) noexcept;
      ~KernelArgPin();

      KernelArgPin(const KernelArgPin&) = delete;
      KernelArgPin& operator=(const KernelArgPin&) = delete;

    private:
      friend class QDPCache;
      KernelArgPin(QDPCache& cache, const std::vector<int>& ids);
      void release();

      QDPCache* cache_;
      std::vector<int> ids_;
    };
    
#if defined QDP_BACKEND_ROCM
    typedef std::vector<unsigned char> KernelArgs_t;
#elif defined QDP_BACKEND_CUDA
    typedef std::vector<void*> KernelArgs_t;
#elif defined QDP_BACKEND_L0
    typedef std::vector< std::pair< int , void* > > KernelArgs_t;
#else
#error "No backend specified"
#endif

    union HostArgTypes {
      void *  ptr;
      float   f32;
      int     i32;
      int64_t i64;
      double  f64;
      bool    i1;
    };
    typedef std::vector<HostArgTypes> HostArgs_t;

    
    QDPCache();

    KernelArgPin pin_kernel_args(const std::vector<int>& ids);
    KernelArgs_t get_kernel_args(std::vector<int>& ids , bool for_kernel = true );
    HostArgs_t get_host_args(std::vector<int>& ids );
    
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

    int addDeviceStatic( void** ptr, size_t n_bytes , bool track_ptr = false );
    void signoffViaPtr( void* ptr );
    int addDeviceStatic( size_t n_bytes );
    int addDeviceStaticDirect( void** ptr, size_t n_bytes );

    struct DirectDeviceStats
    {
      size_t current_bytes = 0;
      size_t peak_bytes = 0;
      size_t largest_allocation = 0;
      size_t allocations = 0;
      size_t active = 0;
      size_t peak_active = 0;
    };
    const DirectDeviceStats& get_direct_device_stats() const { return direct_device_stats; }
    


    void zero_rep( int id );
    void copyD2H(int id);
    

    int add( size_t size );
    int addConverted( size_t size, size_t host_size, ConvertFptr func , int layout_ref , Shape shape , JitDataFormat data_format );
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
    bool isOnHost(int id);

    
  private:
    DirectDeviceStats direct_device_stats;

    int add_pool( size_t size,
		  Flags flags,
		  Status st,
		  const void* ptr_host,
		  const void* ptr_dev,
		  ConvertFptr func ,
		  JitDataFormat data_format = JitDataFormat() ,
		  int layout_ref = -1 ,
		  Shape shape = Shape() ,
		  size_t host_size = 0 );

    void printInfo(const Entry& e);
    std::string stringStatus( Status s );

    Entry& getActiveEntry(int id);
    const Entry& getActiveEntry(int id) const;
    static bool hasFlag(Flags flags, Flags bit);
    void pinIds(const std::vector<int>& ids);
    void unpinIds(const std::vector<int>& ids);

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

    std::vector<unsigned char> staging_buffer;
  };

  QDPCache& QDP_get_global_cache();

  QDPCache::Flags operator|(QDPCache::Flags a, QDPCache::Flags b);

  class JitParam
  {
  private:
    int id;
  public:
    JitParam( int id ): id(id) {}
    JitParam( const JitParam& ) = delete;
    JitParam& operator=( const JitParam& ) = delete;
    ~JitParam() {
      QDP_get_global_cache().signoff(id);
    }
    int get_id() const { return id; }
  };

}
#endif
