#ifndef QDP_CACHE
#define QDP_CACHE

#include <iostream>
#include <vector>
#include <stack>
#include <list>


namespace QDP 
{
  template<class T> class multi1d;

  bool qdp_cache_get_cache_verbose();
  void qdp_cache_set_cache_verbose(bool b);

    
  class QDPCache
  {
  public:
    enum Flags {
      Empty = 0,
      OwnHostMemory = 1,
      OwnDeviceMemory = 2,
      JitParam = 4,
      Static = 8,
      Multi = 16,
      Array = 32
    };

    enum class Status       { undef , host , device };
    enum class JitParamType { float_, int_ , int64_, double_, bool_ };
    typedef void (* LayoutFptr)(bool toDev,void * outPtr,void * inPtr);

    struct ArgKey {
      // The default constructor is needed in the custom streaming kernel where multi1d<ArgKey> is used -- don't see a simple way around it
      ArgKey(): id(-1), elem(-1) {}
      ArgKey(int id): id(id), elem(-1) {}
      ArgKey(int id,int elem): id(id), elem(elem) {}
      int id;
      int elem;
    };
    
    struct Entry 
    {
      union JitParamUnion {
	void *  ptr;
	float   float_;
	int     int_;
	int64_t int64_;
	double  double_;
	bool    bool_;
      };

      int    Id;
      size_t size;
      size_t elem_size;
      Flags  flags;
      void*  hstPtr;  // NULL if not allocated
      void*  devPtr;  // NULL if not allocated
      std::list<int>::iterator iterTrack;
      LayoutFptr fptr;
      JitParamUnion param;
      JitParamType param_type;
      std::vector<ArgKey> multi;
      std::vector<Status> status_vec;
      std::vector<void* > karg_vec;
    };


    
    QDPCache();

    std::vector<void*> get_kernel_args(std::vector<ArgKey>& ids , bool for_kernel = true );
    void backup_last_kernel_args();
    
    void printInfo(int id);

    int addJitParamFloat(float i);
    int addJitParamDouble(double i);
    int addJitParamInt(int i);
    int addJitParamInt64(int64_t i);
    int addJitParamBool(bool i);

    // track_ptr - this enables to sign off via the pointer (needed for QUDA, where we hijack cudaMalloc)
    int addDeviceStatic( void** ptr, size_t n_bytes , bool track_ptr = false );
    void signoffViaPtr( void* ptr );
    int addDeviceStatic( size_t n_bytes );
    
    int add( size_t size, Flags flags, Status st, const void* ptr_host, const void* ptr_dev, LayoutFptr func );
    int addArray( size_t element_size , int num_elements );

    int addMulti( const multi1d<QDPCache::ArgKey>& ids );
    
    // Wrappers to the previous interface
    int registrate( size_t size, unsigned flags, LayoutFptr func );
    int registrateOwnHostMem( size_t size, const void* ptr , LayoutFptr func );
    
    void signoff(int id);
    void assureOnHost(int id);
    void assureOnHost(int id, int elem_num);

    void  getHostPtr(void ** ptr , int id);
    void* getHostArrayPtr( int id , int elem );
    
    size_t getSize(int id);
    void printLockSet();
    
    void suspend();
    void resume();

    void   setPoolSize(size_t s);
    size_t getPoolSize();
    void   enableMemset(unsigned val);

    std::vector<QDPCache::Entry>&  get__vec_backed();
    
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
    std::vector<Entry>       vecEntry;
    std::stack<int>          stackFree;
    std::list<int>           lstTracker;

    std::vector<QDPCache::ArgKey> __ids_last;
    std::vector<QDPCache::Entry> __vec_backed;
    std::vector<void*>  __hst_ptr;
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

