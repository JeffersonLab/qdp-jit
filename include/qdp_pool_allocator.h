#ifndef QDP_POOL_ALLOCATOR_H
#define QDP_POOL_ALLOCATOR_H

#include <list>
#include <map>


namespace QDP
{

  class QDPPoolAllocator {
    struct entry_t
    {
      void * ptr;
      size_t size;
      int    id;
      bool   allocated;
    };
    typedef std::list< entry_t >           listEntry_t;
    typedef std::list< entry_t >::iterator iterEntry_t;

  public:
    QDPPoolAllocator() {}
    ~QDPPoolAllocator();

    size_t getPoolSize();
    void setPoolSize(size_t s);

    bool allocate( void** ptr , size_t n_bytes , int id );
    void free(const void *mem);

    void enableMemset(unsigned val);

    size_t get_max_allocated() { return max_allocated; }
    size_t free_mem();
    void print_pool();

  private:
    friend class QDPCache;

    QDPPoolAllocator(const QDPPoolAllocator&);                 // Prevent copy-construction
    QDPPoolAllocator& operator=(const QDPPoolAllocator&);

    void allocateInternalBuffer();

    std::map<size_t,size_t>& get_count() { return count; }

    bool               bufferAllocated = false;
    void *             poolPtr = nullptr;
    size_t             poolSize = 0;
    listEntry_t        listEntry;
    size_t             max_allocated = 0;
    size_t             total_allocated = 0;

    std::map<size_t,size_t> count;

    bool     setMemory = false;
    unsigned setMemoryVal = 0;
  };

} // namespace QDP

#endif
