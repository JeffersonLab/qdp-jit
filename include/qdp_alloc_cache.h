#ifndef QDP_ALLOC_CACHE
#define QDP_ALLOC_CACHE

#include <string>
#include <list>
#include <iostream>
#include <algorithm>

using namespace std;


namespace QDP
{

  class AllocCache
  {
    std::map< size_t , std::vector<void*> > alloc_free;
    std::map< void* , size_t > alloc_inuse;

  public:
    bool allocate ( void** ptr , size_t n_bytes , int id );
    void free(void *ptr);
    void freeAllUnused();
  };


  AllocCache& QDP_get_global_alloc_cache();

    
}
#endif
