#include "qdp.h"

#include <string>
#include <list>
#include <iostream>
#include <algorithm>


namespace QDP
{
  bool AllocCache::allocate ( void** ptr , size_t n_bytes , int id )
  {
    if (n_bytes == 0)
      {
	*ptr = nullptr;
	return true;
      }
      
    if ( alloc_free.count( n_bytes ) > 0 )
      {
	if ( alloc_free.at( n_bytes ).size() > 0 )
	  {
	    *ptr = alloc_free.at( n_bytes ).back();
	    alloc_free.at( n_bytes ).pop_back();
	    alloc_inuse[ *ptr ] = n_bytes;

	    return true;
	  }
      }
	
    bool ret = gpu_malloc_managed( ptr , n_bytes );

    if (ret)
      {
	// Safety check
	if ( alloc_inuse.count( *ptr ) > 0 )
	  {
	    QDPIO::cout << "AllocCache::allocate, address already in use! size = " << n_bytes << std::endl;
	    QDP_abort(1);
	  }
	  
	alloc_inuse[ *ptr ] = n_bytes;
      }

    return ret;
  }

    
  void AllocCache::free(void *ptr)
  {
    if (ptr == nullptr)
      return;
      
    if ( alloc_inuse.count( ptr ) > 0 )
      {
	alloc_free[ alloc_inuse.at( ptr ) ].push_back( ptr );
	alloc_inuse.erase( ptr );
      }
    else
      {
	std::cout << "AllocCache::free, address not in use! ptr " << ptr << std::endl;
	QDP_abort(1);
      }
  }


  void AllocCache::freeAllUnused()
  {
    for (auto& i : alloc_free)
      {
	while (i.second.size() > 0)
	  {
	    gpu_free( i.second.back() );
	    i.second.pop_back();
	  }
      }
  }


  namespace {
    AllocCache* __global_alloc_cache;
  }

  AllocCache& QDP_get_global_alloc_cache()
  {
    if (!__global_alloc_cache) {
      __global_alloc_cache = new AllocCache();
    }
    return *__global_alloc_cache;
  }

  
}
