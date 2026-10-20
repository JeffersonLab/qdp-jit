#include "qdp.h"

#include <algorithm>


namespace QDP
{

  QDPPoolAllocator::~QDPPoolAllocator()
  {
    if (bufferAllocated)
      {
	gpu_free( poolPtr );
	bufferAllocated = false;
      }
  }


  void QDPPoolAllocator::allocateInternalBuffer()
  {
    if ( bufferAllocated )
      {
	QDPIO::cerr << "pool memory already allocated" << std::endl;
	QDP_abort(1);
      }

    if ( listEntry.size() > 0 )
      {
	QDPIO::cerr << "Pool allocator: list of entries not zero" << std::endl;
	QDP_abort(1);
      }

    if ( !gpu_malloc( &poolPtr , poolSize ) )
      {
	QDPIO::cerr << "Pool allocator: could not allocate the pool of " << poolSize << " bytes" << std::endl;
	QDP_abort(1);
      }

    entry_t e;
    e.ptr = poolPtr;
    e.size = poolSize;
    e.id = -1;
    e.allocated = false;

    listEntry.push_back(e);

    if (setMemory)
      {
	QDPIO::cout << "Initializing pool memory with value = " << setMemoryVal << "\n";

	gpu_memset( poolPtr , setMemoryVal , poolSize );
      }

    bufferAllocated = true;
  }


  bool QDPPoolAllocator::allocate( void ** ptr , size_t n_bytes , int id )
  {
    if (n_bytes == 0)
      {
        *ptr = nullptr;
        return true;
      }

    if (!bufferAllocated)
      allocateInternalBuffer();

    size_t alignment = jit_config_get_pool_alignment();

    size_t size = (n_bytes + alignment - 1) & ~(alignment - 1);

    if (size > poolSize)
      {
	QDPIO::cerr << "Pool allocator: requested size = " << n_bytes << " (aligned " << size << ") exceeds the pool size = " << poolSize << std::endl;
	QDP_abort(1);
      }

    iterEntry_t candidate = std::find_if( listEntry.begin() , listEntry.end() ,
					  [size](const entry_t& ent) { return !ent.allocated && ent.size >= size; } );

    if ( candidate == listEntry.end() )
      {
	return false;
      }

    if (candidate->size == size)
      {
	candidate->allocated = true;
	candidate->id = id;

	*ptr = candidate->ptr;
      }
    else
      {
	entry_t e;
	e.ptr = candidate->ptr;
	e.size = size;
	e.id = id;
	e.allocated = true;

	candidate->ptr = (void*)( (size_t)(candidate->ptr) + size );
	candidate->id = -1;
	candidate->size = candidate->size - size;

	listEntry.insert( candidate , e );

	*ptr = e.ptr;
      }

    count[n_bytes]++;

    total_allocated += size;

    if ( total_allocated > max_allocated )
      {
	max_allocated = total_allocated;
      }

    return true;
  }


  void QDPPoolAllocator::free(const void *mem)
  {
    if (mem == nullptr)
      return;

    iterEntry_t p = std::find_if( listEntry.begin() , listEntry.end() , [mem](const entry_t& i){ return i.ptr == mem; } );

    if ( p == listEntry.end() )
      {
	QDPIO::cerr << "pool allocator: free: address not found " << mem << std::endl;
	QDP_abort(1);
      }

    p->allocated = false;

    if ( total_allocated < p->size )
      {
	QDPIO::cout << "pool internal error. total_allocated < p->size." << std::endl;
	QDP_abort(1);
      }

    total_allocated -= p->size;

    if ( p != listEntry.begin() )
      {
	iterEntry_t prev = p;

	prev--;

	if (!prev->allocated)
	  {
	    prev->size += p->size;
	    listEntry.erase(p);
	    p = prev;
	  }
      }

    if ( p != --listEntry.end() )
      {
	iterEntry_t next = p;

	next++;

	if (!next->allocated)
	  {
	    p->size += next->size;
	    listEntry.erase(next);
	  }
      }
  }


  size_t QDPPoolAllocator::free_mem()
  {
    size_t free = 0;

    for ( auto& e : listEntry )
      {
	if ( ! e.allocated )
	  free += e.size;
      }

    return free;
  }


  void QDPPoolAllocator::print_pool()
  {
    QDPIO::cout << "-------------pool-----------\n";
    for ( auto& e : listEntry )
      {
	QDPIO::cout << e.size << "\t";
	if ( e.allocated )
	  QDPIO::cout << "A";
	QDPIO::cout << std::endl;
      }
    QDPIO::cout << "----------------------------\n";
  }


  void QDPPoolAllocator::setPoolSize(size_t s)
  {
    poolSize = s;
  }


  void QDPPoolAllocator::enableMemset(unsigned val)
  {
    setMemory = true;
    setMemoryVal = val;
  }


  size_t QDPPoolAllocator::getPoolSize()
  {
    return poolSize;
  }

} // namespace QDP
