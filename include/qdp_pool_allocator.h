// -*- C++ -*-

#ifndef QDP_POOL_ALLOCATOR
#define QDP_POOL_ALLOCATOR

#include <string>
#include <list>
#include <iostream>
#include <algorithm>

using namespace std;


namespace QDP
{

  template<class Allocator>
  class QDPPoolAllocator {
    struct entry_t
    {
      void * ptr;
      size_t size;
      int    id;
      void * host_ptr;
      bool allocated;
      bool fixed;
    };
    typedef          std::list< entry_t >           listEntry_t;
    typedef typename std::list< entry_t >::iterator iterEntry_t;


  public:
    QDPPoolAllocator(): bufferAllocated(false) {}

    ~QDPPoolAllocator()
    { 
      freeInternalBuffer();
    }

    void registerMemory();
    void unregisterMemory();

    size_t getPoolSize();

    bool allocate      ( void** ptr , size_t n_bytes , int id );
    bool allocate_fixed( void** ptr , size_t n_bytes , int id );

    void free(const void *mem);
    void setPoolSize(size_t s);

    void enableMemset(unsigned val);



    bool allocateInternalBuffer();

    size_t get_max_allocated()
    {
      return max_allocated;
    }
    
    size_t free_mem()
    {
      size_t free = 0;
	
      for ( auto i = listEntry.begin() ; i != listEntry.end() ; ++i )
	{
	  if ( ! i->allocated )
	    {
	      free += i->size;
	    }
	}

      return free;
    }


    void print_pool()
    {
      QDPIO::cout << "-------------pool-----------\n";
      for ( auto i = listEntry.begin() ; i != listEntry.end() ; ++i )
	{
	  QDPIO::cout << i->size << "\t";
	  if ( i->allocated )
	    {
	      if ( i->fixed )
		QDPIO::cout << "F" << std::endl;
	      else
		QDPIO::cout << "A" << std::endl;
	    }
	  else
	    QDPIO::cout << std::endl;
	}
      QDPIO::cout << "----------------------------\n";
    }

    
    
  private:
    friend class QDPCache;

    QDPPoolAllocator(const QDPPoolAllocator&);                 // Prevent copy-construction
    QDPPoolAllocator& operator=(const QDPPoolAllocator&);

    void freeInternalBuffer();
    bool bufferAllocated;
    
    void *             poolPtr;
    void *             unaligned;
    size_t             poolSize;
    size_t             bytes_allocated;
    listEntry_t        listEntry;
    size_t             max_allocated = 0;
    size_t             total_allocated = 0;

    std::map<size_t,size_t> count;
    
    bool setMemory = false;
    unsigned setMemoryVal;


    std::map<size_t,size_t>& get_count()
    {
      return count;
    }
    

    struct SizeNotAllocated: public std::binary_function< entry_t , size_t , bool >
    {
      bool operator () ( const entry_t & ent , size_t size ) const
      {
	return ( !ent.allocated ) && ( ent.size >= size );
      }
    };


    template<class T>
    bool findNextNotAllocated( T& start , const T& end , size_t size )
    {
      start = std::find_if( start , end , std::bind2nd( SizeNotAllocated(), size ) );
      
      if ( start == end)
	{
	  return false;
	}
      
      return true;
    }






    
    struct MemMatch: public std::binary_function< entry_t , const void * , bool >
    {
      bool operator () ( const entry_t& ent , const void* mem ) const
      {
	return ent.ptr == mem;
      }
    };


    bool findMemMatch( iterEntry_t& match , const void* mem )
    {
      match = std::find_if( listEntry.begin() , listEntry.end(), std::bind2nd( MemMatch(), mem ) );

      return match != listEntry.end();
    }


  };


  template<class Allocator>
  void QDPPoolAllocator<Allocator>::registerMemory() {
    if (!bufferAllocated)
      allocateInternalBuffer();
  }


  template<class Allocator>
  void QDPPoolAllocator<Allocator>::unregisterMemory() {
    if (!bufferAllocated) {
      QDP_error_exit("pool unregisterMemory: not allocated");
    }
  }







  template<class Allocator>
  void QDPPoolAllocator<Allocator>::freeInternalBuffer()
  {
    if (bufferAllocated)
      {
	Allocator::free( unaligned );
	
	bufferAllocated = false;
      }
  }



  template<class Allocator>
  bool QDPPoolAllocator<Allocator>::allocateInternalBuffer()
  {
    //std::cout << "pool internal buffer: using alignment " << jit_config_get_pool_alignment() << std::endl;
      

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

    bool pool_allocated = false;
    size_t orig_size = poolSize;
    
    while ( !pool_allocated && poolSize > (orig_size >> 1) )
      {
	bytes_allocated = poolSize + 2 * jit_config_get_pool_alignment();


	if (Allocator::allocate( (void**)&unaligned , bytes_allocated ))
	  {
	    pool_allocated = true;
	  }
	else
	  {
	    poolSize -= qdp_jit_config_pool_size_decrement();
	    QDPIO::cout << "Pool allocation of " << bytes_allocated << " bytes failed." << std::endl;
	    QDPIO::cout << "Retry with reduced pool size by " << qdp_jit_config_pool_size_decrement() << " bytes." << std::endl;
	  }
      }

    if (!pool_allocated)
      {
	QDPIO::cerr << "Pool allocater could not allocate " << bytes_allocated << "\n";
	return false;
      }
    
    poolPtr = (unsigned char *)( ( (unsigned long)unaligned + (jit_config_get_pool_alignment()-1) ) & ~(jit_config_get_pool_alignment() - 1));

    entry_t e;
    e.ptr = poolPtr;
    e.size = poolSize;
    e.id = -1;
    e.allocated = false;
    
    listEntry.push_back(e);

    if (setMemory)
      {
	if ( sizeof(unsigned) != 4 )
	  {
	    QDPIO::cerr << "Error: Compiler's size for int on the architecture is not 4." << std::endl;
	    QDP_abort(1);
	  }

	QDPIO::cout << "Initializing pool memory with value = " << setMemoryVal << "\n";
	
	gpu_memset( poolPtr , setMemoryVal , poolSize );
      }
    
    bufferAllocated=true;
    
    return true;
  }



  namespace
  {
    template<class T>
    bool findFixed( const T& a )
    {
      return a.allocated && a.fixed;
    }
  }

  



  template<class Allocator>
  bool QDPPoolAllocator<Allocator>::allocate( void ** ptr , size_t n_bytes , int id )
  {
    if (n_bytes == 0)
      {
        *ptr = nullptr;
        return true;
      }

    if (!bufferAllocated)
      {
	if ( !allocateInternalBuffer() )
	  return false;
      }

    size_t alignment = jit_config_get_pool_alignment();

    size_t size = (n_bytes + (alignment) - 1) & ~((alignment) - 1);

    if (size==0)
      {
	QDPIO::cout << "pool allocator requested size 0." << std::endl;
	QDP_abort(1);
      }

    //QDPIO::cout << "pool allocate fixed: size = " << n_bytes << ", after alignment requirements = " << size << std::endl;

    if (size > poolSize)
      {
	QDPIO::cout << "Pool allocator: requested size = " << size << ", poolsize = " << poolSize << std::endl;
	return false;
      }

    

    iterEntry_t candidate = listEntry.begin();
    
    if ( !findNextNotAllocated( candidate , listEntry.end() , size ) )
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
	e.fixed = false;
	
	candidate->ptr = (void*)( (size_t)(candidate->ptr) + size );
	candidate->id = -1;
	candidate->size = candidate->size - size;

	if ( candidate->size == 0 )
	  {
	    QDPIO::cerr << "pool: some bizzare error. " << std::endl;
	    QDP_abort(1);
	  }

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



  template<class Allocator>
  bool QDPPoolAllocator<Allocator>::allocate_fixed( void ** ptr , size_t n_bytes , int id )
  {
    if (n_bytes == 0)
      {
        *ptr = nullptr;
        return true;
      }

    if (!bufferAllocated)
      {
	if ( !allocateInternalBuffer() )
	  return false;
      }

    size_t alignment = jit_config_get_pool_alignment();

    size_t size = (n_bytes + (alignment) - 1) & ~((alignment) - 1);

    if ( size == 0 )
      {
	QDPIO::cout << "pool allocator fixed requested size 0." << std::endl;
	QDP_abort(1);
      }
      
    //QDPIO::cout << "pool allocate fixed: size = " << n_bytes << ", after alignment requirements = " << size << std::endl;

    if (size > poolSize)
      {
	QDPIO::cout << "Pool allocator fixed: requested size = " << size << ", poolsize = " << poolSize << std::endl;
	return false;
      }



    auto candidate = listEntry.rbegin();
    
    if ( !findNextNotAllocated( candidate , listEntry.rend() , size ) )
      {
	return false;
      }


    if (candidate->size == size)
      {
	candidate->allocated = true;
	candidate->id = id;
	candidate->fixed = true;
	
	*ptr = candidate->ptr;
      }
    else
      {
	entry_t e;
	e.ptr = candidate->ptr;
	e.size = candidate->size - size;
	e.id = -1;
	e.allocated = false;

	candidate->ptr = (void*)( (size_t)(candidate->ptr) + candidate->size - size );
	candidate->id = id;
	candidate->size = size;
	candidate->fixed = true;
	candidate->allocated = true;

	if ( e.size == 0 )
	  {
	    QDPIO::cerr << "pool fixed: some bizzare error. " << std::endl;
	    QDP_abort(1);
	  }

	listEntry.insert( --candidate.base() , e );

	*ptr = candidate->ptr;
      }

    total_allocated += size;
    
    if ( total_allocated > max_allocated )
      {
	max_allocated = total_allocated;
      }

    return true;
  }

  



  template<class Allocator>
  void QDPPoolAllocator<Allocator>::free(const void *mem)
  {
    if (mem == nullptr)
      return;

    iterEntry_t p;
    
    if (!findMemMatch( p , mem ))
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
	typename listEntry_t::iterator prev = p;
	
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
	typename listEntry_t::iterator next = p;

	next++;
	
	if (!next->allocated)
	  {
	    p->size += next->size;
	    listEntry.erase(next);
	  }
      }

    return;
  }



  template<class Allocator>
  void QDPPoolAllocator<Allocator>::setPoolSize(size_t s)
  {
    //QDP_info_primary("Pool allocator: set pool size %lu bytes" , (unsigned long)s );
    poolSize = s;
  }

  template<class Allocator>
  void QDPPoolAllocator<Allocator>::enableMemset(unsigned val)
  {
    setMemory = true;
    setMemoryVal = val;
  }


  template<class Allocator>
  size_t QDPPoolAllocator<Allocator>::getPoolSize()
  {
    return poolSize;
  }




} // namespace QDP



#endif


