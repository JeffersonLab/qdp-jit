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

    bool allocate( void** ptr , size_t n_bytes , int id );

    void free(const void *mem);
    void setPoolSize(size_t s);

    void enableMemset(unsigned val);


    void defrag();
    int defrag_occurrences() { return defrags; }


    bool allocateInternalBuffer();


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
    int                defrags = 0;

    bool setMemory = false;
    unsigned setMemoryVal;




    struct SizeNotAllocated: public std::binary_function< entry_t , size_t , bool >
    {
      bool operator () ( const entry_t & ent , size_t size ) const
      {
	return ( !ent.allocated ) && ( ent.size >= size );
      }
    };

    
    bool findNextNotAllocated( iterEntry_t& start , size_t size )
    {
      iterEntry_t save = start;

      start = std::find_if( start , listEntry.end(), std::bind2nd( SizeNotAllocated(), size ) );
      
      if ( start == listEntry.end())
	{
	  start = std::find_if( listEntry.begin() , save , std::bind2nd( SizeNotAllocated(), size ) );
	  if ( start == save)
	    {
	      return false;
	    }
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

    bytes_allocated = poolSize + 2 * QDP_ALIGNMENT_SIZE;

    if (!Allocator::allocate( (void**)&unaligned , bytes_allocated ))
      {
	QDPIO::cerr << "Pool allocater could not allocate " << bytes_allocated << "\n";
	return false;
      }

    poolPtr = (unsigned char *)( ( (unsigned long)unaligned + (QDP_ALIGNMENT_SIZE-1) ) & ~(QDP_ALIGNMENT_SIZE - 1));

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
	
	gpu_memset( poolPtr , setMemoryVal , poolSize/sizeof(unsigned) );
      }
    
    bufferAllocated=true;
    
    return true;
  }



  template<class Allocator>
  void QDPPoolAllocator<Allocator>::defrag()
  {
    QDPIO::cout << "Defragmentation of pool memory." << "\n";

    defrags++;
    size_t free = 0;
    
    QDPIO::cout << "Copy fields to host" << "\n";
    iterEntry_t i = listEntry.begin();
    while (i != listEntry.end())
      {
	if ( i->allocated )
	  {
	    if ( !( i->host_ptr = malloc( i->size ) ) )
	      {
		QDPIO::cerr << "out of host memory during pool memory defragmentation." << std::endl;
		QDP_abort(1);
	      }

	    gpu_memcpy_d2h( i->host_ptr , i->ptr , i->size );
	    
	    i++;
	  }
	else
	  {
	    free += i->size;
	    
	    listEntry.erase( i++ );
	  }
      }

    QDPIO::cout << "Copy fields back to device" << "\n";
    void* ptr_cur = poolPtr;
    
    i = listEntry.begin();
    while (i != listEntry.end())
      {
	i->ptr = ptr_cur;

	//QDPIO::cout << "h2d dev ptr = " << (size_t)i->ptr << std::endl;

	gpu_memcpy_h2d( i->ptr , i->host_ptr , i->size );

	::free( i->host_ptr );
	
	QDP_get_global_cache().updateDevPtr( i->id , i->ptr );

	*(size_t*)&ptr_cur += i->size;
	i++;
      }

    entry_t e;
    e.ptr = ptr_cur;
    e.size = free;
    e.id = -1;
    e.allocated = false;

    listEntry.push_back( e );
  }


  template<class Allocator>
  bool QDPPoolAllocator<Allocator>::allocate( void ** ptr , size_t n_bytes , int id )
  {
    if (!bufferAllocated)
      {
	if ( !allocateInternalBuffer() )
	  return false;
      }

    //size_t alignment = QDP_ALIGNMENT_SIZE;
    size_t alignment = Allocator::ALIGNMENT_SIZE;

    size_t size = (n_bytes + (alignment) - 1) & ~((alignment) - 1);

#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("Pool allocator: allocate=%lu (resized=%lu)", n_bytes , size );
#endif

    if (size > poolSize)
      {
	QDP_info("Pool allocator: trying to allocate %lu (poolsize=%lu) " , size , poolSize );
	return false;
      }

#ifdef GPU_DEBUG_DEEP
    if (size==0)
      QDP_error_exit("QDPPoolAllocator<Allocator>::allocate ( size == 0 )");
#endif


    iterEntry_t candidate = listEntry.begin();
    
    if ( !findNextNotAllocated( candidate , size ) )
      {
	return false;
#if 0	
	QDPIO::cout << "Could not allocate " << size << " bytes." << std::endl;

	size_t free = free_mem();
	
	QDPIO::cout << "Total free (in chunks): " << free << " bytes." << std::endl;

	if ( size <= free )
	  {
	    defrag();

	    candidate = listEntry.begin();
    
	    if ( !findNextNotAllocated( candidate , size ) )
	      {
		QDPIO::cout << "After defrag: could not allocate " << size << " bytes." << std::endl;
		QDP_abort(1);
		return false;
	      }
	  }
	else
	  {
	    QDPIO::cout << "could not allocate " << size << " bytes." << std::endl;
	    QDP_abort(1);
	    return false;
	  }
#endif
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

	if ( candidate->size == 0 )
	  {
	    QDPIO::cerr << "pool: some bizzare error. " << std::endl;
	    QDP_abort(1);
	  }

	listEntry.insert( candidate , e );

	*ptr = e.ptr;
      }

    return true;
  }




  template<class Allocator>
  void QDPPoolAllocator<Allocator>::free(const void *mem)
  {
    iterEntry_t p;
    
    if (!findMemMatch( p , mem ))
      {
	QDPIO::cerr << "pool allocator: free: address not found " << mem << std::endl;
	QDP_abort(1);
      }

    p->allocated = false;

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


