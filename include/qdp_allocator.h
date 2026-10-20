#ifndef QDP_ALLOCATOR_H
#define QDP_ALLOCATOR_H


#include "qdp_singleton.h"
#include <cstdlib>
#include <new>

namespace QDP
{
  namespace Allocator
  {

    enum MemoryPoolHint { DEFAULT, FAST };

    class QDPDefaultAllocator {
    private:
      QDPDefaultAllocator() {}
      ~QDPDefaultAllocator() {}

      QDPDefaultAllocator(const QDPDefaultAllocator&) = delete;
      QDPDefaultAllocator& operator=(const QDPDefaultAllocator&) = delete;

      friend class QDP::SingletonHolder<QDP::Allocator::QDPDefaultAllocator>;
    public:

      void* allocate(size_t n_bytes, const MemoryPoolHint& mem_pool_hint)
      {
	void* ptr = std::malloc( n_bytes > 0 ? n_bytes : 1 );
	if (!ptr)
	  throw std::bad_alloc();
	return ptr;
      }

      void free(void* mem)
      {
	std::free(mem);
      }

      void pushFunc(const char* func, int line) {}
      void popFunc() {}
    };

    typedef SingletonHolder<QDP::Allocator::QDPDefaultAllocator> theQDPAllocator;

  } // namespace Allocator


  namespace Hints
  {

    template<typename T>
    inline
    void moveToFastMemoryHint(T& x, bool copy=false) {}

    template<typename T>
    inline
    void revertFromFastMemoryHint(T& x, bool copy=false) {}

  } // namespace Hints

} // namespace QDP

#endif
