#include "qdp.h"


namespace QDP
{
  namespace
  {
    // Memory Pool
    int thread_stack = 64 * sizeof(REAL);  //1024;
    bool use_total_pool_size = false;
    size_t pool_size = 0;

    // Kernel Launch
    int threads_per_block = 128;

    // In case memory allocation fails, decrease Pool size by this amount for next try.
    size_t pool_size_decrement = 10 * 1024*1024;   // 10MB
  }

  void jit_config_print()
  {
    QDPIO::cout << "qdp-jit configuration\n";
    QDPIO::cout << "  threads per block                   : " << threads_per_block << "\n";
    if (use_total_pool_size)
    QDPIO::cout << "  memory pool size (user request)     : " << pool_size/1024/1024 << " MB\n";
    else
      {
    QDPIO::cout << "  reserved memory per thread          : " << thread_stack << " bytes\n";
    auto val = gpu_mem_free() - Layout::sitesOnNode() * thread_stack;
    QDPIO::cout << "  resulting memory pool size          : " << val/1024/1024 << " MB\n";
      }
  }


  size_t qdp_jit_config_pool_size_decrement()
  {
    return pool_size_decrement;
  }
  
  
  void jit_config_set_threads_per_block( int t )
  {
    threads_per_block = t;
  }

  int jit_config_get_threads_per_block()
  {
    return threads_per_block;
  }



  void jit_config_set_pool_size( size_t val )
  {
    use_total_pool_size = true;
    pool_size = val;
  }
  
  void jit_config_set_thread_stack( int stack )
  {
    thread_stack = stack;
  }

  size_t jit_config_get_pool_size()
  {
    if (use_total_pool_size)
      {
	return pool_size;
      }
    else
      {
	return gpu_mem_free() - Layout::sitesOnNode() * thread_stack;
      }
  }

}
