#include "qdp.h"


namespace QDP
{
  namespace
  {
    // Memory Pool
    int thread_stack = 64 * sizeof(REAL);  //1024;
    bool use_total_pool_size = false;
    size_t pool_size = 0;
    bool use_defrag = false;
    int max_allocation_size = -1;
    size_t pool_alignment = 128;

    // Kernel Launch
    int threads_per_block = 128;

    // In case memory allocation fails, decrease Pool size by this amount for next try.
    size_t pool_size_decrement = 10 * 1024*1024;   // 10MB

    bool pool_count_allocations = false;

    bool verbose_output = false;
    
#ifdef QDP_BACKEND_ROCM
    int  codegen_opt = 1;
#endif
    
#ifdef QDP_DEEP_LOG
    bool deep_log = false;
    bool deep_log_create = false;
    std::string deep_log_name = "qdp-jit-log.dat";
#endif
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

  bool jit_config_get_verbose_output() { return verbose_output; }
  void jit_config_set_verbose_output(bool v) { verbose_output = v; }

#ifdef QDP_BACKEND_ROCM
  int jit_config_get_codegen_opt() { return codegen_opt; }
  void jit_config_set_codegen_opt(int opt) { codegen_opt = opt; }
#endif
  
  size_t jit_config_get_pool_alignment() { return pool_alignment; }
  void jit_config_set_pool_alignment(size_t size ) { pool_alignment = size; }

  
  int jit_config_get_max_allocation() { return max_allocation_size; }
  void jit_config_set_max_allocation(int size ) { max_allocation_size = size; }
  
  bool jit_config_pool_stats() { return pool_count_allocations; }
  void jit_set_config_pool_stats() { pool_count_allocations = true; }


#ifdef QDP_DEEP_LOG
  bool        jit_config_deep_log() { return deep_log; }
  bool        jit_config_deep_log_create() { return deep_log_create; }
  std::string jit_config_deep_log_name() { return deep_log_name; }
  
  void        jit_config_deep_set( std::string name , bool create )
  {
    deep_log = true;
    deep_log_create = create;
    deep_log_name = name;
  }
#endif

  
  bool qdp_jit_config_defrag()
  {
    return use_defrag;
  }

  void qdp_jit_set_defrag()
  {
    use_defrag = true;
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
