#ifndef QDP_JIT_CONFIG_H
#define QDP_JIT_CONFIG_H

namespace QDP
{
  bool jit_config_get_verbose_output();
  void jit_config_set_verbose_output(bool v);

  size_t jit_config_get_pool_alignment();
  void jit_config_set_pool_alignment(size_t size );

  int jit_config_get_codegen_opt();
  void jit_config_set_codegen_opt(int opt);

  void jit_config_set_pool_size( size_t val );
  void jit_config_set_thread_stack( int stack );
  size_t jit_config_get_pool_size();

  int  jit_config_get_max_allocation();
  void jit_config_set_max_allocation(int size );

  bool qdp_jit_config_defrag();
  void qdp_jit_set_defrag();

  void jit_config_set_threads_per_block( int t );
  int  jit_config_get_threads_per_block();

  size_t qdp_jit_config_pool_size_decrement();
  
  void jit_config_print();

  bool jit_config_pool_stats();
  void jit_set_config_pool_stats();
  
#ifdef QDP_DEEP_LOG
  bool        jit_config_deep_log();
  bool        jit_config_deep_log_create();
  std::string jit_config_deep_log_name();

  void        jit_config_deep_set( std::string name , bool create );
#endif  
}


#endif
