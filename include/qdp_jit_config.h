#ifndef QDP_JIT_CONFIG_H
#define QDP_JIT_CONFIG_H

namespace QDP
{
  void jit_config_set_pool_size( size_t val );
  void jit_config_set_thread_stack( int stack );
  size_t jit_config_get_pool_size();

  void jit_config_set_threads_per_block( int t );
  int  jit_config_get_threads_per_block();


  void jit_config_print();

  
}


#endif
