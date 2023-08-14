#ifndef QDP_JIT_CONFIG_H
#define QDP_JIT_CONFIG_H

namespace QDP
{
  int  qdp_jit_config_get_global_addrspace();
  void qdp_jit_config_set_global_addrspace(int v);

  int  qdp_jit_config_get_local_addrspace();
  void qdp_jit_config_set_local_addrspace(int v);

  int  qdp_jit_config_get_use_gpu();
  void qdp_jit_config_set_use_gpu(int v);

  bool qdp_jit_config_get_opt_shifts();
  void qdp_jit_config_set_opt_shifts(bool v);

  bool jit_config_get_instcombine();
  bool jit_config_get_inline();

  void jit_config_set_instcombine(bool b);
  void jit_config_set_inline(bool b);

  bool jit_config_get_timing_run();
  void jit_config_set_timing_run(bool v);

  int jit_config_get_oscalar_ringbuffer_size();
  void jit_config_set_oscalar_ringbuffer_size(int n);

  bool jit_config_get_tuning();
  void jit_config_set_tuning(bool v);

  bool jit_config_get_tuning_verbose();
  void jit_config_set_tuning_verbose(bool v);

  void jit_config_delayed_message(std::string txt);
  void jit_config_print_delayed_message();

  std::string jit_config_get_tuning_file();
  void jit_config_set_tuning_file( std::string v);

  void jit_config_set_threads_per_block_min( int t );
  void jit_config_set_threads_per_block_max( int t );
  void jit_config_set_threads_per_block_step( int t );
  void jit_config_set_threads_per_block_loops( int t );
  int jit_config_get_threads_per_block_min();
  int jit_config_get_threads_per_block_max();
  int jit_config_get_threads_per_block_step();
  int jit_config_get_threads_per_block_loops();

  bool jit_config_get_verbose_output();
  void jit_config_set_verbose_output(bool v);

  size_t jit_config_get_pool_alignment();
  void jit_config_set_pool_alignment(size_t size );

#ifdef QDP_BACKEND_ROCM
  std::string jit_config_get_prepend_path();
  void        jit_config_set_prepend_path(std::string s);
  int  jit_config_get_codegen_opt();
  void jit_config_set_codegen_opt(int opt);
  void jit_config_add_extra_lib( std::string l );
  std::vector<std::string>& jit_config_get_extra_lib();
#endif

#ifdef QDP_BACKEND_CUDA
  int  jit_config_get_CUDA_FTZ();
  void jit_config_set_CUDA_FTZ(int i);

  int  jit_config_get_max_regs_per_block();
#endif

#if defined(QDP_BACKEND_ROCM) || (QDP_BACKEND_L0)
  bool jit_config_get_keepfiles();
  void jit_config_set_keepfiles(bool v);
#endif

  void jit_config_set_pool_size( size_t val );
  void jit_config_set_thread_stack( int stack );
  size_t jit_config_get_pool_size();

  int  jit_config_get_max_allocation();
  void jit_config_set_max_allocation(int size );

  void jit_config_set_threads_per_block( int t );
  int  jit_config_get_threads_per_block();

  size_t qdp_jit_config_pool_size_decrement();
  
  void jit_config_print();

  bool jit_config_pool_stats();
  void jit_set_config_pool_stats();
  
#ifdef QDP_DEEP_LOG
  size_t jit_config_get_log_events();
  void jit_config_set_log_events( unsigned e );

  double jit_config_get_fuzzfactor();
  void   jit_config_set_fuzzfactor(double i);

  double jit_config_get_tolerance();
  void   jit_config_set_tolerance(double i);

  bool        jit_config_deep_log();
  bool        jit_config_deep_log_create();
  std::string jit_config_deep_log_name();

  void        jit_config_deep_log_start();

  void        jit_config_deep_set( std::string name , bool create );
#endif

#if defined(QDP_USE_PROFILING)
  void qdp_jit_CPU_add( const std::string& pretty );
  std::map<std::string,int>& qdp_jit_CPU_getall();
#endif

  void jit_config_set_gpu_direct(bool g);
  bool jit_config_get_gpu_direct();


}


#endif
