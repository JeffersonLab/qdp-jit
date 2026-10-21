#ifndef QDP_JIT_CONFIG_H
#define QDP_JIT_CONFIG_H

namespace QDP
{
  JitDataOrder  qdp_jit_config_get_default_data_order();
  void          qdp_jit_config_set_default_data_order( JitDataOrder o );
  JitDataFormat qdp_jit_config_get_default_data_format();
  void          qdp_jit_config_set_default_data_format( JitDataFormat fmt );

  class JitDataFormatScope
  {
    JitDataFormat prev;
  public:
    explicit JitDataFormatScope( JitDataFormat fmt ) : prev( qdp_jit_config_get_default_data_format() )
    {
      qdp_jit_config_set_default_data_format( fmt );
    }
    ~JitDataFormatScope() { qdp_jit_config_set_default_data_format( prev ); }
    JitDataFormatScope( const JitDataFormatScope& ) = delete;
    JitDataFormatScope& operator=( const JitDataFormatScope& ) = delete;
  };
  
  bool qdp_jit_config_get_managed();
  void qdp_jit_config_set_managed(bool b);

  // bool qdp_jit_config_get_var_adapt();
  // void qdp_jit_config_set_var_adapt(bool b);

  // bool qdp_jit_config_get_var_warnings();
  // void qdp_jit_config_set_var_warnings(bool b);

  // bool qdp_jit_config_get_var_raise();
  // void qdp_jit_config_set_var_raise(bool b);

  bool qdp_jit_config_get_hostjit();
  void qdp_jit_config_set_hostjit( bool b = true );

  bool qdp_jit_config_get_gpu_sync();
  void qdp_jit_config_set_gpu_sync(bool b);
  
  int  qdp_jit_config_get_use_gpu();
  void qdp_jit_config_set_use_gpu(int v);

  int  qdp_jit_config_get_blocksum();
  void qdp_jit_config_set_blocksum(int v);
  

  bool jit_config_get_timing_run();
  void jit_config_set_timing_run(bool v);

  int jit_config_get_oscalar_ringbuffer_size();
  void jit_config_set_oscalar_ringbuffer_size(int n);

  void jit_config_delayed_message(std::string txt);
  void jit_config_print_delayed_message();

  bool jit_config_get_verbose_output();
  void jit_config_set_verbose_output(bool v);

  bool jit_config_get_debug_layout();
  void jit_config_set_debug_layout(bool v);

  size_t jit_config_get_pool_alignment();
  void jit_config_set_pool_alignment(size_t size );

  int  jit_config_get_codegen_opt();
  void jit_config_set_codegen_opt(int opt);

#ifdef QDP_BACKEND_ROCM
  std::string jit_config_get_libdevice_path();
  std::string jit_config_get_libdevice_name();
  std::string jit_config_get_prepend_path();
  void        jit_config_set_prepend_path(std::string s);
  void jit_config_add_extra_lib( std::string l );
  std::vector<std::string>& jit_config_get_extra_lib();
#endif

#ifdef QDP_BACKEND_CUDA
  int  jit_config_get_CUDA_FTZ();
  void jit_config_set_CUDA_FTZ(int i);

  int  jit_config_get_max_regs_per_block();
#endif

#if defined(QDP_BACKEND_ROCM) || defined(QDP_BACKEND_L0)
  bool jit_config_get_keepfiles();
  void jit_config_set_keepfiles(bool v);
#endif

#ifdef QDP_BACKEND_L0
  std::string jit_config_get_l0_build_flags();
  void        jit_config_set_l0_build_flags(std::string s);
#endif

  void jit_config_set_pool_size( size_t val );
  size_t jit_config_get_pool_size();
  void jit_config_set_pool_reserve( size_t val );
  size_t jit_config_get_pool_reserve();

  int  jit_config_get_max_allocation();
  void jit_config_set_max_allocation(int size );

  void jit_config_set_threads_per_block( int t );
  int  jit_config_get_threads_per_block();

#ifdef QDP_BACKEND_ROCM
  void jit_config_set_workgroup_bound( bool b );
  bool jit_config_get_workgroup_bound();

  void jit_config_set_opt_prepass( bool b );
  bool jit_config_get_opt_prepass();
#endif

  
  void jit_config_print();

  bool jit_config_pool_stats();
  void jit_set_config_pool_stats();
  
#ifdef QDP_DEEP_LOG
  unsigned char jit_config_get_memset_value();
  void jit_config_set_memset_value( unsigned char uc );

  size_t jit_config_get_log_events();
  void jit_config_set_log_events( unsigned e );

  double jit_config_get_fuzzfactor();
  void   jit_config_set_fuzzfactor(double i);

  double jit_config_get_tolerance();
  void   jit_config_set_tolerance(double i);

  void        jit_config_deep_log_start();
  void        jit_config_deep_log_stop();

  bool        jit_config_deep_log();
  bool        jit_config_deep_log_create();
  std::string jit_config_deep_log_name();

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
