#include "qdp.h"


namespace QDP
{
  namespace
  {
#if defined (QDP_HOST_COMPILER)
    bool host_jit = true;
#else
    bool host_jit = false;
#endif
    bool gpu_ctxsync = false;
    
    bool managed = false;
    JitDataFormat default_data_format;

    
    // Memory Pool
    bool use_total_pool_size = false;
    size_t requested_pool_size = 0;
    size_t pool_size = 0;

    bool use_pool_reserve = false;
    size_t pool_reserve = 0;
    size_t effective_pool_reserve = 0;
    double gpu_direct_reserve_fraction = 0.10;
    int max_allocation_size = -1;
    size_t pool_alignment = 256;
    size_t min_total_reserved_GPU_memory = 50*1024*1024; // 50 MB
    
    // In case the Layout is not initialized when the pool size is set
    // use this fraction of free GPU memory to determine the pool size
    double free_mem_fraction = 0.92; 

    enum PoolSetMethod { User , Fraction };
    PoolSetMethod poolSetMethod = PoolSetMethod::Fraction;
    
    bool pool_count_allocations = false;

    bool verbose_output = false;

    bool debug_layout = false;

    //
    int threads_per_block = 128;        // default value
#ifdef QDP_BACKEND_ROCM
    bool workgroup_bound = true;
    bool opt_prepass = true;
#endif

    // Delay output when QDPIO is not ready yet
    std::vector<std::string> delayed_output;

    // Ring buffer size for OScalars
    int oscalar_ringbuffer_size = 100;

    // Timing run?
    bool timing_run = false;

#if defined (QDP_BACKEND_ROCM)
    int  codegen_opt = 1;
#elif defined (QDP_BACKEND_CUDA)
    int  codegen_opt = 3;
#else
    int  codegen_opt = 2;
#endif
    
#ifdef QDP_BACKEND_ROCM
    std::vector<std::string> extra_libs;
    std::string prepend_path = "./";

    std::string libdevice_name = "libomptarget-new-amdgpu-gfx90a-5.4.3.bc";
    std::string libdevice_path = "/libdevice/";
#endif

#ifdef QDP_BACKEND_CUDA
    int CUDA_FTZ = 0;
    int max_regs_per_block = 64*1024;
#endif

#if defined(QDP_BACKEND_ROCM) || defined (QDP_BACKEND_L0)
    bool codegen_keepfiles = false;
#endif

#ifdef QDP_BACKEND_L0
    // 256 registers per thread where it avoids register spills
    std::string l0_build_flags = "-ze-intel-enable-auto-large-GRF-mode";
#endif
    
#ifdef QDP_DEEP_LOG
    unsigned char memset_value = 0;
    bool deep_log = false;
    bool deep_log_create = false;
    std::string deep_log_name = "qdp-jit-log.dat";
    double tolerance = 1.0e-5;
    double fuzzfactor = 50;
    size_t deep_log_events = 0;
#endif

#if ! defined (QDP_BACKEND_CUDA) && ! defined (QDP_BACKEND_ROCM) && ! defined (QDP_BACKEND_L0)
#error "No backend specified"
#endif


    
    int use_gpu = -1;
    
    bool gpu_direct = false;

    //bool var_adapt    = true;
    bool var_warnings = false;
    bool var_raise    = false;

    int type_of_blocksum = 0;
  }


  JitDataOrder qdp_jit_config_get_default_data_order() { return default_data_format.order; }
  void         qdp_jit_config_set_default_data_order( JitDataOrder o )
  {
    default_data_format.order = o;
  }

  JitDataFormat qdp_jit_config_get_default_data_format() { return default_data_format; }
  void          qdp_jit_config_set_default_data_format( JitDataFormat fmt ) { default_data_format = fmt; }

  bool qdp_jit_config_get_gpu_sync() { return gpu_ctxsync; }
  void qdp_jit_config_set_gpu_sync(bool b) { gpu_ctxsync = b; }


  bool qdp_jit_config_get_managed() { return managed; }
  void qdp_jit_config_set_managed(bool b)
  {
    managed = b;
  }


  
  bool qdp_jit_config_get_var_warnings() { return var_warnings; }
  void qdp_jit_config_set_var_warnings(bool b)
  {
    var_warnings = b;
    std::string tmp = "Setting variable Nc/Ns adaptation warnings " + std::to_string( b?1:0 );
    jit_config_delayed_message(tmp);
  }


  bool qdp_jit_config_get_var_raise() { return var_raise; }
  void qdp_jit_config_set_var_raise(bool b)
  {
    var_raise = b;
    std::string tmp = "Setting variable Nc/Ns adaptation raise    " + std::to_string( b?1:0 );
    jit_config_delayed_message(tmp);
  }

  
  bool qdp_jit_config_get_hostjit() { return host_jit; }
  void qdp_jit_config_set_hostjit( bool b ) { host_jit = b; }

  int  qdp_jit_config_get_use_gpu() {return use_gpu;}
  void qdp_jit_config_set_use_gpu(int v) { use_gpu = v; } 


#ifdef QDP_BACKEND_CUDA
  int  jit_config_get_CUDA_FTZ()   { return CUDA_FTZ; }
  void jit_config_set_CUDA_FTZ(int i)   { CUDA_FTZ = i; }

  int  jit_config_get_max_regs_per_block()   { return max_regs_per_block; }
#endif

  bool jit_config_get_timing_run() { return timing_run; }
  void jit_config_set_timing_run(bool v) { timing_run = v; }
  

  void jit_config_delayed_message(std::string txt)
  {
    delayed_output.push_back(txt);
  }

  void jit_config_print_delayed_message()
  {
    for ( auto i : delayed_output )
      {
	QDPIO::cout << i << std::endl;
      }
    delayed_output.clear();
  }
  
  void jit_config_print()
  {
#if QDP_USE_LEXICO_LAYOUT == 1
    QDPIO::cout << "Data layout: lexicographic" << endl;
#elif QDP_USE_CB2_LAYOUT == 1
    QDPIO::cout << "Data layout: 2 checkerboard (red/black)" << endl;
#elif QDP_USE_CB3D_LAYOUT == 1
    QDPIO::cout << "Data layout: 3D checkerboard (red/black)" << endl;
#elif QDP_USE_CB32_LAYOUT == 1
    QDPIO::cout << "Data layout: 32 checkerboard layout (experimental)" << endl;
#endif


    if (qdp_jit_config_get_managed())
      {
	QDPIO::cout << "Using managed GPU memory\n";
      }
    else
      {
	QDPIO::cout << "Memory pool config:\n";

	switch (poolSetMethod) {
	case PoolSetMethod::User:
	  QDPIO::cout << "  memory pool size (user request)     : " << requested_pool_size/1024/1024 << " MB\n";
	  if (pool_size != requested_pool_size)
	    QDPIO::cout << "  resulting memory pool size          : " << pool_size/1024/1024 << " MB\n";
	  break;
	case PoolSetMethod::Fraction:
	  QDPIO::cout << "  memory pool size (per fraction)     : " << pool_size/1024/1024 << " MB\n";
	  break;
	}
	if (effective_pool_reserve > 0)
	  QDPIO::cout << "  device memory reserved outside pool : " << effective_pool_reserve/1024/1024 << " MB\n";
      }
    
    QDPIO::cout << "Floating point base precision (bits)  : " << (int)BASE_PRECISION << "\n";
    QDPIO::cout << "Accurate timing                       : " << (int)jit_config_get_timing_run() << "\n";
    
    QDPIO::cout << "Code generation:\n";
#if defined (QDP_BACKEND_CUDA)
    QDPIO::cout << "  CUDA flush denormals to zero        : " << (int)jit_config_get_CUDA_FTZ() << "\n";
#endif
    QDPIO::cout << "  Standard compiler optimization      : " << (int)jit_config_get_codegen_opt() << "\n";
#if defined (QDP_HOST_COMPILER)
    QDPIO::cout << "  Host JIT for scalar expressions     : " << (qdp_jit_config_get_hostjit() ? "on" : "off") << "\n";
#endif
    if (qdp_jit_config_get_blocksum())
      {
	QDPIO::cout << "  blockSum threads per tile           : " << (int)qdp_jit_config_get_blocksum() << "\n";
      }
    
#ifdef QDP_BACKEND_L0
    QDPIO::cout << "  Level Zero build flags              : " << jit_config_get_l0_build_flags() << "\n";
#endif
#ifdef QDP_BACKEND_ROCM
    QDPIO::cout << "  Pre-pass instead of generic opt     : " << (int)opt_prepass << "\n";
    QDPIO::cout <<"  Temp Files Prepend path              : " <<  jit_config_get_prepend_path() << "\n";
#endif

    if (qdp_jit_config_get_gpu_sync())
      QDPIO::cout << "GPU context sync after kernel launch  : 1\n";
    
    QDPIO::cout << "Launch configuration:\n";
    QDPIO::cout << "  Threads per block                   : " << threads_per_block << "\n";
#ifdef QDP_BACKEND_ROCM
    QDPIO::cout << "  Workgroup-size bound on kernels     : " << (int)workgroup_bound << "\n";
#endif
    QDPIO::cout << "  Using GPU-aware MPI                 : " << (int)jit_config_get_gpu_direct() << "\n";
  }


  bool jit_config_get_verbose_output() { return verbose_output; }
  void jit_config_set_verbose_output(bool v) { verbose_output = v; }

  bool jit_config_get_debug_layout() { return debug_layout; }
  void jit_config_set_debug_layout(bool v) { debug_layout = v; }

  int jit_config_get_oscalar_ringbuffer_size() { return oscalar_ringbuffer_size; }
  void jit_config_set_oscalar_ringbuffer_size(int n) { oscalar_ringbuffer_size = n; }

  int jit_config_get_codegen_opt() { return codegen_opt; }
  void jit_config_set_codegen_opt(int opt) { codegen_opt = opt; }

#ifdef QDP_BACKEND_ROCM
  std::string jit_config_get_libdevice_path() { return libdevice_path; }
  std::string jit_config_get_libdevice_name() { return libdevice_name; }
  
  std::string jit_config_get_prepend_path() { return prepend_path; }
  void        jit_config_set_prepend_path(std::string s) { prepend_path = s; }
  
  void jit_config_add_extra_lib( std::string l )
  {
    extra_libs.push_back( l );
  }
  std::vector<std::string>& jit_config_get_extra_lib()
  {
    return extra_libs;
  }
#endif

  
#if defined(QDP_BACKEND_ROCM) || defined (QDP_BACKEND_L0)
  bool jit_config_get_keepfiles() { return codegen_keepfiles; }
  void jit_config_set_keepfiles(bool v) { codegen_keepfiles = v; }
#endif

#ifdef QDP_BACKEND_L0
  std::string jit_config_get_l0_build_flags() { return l0_build_flags; }
  void        jit_config_set_l0_build_flags(std::string s) { l0_build_flags = s; }
#endif

  size_t jit_config_get_pool_alignment() { return pool_alignment; }
  void jit_config_set_pool_alignment(size_t size ) { pool_alignment = size; }

  
  int jit_config_get_max_allocation() { return max_allocation_size; }
  void jit_config_set_max_allocation(int size ) { max_allocation_size = size; }

  int  qdp_jit_config_get_blocksum() { return type_of_blocksum; }
  void qdp_jit_config_set_blocksum(int v) { type_of_blocksum = v;};

  
  bool jit_config_pool_stats() { return pool_count_allocations; }
  void jit_set_config_pool_stats() { pool_count_allocations = true; }


#ifdef QDP_DEEP_LOG
  unsigned char jit_config_get_memset_value() { return memset_value; }
  void jit_config_set_memset_value( unsigned char uc ) { memset_value = uc; }
  
  
  size_t jit_config_get_log_events() { return deep_log_events; }
  void jit_config_set_log_events( unsigned e ) { deep_log_events = e; }

  double jit_config_get_fuzzfactor()           { return fuzzfactor; }
  void   jit_config_set_fuzzfactor(double i)   { fuzzfactor = i; }
  
  double jit_config_get_tolerance()           { return tolerance; }
  void   jit_config_set_tolerance(double i)   { tolerance = i; }

  bool        jit_config_deep_log() { return deep_log; }
  bool        jit_config_deep_log_create() { return deep_log_create; }
  std::string jit_config_deep_log_name() { return deep_log_name; }

  void        jit_config_deep_log_start() { deep_log = true; }
  void        jit_config_deep_log_stop() { deep_log = false; }

  void        jit_config_deep_set( std::string name , bool create )
  {
    deep_log = true;
    deep_log_create = create;
    deep_log_name = name;
  }
#endif

  
  

  
  
  void jit_config_set_threads_per_block( int t )
  {
    threads_per_block = t;
  }

  int jit_config_get_threads_per_block()
  {
    return threads_per_block;
  }

#ifdef QDP_BACKEND_ROCM
  void jit_config_set_workgroup_bound( bool b ) { workgroup_bound = b; }
  bool jit_config_get_workgroup_bound() { return workgroup_bound; }

  void jit_config_set_opt_prepass( bool b ) { opt_prepass = b; }
  bool jit_config_get_opt_prepass() { return opt_prepass; }
#endif

  void jit_config_set_pool_size( size_t val )
  {
    use_total_pool_size = true;
    requested_pool_size = val;
  }

  void jit_config_set_pool_reserve( size_t val )
  {
    use_pool_reserve = true;
    pool_reserve = val;
  }

  size_t jit_config_get_pool_reserve()
  {
    return effective_pool_reserve;
  }
  
  size_t jit_config_get_pool_size()
  {
    const size_t free_memory = gpu_mem_free();

    if (use_pool_reserve)
      effective_pool_reserve = pool_reserve;
    else if (jit_config_get_gpu_direct())
      effective_pool_reserve = (size_t)((double)free_memory * gpu_direct_reserve_fraction);
    else
      effective_pool_reserve = 0;

    if (effective_pool_reserve >= free_memory)
      {
	QDP_error_exit("Device memory reserve outside the pool (%zu MB) is not smaller than the free device memory (%zu MB)",
		       effective_pool_reserve/1024/1024 , free_memory/1024/1024 );
      }

    const size_t maximum_pool_size = free_memory - effective_pool_reserve;

    if (use_total_pool_size)
      {
	poolSetMethod = PoolSetMethod::User;
	pool_size = requested_pool_size;

	if (effective_pool_reserve > 0 && pool_size > maximum_pool_size)
	  {
	    QDPIO::cout << "Reducing the memory pool from " << pool_size/1024/1024 << " MB to "
			<< maximum_pool_size/1024/1024 << " MB to keep " << effective_pool_reserve/1024/1024
			<< " MB outside the pool" << std::endl;
	    pool_size = maximum_pool_size;
	  }
      }
    else
      {
	poolSetMethod = PoolSetMethod::Fraction;
	pool_size = std::min( (size_t)((double)free_memory * free_mem_fraction) , maximum_pool_size );
      }

    return pool_size;
  }


#if defined(QDP_USE_PROFILING)
  namespace
  {
    std::map<std::string,int> _map_cpu;
  }

  void qdp_jit_CPU_add( const std::string& pretty )
  {
    _map_cpu[ pretty ]++;
  }
  
  std::map<std::string,int>& qdp_jit_CPU_getall()
  {
    return _map_cpu;
  }
#endif


  void jit_config_set_gpu_direct(bool g)
  {
    gpu_direct = g;
  }
  
  bool jit_config_get_gpu_direct()
  {
    return gpu_direct;
  }

  
}
