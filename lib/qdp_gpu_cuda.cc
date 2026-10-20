// -*- c++ -*-



#include "qdp_config_internal.h" 
#include "qdp.h"

#include <iostream>
#include <string>
#include <sstream>
#include <iterator>

#include "cuda.h"


#include <unistd.h>


#include "cudaProfiler.h"

namespace {
  std::string buffer;
  std::stringstream delayed(buffer);
}

namespace QDP {

  void gpu_print_info()
  {
    QDPIO::cout << delayed.str() << std::endl;
  }

  namespace {
#if defined (CUDA_VERSION)
    int CUDA_version_major = CUDA_VERSION / 1000;
    int CUDA_version_minor = (CUDA_VERSION % 1000)/10;
#else
    int CUDA_version_major = 0;
    int CUDA_version_minor = 0;
#endif
    
    CUdevice cuDevice;
    CUcontext cuContext;

    CUevent evStart;
    CUevent evStop;
    //CUstream prefetchStream;

    int deviceCount;


    unsigned device;
    std::string envvar;
    bool GPUDirect;
    bool syncDevice;
    unsigned maxKernelArg;

    unsigned smem;
    
    unsigned max_gridx;
    unsigned max_gridy;
    unsigned max_gridz;

    unsigned max_blockx;
    unsigned max_blocky;
    unsigned max_blockz;

    unsigned major;
    unsigned minor;

    size_t mem_free, mem_total;

    
    size_t roundDown2pow(size_t x)
    {
      size_t s=1;
      while (s<=x) s <<= 1;
      s >>= 1;
      return s;
    }
  }
  

  int gpu_SDK_version_major()
  {
    return CUDA_version_major;
  }

  int gpu_SDK_version_minor()
  {
    return CUDA_version_minor;
  }

  
  void gpu_auto_detect();



  void CudaCheckResult(CUresult result)
  {
    if (result != CUDA_SUCCESS) {
      const char *errStr;
      cuGetErrorString(result, &errStr);
      const char *errName;
      cuGetErrorName(result, &errName);
      QDP_error_exit("CUDA error %d %s (%s)", (int)result , errName, errStr );
    }
  }


  void gpu_create_events()
  {
    CUresult res = cuEventCreate ( &evStart, 0 );
    if (res != CUDA_SUCCESS)
      {
	printf("error event creation start\n");
	QDP_abort(1);
      }
    res = cuEventCreate ( &evStop, 0 );
    if (res != CUDA_SUCCESS)
      {
	printf("error event creation stop\n");
	QDP_abort(1);
      }
  }

  void gpu_record_start()
  {
    CUresult res = cuEventRecord ( evStart, 0 );
    if (res != CUDA_SUCCESS)
      {
	QDPIO::cout << "error event record start\n";
	QDP_abort(1);
      }
  }


  float gpu_record_stop_sync_time()
  {
    // stop
    CUresult res = cuEventRecord ( evStop, 0 );
    if (res != CUDA_SUCCESS)
      {
	QDPIO::cout << "error event record stop\n";
	QDP_abort(1);
      }

    // sync
    res = cuEventSynchronize ( evStop );
    if (res != CUDA_SUCCESS)
      {
	QDPIO::cout << "error event sync stop\n";
	QDP_abort(1);
      }

    // get time
    float pMilliseconds;
    res = cuEventElapsedTime( &pMilliseconds, evStart, evStop );
    if (res != CUDA_SUCCESS)
      {
	QDPIO::cout << "error event get time\n";
	QDP_abort(1);
      }

    return pMilliseconds;
  }


  void CudaRes(const std::string& s,CUresult ret);
  int CudaGetConfig(int what);

  size_t gpu_mem_free()
  {
    size_t total;
    CUresult ret = cuMemGetInfo( &mem_free , &total );
    CudaRes("cuMemGetInfo",ret);
    return mem_free;
  }


  size_t gpu_get_max_resident_threads()
  {
    return (size_t)CudaGetConfig( CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT ) *
      (size_t)CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR );
  }
  
  size_t gpu_mem_total()
  {
    return mem_total;
  }
  
  
  std::map< JitFunction::Func_t , std::string > mapCUFuncPTX;

  std::string getPTXfromCUFunc(JitFunction& f) {
    return mapCUFuncPTX[f.get_function()];
  }


  
  int CudaGetAttributesLocalSize( JitFunction& f )
  {
    int local_mem = 0;
    cuFuncGetAttribute( &local_mem , CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES , (CUfunction)f.get_function() );
    return local_mem;
  }



  JitResult gpu_launch_kernel( JitFunction& f, 
			       unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
			       unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
			       unsigned int  sharedMemBytes, QDPCache::KernelArgs_t kernelArgs , bool set_l0_event )
  {
    if (gpu_get_record_stats() && Layout::primaryNode())
      {
	gpu_record_start();
      }
    
    CUresult res = cuLaunchKernel((CUfunction)f.get_function(), gridDimX, gridDimY, gridDimZ, 
				  blockDimX, blockDimY, blockDimZ, 
				  sharedMemBytes, 0, (void**)&kernelArgs[0], 0 );

    if (qdp_jit_config_get_gpu_sync())
      {
	CUresult sync_ret = cuCtxSynchronize();

	if (sync_ret != CUDA_SUCCESS) {
	  const char* error_name = nullptr;
	  const char* error_string = nullptr;
      
	  cuGetErrorName(sync_ret, &error_name);
	  cuGetErrorString(sync_ret, &error_string);
      
	  QDPIO::cerr
	    << "CUDA error AFTER launching kernel " << f.get_kernel_name() << "\n"
	    << "CUDA error: "
	    << (error_name ? error_name : "unknown") << "\n"
	    << "Description: "
	    << (error_string ? error_string : "unknown") << "\n";
      
	  QDP_abort(1);
	}
      }
    
    if (gpu_get_record_stats() && Layout::primaryNode())
      {
	float time = gpu_record_stop_sync_time();
	f.add_timing( time );
      }
  
    JitResult ret;

    if ( res != CUDA_SUCCESS )
      {
	const char* errstr;
	cuGetErrorString(res, &errstr);
	printf("CUDA Driver Failure: returned 0x%x (%s)\n", res, errstr);
      }

    switch (res) {
    case CUDA_SUCCESS:
      ret = JitResult::JitSuccess;
      break;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
      ret = JitResult::JitResource;
      break;
    case CUDA_ERROR_OUT_OF_MEMORY:
      ret = JitResult::JitOutOfMemory;
      break;
    default:
      ret = JitResult::JitError;
    }
    
    return ret;
  }



  void CudaRes(const std::string& s,CUresult ret) {
    if (ret != CUDA_SUCCESS) {
      const char *errStr;
      cuGetErrorString(ret, &errStr);
      std::cout << s << " Error: " << errStr << "\n";
      exit(1);
    }
  }



  int CudaGetConfig(int what)
  {
    int data;
    CUresult ret;
    ret = cuDeviceGetAttribute( &data, (CUdevice_attribute)what , cuDevice );
    CudaRes("cuDeviceGetAttribute",ret);
    return data;
  }

  void CudaGetSM(int* maj,int* min) {
    CUresult ret;
    ret = cuDeviceGetAttribute(maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice );
    CudaRes("cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)",ret);
    ret = cuDeviceGetAttribute(min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice );
    CudaRes("cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)",ret);
  }

  void gpu_init() {
    cuInit(0);

    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) { 
      std::cout << "There is no device supporting CUDA.\n"; 
      exit(1); 
    }
  }

  
  void gpu_done() {
  }
    
  
  std::string gpu_get_arch()
  {
    int major = gpu_getMajor();
    int minor = gpu_getMinor();

    std::string str_compute = "sm_" + std::to_string( major * 10 + minor );

    return str_compute;
  }



  void gpu_set_device(int dev)
  {
    CUresult ret;


    ret = cuDeviceGet(&cuDevice, dev);
    CudaRes(__func__,ret);

    ret = cuDevicePrimaryCtxRetain(&cuContext, cuDevice);
    CudaRes(__func__,ret);

    ret = cuCtxPushCurrent(cuContext);
    CudaRes(__func__,ret);

    //std::cout << "creating CUDA events\n";
    gpu_create_events();
    //cuStreamCreate ( &prefetchStream, CU_STREAM_NON_BLOCKING);

    gpu_auto_detect();
  }


  void gpu_get_device_props() {
    smem = CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK );
    max_gridx = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X ) );
    max_gridy = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y ) );
    max_gridz = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z ) );
    max_blockx = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X ) );
    max_blocky = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y ) );
    max_blockz = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z ) );

    int ma,mi;
    CudaGetSM(&ma,&mi);
    
    major = ma;
    minor = mi;

    // hack to allow overrding gpu arch passed to llvmset
    char *qdpjit_cuda_arch = getenv("QDPJIT_ARCH");
    if (qdpjit_cuda_arch) {
      int arch = atoi(qdpjit_cuda_arch);
      major = arch/10;
      minor = arch%10;
      delayed << "Overriding Compute capability for JIT: " << qdpjit_cuda_arch << "\n";
    }
    
    delayed << "  Compute capability (major,minor)    : " << major << "," << minor << "\n";
    delayed << "  Shared memory                       : " << smem/1024  << " KB\n";
    delayed << "  Max grid  (x,y,z)                   : (" << max_gridx << "," << max_gridy << "," << max_gridz << ")\n";
    delayed << "  Max block (x,y,z)                   : (" << max_blockx << "," << max_blocky << "," << max_blockz << ")\n";
  }


  
  void gpu_auto_detect()
  {
    CUresult ret;

    delayed << "GPU properties\n";
    
    // Get the device properties
    gpu_get_device_props();
    
    ret = cuMemGetInfo( &mem_free , &mem_total );
    CudaRes("cuMemGetInfo",ret);

    delayed << "  GPU memory (free,total)             : " << mem_free/1024/1024 << "/" << mem_total/1024/1024 << " MB\n";


    ret = cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1);
    CudaRes("cuCtxSetCacheConfig",ret);
  }


  int gpu_get_device_count()
  {
    return deviceCount;
  }



  void gpu_host_alloc(void **mem , const size_t size)
  {
    CUresult ret;
    ret = cuMemHostAlloc(mem,size,0);
    CudaRes("cudaHostAlloc",ret);
  }


  void gpu_host_free(void *mem)
  {
    if (!QDP_isInitialized())
      {
	//std::cout << "host free, QDP not initialized" << std::endl;
	return;
      }
    CUresult ret;
    ret = cuMemFreeHost(mem);
    CudaRes("cuMemFreeHost",ret);
  }





  void gpu_memcpy( void * dest , const void * src , size_t size )
  {
    CUresult ret;
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("CudaMemcpy dest=%p src=%p size=%d" ,  dest , src , size );
#endif
    ret = cuMemcpy((CUdeviceptr)const_cast<void*>(dest), (CUdeviceptr)const_cast<void*>(src), size);
    CudaRes("cuMemcpy",ret);
  }

  bool gpu_malloc_managed(void **mem , size_t size )
  {
    CUresult ret;
    ret = cuMemAllocManaged( (CUdeviceptr*)mem , size , CU_MEM_ATTACH_GLOBAL );
    return ret == CUDA_SUCCESS;
  }

  
  void gpu_memcpy_h2d( void * dest , const void * src , size_t size )
  {
    CUresult ret;
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("CudaMemcpyH2D dest=%p src=%p size=%d" ,  dest , src , size );
#endif
    ret = cuMemcpyHtoD((CUdeviceptr)const_cast<void*>(dest), src, size);
    CudaRes("cuMemcpyH2D",ret);
  }

  void gpu_memcpy_d2h( void * dest , const void * src , size_t size )
  {
    CUresult ret;
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("CudaMemcpyD2H dest=%p src=%p size=%d" ,  dest , src , size );
#endif
    ret = cuMemcpyDtoH( dest, (CUdeviceptr)const_cast<void*>(src), size);
    CudaRes("cuMemcpyD2H",ret);
  }

  
  bool gpu_malloc(void **mem , size_t size )
  {
    CUresult ret;
    ret = cuMemAlloc( (CUdeviceptr*)mem,size);
    return ret == CUDA_SUCCESS;
  }

  
  void gpu_free(const void *mem )
  {
    if (!QDP_isInitialized())
      {
	return;
      }

    CUresult ret;
    ret = cuMemFree((CUdeviceptr)const_cast<void*>(mem));
    CudaRes("cuMemFree",ret);
  }



  void gpu_memset( void * dest , unsigned char val , size_t N )
  {
    CUresult ret;
    ret = cuMemsetD8((CUdeviceptr)dest, val, N);
    CudaRes("cuMemsetD8",ret);
  }


  
  void gpu_sync()
  {
    CUresult ret = cuCtxSynchronize();
    CudaRes("cuCtxSynchronize",ret);
  }


  int cuda_get_attribute( CUfunction_attribute a , CUfunction f )
  {
    CUresult ret;

    int tmp;
    ret = cuFuncGetAttribute ( &tmp , a , f );
    if (ret != CUDA_SUCCESS)
    {
      QDPIO::cerr << "Error getting function attribute: num regs.";
      QDP_abort(1);
    }
    return tmp;
  }


  bool get_jitf( JitFunction& func, const std::string& kernel_ptx , const std::string& kernel_name , const std::string& pretty , const std::string& str_compute )
  {
    CUresult ret;
    CUmodule cuModule;

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "Kernel PTX:\n" << kernel_ptx << "\n-----\n\n";
	QDPIO::cout << "Kernel name:\n" << kernel_name << "\n-----\n\n";
      }
    
    func.set_kernel_name( kernel_name );
    func.set_pretty( pretty );
    
    constexpr size_t log_size = 16384;

    char error_log[log_size] = {};
    char info_log[log_size] = {};

    CUjit_option options[] = {
      CU_JIT_ERROR_LOG_BUFFER,
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_INFO_LOG_BUFFER,
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_LOG_VERBOSE
    };

    void* option_values[] = {
      error_log,
      reinterpret_cast<void*>(log_size),
      info_log,
      reinterpret_cast<void*>(log_size),
      reinterpret_cast<void*>(1)
    };

    ret = cuModuleLoadDataEx(&cuModule,
			     static_cast<const void*>(kernel_ptx.c_str()),
			     sizeof(options) / sizeof(options[0]),
			     options,
			     option_values);

    if (ret != CUDA_SUCCESS) {
      const char* error_name = nullptr;
      const char* error_string = nullptr;

      cuGetErrorName(ret, &error_name);
      cuGetErrorString(ret, &error_string);

      QDPIO::cerr
	<< "\ncuModuleLoadDataEx failed\n"
	<< "Kernel: " << kernel_name << "\n"
	<< "CUDA error: "
	<< (error_name ? error_name : "unknown") << "\n"
	<< "Description: "
	<< (error_string ? error_string : "unknown") << "\n"
	<< "\nCUDA JIT error log:\n"
	<< error_log << "\n"
	<< "\nCUDA JIT info log:\n"
	<< info_log << "\n";

      {
	QDPIO::cerr << "Writing failing PTX to failing.ptx\n";
	std::ofstream out("failing.ptx");
	out << kernel_ptx;
      }
      
      QDP_abort(1);
    }


    
    if (ret != CUDA_SUCCESS) {
      QDPIO::cerr << "Error loading external data.\n";
      raise(SIGSEGV);
      QDP_abort(1);
    }

    CUfunction cuf;
    ret = cuModuleGetFunction( &cuf , cuModule , kernel_name.c_str() );
    if (ret != CUDA_SUCCESS) {
      QDPIO::cerr << "Error getting function: " << kernel_name << std::endl;
      QDPIO::cerr << std::endl;
      raise(SIGSEGV);
      QDP_abort(1);
    }

    func.set_function( cuf );
    
    mapCUFuncPTX[func.get_function()] = kernel_ptx;

    func.set_regs ( cuda_get_attribute( CU_FUNC_ATTRIBUTE_NUM_REGS , cuf ) );
    func.set_stack( cuda_get_attribute( CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES , cuf ) );
    func.set_cmem ( cuda_get_attribute( CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES , cuf ) );
    
    return true;
  }



  kernel_geom_t getGeom(int numSites , int threadsPerBlock)
  {
    kernel_geom_t geom_host;

    int64_t num_sites = numSites;
  
    int64_t M = gpu_getMaxGridX() * threadsPerBlock;
    int64_t Nblock_y = (num_sites + M-1) / M;

    int64_t P = threadsPerBlock;
    int64_t Nblock_x = (num_sites + P-1) / P;

    geom_host.threads_per_block = threadsPerBlock;
    geom_host.Nblock_x = Nblock_x;
    geom_host.Nblock_y = Nblock_y;
    return geom_host;
  }


  size_t gpu_getMaxGridX()  {return max_gridx;}
  size_t gpu_getMaxGridY()  {return max_gridy;}
  size_t gpu_getMaxGridZ()  {return max_gridz;}

  size_t gpu_getMaxBlockX()  {return max_blockx;}
  size_t gpu_getMaxBlockY()  {return max_blocky;}
  size_t gpu_getMaxBlockZ()  {return max_blockz;}
  
  size_t gpu_getMaxSMem()  {return smem;}

  unsigned gpu_getMajor() { return major; }
  unsigned gpu_getMinor() { return minor; }
  
  
}


