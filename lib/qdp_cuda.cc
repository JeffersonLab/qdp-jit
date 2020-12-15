// -*- c++ -*-



#include "qdp_config_internal.h" 
#include "qdp.h"

#include <iostream>
#include <string>

#include "cuda.h"

#include "cudaProfiler.h"

using namespace std;


namespace {
  int max_local_size = 0;
  int max_local_usage = 0;
  size_t total_free = 0;
}


namespace QDP {

  std::map< JitFunction::Func_t , std::string > mapCUFuncPTX;

  std::string getPTXfromCUFunc(JitFunction f) {
    return mapCUFuncPTX[f.getFunction()];
  }


  int CudaGetMaxLocalSize() { return max_local_size; }
  int CudaGetMaxLocalUsage() { return max_local_usage; }
  size_t CudaGetInitialFreeMemory() { return total_free; }

  CUevent * QDPevCopied;

  CUdevice cuDevice;
  CUcontext cuContext;

  std::map<CUresult,std::string> mapCuErrorString= {
    {CUDA_SUCCESS,"CUDA_SUCCESS"},
    {CUDA_ERROR_INVALID_VALUE,"CUDA_ERROR_INVALID_VALUE"},
    {CUDA_ERROR_OUT_OF_MEMORY,"CUDA_ERROR_OUT_OF_MEMORY"},
    {CUDA_ERROR_NOT_INITIALIZED,"CUDA_ERROR_NOT_INITIALIZED"},
    {CUDA_ERROR_DEINITIALIZED,"CUDA_ERROR_DEINITIALIZED"},
    {CUDA_ERROR_PROFILER_DISABLED,"CUDA_ERROR_PROFILER_DISABLED"},
    {CUDA_ERROR_PROFILER_NOT_INITIALIZED,"CUDA_ERROR_PROFILER_NOT_INITIALIZED"},
    {CUDA_ERROR_PROFILER_ALREADY_STARTED,"CUDA_ERROR_PROFILER_ALREADY_STARTED"},
    {CUDA_ERROR_PROFILER_ALREADY_STOPPED,"CUDA_ERROR_PROFILER_ALREADY_STOPPED"},
    {CUDA_ERROR_NO_DEVICE,"CUDA_ERROR_NO_DEVICE"},
    {CUDA_ERROR_INVALID_DEVICE,"CUDA_ERROR_INVALID_DEVICE"},
    {CUDA_ERROR_INVALID_IMAGE,"CUDA_ERROR_INVALID_IMAGE"},
    {CUDA_ERROR_INVALID_CONTEXT,"CUDA_ERROR_INVALID_CONTEXT"},
    {CUDA_ERROR_CONTEXT_ALREADY_CURRENT,"CUDA_ERROR_CONTEXT_ALREADY_CURRENT"},
    {CUDA_ERROR_MAP_FAILED,"CUDA_ERROR_MAP_FAILED"},
    {CUDA_ERROR_UNMAP_FAILED,"CUDA_ERROR_UNMAP_FAILED"},
    {CUDA_ERROR_ARRAY_IS_MAPPED,"CUDA_ERROR_ARRAY_IS_MAPPED"},
    {CUDA_ERROR_ALREADY_MAPPED,"CUDA_ERROR_ALREADY_MAPPED"},
    {CUDA_ERROR_NO_BINARY_FOR_GPU,"CUDA_ERROR_NO_BINARY_FOR_GPU"},
    {CUDA_ERROR_ALREADY_ACQUIRED,"CUDA_ERROR_ALREADY_ACQUIRED"},
    {CUDA_ERROR_NOT_MAPPED,"CUDA_ERROR_NOT_MAPPED"},
    {CUDA_ERROR_NOT_MAPPED_AS_ARRAY,"CUDA_ERROR_NOT_MAPPED_AS_ARRAY"},
    {CUDA_ERROR_NOT_MAPPED_AS_POINTER,"CUDA_ERROR_NOT_MAPPED_AS_POINTER"},
    {CUDA_ERROR_ECC_UNCORRECTABLE,"CUDA_ERROR_ECC_UNCORRECTABLE"},
    {CUDA_ERROR_UNSUPPORTED_LIMIT,"CUDA_ERROR_UNSUPPORTED_LIMIT"},
    {CUDA_ERROR_CONTEXT_ALREADY_IN_USE,"CUDA_ERROR_CONTEXT_ALREADY_IN_USE"},
    {CUDA_ERROR_INVALID_SOURCE,"CUDA_ERROR_INVALID_SOURCE"},
    {CUDA_ERROR_FILE_NOT_FOUND,"CUDA_ERROR_FILE_NOT_FOUND"},
    {CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,"CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND"},
    {CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,"CUDA_ERROR_SHARED_OBJECT_INIT_FAILED"},
    {CUDA_ERROR_OPERATING_SYSTEM,"CUDA_ERROR_OPERATING_SYSTEM"},
    {CUDA_ERROR_INVALID_HANDLE,"CUDA_ERROR_INVALID_HANDLE"},
    {CUDA_ERROR_NOT_FOUND,"CUDA_ERROR_NOT_FOUND"},
    {CUDA_ERROR_NOT_READY,"CUDA_ERROR_NOT_READY"},
    {CUDA_ERROR_LAUNCH_FAILED,"CUDA_ERROR_LAUNCH_FAILED"},
    {CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,"CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"},
    {CUDA_ERROR_LAUNCH_TIMEOUT,"CUDA_ERROR_LAUNCH_TIMEOUT"},
    {CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,"CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"},
    {CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,"CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED"},
    {CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,"CUDA_ERROR_PEER_ACCESS_NOT_ENABLED"},
    {CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,"CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE"},
    {CUDA_ERROR_CONTEXT_IS_DESTROYED,"CUDA_ERROR_CONTEXT_IS_DESTROYED"},
    {CUDA_ERROR_ASSERT,"CUDA_ERROR_ASSERT"},
    {CUDA_ERROR_TOO_MANY_PEERS,"CUDA_ERROR_TOO_MANY_PEERS"},
    {CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,"CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED"},
    {CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,"CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED"},
    {CUDA_ERROR_UNKNOWN,"CUDA_ERROR_UNKNOWN"}};



  void CudaLaunchKernel( JitFunction f, 
			 unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
			 unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
			 unsigned int  sharedMemBytes, int hStream, void** kernelParams, void** extra )
  {
    //if ( blockDimX * blockDimY * blockDimZ > 0  &&  gridDimX * gridDimY * gridDimZ > 0 ) {
    
    JitResult result = CudaLaunchKernelNoSync(f, gridDimX, gridDimY, gridDimZ, 
					     blockDimX, blockDimY, blockDimZ, 
					     sharedMemBytes, 0, kernelParams, extra);
    if (result != JitResult::JitSuccess) {
      QDP_error_exit("CUDA launch error (CudaLaunchKernel): grid=(%u,%u,%u), block=(%u,%u,%u), shmem=%u",
		     gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes );

      CUresult result = cuCtxSynchronize();
      if (result != CUDA_SUCCESS) {

	if (mapCuErrorString.count(result)) 
	  std::cout << " Error: " << mapCuErrorString.at(result) << "\n";
	else
	  std::cout << " Error: (not known)\n";
      }      
      QDP_error_exit("CUDA launch error (CudaLaunchKernel, on sync): grid=(%u,%u,%u), block=(%u,%u,%u), shmem=%u",
		     gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes );
    }

  }

  
  int CudaGetAttributesLocalSize( JitFunction f )
  {
    int local_mem = 0;
    cuFuncGetAttribute( &local_mem , CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES , (CUfunction)f.getFunction() );
    return local_mem;
  }


  namespace {
    std::vector<unsigned> __kernel_geom;
    JitFunction            __kernel_ptr;
  }

  std::vector<unsigned> get_backed_kernel_geom() { return __kernel_geom; }
  JitFunction            get_backed_kernel_ptr() { return __kernel_ptr; }


  JitResult CudaLaunchKernelNoSync( JitFunction f, 
				    unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
				    unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
				    unsigned int  sharedMemBytes, int hStream, void** kernelParams, void** extra  )
  {
     // QDP_info("CudaLaunchKernelNoSync: grid=(%u,%u,%u), block=(%u,%u,%u), shmem=%u",
     // 	      gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes );
    // QDPIO::cout << "JitFunction = " << (size_t)(void*)f << "\n";
      


    //QDPIO::cout << "local mem (bytes) = " << num_threads << "\n";
    //
    
    CUresult res = cuLaunchKernel((CUfunction)f.getFunction(), gridDimX, gridDimY, gridDimZ, 
				  blockDimX, blockDimY, blockDimZ, 
				  sharedMemBytes, 0, kernelParams, extra);

    if (res == CUDA_SUCCESS)
      {
	//QDPIO::cout << "CUDA_SUCCESS\n";
	if (qdp_cache_get_launch_verbose())
	  {
	    QDP_info("CudaLaunchKernelNoSync: grid=(%u,%u,%u), block=(%u,%u,%u), shmem=%u",
		     gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes );
	  }
	
      }
    else
      {
	//QDPIO::cout << "no CUDA_SUCCESS " << mapCuErrorString[res] << "\n";
      }

    JitResult ret;

    switch (res) {
    case CUDA_SUCCESS:
      ret = JitResult::JitSuccess;
      break;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
      ret = JitResult::JitResource;
      break;
    default:
      ret = JitResult::JitError;
    }

    return ret;
  }


    

  void CudaCheckResult(CUresult result) {
    if (result != CUDA_SUCCESS) {
      QDP_info("CUDA error %d (%s)", (int)result , mapCuErrorString[result].c_str());
    }
  }


  void CudaRes(const std::string& s,CUresult ret) {
    if (ret != CUDA_SUCCESS) {
      if (mapCuErrorString.count(ret)) 
	std::cout << s << " Error: " << mapCuErrorString.at(ret) << "\n";
      else
	std::cout << s << " Error: (not known)\n";
      exit(1);
    }
  }



  int CudaAttributeNumRegs( JitFunction f ) {
    int pi;
    CUresult res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_NUM_REGS , (CUfunction)f.getFunction() );
    CudaRes("CudaAttributeNumRegs",res);
    return pi;
  }

  int CudaAttributeLocalSize( JitFunction f ) {
    int pi;
    CUresult res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES , (CUfunction)f.getFunction() );
    CudaRes("CudaAttributeLocalSize",res);
    return pi;
  }

  int CudaAttributeConstSize( JitFunction f ) {
    int pi;
    CUresult res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES , (CUfunction)f.getFunction() );
    CudaRes("CudaAttributeConstSize",res);
    return pi;
  }


  void CudaProfilerInitialize()
  {
    CUresult res;
    std::cout << "CUDA Profiler Initializing ...\n";
    res = cuProfilerInitialize( "prof.cfg" , "prof.out" , CU_OUT_CSV );
    CudaRes("cuProfilerInitialize",res);
  }

  void CudaProfilerStart()
  {
    CUresult res;
    res = cuProfilerStart();
    CudaRes("cuProfilerStart",res);
  }

  void CudaProfilerStop()
  {
    CUresult res;
    res = cuProfilerStop();
    CudaRes("cuProfilerStop",res);
  }




  //int CudaGetConfig(CUdevice_attribute what)
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

  void CudaInit() {
    //QDP_info_primary("CUDA initialization");
    cuInit(0);

    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) { 
      std::cout << "There is no device supporting CUDA.\n"; 
      exit(1); 
    }
  }








  void CudaSetDevice(int dev)
  {
    CUresult ret;

    QDP_info_primary("trying to get device %d",dev);
    ret = cuDeviceGet(&cuDevice, dev);
    CudaRes(__func__,ret);

    QDP_info_primary("trying to grab pre-existing context",dev);
    ret = cuCtxGetCurrent(&cuContext);
    
    if (ret != CUDA_SUCCESS || cuContext == NULL) {
      QDP_info_primary("trying to create a context");
      ret = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, cuDevice);
    }
    CudaRes(__func__,ret);

  }

  void CudaMemGetInfo(size_t *free,size_t *total)
  {
    CUresult ret = cuMemGetInfo(free, total);
    CudaRes("cuMemGetInfo",ret);
  }


  void CudaGetDeviceProps()
  {
    CUresult ret;

    gpu_autoDetect();

    size_t free, total;
    ret = cuMemGetInfo(&free, &total);
    CudaRes("cuMemGetInfo",ret);
    total_free = free;

    QDP_info_primary("GPU memory: free = %lld (%f MB),  total = %lld (%f MB)",
		     (unsigned long long)free , (float)free/1024./1024.,
		     (unsigned long long)total, (float)total/1024./1024. );
    if (!setPoolSize) {

      size_t val = (size_t)((double)(0.90) * (double)free);

      int val_in_MiB = val/1024/1024;

      if (val_in_MiB < 1)
	QDP_error_exit("Less than 1 MiB device memory available. Giving up.");

      float val_min = (float)val_in_MiB;

      QDPInternal::globalMinValue( &val_min );

      if ( val_min > (float)val_in_MiB )
	QDP_error_exit("Inconsistency: Global minimum %f larger than local value %d.",val_min,val_in_MiB);

      if ( val_min < (float)val_in_MiB ) {
	QDP_info("Global minimum %f of available GPU memory smaller than local value %d. Using global minimum.",val_min,val_in_MiB);
      }
      int val_min_int = (int)val_min;
      QDP_info_primary("Using device memory pool size: %d MB",(int)val_min_int);

      //CUDADevicePoolAllocator::Instance().setPoolSize( ((size_t)val_min_int) * 1024 * 1024 );
      QDP_get_global_cache().setPoolSize( ((size_t)val_min_int) * 1024 * 1024 );

      setPoolSize = true;
    } else {
      //QDP_info_primary("Using device pool size: %d MiB",(int)(CUDADevicePoolAllocator::Instance().getPoolSize()/1024/1024));
      QDP_info_primary("Using device pool size: %d MiB",(int)(QDP_get_global_cache().getPoolSize()/1024/1024));
    }


    ret = cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1);
    CudaRes("cuCtxSetCacheConfig",ret);
  }



  void CudaGetDeviceCount(int * count)
  {
    cuDeviceGetCount( count );
  }


  bool CudaHostRegister(void * ptr , size_t size)
  {
    CUresult ret;
    int flags = 0;
    QDP_info_primary("CUDA host register ptr=%p (%u) size=%lu (%u)",ptr,(unsigned)((size_t)ptr%4096) ,(unsigned long)size,(unsigned)((size_t)size%4096));
    ret = cuMemHostRegister(ptr, size, flags);
    CudaRes("cuMemHostRegister",ret);
    return true;
  }

  
  void CudaHostUnregister(void * ptr )
  {
    CUresult ret;
    ret = cuMemHostUnregister(ptr);
    CudaRes("cuMemHostUnregister",ret);
  }
  

  bool CudaHostAlloc(void **mem , const size_t size, const int flags)
  {
    CUresult ret;
    ret = cuMemHostAlloc(mem,size,flags);
    CudaRes("cudaHostAlloc",ret);
    return ret == CUDA_SUCCESS;
  }


  void CudaHostFree(void *mem)
  {
    CUresult ret;
    ret = cuMemFreeHost(mem);
    CudaRes("cuMemFreeHost",ret);
  }





  void CudaMemcpyH2D( void * dest , const void * src , size_t size )
  {
    CUresult ret;
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("CudaMemcpyH2D dest=%p src=%p size=%d" ,  dest , src , size );
#endif
    ret = cuMemcpyHtoD((CUdeviceptr)const_cast<void*>(dest), src, size);
    CudaRes("cuMemcpyH2D",ret);
  }

  void CudaMemcpyD2H( void * dest , const void * src , size_t size )
  {
    CUresult ret;
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("CudaMemcpyD2H dest=%p src=%p size=%d" ,  dest , src , size );
#endif
    ret = cuMemcpyDtoH( dest, (CUdeviceptr)const_cast<void*>(src), size);
    CudaRes("cuMemcpyD2H",ret);
  }


  bool CudaMalloc(void **mem , size_t size )
  {
    CUresult ret;
#ifndef QDP_USE_CUDA_MANAGED_MEMORY
    ret = cuMemAlloc( (CUdeviceptr*)mem,size);
#else
    ret = cuMemAllocManaged( (CUdeviceptr*)mem, size, CU_MEM_ATTACH_GLOBAL ); 
#endif

#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep( "CudaMalloc %p", *mem );
#endif

#ifndef  QDP_USE_CUDA_MANAGED_MEMORY
    //CudaRes("cuMemAlloc",ret);
#else 
    //CudaRes("cuMemAllocManaged", ret);
#endif

    return ret == CUDA_SUCCESS;
  }



  
  void CudaFree(const void *mem )
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep( "CudaFree %p", mem );
#endif
    CUresult ret;
    ret = cuMemFree((CUdeviceptr)const_cast<void*>(mem));
    CudaRes("cuMemFree",ret);
  }

  void CudaThreadSynchronize()
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep( "cudaThreadSynchronize" );
#endif
    cuCtxSynchronize();
  }

  void CudaDeviceSynchronize()
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep( "cudaDeviceSynchronize" );
#endif
    CUresult ret = cuCtxSynchronize();
    CudaRes("cuCtxSynchronize",ret);
  }

  bool CudaCtxSynchronize()
  {
    CUresult ret = cuCtxSynchronize();
    return ret == CUDA_SUCCESS;
  }

  void CudaMemset( void * dest , unsigned val , size_t N )
  {
    CUresult ret;
    ret = cuMemsetD32((CUdeviceptr)const_cast<void*>(dest), val, N);
    CudaRes("cuMemsetD32",ret);
  }




  JitFunction get_fptr_from_ptx( const char* fname , const std::string& kernel )
  {
    JitFunction func;
    CUresult ret;
    CUmodule cuModule;

    CUresult ret_sync = cuCtxSynchronize();
    if (ret_sync != CUDA_SUCCESS) {
      QDPIO::cerr << "Error on sync before loading image.\n";

      CudaCheckResult(ret_sync);

      const char* pStr_name;
      cuGetErrorName ( ret, &pStr_name );

      QDPIO::cerr << "Error: " << pStr_name << "\n";
	
      const char* pStr_string;
      cuGetErrorString ( ret, &pStr_string );

      QDPIO::cerr << "String: " << pStr_string << "\n";

      QDP_error_exit("Sync failed right before loading the PTX module.");
    }

    ret = cuModuleLoadData(&cuModule, (const void *)kernel.c_str());
    //ret = cuModuleLoadDataEx( &cuModule , ptx_kernel.c_str() , 0 , 0 , 0 );

    if (ret) {
      if (Layout::primaryNode()) {

	QDPIO::cerr << "Error loading external data.\n";
	
	const char* pStr_name;
	cuGetErrorName ( ret, &pStr_name );

	QDPIO::cerr << "Error: " << pStr_name << "\n";
	
	const char* pStr_string;
	cuGetErrorString ( ret, &pStr_string );

	QDPIO::cerr << "String: " << pStr_string << "\n";

	QDPIO::cerr << "Dumping kernel to " << fname << "\n";
#if 1
	std::ofstream out(fname);
	out << kernel;
	out.close();
	//llvm_module_dump();
#endif
	QDP_error_exit("Abort.");
      }
    }

    ret = cuModuleGetFunction( (CUfunction*)&func.getFunction(), cuModule, "main");
    if (ret)
      QDP_error_exit("Error returned from cuModuleGetFunction. Abort.");

    mapCUFuncPTX[func.getFunction()] = kernel;

    return func;
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




  
  namespace
  {
    size_t roundDown2pow(size_t x) {
      size_t s=1;
      while (s<=x) s <<= 1;
      s >>= 1;
      return s;
    }

        bool boolNoReadSM;
    unsigned device;
    std::string envvar;
    bool GPUDirect;
    bool syncDevice;
    bool asyncTransfers;
    bool unifiedAddressing;
    bool divRnd;
    unsigned maxKernelArg;

    unsigned smem;
    unsigned smem_default;
    int      sm_count;
    
    unsigned max_gridx;
    unsigned max_gridy;
    unsigned max_gridz;

    unsigned max_blockx;
    unsigned max_blocky;
    unsigned max_blockz;

    unsigned major;
    unsigned minor;

    int defaultGPU = -1;

  }


  void gpu_autoDetect() {
    unifiedAddressing = CudaGetConfig(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING) == 1;
    asyncTransfers = CudaGetConfig(CU_DEVICE_ATTRIBUTE_GPU_OVERLAP) == 1;
    sm_count = CudaGetConfig( CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT );
    smem = CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK );
    smem_default = 0;
    max_gridx = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X ) );
    max_gridy = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y ) );
    max_gridz = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z ) );
    max_blockx = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X ) );
    max_blocky = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y ) );
    max_blockz = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z ) );

#ifdef QDP_CUDA_SPECIAL
    QDPIO::cout << "Setting max gridx for CUDA special functions\n";
    cuda_special_set_maxgridx( max_gridx );
#endif
    
    int ma,mi;
    if (!boolNoReadSM)
      CudaGetSM(&ma,&mi);
    major = ma;
    minor = mi;
    divRnd = major >= 2;

    QDP_info_primary("Compute capability (major)              = %d",major);
    QDP_info_primary("Compute capability (minor)              = %d",minor);
    QDP_info_primary("Divide with IEEE 754 compliant rounding = %d",divRnd);
    QDP_info_primary("Sqrt with IEEE 754 compliant rounding   = %d",divRnd);
    QDP_info_primary("unified addr                            = %d",unifiedAddressing ? 1 : 0);
    QDP_info_primary("asyncTransfers                          = %d",asyncTransfers ? 1 : 0);
    QDP_info_primary("smem                                    = %d",smem);
    QDP_info_primary("max_gridx                               = %d",max_gridx);
    QDP_info_primary("max_gridy                               = %d",max_gridy);
    QDP_info_primary("max_gridz                               = %d",max_gridz);
    QDP_info_primary("max_blockx                              = %d",max_blockx);
    QDP_info_primary("max_blocky                              = %d",max_blocky);
    QDP_info_primary("max_blockz                              = %d",max_blockz);
    QDP_info_primary("SM count                                = %d",sm_count);
  }

  void gpu_setSM(int sm) {
    QDP_info_primary("Compiling LLVM IR to PTX for compute capability sm_%d (instead of autodetect)",sm);
    major = sm / 10;
    minor = sm % 10;
    boolNoReadSM = true;
  }

  void gpu_setDefaultGPU(int ngpu) {
    defaultGPU = ngpu;
  }


  int  gpu_getDefaultGPU() { return defaultGPU; }
  
  size_t gpu_getMaxGridX()  {return max_gridx;}
  size_t gpu_getMaxGridY()  {return max_gridy;}
  size_t gpu_getMaxGridZ()  {return max_gridz;}

  size_t gpu_getMaxBlockX()  {return max_blockx;}
  size_t gpu_getMaxBlockY()  {return max_blocky;}
  size_t gpu_getMaxBlockZ()  {return max_blockz;}
  
  size_t gpu_getMaxSMem()  {return smem;}
  size_t gpu_getDefaultSMem()  {return smem_default;}

  bool gpu_getSyncDevice() { return syncDevice; }
  bool gpu_getGPUDirect() { return GPUDirect; }
  void gpu_setSyncDevice(bool sync) { 
    QDP_info_primary("Setting device sync = %d",(int)sync);
    syncDevice = sync;
  };
  void gpu_setGPUDirect(bool direct) {
    QDP_info_primary("Setting GPU Direct = %d",(int)direct);
    GPUDirect = direct;
  };
  
  unsigned gpu_getMajor() { return major; }
  unsigned gpu_getMinor() { return minor; }
  



  
}


