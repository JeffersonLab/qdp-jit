// -*- c++ -*-


#include <iostream>

#include "qdp_config_internal.h" 

#include "qdp.h"
// #include "qdp_init.h"
// #include "qdp_deviceparams.h"
// #include "qdp_cuda.h"
// #include "cuda.h"

#include <string>

#include "cudaProfiler.h"

using namespace std;


namespace {
  int max_local_size = 0;
  int max_local_usage = 0;
  size_t total_free = 0;
}


namespace QDP {

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



  void CudaLaunchKernel( CUfunction f, 
			 unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
			 unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
			 unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra )
  {
    //if ( blockDimX * blockDimY * blockDimZ > 0  &&  gridDimX * gridDimY * gridDimZ > 0 ) {
    
    CUresult result = CudaLaunchKernelNoSync(f, gridDimX, gridDimY, gridDimZ, 
					     blockDimX, blockDimY, blockDimZ, 
					     sharedMemBytes, 0, kernelParams, extra);
    if (result != CUDA_SUCCESS) {
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

    if (DeviceParams::Instance().getSyncDevice()) {  
      //CudaDeviceSynchronize();
    }
  }

  
  int CudaGetAttributesLocalSize( CUfunction f )
  {
    int local_mem = 0;
    cuFuncGetAttribute( &local_mem , CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES , f );
    return local_mem;
  }


  namespace {
    std::vector<unsigned> __kernel_geom;
    CUfunction            __kernel_ptr;
  }

  std::vector<unsigned> get_backed_kernel_geom() { return __kernel_geom; }
  CUfunction            get_backed_kernel_ptr() { return __kernel_ptr; }


  CUresult CudaLaunchKernelNoSync( CUfunction f, 
				   unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
				   unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
				   unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra  )
  {
    // QDP_info("CudaLaunchKernelNoSync: grid=(%u,%u,%u), block=(%u,%u,%u), shmem=%u",
    // 	     gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes );
    // QDPIO::cout << "CUfunction = " << (size_t)(void*)f << "\n";
      


    //QDPIO::cout << "local mem (bytes) = " << num_threads << "\n";
    //
    
    auto res = cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, 
			      blockDimX, blockDimY, blockDimZ, 
			      sharedMemBytes, 0, kernelParams, extra);

    if (res == CUDA_SUCCESS)
      {
	//QDPIO::cout << "CUDA_SUCCESS\n";
        if (qdp_cache_get_pool_bisect())
	  {
	    int num_threads = blockDimX * blockDimY * blockDimZ * gridDimX * gridDimY * gridDimZ;
	    int local = CudaGetAttributesLocalSize( f );

	    // Total local memory for this kernel launch
	    int local_use = local * DeviceParams::Instance().getSMcount() * blockDimZ * blockDimY * blockDimX;
	    
	    if (local_use > max_local_usage)
	      {
		QDP_get_global_cache().backup_last_kernel_args();
		__kernel_geom.clear();
		__kernel_geom.push_back(gridDimX);
		__kernel_geom.push_back(gridDimY);
		__kernel_geom.push_back(gridDimZ);
		__kernel_geom.push_back(blockDimX);
		__kernel_geom.push_back(blockDimY);
		__kernel_geom.push_back(blockDimZ);
		__kernel_geom.push_back(sharedMemBytes);
		__kernel_ptr = f;
	      }
      
	    max_local_size = local > max_local_size ? local : max_local_size;
	    max_local_usage = local_use > max_local_usage ? local_use : max_local_usage;
	  }
      }
    else
      {
	//QDPIO::cout << "no CUDA_SUCCESS " << mapCuErrorString[res] << "\n";
      }

    return res;
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



  int CudaAttributeNumRegs( CUfunction f ) {
    int pi;
    CUresult res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_NUM_REGS , f );
    CudaRes("CudaAttributeNumRegs",res);
    return pi;
  }

  int CudaAttributeLocalSize( CUfunction f ) {
    int pi;
    CUresult res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES , f );
    CudaRes("CudaAttributeLocalSize",res);
    return pi;
  }

  int CudaAttributeConstSize( CUfunction f ) {
    int pi;
    CUresult res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES , f );
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

    DeviceParams::Instance().autoDetect();

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
      QDP_get_global_cache().get_allocator().setPoolSize( ((size_t)val_min_int) * 1024 * 1024 );

      setPoolSize = true;
    } else {
      //QDP_info_primary("Using device pool size: %d MiB",(int)(CUDADevicePoolAllocator::Instance().getPoolSize()/1024/1024));
      QDP_info_primary("Using device pool size: %d MiB",(int)(QDP_get_global_cache().get_allocator().getPoolSize()/1024/1024));
    }

    // int major = DeviceParams::Instance().getMajor();
    // int minor = DeviceParams::Instance().getMinor();
    // PTX::ptx_type_matrix = PTX::create_ptx_type_matrix();

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

  
}


