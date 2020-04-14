// -*- c++ -*-



#include "qdp_config_internal.h" 
#include "qdp.h"

#include <iostream>
#include <string>

//#include "cudaProfiler.h"

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

using namespace std;


namespace {
  int max_local_size = 0;
  int max_local_usage = 0;
  size_t total_free = 0;
}


namespace QDP {

  int HipGetMaxLocalSize() { return max_local_size; }
  int HipGetMaxLocalUsage() { return max_local_usage; }
  size_t HipGetInitialFreeMemory() { return total_free; }

  // CUdevice cuDevice;
  // CUcontext ;

  //hipDevice_t hipDevice;
  //hipCtx_t hipContext;


  std::map<hipError_t,std::string> mapHipErrorString= {
    {hipSuccess ,"Successful completion."},
    {hipErrorInvalidContext ,"Produced when input context is invalid."},
    {hipErrorInvalidKernelFile , "In CUDA DRV, it is CUDA_ERROR_INVALID_PTX."},
    {hipErrorMemoryAllocation ,"Memory allocation error."},
    {hipErrorInitializationError ,"TODO comment from hipErrorInitializationError."},
    {hipErrorLaunchFailure ,"An exception occurred on the device while executing a kernel."},
    {hipErrorLaunchOutOfResources ,"Out of resources error."},
    {hipErrorInvalidDevice , "DeviceID must be in range 0...#compute-devices."},
    {hipErrorInvalidValue , "One or more of the parameters passed to the API call is NULL or not in an acceptable range."},
    {hipErrorInvalidDevicePointer ,"Invalid Device Pointer."},
    {hipErrorInvalidMemcpyDirection ,"Invalid memory copy direction."},
    {hipErrorUnknown ,"Unknown error."},
    {hipErrorInvalidResourceHandle ,"Resource handle (hipEvent_t or hipStream_t) invalid."},
    {hipErrorNotReady ,"Indicates that asynchronous operations enqueued earlier are not ready. This is not actually an error, but is used to distinguish from hipSuccess (which indicates completion). APIs that return this error include hipEventQuery and hipStreamQuery."},
    {hipErrorNoDevice ,"Call to hipGetDeviceCount returned 0 devices."},
    {hipErrorPeerAccessAlreadyEnabled ,"Peer access was already enabled from the current device."},
    {hipErrorPeerAccessNotEnabled ,"Peer access was never enabled from the current device."},
    {hipErrorRuntimeMemory ,"HSA runtime memory call returned error. Typically not seen in production systems."},
    {hipErrorRuntimeOther ,"HSA runtime call other than memory returned error. Typically not seen in production systems."},
    {hipErrorHostMemoryAlreadyRegistered ,"Produced when trying to lock a page-locked memory."},
    {hipErrorHostMemoryNotRegistered ,"Produced when trying to unlock a non-page-locked memory."},
    {hipErrorMapBufferObjectFailed ,"Produced when the IPC memory attach failed from ROCr."},
    {hipErrorTbd ,"Marker that more error codes are needed. "}
  };



  void HipLaunchKernel( CUfunction f, 
			 unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
			 unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
			 unsigned int  sharedMemBytes, void** kernelParams, void** extra )
  {
    QDP_error_exit("HipLaunchKernel, fixme\n");
#if 0
    //if ( blockDimX * blockDimY * blockDimZ > 0  &&  gridDimX * gridDimY * gridDimZ > 0 ) {
    
    hipError_t result = HipLaunchKernelNoSync(f, gridDimX, gridDimY, gridDimZ, 
					     blockDimX, blockDimY, blockDimZ, 
					     sharedMemBytes, 0, kernelParams, extra);
    if (result != hipSuccess) {
      QDP_error_exit("CUDA launch error (HipLaunchKernel): grid=(%u,%u,%u), block=(%u,%u,%u), shmem=%u",
		     gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes );

      hipError_t result = cuCtxSynchronize();
      if (result != hipSuccess) {

	if (mapHipErrorString.count(result)) 
	  std::cout << " Error: " << mapHipErrorString.at(result) << "\n";
	else
	  std::cout << " Error: (not known)\n";
      }      
      QDP_error_exit("CUDA launch error (HipLaunchKernel, on sync): grid=(%u,%u,%u), block=(%u,%u,%u), shmem=%u",
		     gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes );
    }

    if (DeviceParams::Instance().getSyncDevice()) {  
      //HipDeviceSynchronize();
    }
#endif
  }


  
  // namespace {
  //   std::vector<unsigned> __kernel_geom;
  //   //CUfunction            __kernel_ptr;
  // }

  // std::vector<unsigned> get_backed_kernel_geom() { return __kernel_geom; }
  // CUfunction            get_backed_kernel_ptr() { 
  //   QDP_error_exit("get_backed_kernel_ptr, fixme\n");
  //   return __kernel_ptr; 
  // }



  hipError_t HipLaunchKernelNoSync( 
				   unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
				   unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
				   unsigned int  sharedMemBytes, void** kernelParams, void** extra  )
  {
    QDP_error_exit("HipLaunchKernelNoSync, fixme\n");
#if 0
     // QDP_info("HipLaunchKernelNoSync: grid=(%u,%u,%u), block=(%u,%u,%u), shmem=%u",
     // 	      gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes );
    // QDPIO::cout << "CUfunction = " << (size_t)(void*)f << "\n";
      


    //QDPIO::cout << "local mem (bytes) = " << num_threads << "\n";
    //
    
    auto res = cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, 
			      blockDimX, blockDimY, blockDimZ, 
			      sharedMemBytes, 0, kernelParams, extra);

    if (res == hipSuccess)
      {
	//QDPIO::cout << "hipSuccess\n";
	if (qdp_cache_get_launch_verbose())
	  {
	    QDP_info("HipLaunchKernelNoSync: grid=(%u,%u,%u), block=(%u,%u,%u), shmem=%u",
		     gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes );
	  }
	
        if (qdp_cache_get_pool_bisect())
	  {
	    int local = HipGetAttributesLocalSize( f );

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
	//QDPIO::cout << "no hipSuccess " << mapHipErrorString[res] << "\n";
      }
#endif
    hipError_t res;
    return res;
  }



    

  void HipCheckResult(hipError_t result) {
    if (result != hipSuccess) {
      QDP_info("ROCM error %d (%s)", (int)result , mapHipErrorString[result].c_str());
    }
  }


  void HipRes(const std::string& s,hipError_t ret) {
    if (ret != hipSuccess) {
      if (mapHipErrorString.count(ret)) 
	std::cout << s << " Error: " << mapHipErrorString.at(ret) << "\n";
      else
	std::cout << s << " Error: (not known)\n";
      exit(1);
    }
  }


#if 0
  int HipAttributeNumRegs( CUfunction f ) {
    int pi;
    hipError_t res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_NUM_REGS , f );
    HipRes("HipAttributeNumRegs",res);
    return pi;
  }

  int HipAttributeLocalSize( CUfunction f ) {
    int pi;
    hipError_t res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES , f );
    HipRes("HipAttributeLocalSize",res);
    return pi;
  }

  int HipAttributeConstSize( CUfunction f ) {
    int pi;
    hipError_t res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES , f );
    HipRes("HipAttributeConstSize",res);
    return pi;
  }
#endif


#if 0
  void HipProfilerInitialize()
  {
    hipError_t res;
    std::cout << "CUDA Profiler Initializing ...\n";
    res = cuProfilerInitialize( "prof.cfg" , "prof.out" , CU_OUT_CSV );
    HipRes("cuProfilerInitialize",res);
  }

  void HipProfilerStart()
  {
    hipError_t res;
    res = cuProfilerStart();
    HipRes("cuProfilerStart",res);
  }

  void HipProfilerStop()
  {
    hipError_t res;
    res = cuProfilerStop();
    HipRes("cuProfilerStop",res);
  }
#endif



  //int HipGetConfig(CUdevice_attribute what)
  int HipGetConfig(hipDeviceAttribute_t what)
  {
    int data;
    hipError_t ret;

    QDPIO::cout << "HipGetConfig using device 0 (hard-coded!!)\n";

    ret = hipDeviceGetAttribute( &data, what , 0 );
    HipRes("cuDeviceGetAttribute",ret);
    return data;
  }


  void HipInit() {
    //QDP_info_primary("CUDA initialization");
    //cuInit(0);
    std::cout << "Not doing any device init\n";

    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);
    if (deviceCount == 0) { 
      std::cout << "There is no device supporting ROCM.\n";
      exit(1); 
    }
  }








  void HipSetDevice(int dev)
  {
    hipError_t ret;

    std::cout << "Skipping trying to create a context as it is marked deprecated\n";
    // QDP_info_primary("trying to create a context on device %d",dev);
    // ret = hipCtxCreate( &hipContext , 0 , dev);
    // HipRes("hipCtxCreate",ret);

    std::cout << "Skipping trying to get the device as it is marked deprecated\n";

    // QDP_info_primary("trying to get device with the current context");
    // ret = hipCtxGetDevice(&hipDevice);
    // HipRes("hipCtxGetDevice",ret);

    std::cout << "Skipping trying to get a context\n";
    // QDP_info_primary("trying to grab pre-existing context",dev);
    // ret = cuCtxGetCurrent(&cuContext);
    
    // QDPIO::cout << "trying to create a context. Using zero flags, was CU_CTX_MAP_HOST\n";
    // ret = hipCtxCreate( &hipContext, 0 , hipDevice);
    // HipRes("hipCtxCreate",ret);
  }



  void HipMemGetInfo(size_t *free,size_t *total)
  {
    hipError_t ret = hipMemGetInfo(free, total);
    HipRes("hipMemGetInfo",ret);
  }


  void HipGetDeviceProps()
  {
    hipError_t ret;

    DeviceParams::Instance().autoDetect();

    size_t free, total;
    ret = hipMemGetInfo(&free, &total);
    HipRes("hipMemGetInfo",ret);
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

    // int major = DeviceParams::Instance().getMajor();
    // int minor = DeviceParams::Instance().getMinor();
    // PTX::ptx_type_matrix = PTX::create_ptx_type_matrix();

    ret = hipCtxSetCacheConfig( hipFuncCachePreferL1 );
    HipRes("hipCtxSetCacheConfig",ret);
  }



  void HipGetDeviceCount(int * count)
  {
    hipGetDeviceCount( count );
  }


  bool HipHostRegister(void * ptr , size_t size)
  {
    hipError_t ret;
    int flags = 0;
    QDP_info_primary("ROCM host register ptr=%p (%u) size=%lu (%u)",ptr,(unsigned)((size_t)ptr%4096) ,(unsigned long)size,(unsigned)((size_t)size%4096));
    ret = hipHostRegister(ptr, size, flags);
    HipRes("hipHostRegister",ret);
    return true;
  }

  
  void HipHostUnregister(void * ptr )
  {
    hipError_t ret;
    ret = hipHostUnregister(ptr);
    HipRes("hipHostUnregister",ret);
  }
  

  bool HipHostAlloc(void **mem , const size_t size, const int flags)
  {
    hipError_t ret;
    ret = hipHostMalloc(mem,size,flags);
    HipRes("hipHostMalloc",ret);
    return ret == hipSuccess;
  }


  void HipHostFree(void *mem)
  {
    hipError_t ret;
    ret = hipHostFree(mem);
    HipRes("hipHostFree",ret);
  }





  void HipMemcpyH2D( void * dest , const void * src , size_t size )
  {
    hipError_t ret;
    ret = hipMemcpyHtoD((hipDeviceptr_t)const_cast<void*>(dest), const_cast<void*>(src) , size);
    HipRes("hipMemcpyH2D",ret);
  }

  void HipMemcpyD2H( void * dest , const void * src , size_t size )
  {
    hipError_t ret;
    ret = hipMemcpyDtoH( dest, (hipDeviceptr_t)const_cast<void*>(src), size);
    HipRes("hipMemcpyD2H",ret);
  }


  bool HipMalloc(void **mem , size_t size )
  {
    hipError_t ret;
    ret = hipMalloc( mem , size );
    return ret == hipSuccess;
  }



  
  void HipFree(const void *mem )
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep( "HipFree %p", mem );
#endif
    hipError_t ret;
    ret = hipFree( const_cast<void*>(mem));
    HipRes("hipFree",ret);
  }


  void HipDeviceSynchronize()
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep( "cudaDeviceSynchronize" );
#endif
    hipError_t ret = hipCtxSynchronize();
    HipRes("hipCtxSynchronize",ret);
  }

  bool HipCtxSynchronize()
  {
    hipError_t ret = hipCtxSynchronize();
    return ret == hipSuccess;
  }

  void HipMemset( void * dest , unsigned val , size_t N )
  {
    hipError_t ret;
    ret = hipMemset( dest, val, N);
    HipRes("hipMemset",ret);
  }

  
}


