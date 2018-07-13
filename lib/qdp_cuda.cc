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



namespace QDP {

  CUstream * QDPcudastreams;
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
    //QDPIO::cout << "kernel launch (manual)..\n";
#if 0
    QDP_get_global_cache().releasePrevLockSet();
    QDP_get_global_cache().beginNewLockSet();
    return;
#endif

#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("CudaLaunchKernel ... ");
#endif

    //std::cout << "shmem = " << sharedMemBytes << "\n";

    // std::cout << "CudaLaunchKernel:"
    // 	      << "  grid = " << gridDimX   << " " << gridDimY  << " " << gridDimZ   
    // 	      << "  block = " << blockDimX  << " " <<  blockDimY  << " " << blockDimZ   << "  shmem = " << sharedMemBytes << "\n";

    // CudaSyncTransferStream();
    // CudaSyncKernelStream();

    // This call is async
#if 1
    if ( blockDimX * blockDimY * blockDimZ > 0  &&  gridDimX * gridDimY * gridDimZ > 0 ) {
      CUresult result = cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, 
				       blockDimX, blockDimY, blockDimZ, 
				       sharedMemBytes, QDPcudastreams[KERNEL], kernelParams, extra);
      if (result != CUDA_SUCCESS) {
	QDP_error_exit("CUDA launch error (CudaLaunchKernel): grid=(%u,%u,%u), block=(%u,%u,%u), shmem=%u",
		       gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes );
      }
    } else {
      //std::cout << "skipping kernel launch due to zero block!!!\n";
    }
#endif

    QDP_get_global_cache().releasePrevLockSet();
    QDP_get_global_cache().beginNewLockSet();

#ifdef GPU_DEBUG_DEEP
    QDP_get_global_cache().printLockSets();
#endif

    // For now, pull the brakes
    // I've seen the GPU running away from CPU thread
    // This call is probably too much, but it's safe to call it.
#if 1
    CUresult result = cuCtxSynchronize();
    if (result != CUDA_SUCCESS) {

      if (mapCuErrorString.count(result)) 
	std::cout << " Error: " << mapCuErrorString.at(result) << "\n";
      else
	std::cout << " Error: (not known)\n";
      
      QDP_error_exit("CUDA launch error (CudaLaunchKernel, on sync): grid=(%u,%u,%u), block=(%u,%u,%u), shmem=%u",
		     gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes );
    }
#endif
    //CudaDeviceSynchronize();

    if (DeviceParams::Instance().getSyncDevice()) {  
      QDP_info_primary("Pulling the brakes: device sync after kernel launch!");
      //CudaDeviceSynchronize();
    }
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








  void * CudaGetKernelStream() {
    return (void*)&QDPcudastreams[KERNEL];
  }

  void CudaCreateStreams() {
    QDPcudastreams = new CUstream[2];
    for (int i=0; i<2; i++) {
      QDP_info_primary("JIT: Creating CUDA stream %d",i);
      cuStreamCreate(&QDPcudastreams[i],0);
    }
    QDP_info_primary("JIT: Creating CUDA event for transfers");
    QDPevCopied = new CUevent;
    cuEventCreate(QDPevCopied,CU_EVENT_BLOCKING_SYNC);
  }

  void CudaSyncKernelStream() {
    CUresult ret = cuStreamSynchronize(QDPcudastreams[KERNEL]);
    CudaRes("cuStreamSynchronize",ret);    
  }

  void CudaSyncTransferStream() {
    CUresult ret = cuStreamSynchronize(QDPcudastreams[TRANSFER]);
    CudaRes("cuStreamSynchronize",ret);    
  }

  void CudaRecordAndWaitEvent() {
    cuEventRecord( *QDPevCopied , QDPcudastreams[TRANSFER] );
    cuStreamWaitEvent( QDPcudastreams[KERNEL] , *QDPevCopied , 0);
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



  void CudaGetDeviceProps()
  {
    CUresult ret;

    DeviceParams::Instance().autoDetect();

    size_t free, total;
    ret = cuMemGetInfo(&free, &total);
    CudaRes("cuMemGetInfo",ret);

    QDP_info_primary("GPU memory: free = %lld,  total = %lld",(unsigned long long)free ,(unsigned long long)total);
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
      QDP_info_primary("Using device pool size: %d MiB",(int)val_min_int);

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



#if 0
  void CudaMemcpy( const void * dest , const void * src , size_t size)
  {
    CUresult ret;
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("cudaMemcpy dest=%p src=%p size=%d" ,  dest , src , size );
#endif

    ret = cuMemcpy((CUdeviceptr)const_cast<void*>(dest),
		   (CUdeviceptr)const_cast<void*>(src),
		   size);

    CudaRes("cuMemcpy",ret);
  }

  void CudaMemcpyAsync( const void * dest , const void * src , size_t size )
  {
    CUresult ret;
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("cudaMemcpy dest=%p src=%p size=%d" ,  dest , src , size );
#endif

    if (DeviceParams::Instance().getAsyncTransfers()) {
      ret = cuMemcpyAsync((CUdeviceptr)const_cast<void*>(dest),
			  (CUdeviceptr)const_cast<void*>(src),
			  size,QDPcudastreams[TRANSFER]);
    } else {
      std::cout << "using sync copy\n";
      ret = cuMemcpy((CUdeviceptr)const_cast<void*>(dest),
		     (CUdeviceptr)const_cast<void*>(src),
		     size);
    }

    CudaRes("cuMemcpyAsync",ret);
  }
#endif

  
  void CudaMemcpyH2DAsync( void * dest , const void * src , size_t size )
  {
    CUresult ret;
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("CudaMemcpyH2DAsync dest=%p src=%p size=%d" ,  dest , src , size );
#endif

    if (DeviceParams::Instance().getAsyncTransfers()) {
      ret = cuMemcpyHtoDAsync((CUdeviceptr)const_cast<void*>(dest),
			      src,
			      size,QDPcudastreams[TRANSFER]);
    } else {
      std::cout << "using sync H2D copy\n";
      ret = cuMemcpyHtoD((CUdeviceptr)const_cast<void*>(dest),
			 src,
			 size);
    }

    CudaRes("cuMemcpyH2DAsync",ret);
  }

  void CudaMemcpyD2HAsync( void * dest , const void * src , size_t size )
  {
    CUresult ret;
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("CudaMemcpyD2HAsync dest=%p src=%p size=%d" ,  dest , src , size );
#endif

    if (DeviceParams::Instance().getAsyncTransfers()) {
      ret = cuMemcpyDtoHAsync( dest,
			      (CUdeviceptr)const_cast<void*>(src),
			      size,QDPcudastreams[TRANSFER]);
    } else {
      std::cout << "using sync D2H copy\n";
      ret = cuMemcpyDtoH( const_cast<void*>(dest),
			 (CUdeviceptr)const_cast<void*>(src),
			 size);
    }

    CudaRes("cuMemcpyH2DAsync",ret);
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
    CudaRes("cuMemAlloc",ret);
#else 
    CudaRes("cuMemAllocManaged", ret);
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

}


