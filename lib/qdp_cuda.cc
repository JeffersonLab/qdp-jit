// -*- c++ -*-


#include <iostream>

#include "qdp_config_internal.h" 

#include "qdp_cuda.h"
#include "qdp_init.h"
#include "cuda.h"

using namespace std;



namespace QDP {

  void CudaRes(const std::string& s,CUresult ret) {
    if (ret != CUDA_SUCCESS) {
      std::cout << "cuda error: " << s << "\n";
      exit(1);
    }
  }


  void CudaInit() {
    std::cout << "cuda init\n";
    cuInit(0);

    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) { 
      std::cout << "There is no device supporting CUDA.\n"; 
      exit(1); 
    }
  }





  CUstream * QDPcudastreams;
  CUevent * QDPevCopied;

  CUdevice cuDevice;
  CUcontext cuContext;


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
    cuStreamSynchronize(QDPcudastreams[KERNEL]);
  }

  void CudaSyncTransferStream() {
    cuStreamSynchronize(QDPcudastreams[TRANSFER]);
  }

  void CudaRecordAndWaitEvent() {
    cuEventRecord( *QDPevCopied , QDPcudastreams[TRANSFER] );
    cuStreamWaitEvent( QDPcudastreams[KERNEL] , *QDPevCopied , 0);
  }

  void CudaSetDevice(int dev)
  {
    CUresult ret;
    std::cout << "trying to get device " << dev << "\n";
    ret = cuDeviceGet(&cuDevice, dev);
    CudaRes("",ret);
    std::cout << "trying to create a context \n";
    ret = cuCtxCreate(&cuContext, 0, cuDevice);
    CudaRes("",ret);
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


  void CudaHostFree(const void *mem)
  {
    CUresult ret;
    ret = cuMemFreeHost((void *)mem);
    CudaRes("cuMemFreeHost",ret);
  }




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

    ret = cuMemcpyAsync((CUdeviceptr)const_cast<void*>(dest),
			(CUdeviceptr)const_cast<void*>(src),
			size,QDPcudastreams[TRANSFER]);

    CudaRes("cuMemcpyAsync",ret);
  }


  bool CudaMalloc(void **mem , size_t size )
  {
    CUresult ret;
    ret = cuMemAlloc( (CUdeviceptr*)mem,size);
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep( "CudaMalloc %p", *mem );
#endif
    CudaRes("cuMemAlloc",ret);
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
    cuCtxSynchronize();
  }

}


