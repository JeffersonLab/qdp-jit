#ifndef QDP_CUDA_H
#define QDP_CUDA_H


#include <map>


namespace QDP {

  enum class JitResult { JitSuccess , JitError , JitResource };

  std::vector<unsigned> get_backed_kernel_geom();
  JitFunction            get_backed_kernel_ptr();

  //void CudaCheckResult(CUresult ret);

  void CudaInit();
  //int CudaGetConfig(CUdevice_attribute what);
  int CudaGetConfig(int what);
  void CudaGetSM(int* maj,int* min);

  void CudaLaunchKernel( JitFunction f, 
			 unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
			 unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
			 unsigned int  sharedMemBytes, int hStream, void** kernelParams, void** extra );

  JitResult CudaLaunchKernelNoSync( JitFunction f, 
				    unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
				    unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
				    unsigned int  sharedMemBytes, int hStream, void** kernelParams, void** extra );

  int CudaGetMaxLocalSize();
  int CudaGetMaxLocalUsage();
  size_t CudaGetInitialFreeMemory();
  
  int CudaAttributeNumRegs( JitFunction f );
  int CudaAttributeLocalSize( JitFunction f );
  int CudaAttributeConstSize( JitFunction f );

  bool CudaHostRegister(void * ptr , size_t size);
  void CudaHostUnregister(void * ptr );
  void CudaMemGetInfo(size_t *free,size_t *total);

  bool CudaHostAlloc(void **mem , const size_t size, const int flags);
  void CudaHostAllocWrite(void **mem , size_t size);
  void CudaHostFree(void *mem);

  void CudaSetDevice(int dev);
  void CudaGetDeviceCount(int * count);
  void CudaGetDeviceProps();

  void CudaMemcpyH2D( void * dest , const void * src , size_t size );
  void CudaMemcpyD2H( void * dest , const void * src , size_t size );

  bool CudaMalloc( void **mem , const size_t size );
  void CudaFree( const void *mem );

  void CudaThreadSynchronize();
  void CudaDeviceSynchronize();

  bool CudaCtxSynchronize();

  void CudaProfilerInitialize();
  void CudaProfilerStart();
  void CudaProfilerStop();

  void CudaMemset( void * dest , unsigned val , size_t N );

  JitFunction get_fptr_from_ptx( const char* fname , const std::string& kernel );
  std::string getPTXfromCUFunc(JitFunction f);

}

#endif
