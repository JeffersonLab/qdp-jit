#ifndef QDP_HIP_H
#define QDP_HIP_H


#include <map>


#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>


namespace QDP {

  extern std::map<hipError_t,std::string> mapCuErrorString;

  /* std::vector<unsigned> get_backed_kernel_geom(); */
  /* CUfunction            get_backed_kernel_ptr(); */

  void HipCheckResult(hipError_t ret);

  void HipInit();

  int HipGetConfig(hipDeviceAttribute_t what);

  void HipLaunchKernel( CUfunction f, 
			 unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
			 unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
			 unsigned int  sharedMemBytes, void** kernelParams, void** extra );

  hipError_t HipLaunchKernelNoSync( CUfunction f, 
				   unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
				   unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
				   unsigned int  sharedMemBytes, void** kernelParams, void** extra );

  int HipGetMaxLocalSize();
  int HipGetMaxLocalUsage();
  size_t HipGetInitialFreeMemory();
  
  int HipAttributeNumRegs( CUfunction f );
  int HipAttributeLocalSize( CUfunction f );
  int HipAttributeConstSize( CUfunction f );

  bool HipHostRegister(void * ptr , size_t size);
  void HipHostUnregister(void * ptr );
  void HipMemGetInfo(size_t *free,size_t *total);

  bool HipHostAlloc(void **mem , const size_t size, const int flags);
  void HipHostAllocWrite(void **mem , size_t size);
  void HipHostFree(void *mem);

  void HipSetDevice(int dev);
  void HipGetDeviceCount(int * count);
  void HipGetDeviceProps();

  void HipMemcpyH2D( void * dest , const void * src , size_t size );
  void HipMemcpyD2H( void * dest , const void * src , size_t size );

  bool HipMalloc( void **mem , const size_t size );
  void HipFree( const void *mem );

  void HipDeviceSynchronize();

  bool HipCtxSynchronize();

  void HipProfilerInitialize();
  void HipProfilerStart();
  void HipProfilerStop();

  void HipMemset( void * dest , unsigned val , size_t N );
}

#endif
