#ifndef QDP_GPU_H
#define QDP_GPU_H


#include <map>


namespace QDP {

  enum class JitResult { JitSuccess , JitError , JitResource };

#ifdef QDP_DEEP_LOG
  void gpu_deep_logger( void* host_ptr , std::string type_W , int size_T , int start , int count , std::string pretty );
  void gpu_deep_logger_close();
#endif

  void gpu_create_events();
  void gpu_record_start();
  void gpu_record_stop();
  void gpu_event_sync();
  float gpu_get_time();

  
  void gpu_init();
  
  std::string gpu_get_arch();

  void gpu_auto_detect();

  size_t gpu_mem_free();
  size_t gpu_mem_total();

  int gpu_get_device_count();

  void gpu_set_device(int dev);

  void gpu_host_alloc(void **mem , const size_t size);
  void gpu_host_free(void *mem);

  void gpu_memcpy_h2d( void * dest , const void * src , size_t size );
  void gpu_memcpy_d2h( void * dest , const void * src , size_t size );

  bool gpu_malloc( void **mem , const size_t size );
  void gpu_free( const void *mem );
  void gpu_prefetch(void *mem,   size_t  size);
  void gpu_memset( void * dest , unsigned val , size_t N );

  JitResult gpu_launch_kernel( JitFunction& f, 
			       unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
			       unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
			       unsigned int  sharedMemBytes, QDPCache::KernelArgs_t kernelArgs );

#if 0
  int CudaAttributeNumRegs( JitFunction& f );
  int CudaAttributeLocalSize( JitFunction& f );
  int CudaAttributeConstSize( JitFunction& f );
  void CudaProfilerInitialize();
  void CudaProfilerStart();
  void CudaProfilerStop();
#endif
  
  

  //JitFunction get_fptr_from_ptx( const char* fname , const std::string& kernel );
  bool get_jitf( JitFunction& func, const std::string& kernel , const std::string& func_name , const std::string& pretty , const std::string& compute );

  std::string getPTXfromCUFunc(JitFunction& f);


  void gpu_sync();


  struct kernel_geom_t {
    int threads_per_block;
    int Nblock_x;
    int Nblock_y;
  };

  kernel_geom_t getGeom(int numSites , int threadsPerBlock);

  size_t gpu_getMaxGridX();
  size_t gpu_getMaxGridY();
  size_t gpu_getMaxGridZ();

  size_t gpu_getMaxBlockX();
  size_t gpu_getMaxBlockY();
  size_t gpu_getMaxBlockZ();
  
  size_t gpu_getMaxSMem();

  unsigned gpu_getMajor();
  unsigned gpu_getMinor();
  


  
}

#endif
