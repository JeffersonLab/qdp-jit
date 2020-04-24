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


#if 0
  void HipLaunchKernel( JitFunction f, 
			 unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
			 unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
			 unsigned int  sharedMemBytes, void* kernelParams)
  {
    //if ( blockDimX * blockDimY * blockDimZ > 0  &&  gridDimX * gridDimY * gridDimZ > 0 ) {
    
    JitResult result = HipLaunchKernelNoSync(f,
					     gridDimX, gridDimY, gridDimZ, 
					     blockDimX, blockDimY, blockDimZ, 
					     sharedMemBytes, kernelParams);

    if (result != JitResult::JitSuccess) {
      QDP_error_exit("HIP launch error (HipLaunchKernel): grid=(%u,%u,%u), block=(%u,%u,%u), shmem=%u",
		     gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes );
    }

    // if (DeviceParams::Instance().getSyncDevice()) {  
    //   //HipDeviceSynchronize();
    // }
  }
#endif

  
  // namespace {
  //   std::vector<unsigned> __kernel_geom;
  //   //JitFunction            __kernel_ptr;
  // }

  // std::vector<unsigned> get_backed_kernel_geom() { return __kernel_geom; }
  // JitFunction            get_backed_kernel_ptr() { 
  //   QDP_error_exit("get_backed_kernel_ptr, fixme\n");
  //   return __kernel_ptr; 
  // }



  JitResult HipLaunchKernelNoSync( JitFunction f,
				   unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
				   unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
				   unsigned int  sharedMemBytes, std::vector<unsigned char>& vec_ptr  )
  {
    QDP_info("HipLaunchKernelNoSync: grid=(%u,%u,%u), block=(%u,%u,%u), shmem=%u",
	     gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes );
    // QDPIO::cout << "JitFunction = " << (size_t)(void*)f << "\n";
      


    //QDPIO::cout << "local mem (bytes) = " << num_threads << "\n";
    //
    
    // auto res = cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, 
    // 			      blockDimX, blockDimY, blockDimZ, 
    // 			      sharedMemBytes, 0, kernelParams, extra);


    //= (std::vector<unsigned char>*)kernelParams;
#if 0
    auto size = vec_ptr.size();
    std::cout << "HipLaunchKernelNoSync: kernel params size: " << size << "\n";
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, vec_ptr.data(),
		      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
		      HIP_LAUNCH_PARAM_END};
#else
    size_t N = 16;

    bool p_ordered = true;
    int  p_th_count = N;
    int  p_start = 0;
    int  p_end = 15;
    bool p_do_site_perm = false;
    int* p_site_table   = NULL;
    bool* p_member_array = NULL;

    hipDeviceptr_t dev_ptr_dest;
    hipDeviceptr_t dev_ptr_a;
    hipDeviceptr_t dev_ptr_b;

    float* ptr_dest = new float[N];
    float* ptr_a = new float[N];
    float* ptr_b = new float[N];

    for (int i = 0 ; i < N ; ++i )
      {
	ptr_dest[i] = 0.0;
	ptr_a[i] = 2.0;
	ptr_b[i] = 3.0;
      }

    hipMalloc(&dev_ptr_dest, sizeof(float)*N);
    hipMalloc(&dev_ptr_a, sizeof(float)*N);
    hipMalloc(&dev_ptr_b, sizeof(float)*N);

    hipMemcpyHtoD(dev_ptr_dest, ptr_dest, sizeof(float)*N );
    hipMemcpyHtoD(dev_ptr_a, ptr_a, sizeof(float)*N );
    hipMemcpyHtoD(dev_ptr_b, ptr_b, sizeof(float)*N );


    struct { 
      bool  p_ordered;
      int   p_th_count;
      int   p_start;
      int   p_end;
      bool  p_do_site_perm;
      int*  p_site_table;
      bool* p_member_array;
      hipDeviceptr_t ptr_dest; 
      hipDeviceptr_t ptr_a; 
      hipDeviceptr_t ptr_b; 
    } args{     
      p_ordered,
	p_th_count,
	p_start,
	p_end,
	p_do_site_perm,
	p_site_table,
	p_member_array,
	dev_ptr_dest,
	dev_ptr_a,
	dev_ptr_b
	};


    std::cout << "size of args: " << sizeof(args) << "\n";
    std::cout << "size of bool: " << sizeof(bool) << "\n";
    std::cout << "size of int: " << sizeof(int) << "\n";
    std::cout << "size of int*: " << sizeof(int*) << "\n";
    std::cout << "size of bool*: " << sizeof(bool*) << "\n";
    std::cout << "size of hipDeviceptr_t: " << sizeof(hipDeviceptr_t) << "\n";

    std::cout << "was: " << "\n";
    //args.ptr_dest = (hipDeviceptr_t)(*(void**)&(*vec_ptr)[30]);
    //std::cout << "now: " << args.ptr_dest << "\n";

    std::cout << "p_ordered: " << args.p_ordered << "\n";
    std::cout << "p_th_count: " << args.p_th_count << "\n";
    std::cout << "p_start: " << args.p_start << "\n";
    std::cout << "p_end: " << args.p_end << "\n";
    std::cout << "p_do_site_perm: " << args.p_do_site_perm << "\n";
    std::cout << "p_site_table: " << args.p_site_table << "\n";
    std::cout << "p_member_array: " << args.p_member_array << "\n";
    std::cout << "ptr_dest: " << args.ptr_dest << "\n";
    std::cout << "ptr_a: " << args.ptr_a << "\n";
    std::cout << "ptr_b: " << args.ptr_b << "\n";

    memcpy( &args.p_ordered , &vec_ptr[0] , vec_ptr.size() );

    // for (int i = 0 ; i < 54 ; ++i)
    //   ((unsigned int*)&args.p_ordered)[i] = vec_ptr[i];


    // args.p_ordered=*(bool*)&vec_ptr.data()[0];
    // args.p_th_count=*(int*)&vec_ptr.data()[1];
    // args.p_start=*(int*)&vec_ptr.data()[5];
    // args.p_end=*(int*)&vec_ptr.data()[9];
    // args.p_do_site_perm=*(bool*)&vec_ptr.data()[13];
    // args.p_site_table=*(int**)&vec_ptr.data()[14];
    // args.p_member_array=*(bool**)&vec_ptr.data()[22];
    // args.ptr_dest=*(void**)&vec_ptr.data()[30];
    // args.ptr_a=*(void**)&vec_ptr.data()[38];
    // args.ptr_b=*(void**)&vec_ptr.data()[46];



    std::cout << "now: " << "\n";

    std::cout << "p_ordered: " << args.p_ordered << "\n";
    std::cout << "p_th_count: " << args.p_th_count << "\n";
    std::cout << "p_start: " << args.p_start << "\n";
    std::cout << "p_end: " << args.p_end << "\n";
    std::cout << "p_do_site_perm: " << args.p_do_site_perm << "\n";
    std::cout << "p_site_table: " << args.p_site_table << "\n";
    std::cout << "p_member_array: " << args.p_member_array << "\n";
    std::cout << "ptr_dest: " << args.ptr_dest << "\n";
    std::cout << "ptr_a: " << args.ptr_a << "\n";
    std::cout << "ptr_b: " << args.ptr_b << "\n";

    auto size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
		      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
		      HIP_LAUNCH_PARAM_END};

    // QDPIO::cout << "Launching kernel from copied JitFunction..\n";

    // hipModuleLaunchKernel((hipFunction_t)tmp.getFunction(), 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr,
    //  			  config);
#endif


    // std::cout << "kernel args before launch:\n";
    // std::cout << "p_ordered: " << (*(bool*)&vec_ptr.data()[0]) << "\n";
    // std::cout << "p_th_count: " << (*(int*)&vec_ptr.data()[1]) << "\n";
    // std::cout << "p_start: " << (*(int*)&vec_ptr.data()[5]) << "\n";
    // std::cout << "p_end: " << (*(int*)&vec_ptr.data()[9]) << "\n";
    // std::cout << "p_do_site_perm: " << (*(bool*)&vec_ptr.data()[13]) << "\n";
    // std::cout << "p_site_table: " << (*(void**)&vec_ptr.data()[14]) << "\n";
    // std::cout << "p_member_array: " << (*(void**)&vec_ptr.data()[22]) << "\n";
    // std::cout << "ptr_dest: " << (*(void**)&vec_ptr.data()[30]) << "\n";
    // std::cout << "ptr_a: " << (*(void**)&vec_ptr.data()[38]) << "\n";
    // std::cout << "ptr_b: " << (*(void**)&vec_ptr.data()[46]) << "\n";



    hipError_t res = hipModuleLaunchKernel((hipFunction_t)f.getFunction(),  
					   gridDimX, gridDimY, gridDimZ, 
					   blockDimX, blockDimY, blockDimZ, 
					   sharedMemBytes, nullptr, nullptr, config);

#if 0
    QDPIO::cout << "..done!\n";

    hipMemcpyDtoH(ptr_dest, dev_ptr_dest, sizeof(float)*N );

    for (int i = 0 ; i < N ; ++i )
      QDPIO::cout << ptr_dest[i] << " ";
    QDPIO::cout << "\n";

    delete[] ptr_dest;
    delete[] ptr_a;
    delete[] ptr_b;
#endif

    if (res == hipSuccess)
      {
	QDPIO::cout << "hipSuccess\n";
#if 0
	if (qdp_cache_get_launch_verbose())
	  {
	    QDP_info("HipLaunchKernelNoSync: grid=(%u,%u,%u), block=(%u,%u,%u), shmem=%u",
		     gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes );
	  }
#endif
#if 0	
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
#endif
      }
    else
      {
	QDPIO::cout << "no hipSuccess " << mapHipErrorString[res] << "\n";
      }

    //HipDeviceSynchronize();



    JitResult ret;

    switch (res) {
    case hipSuccess:
      std::cout << "returning: JitResult::JitSuccess\n";
      ret = JitResult::JitSuccess;
      break;
    case hipErrorLaunchOutOfResources:
      std::cout << "returning: JitResult::JitResource\n";
      ret = JitResult::JitResource;
      break;
    default:
      std::cout << "returning: JitResult::JitError\n";
      ret = JitResult::JitError;
    }

    return ret;
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
  int HipAttributeNumRegs( JitFunction f ) {
    int pi;
    hipError_t res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_NUM_REGS , f );
    HipRes("HipAttributeNumRegs",res);
    return pi;
  }

  int HipAttributeLocalSize( JitFunction f ) {
    int pi;
    hipError_t res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES , f );
    HipRes("HipAttributeLocalSize",res);
    return pi;
  }

  int HipAttributeConstSize( JitFunction f ) {
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

    QDPIO::cout << "copy H -> D: ";
    for ( int s = 0 ; s < Layout::sitesOnNode() ; s++ ) 
      {
	float* ptr = (float*)src;
	QDPIO::cout << ptr[s] << " ";
      }
    QDPIO::cout << "\n";

    HipRes("hipMemcpyH2D",ret);
  }

  void HipMemcpyD2H( void * dest , const void * src , size_t size )
  {
    hipError_t ret;
    ret = hipMemcpyDtoH( dest, (hipDeviceptr_t)const_cast<void*>(src), size);

    QDPIO::cout << "copy D -> H: ";
    for ( int s = 0 ; s < Layout::sitesOnNode() ; s++ ) 
      {
	float* ptr = (float*)dest;
	QDPIO::cout << ptr[s] << " ";
      }
    QDPIO::cout << "\n";

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


