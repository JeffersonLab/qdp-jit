// -*- c++ -*-



#include "qdp_config_internal.h" 
#include "qdp.h"

#include <iostream>
#include <string>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <unistd.h>


//#include "cudaProfiler.h"


namespace QDP {

  namespace {
    // CUevent evStart;
    // CUevent evStop;

    int deviceCount;
    int deviceId;     // the device we use
    int gcnArch;

    size_t mem_free, mem_total;

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

    int defaultGPU = -1;

    size_t roundDown2pow(size_t x) {
      size_t s=1;
      while (s<=x) s <<= 1;
      s >>= 1;
      return s;
    }
  }
  

  void gpu_create_events()
  {
#if 0
    std::cout << "creating CUDA events\n";

    hipError_t res = cuEventCreate ( &evStart, 0 );
    if (res != hipSuccess)
      {
	QDPIO::cout << "error event creation start\n";
	QDP_abort(1);
      }
    res = cuEventCreate ( &evStop, 0 );
    if (res != hipSuccess)
      {
	QDPIO::cout << "error event creation stop\n";
	QDP_abort(1);
      }
#endif
  }

  void gpu_record_start()
  {
#if 0
    hipError_t res = cuEventRecord ( evStart, 0 );
    if (res != hipSuccess)
      {
	QDPIO::cout << "error event record start\n";
	QDP_abort(1);
      }
#endif
  }

  void gpu_record_stop()
  {
#if 0
    hipError_t res = cuEventRecord ( evStop, 0 );
    if (res != hipSuccess)
      {
	QDPIO::cout << "error event record stop\n";
	QDP_abort(1);
      }
#endif
  }

  void gpu_event_sync()
  {
#if 0
    hipError_t res = cuEventSynchronize ( evStop );
    if (res != hipSuccess)
      {
	QDPIO::cout << "error event sync stop\n";
	QDP_abort(1);
      }
#endif
  }


  float gpu_get_time()
  {
#if 0
    float pMilliseconds;
    hipError_t res = cuEventElapsedTime( &pMilliseconds, evStart, evStop );
    if (res != hipSuccess)
      {
	QDPIO::cout << "error event get time\n";
	QDP_abort(1);
      }
    return pMilliseconds;
#endif
    return 0.;
  }




  
  
  std::map< JitFunction::Func_t , std::string > mapCUFuncPTX;

  std::string getPTXfromCUFunc(JitFunction& f) {
    return mapCUFuncPTX[f.get_function()];
  }



  std::map<hipError_t,std::string> mapCuErrorString= {
    {hipSuccess 	,"Successful completion."},
    {hipErrorInvalidContext 	,"Produced when input context is invalid."},
    {hipErrorInvalidKernelFile 	,"In CUDA DRV, it is CUDA_ERROR_INVALID_PTX."},
    {hipErrorMemoryAllocation 	,"Memory allocation error."},
    {hipErrorInitializationError 	,"TODO comment from hipErrorInitializationError."},
    {hipErrorLaunchFailure 	,"An exception occurred on the device while executing a kernel."},
    {hipErrorLaunchOutOfResources 	,"Out of resources error."},
    {hipErrorInvalidDevice 	,"DeviceID must be in range 0...#compute-devices."},
    {hipErrorInvalidValue 	,"One or more of the parameters passed to the API call is NULL or not in an acceptable range."},
    {hipErrorInvalidDevicePointer 	,"Invalid Device Pointer."},
    {hipErrorInvalidMemcpyDirection 	,"Invalid memory copy direction."},
    {hipErrorUnknown 	,"Unknown error."},
    {hipErrorInvalidResourceHandle 	,"Resource handle (hipEvent_t or hipStream_t) invalid."},
    {hipErrorNotReady 	,"Indicates that asynchronous operations enqueued earlier are not ready. This is not actually an error, but is used to distinguish from hipSuccess (which indicates completion). APIs that return this error include hipEventQuery and hipStreamQuery."},
    {hipErrorNoDevice 	,"Call to hipGetDeviceCount returned 0 devices."},
    {hipErrorPeerAccessAlreadyEnabled 	,"Peer access was already enabled from the current device."},
    {hipErrorPeerAccessNotEnabled 	,"Peer access was never enabled from the current device."},
    {hipErrorRuntimeMemory 	,"HSA runtime memory call returned error. Typically not seen in production systems."},
    {hipErrorRuntimeOther 	,"HSA runtime call other than memory returned error. Typically not seen in production systems."},
    {hipErrorHostMemoryAlreadyRegistered 	,"Produced when trying to lock a page-locked memory."},
    {hipErrorHostMemoryNotRegistered 	,"Produced when trying to unlock a non-page-locked memory."},
    {hipErrorMapBufferObjectFailed 	,"Produced when the IPC memory attach failed from ROCr."},
    {hipErrorTbd 	,"Marker that more error codes are needed. "} };

  
  int CudaGetAttributesLocalSize( JitFunction& f )
  {
#if 0
    int local_mem = 0;
    cuFuncGetAttribute( &local_mem , CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES , (CUfunction)f.get_function() );
    return local_mem;
#endif
    return 0;
  }



  JitResult gpu_launch_kernel( JitFunction& f, 
			       unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
			       unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
			       unsigned int  sharedMemBytes, int hStream, void** kernelParams, void** extra  )
  {
#if 0
    if (gpu_get_record_stats() && Layout::primaryNode())
      {
	gpu_record_start();
      }
    
    hipError_t res = cuLaunchKernel((CUfunction)f.get_function(), gridDimX, gridDimY, gridDimZ, 
				  blockDimX, blockDimY, blockDimZ, 
				  sharedMemBytes, 0, kernelParams, extra);
    
    if (gpu_get_record_stats() && Layout::primaryNode())
      {
	gpu_record_stop();
	gpu_event_sync();
	float time = gpu_get_time();
	f.add_timing( time );
      }
  
    JitResult ret;

    switch (res) {
    case hipSuccess:
      ret = JitResult::JitSuccess;
      break;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
      ret = JitResult::JitResource;
      break;
    default:
      ret = JitResult::JitError;
    }
#endif
    JitResult ret;
    return ret;
  }


    

  void CudaCheckResult(hipError_t result) {
    if (result != hipSuccess) {
      QDP_info("CUDA error %d (%s)", (int)result , mapCuErrorString[result].c_str());
    }
  }


  void CheckError(const std::string& s,hipError_t ret) {
    if (ret != hipSuccess) {
      if (mapCuErrorString.count(ret)) 
	std::cout << s << " Error: " << mapCuErrorString.at(ret) << "\n";
      else
	std::cout << s << " Error: (not known)\n";
      exit(1);
    }
  }


#if 0
  int CudaAttributeNumRegs( JitFunction& f ) {
    int pi;
    hipError_t res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_NUM_REGS , (CUfunction)f.get_function() );
    CheckError("CudaAttributeNumRegs",res);
    return pi;
  }

  int CudaAttributeLocalSize( JitFunction& f ) {
    int pi;
    hipError_t res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES , (CUfunction)f.get_function() );
    CheckError("CudaAttributeLocalSize",res);
    return pi;
  }

  int CudaAttributeConstSize( JitFunction& f ) {
    int pi;
    hipError_t res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES , (CUfunction)f.get_function() );
    CheckError("CudaAttributeConstSize",res);
    return pi;
  }

  void CudaProfilerInitialize()
  {
    hipError_t res;
    std::cout << "CUDA Profiler Initializing ...\n";
    res = cuProfilerInitialize( "prof.cfg" , "prof.out" , CU_OUT_CSV );
    CheckError("cuProfilerInitialize",res);
  }

  void CudaProfilerStart()
  {
    hipError_t res;
    res = cuProfilerStart();
    CheckError("cuProfilerStart",res);
  }

  void CudaProfilerStop()
  {
    hipError_t res;
    res = cuProfilerStop();
    CheckError("cuProfilerStop",res);
  }
#endif



  void gpu_init() {
    // no need to initialize in ROCm

    std::cout << "gpu_init\n";
    
    hipError_t ret = hipGetDeviceCount(&deviceCount);
    CheckError("hipGetDeviceCount",ret);
    
    if (deviceCount == 0)
      { 
	std::cout << "There is no device supporting ROCm.\n"; 
	exit(1); 
      }
  }








  void gpu_set_device(int dev)
  {
    hipError_t ret;

    std::cout << "trying to get device " << dev << "\n";
    
    ret = hipSetDevice(dev);
    CheckError("hitSetDevice",ret);

#if 0
    QDP_info_primary("trying to grab pre-existing context",dev);
    ret = cuCtxGetCurrent(&cuContext);
    
    if (ret != hipSuccess || cuContext == NULL) {
      QDP_info_primary("trying to create a context");
      ret = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, cuDevice);
    }
    CheckError(__func__,ret);
#endif
    
    gpu_create_events();
  }



  void gpu_get_device_props() {

    hipDeviceProp_t prop;
    hipError_t ret = hipGetDeviceProperties ( &prop , deviceId );
      
    smem = prop.sharedMemPerBlock;
    max_gridx = roundDown2pow( prop.maxGridSize[0] );
    max_gridy = roundDown2pow( prop.maxGridSize[1] );
    max_gridz = roundDown2pow( prop.maxGridSize[2] );
    max_blockx = roundDown2pow( prop.maxThreadsDim[0] );
    max_blocky = roundDown2pow( prop.maxThreadsDim[1] );
    max_blockz = roundDown2pow( prop.maxThreadsDim[2] );
    gcnArch = prop.gcnArch;
    
#ifdef QDP_CUDA_SPECIAL
    QDPIO::cout << "Setting max gridx for CUDA special functions\n";
    cuda_special_set_maxgridx( max_gridx );
#endif
    
    QDPIO::cout << "GPU autodetect\n";
    QDPIO::cout << "  Device name                         : " << std::string( prop.name ) << "\n";
    QDPIO::cout << "  GCN architecture                    : gfx" << prop.gcnArch << "\n";
    QDPIO::cout << "  Shared memory                       : " << smem/1024  << " KB\n";
    QDPIO::cout << "  Max grid  (x,y,z)                   : (" << max_gridx << "," << max_gridy << "," << max_gridz << ")\n";
    QDPIO::cout << "  Max block (x,y,z)                   : (" << max_blockx << "," << max_blocky << "," << max_blockz << ")\n";
  }

  
  size_t gpu_mem_free()
  {
    return mem_free;
  }
  
  size_t gpu_mem_total()
  {
    return mem_total;
  }


  void gpu_auto_detect()
  {
    hipError_t ret;

    // get device props
    gpu_get_device_props();
  
    ret = hipMemGetInfo (&mem_free, &mem_total);
    CheckError("hipMemGetInfo",ret);

    QDPIO::cout << "  GPU memory (free,total)             : " << mem_free/1024/1024 << "/" << mem_total/1024/1024 << " MB\n";

    QDPIO::cout << "  threads per block                   : " << jit_util_get_threads_per_block() << "\n";

#if 0
    ret = cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1);
    CheckError("cuCtxSetCacheConfig",ret);
#endif
  }


  int gpu_get_device_count()
  {
    return deviceCount;
  }




  void gpu_host_alloc(void **mem , const size_t size)
  {
    hipError_t ret;
    ret = hipHostMalloc ( mem , size , 0 );
    CheckError("hipHostMalloc",ret);
  }


  void gpu_host_free(void *mem)
  {
    hipError_t ret;
    ret = hipHostFree ( mem );
    CheckError("hipHostFree",ret);
  }





  void gpu_memcpy_h2d( void * dest , const void * src , size_t size )
  {
    hipError_t ret;
    ret = hipMemcpyHtoD( (hipDeviceptr_t)const_cast<void*>(dest) , (void*)src , size );
    CheckError("hipMemcpyHtoD",ret);
  }

  void gpu_memcpy_d2h( void * dest , const void * src , size_t size )
  {
    hipError_t ret;
    ret = hipMemcpyDtoH( dest , (hipDeviceptr_t)const_cast<void*>(src) , size );
    CheckError("hipMemcpyDtoH",ret);
  }


  bool gpu_malloc(void **mem , size_t size )
  {
    hipError_t ret;
    ret = hipMalloc ( mem , size);
    return ret == hipSuccess;
  }

  void gpu_free(const void *mem )
  {
    hipError_t ret;
    ret = hipFree( (void*)mem );
    CheckError("hipFree",ret);
  }



  void gpu_memset( void * dest , unsigned val , size_t N )
  {
    hipError_t ret;
    ret = hipMemset ( dest , val , N );
    CheckError("hipMemset",ret);
  }


  std::string gpu_get_arch()
  {
    return "gfx" + std::to_string(gcnArch);
  }



  void get_jitf( JitFunction& func, const std::string& kernel_ptx , const std::string& kernel_name , const std::string& pretty , const std::string& str_compute )
  {
#if 0
    hipError_t ret;
    CUmodule cuModule;

    func.set_kernel_name( kernel_name );
    func.set_pretty( pretty );
    
    ret = cuModuleLoadData(&cuModule, (const void *)kernel_ptx.c_str());

    if (ret != hipSuccess) {
      QDPIO::cerr << "Error loading external data.\n";
      QDP_abort(1);
    }

    CUfunction cuf;
    ret = cuModuleGetFunction( &cuf , cuModule , kernel_name.c_str() );
    if (ret != hipSuccess) {
      QDPIO::cerr << "Error getting function.";
      QDP_abort(1);
    }

    func.set_function( cuf );
    
    mapCUFuncPTX[func.get_function()] = kernel_ptx;

    if ( gpu_get_record_stats() && Layout::primaryNode() )
      {
	std::string ptxpath  = "kernel_n" + std::to_string( Layout::nodeNumber() ) + "_p" + std::to_string( ::getpid() ) + ".ptx";
	std::string sasspath = "kernel_n" + std::to_string( Layout::nodeNumber() ) + "_p" + std::to_string( ::getpid() ) + ".sass";
      
	std::ofstream f(ptxpath);
	f << kernel_ptx;
	f.close();

	FILE *fp;
	char buf[1024];

	string cmd = "ptxas -v --gpu-name " + str_compute + " " + ptxpath + " -o " + sasspath + " 2>&1";
      
	fp = popen( cmd.c_str() , "r" );
      
	if (fp == NULL)
	  {
	    QDPIO::cerr << "Stats error: Failed to run command via popen\n";
	    return;
	  }

	std::ostringstream output;
	while (fgets(buf, sizeof(buf), fp) != NULL) {
	  output << buf;
	}
      
	pclose(fp);

	std::istringstream iss(output.str());

	std::vector<std::string> words;
	std::copy(istream_iterator<string>(iss),
		  istream_iterator<string>(),
		  std::back_inserter(words));

	if ( words.size() < 32 )
	  {
	    std::cerr << "Couldn't read all tokens (output of ptxas has changed?)\n";
	    std::cerr << "----- Output -------\n";
	    std::cerr << output.str() << "\n";
	    std::cerr << "--------------------\n";
	    return;
	  }

	//
	// Usually the output of ptxas -v looks like
	//
	// ptxas info    : 0 bytes gmem
	// ptxas info    : Compiling entry function 'sum0' for 'sm_50'
	// ptxas info    : Function properties for sum0
	//     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
	// ptxas info    : Used 12 registers, 344 bytes cmem[0]
	//
	// But for functions that use calls to other functions like
	// into the libdevice the first line might like more like
	//
	// ptxas info    : 0 bytes gmem, 24 bytes cmem[3]

	int pos_add = 0;
	if (words[5] == "gmem,")
	  {
	    pos_add = 3;
	  }
	
	const int pos_stack = 22 + pos_add;
	const int pos_store = 26 + pos_add;
	const int pos_loads = 30 + pos_add;
	const int pos_regs = 38 + pos_add;
	const int pos_cmem = 40 + pos_add;

	func.set_stack( std::atoi( words[ pos_stack ].c_str() ) );
	func.set_spill_store( std::atoi( words[ pos_store ].c_str() ) );
	func.set_spill_loads( std::atoi( words[ pos_loads ].c_str() ) );
	func.set_regs( std::atoi( words[ pos_regs ].c_str() ) );
	func.set_cmem( std::atoi( words[ pos_cmem ].c_str() ) );

#if 1
	// Zero encountered ??
	if (func.get_regs() == 0)
	  {
	    std::cerr << "----- zero regs encountered -----\n";
	    std::cerr << output.str() << "\n";
	    std::cerr << "----------------------------\n";
	  }
#endif
	
	// QDPIO::cout << "----- Kernel stats ------\n";
	// QDPIO::cout << "kernel_stack       = "<< kernel_stack << "\n";
	// QDPIO::cout << "kernel_spill_store = "<< kernel_spill_store << "\n";
	// QDPIO::cout << "kernel_spill_loads = "<< kernel_spill_loads << "\n";
	// QDPIO::cout << "kernel_regs        = "<< kernel_regs << "\n";
	// QDPIO::cout << "kernel_cmem        = "<< kernel_cmem << "\n";
      
#if 0      
	string cmd = "ptxas --gpu-name " + compute + " " + ptxpath + " -o " + sasspath;
	if (system(cmd.c_str()) == 0) {
	  cmd = "nvdisasm " + sasspath;
	  int ret = system(cmd.c_str());
	  (void)ret;  // Don't care if it fails
	}
#endif
      }
#endif
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





  void gpu_set_default_GPU(int ngpu) {
    defaultGPU = ngpu;
  }


  int  gpu_get_default_GPU() { return defaultGPU; }
  
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


