// -*- c++ -*-



#include "qdp_config_internal.h" 
#include "qdp.h"

#include <iostream>
#include <string>

#include "cuda.h"

#ifdef QDP_THRUSTALIGN
#include "cuda_runtime.h"
#endif

#include <unistd.h>


#include "cudaProfiler.h"

#ifdef QDP_DEEP_LOG
#include <signal.h>
#endif

namespace QDP {

  namespace {
    CUdevice cuDevice;
    CUcontext cuContext;
#ifdef QDP_THRUSTALIGN
    bool deviceSet = false;
#endif

    CUevent evStart;
    CUevent evStop;

    int deviceCount;


    unsigned device;
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

    size_t mem_free, mem_total;

    
    size_t roundDown2pow(size_t x)
    {
      size_t s=1;
      while (s<=x) s <<= 1;
      s >>= 1;
      return s;
    }
  }
  




  void CudaCheckResult(CUresult result)
  {
    if (result != CUDA_SUCCESS) {
      const char *errStr;
      cuGetErrorString(result, &errStr);
      const char *errName;
      cuGetErrorName(result, &errName);
      QDP_error_exit("CUDA error %d %s (%s)", (int)result , errName, errStr );
    }
  }

#ifdef QDP_THRUSTALIGN
  void CudaCheckResult(cudaError_t result)
  {
    if (result != cudaSuccess) {
      QDP_error_exit("CUDA error %s: %s", cudaGetErrorName(result) , cudaGetErrorString(result));
    }
  }
#endif

  void gpu_create_events()
  {
#ifdef QDP_THRUSTALIGN
    CudaCheckResult(cuCtxSetCurrent(cuContext));
#endif
    CUresult res = cuEventCreate ( &evStart, 0 );
    if (res != CUDA_SUCCESS)
      {
	QDPIO::cout << "error event creation start\n";
	QDP_abort(1);
      }
    res = cuEventCreate ( &evStop, 0 );
    if (res != CUDA_SUCCESS)
      {
	QDPIO::cout << "error event creation stop\n";
	QDP_abort(1);
      }
  }

  void gpu_record_start()
  {
#ifdef QDP_THRUSTALIGN
    CudaCheckResult(cuCtxSetCurrent(cuContext));
#endif
    CUresult res = cuEventRecord ( evStart, 0 );
    if (res != CUDA_SUCCESS)
      {
	QDPIO::cout << "error event record start\n";
	QDP_abort(1);
      }
  }

  void gpu_record_stop()
  {
#ifdef QDP_THRUSTALIGN
    CudaCheckResult(cuCtxSetCurrent(cuContext));
#endif
    CUresult res = cuEventRecord ( evStop, 0 );
    if (res != CUDA_SUCCESS)
      {
	QDPIO::cout << "error event record stop\n";
	QDP_abort(1);
      }
  }

  void gpu_event_sync()
  {
#ifdef QDP_THRUSTALIGN
    CudaCheckResult(cuCtxSetCurrent(cuContext));
#endif
    CUresult res = cuEventSynchronize ( evStop );
    if (res != CUDA_SUCCESS)
      {
	QDPIO::cout << "error event sync stop\n";
	QDP_abort(1);
      }
  }


  float gpu_get_time()
  {
    float pMilliseconds;
#ifdef QDP_THRUSTALIGN
    CudaCheckResult(cuCtxSetCurrent(cuContext));
#endif
    CUresult res = cuEventElapsedTime( &pMilliseconds, evStart, evStop );
    if (res != CUDA_SUCCESS)
      {
	QDPIO::cout << "error event get time\n";
	QDP_abort(1);
      }
    return pMilliseconds;
  }




  size_t gpu_mem_free()
  {
    return mem_free;
  }
  
  size_t gpu_mem_total()
  {
    return mem_total;
  }
  
  
  std::map< JitFunction::Func_t , std::string > mapCUFuncPTX;

  std::string getPTXfromCUFunc(JitFunction& f) {
    return mapCUFuncPTX[f.get_function()];
  }






  
  int CudaGetAttributesLocalSize( JitFunction& f )
  {
    int local_mem = 0;
#ifdef QDP_THRUSTALIGN
    CudaCheckResult(cuCtxSetCurrent(cuContext));
#endif
    cuFuncGetAttribute( &local_mem , CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES , (CUfunction)f.get_function() );
    return local_mem;
  }


#ifdef QDP_DEEP_LOG
  namespace {
    std::unique_ptr<std::fstream> logger_cmp;
  }

  void gpu_deep_logger_close()
  {
    if (logger_cmp)
      {
	std::cout << "closing log file" << std::endl;
	logger_cmp->close();
      }
  }


  void gpu_deep_logger_create( void* host_ptr , std::string type_W , int size_T , int start , int count , std::string pretty )
  {
    if (!logger_cmp)
      {
	std::cout << "creating log object ..." << std::endl;
	logger_cmp.reset( new std::fstream );
	std::cout << "opening log file ..." << std::endl;
	logger_cmp->open("ref.dat", ios::out | ios::binary);
	if(!(*logger_cmp))
	  {
	    std::cout << "Cannot open log file!" << endl;
	    QDP_abort(1);
	  }
      }
    
    char* host_pos = (char*)host_ptr;

    int size_W;
	    
    if ( type_W == "i" || type_W == "f" )
      {
	size_W = 4;
      }
    else if ( type_W == "b" )
      {
	size_W = 1;
      }
    else if ( type_W == "d" )
      {
	size_W = 8;
      }
    else
      {
	std::cout << " size not known : " << type_W << std::endl;
	QDP_abort(1);
      }
	
    if (size_T % size_W)
      {
	std::cout << " size_T not divisable by size_W" << std::endl;
	QDP_abort(1);
      }

    std::cout << "logger: start = " << start << "\tcount = " << count << "\ttype_W = " << type_W << "\tsize_T = " << size_T;
    std::cout << "\t\tdoing " << count * (size_T/size_W) << " writes\n";
    
    //std::cout << pretty << std::endl;
    

    
    for (int i = 0 ; i < count ; ++i )
      {
	for (int q = 0 ; q < size_T/size_W ; ++q )
	  {
	    if (type_W == "i")
	      {
		//std::cout << *(int*)host_pos << " ";
		logger_cmp->write( host_pos, 4 );
		host_pos += 4;
	      }
	    else if(type_W == "f")
	      {
		//std::cout << *(float*)host_pos << " ";
		logger_cmp->write( host_pos, 4 );
		host_pos += 4;
	      }
	    else if(type_W == "b")
	      {
		//std::cout << *(bool*)host_pos << " ";
		logger_cmp->write( host_pos, 1 );
		host_pos += 1;
	      }
	    else if(type_W == "d")
	      {
		//std::cout << *(bool*)host_pos << " ";
		logger_cmp->write( host_pos, 8 );
		host_pos += 8;
	      }
	    else
	      {
		std::cout << " size not known: " << type_W << std::endl;
		std::cout << pretty << std::endl;
		QDP_abort(1); break;
	      }
	  }
      }
    //std::cout << std::endl;
  }


  
  void gpu_deep_logger_compare( void* host_ptr , std::string type_W , int size_T , int start , int count , std::string pretty )
  {
    if (!logger_cmp)
      {
	std::cout << "creating log object ..." << std::endl;
	logger_cmp.reset( new std::fstream );
	std::cout << "opening log file ..." << std::endl;
	logger_cmp->open("ref.dat", ios::in | ios::binary);
	if(!(*logger_cmp))
	  {
	    std::cout << "Cannot open log file!" << endl;
	    QDP_abort(1);
	  }
      }

    char* host_pos = (char*)host_ptr;
	
    int size_W;
	
    if ( type_W == "i" || type_W == "f" )
      {
	size_W = 4;
      }
    else if ( type_W == "b" )
      {
	size_W = 1;
      }
    else if ( type_W == "d" )
      {
	size_W = 8;
      }
    else
      {
	std::cout << " size not known: " << type_W << std::endl;
	std::cout << " pretty: " << pretty << std::endl;
	QDP_abort(1);
      }
	
    if (size_T % size_W)
      {
	std::cout << " size_T not divisable by size_W" << std::endl;
	QDP_abort(1);
      }

    //std::cout << "doing " << count * (size_T/size_W) << " reads\n";
    std::cout << "logger: start = " << start << "\tcount = " << count << "\ttype_W = " << type_W << "\tsize_T = " << size_T;
    std::cout << "\tdoing " << count * (size_T/size_W) << " reads\t\t";

    int print_count = 4;
    
    for (int i = 0 ; i < count; ++i )
      {
	for (int q = 0 ; q < size_T/size_W ; ++q )
	  {
	    char buf[8];
	    logger_cmp->read(buf,size_W);

	    if (type_W == "i")
	      {
		int cur = *(int*)host_pos;
		int ref = *(int*)(&buf[0]);
		if (--print_count >= 0) std::cout << cur << " ";
		if (cur != ref)
		  {
		    std::cout << "\nmismatch int: index = " << i << " q = " << q << " fstart = " << start << ": cur = " << cur << "  ref = " << ref << std::endl;
		    std::cout << pretty << std::endl;
		    raise(SIGSEGV);
		  }
		host_pos += 4;
	      }
	    else if(type_W == "f")
	      {
		float cur = *(float*)host_pos;
		float ref = *(float*)(&buf[0]);
		if (--print_count >= 0) std::cout << cur << " ";
		if (cur != ref)
		  {
		    std::cout << "\nmismatch float: index = " << i << " q = " << q << " fstart = " << start << ": cur = " << cur << "  ref = " << ref << std::endl;
		    std::cout << pretty << std::endl;
		    raise(SIGSEGV);
		  }
		host_pos += 4;
	      }
	    else if(type_W == "d")
	      {
		double cur = *(double*)host_pos;
		double ref = *(double*)(&buf[0]);
		if (--print_count >= 0) std::cout << cur << " ";
		if (cur != ref)
		  {
		    std::cout << "\nmismatch double: index = " << i << " q = " << q << " fstart = " << start << ": cur = " << cur << "  ref = " << ref << std::endl;
		    std::cout << pretty << std::endl;
		    raise(SIGSEGV);
		  }
		host_pos += 8;
	      }
	    else if(type_W == "b")
	      {
		bool cur = *(bool*)host_pos;
		bool ref = *(bool*)(&buf[0]);
		if (--print_count >= 0) std::cout << cur << " ";
		if (cur != ref)
		  {
		    std::cout << "\nmismatch bool: index = " << i << " q = " << q << " fstart = " << start << ": cur = " << cur << "  ref = " << ref << std::endl;
		    std::cout << pretty << std::endl;
		    raise(SIGSEGV);
		  }
		host_pos += 1;
	      }
	    else
	      {
		std::cout << " size not known" << std::endl; QDP_abort(1); break;
	      }
		
	  }
      }
    std::cout << std::endl;
  }


  void gpu_deep_logger( void* host_ptr , std::string type_W , int size_T , int start , int count , std::string pretty )
  {
    if ( jit_config_deep_log_create() )
      {
	gpu_deep_logger_create( host_ptr , type_W , size_T , start , count , pretty );
      }
    else
      {
	gpu_deep_logger_compare( host_ptr , type_W , size_T , start , count , pretty );
      }
  }
#endif  


  JitResult gpu_launch_kernel( JitFunction& f, 
			       unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
			       unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
			       unsigned int  sharedMemBytes, QDPCache::KernelArgs_t kernelArgs )
  {
    if (gpu_get_record_stats() && Layout::primaryNode())
      {
	gpu_record_start();
      }
    
#ifdef QDP_THRUSTALIGN
    CudaCheckResult(cuCtxSetCurrent(cuContext));
#endif
    CUresult res = cuLaunchKernel((CUfunction)f.get_function(), gridDimX, gridDimY, gridDimZ, 
				  blockDimX, blockDimY, blockDimZ, 
				  sharedMemBytes, 0, (void**)&kernelArgs[0], 0 );
    
    if (gpu_get_record_stats() && Layout::primaryNode())
      {
	gpu_record_stop();
	gpu_event_sync();
	float time = gpu_get_time();
	f.add_timing( time );
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


#ifdef QDP_DEEP_LOG
    if (jit_config_deep_log())
      {
	//size_t field_size = f.count * f.size_T;

	size_t field_size = QDP_get_global_cache().getSize( f.get_dest_id() );

	if (f.count > 0)
	  {
	    void* host_ptr;

	    if ( ! (host_ptr = malloc( field_size )) )
	      {
		QDPIO::cout << "Cannot allocate host memory!" << endl;
		QDP_abort(1);
	      }

	    
	    std::vector<QDPCache::ArgKey> vec_id;
	    vec_id.push_back( f.get_dest_id() );
	    std::vector<void*> vec_ptrs = QDP_get_global_cache().get_dev_ptrs( vec_id );
	    void* dev_ptr = vec_ptrs.at(0);

	    //std::cout << "d2h: start = " << f.start << "  count = " << f.count << "  size_T = " << f.size_T << "   \t";
    
	    gpu_memcpy_d2h( host_ptr , dev_ptr , field_size );

	    gpu_deep_logger( host_ptr , f.type_W , f.size_T , f.start , f.count , f.get_pretty() );

	    free( host_ptr );
	  }
	else
	  {
	    QDPIO::cout << "zero count. " << f.get_pretty() << std::endl;
	  }
      }
#endif
    
    return ret;
  }


    


  void CudaRes(const std::string& s,CUresult ret) {
    if (ret != CUDA_SUCCESS) {
      const char *errStr;
      cuGetErrorString(ret, &errStr);
      std::cout << s << " Error: " << errStr << "\n";
      exit(1);
    }
  }


#if 0
  int CudaAttributeNumRegs( JitFunction& f ) {
    int pi;
    CUresult res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_NUM_REGS , (CUfunction)f.get_function() );
    CudaRes("CudaAttributeNumRegs",res);
    return pi;
  }

  int CudaAttributeLocalSize( JitFunction& f ) {
    int pi;
    CUresult res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES , (CUfunction)f.get_function() );
    CudaRes("CudaAttributeLocalSize",res);
    return pi;
  }

  int CudaAttributeConstSize( JitFunction& f ) {
    int pi;
    CUresult res;
    res = cuFuncGetAttribute ( &pi, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES , (CUfunction)f.get_function() );
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
#endif


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

  void gpu_init() {
    cuInit(0);

    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) { 
      std::cout << "There is no device supporting CUDA.\n"; 
      exit(1); 
    }
  }

  std::string gpu_get_arch()
  {
    int major = gpu_getMajor();
    int minor = gpu_getMinor();

    std::string str_compute = "sm_" + std::to_string( major * 10 + minor );

    return str_compute;
  }





  void gpu_set_device(int dev)
  {
#ifdef QDP_THRUSTALIGN
    CUresult ret;

    if (cudaSetDevice(dev) != cudaSuccess)
    {
      std::cout << "Error setting device.\n"; 
      exit(1);
    }

    ret = cuDeviceGet(&cuDevice, dev);
    CudaRes(__func__,ret);

    ret = cuDevicePrimaryCtxRetain(&cuContext, dev);
    CudaRes(__func__,ret);

    ret = cuCtxPushCurrent(cuContext);
    CudaRes(__func__,ret);
#else
    CUresult ret;

    ret = cuDeviceGet(&cuDevice, dev);
    CudaRes(__func__,ret);

    ret = cuCtxGetCurrent(&cuContext);
    
    if (ret != CUDA_SUCCESS || cuContext == NULL) 
    {
      ret = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, cuDevice);
    }
    CudaRes(__func__,ret);
#endif

    //std::cout << "creating CUDA events\n";
    gpu_create_events();
  }


  void gpu_get_device_props() {
    smem = CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK );
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
    CudaGetSM(&ma,&mi);
    
    major = ma;
    minor = mi;
    
    QDPIO::cout << "  Compute capability (major,minor)    : " << major << "," << minor << "\n";
    QDPIO::cout << "  Shared memory                       : " << smem/1024  << " KB\n";
    QDPIO::cout << "  Max grid  (x,y,z)                   : (" << max_gridx << "," << max_gridy << "," << max_gridz << ")\n";
    QDPIO::cout << "  Max block (x,y,z)                   : (" << max_blockx << "," << max_blocky << "," << max_blockz << ")\n";
  }


  
  void gpu_auto_detect()
  {
    CUresult ret;

    QDPIO::cout << "GPU properties\n";

    // Get the device properties
    gpu_get_device_props();
    
    ret = cuMemGetInfo( &mem_free , &mem_total );
    CudaRes("cuMemGetInfo",ret);

    QDPIO::cout << "  GPU memory (free,total)             : " << mem_free/1024/1024 << "/" << mem_total/1024/1024 << " MB\n";


    ret = cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1);
    CudaRes("cuCtxSetCacheConfig",ret);
  }


  int gpu_get_device_count()
  {
    return deviceCount;
  }



  void gpu_host_alloc(void **mem , const size_t size)
  {
    CUresult ret;
#ifdef QDP_THRUSTALIGN
    CudaCheckResult(cuCtxSetCurrent(cuContext));
#endif
    ret = cuMemHostAlloc(mem,size,0);
    CudaRes("cudaHostAlloc",ret);
  }


  void gpu_host_free(void *mem)
  {
    if (!QDP_isInitialized())
      {
	//std::cout << "host free, QDP not initialized" << std::endl;
	return;
      }
    CUresult ret;
#ifdef QDP_THRUSTALIGN
    CudaCheckResult(cuCtxSetCurrent(cuContext));
#endif
    ret = cuMemFreeHost(mem);
    CudaRes("cuMemFreeHost",ret);
  }





  void gpu_memcpy_h2d( void * dest , const void * src , size_t size )
  {
    CUresult ret;
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("CudaMemcpyH2D dest=%p src=%p size=%d" ,  dest , src , size );
#endif
    //std::cout << "H2D " << size << std::endl;
#ifdef QDP_THRUSTALIGN
    CudaCheckResult(cuCtxSetCurrent(cuContext));
#endif
    ret = cuMemcpyHtoD((CUdeviceptr)const_cast<void*>(dest), src, size);
    CudaRes("cuMemcpyH2D",ret);
  }

  void gpu_memcpy_d2h( void * dest , const void * src , size_t size )
  {
    CUresult ret;
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("CudaMemcpyD2H dest=%p src=%p size=%d" ,  dest , src , size );
#endif
    //std::cout << "D2H " << size << std::endl;
#ifdef QDP_THRUSTALIGN
    CudaCheckResult(cuCtxSetCurrent(cuContext));
#endif
    ret = cuMemcpyDtoH( dest, (CUdeviceptr)const_cast<void*>(src), size);
    CudaRes("cuMemcpyD2H",ret);
  }


  bool gpu_malloc(void **mem , size_t size )
  {
#ifdef QDP_THRUSTALIGN

if (size == 0) *mem = nullptr;
    CUresult ret;
    CudaCheckResult(cuCtxSetCurrent(cuContext));
#ifndef QDP_USE_CUDA_MANAGED_MEMORY
    CudaCheckResult(cudaMalloc(mem, size));
#else
    CudaCheckResult(cudaMallocManaged(mem, size, cudaMemAttachGlobal));
#endif
    return ret == CUDA_SUCCESS;

#else

    CUresult ret;
#ifndef QDP_USE_CUDA_MANAGED_MEMORY
    ret = cuMemAlloc( (CUdeviceptr*)mem,size);
#else
    ret = cuMemAllocManaged( (CUdeviceptr*)mem, size, CU_MEM_ATTACH_GLOBAL ); 
#endif
    return ret == CUDA_SUCCESS;
#endif
  }



  
  void gpu_free(const void *mem )
  {
#ifdef QDP_THRUSTALIGN
	  if (mem == nullptr) return;
#endif

    if (!QDP_isInitialized())
      {
	//std::cout << "GPU free, QDP not initialized" << std::endl;
	return;
      }

    CUresult ret;
#ifdef QDP_THRUSTALIGN
    CudaCheckResult(cuCtxSetCurrent(cuContext));
    CudaCheckResult(cudaFree((void*)mem));
#else
    ret = cuMemFree((CUdeviceptr)const_cast<void*>(mem));
    CudaRes("cuMemFree",ret);
#endif
  }



  void gpu_memset( void * dest , unsigned val , size_t N )
  {
    CUresult ret;
#ifdef QDP_THRUSTALIGN
    CudaCheckResult(cuCtxSetCurrent(cuContext));
#endif
    ret = cuMemsetD32((CUdeviceptr)const_cast<void*>(dest), val, N);
    CudaRes("cuMemsetD32",ret);
  }


  
  void gpu_sync()
  {
    CUresult ret = cuCtxSynchronize();
    CudaRes("cuCtxSynchronize",ret);
  }




  bool get_jitf( JitFunction& func, const std::string& kernel_ptx , const std::string& kernel_name , const std::string& pretty , const std::string& str_compute )
  {
    CUresult ret;
    CUmodule cuModule;

    func.set_kernel_name( kernel_name );
    func.set_pretty( pretty );
    
#ifdef QDP_THRUSTALIGN
    CudaCheckResult(cuCtxSetCurrent(cuContext));
#endif
    ret = cuModuleLoadData(&cuModule, (const void *)kernel_ptx.c_str());

    if (ret != CUDA_SUCCESS) {
      QDPIO::cerr << "Error loading external data.\n";
      //raise(SIGSEGV);
      QDP_abort(1);
    }

    CUfunction cuf;
    ret = cuModuleGetFunction( &cuf , cuModule , kernel_name.c_str() );
    if (ret != CUDA_SUCCESS) {
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
	    return false;
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
	    return false;
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
    return true;
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


