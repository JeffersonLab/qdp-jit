// -*- c++ -*-



#include "qdp_config_internal.h" 
#include "qdp.h"

#include <iostream>
#include <string>

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

#include <unistd.h>


#ifdef QDP_USE_ROCM_STATS
#include "amd_comgr.h"
#endif


namespace QDP {

  namespace {
    hipEvent_t evStart;
    hipEvent_t evStop;

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

    size_t roundDown2pow(size_t x) {
      size_t s=1;
      while (s<=x) s <<= 1;
      s >>= 1;
      return s;
    }
  }

  void gpu_auto_detect();


#ifdef QDP_USE_ROCM_STATS
    namespace
    {
      bool amd_comgr_check( std::string name , amd_comgr_status_t status)
      {
	if (status == AMD_COMGR_STATUS_SUCCESS)
	  {
	    return false;
	  }
	else
	  {
	    const char *status_str;
	    status = amd_comgr_status_string(status, &status_str);
	    QDPIO::cerr << name << ": " << status_str;
	    return true;
	  }
      }


      struct UserVal_t
      {
	std::string searchkey;
	amd_comgr_metadata_node_t retval;
      };


      std::string get_metadata_string( amd_comgr_metadata_node_t md0 )
      {
	amd_comgr_status_t status;
	amd_comgr_metadata_kind_t kind;
	status = amd_comgr_get_metadata_kind( md0 , &kind );
	if (amd_comgr_check( "amd_comgr_get_metadata_kind", status ))
	  {
	    QDPIO::cerr << "X amd_comgr_get_metadata_kind error\n";
	    QDP_abort(1);
	  }
	if (kind != AMD_COMGR_METADATA_KIND_STRING)
	  {
	    QDPIO::cerr << "X key metadata is not a string\n";
	    QDP_abort(1);
	  }


	size_t size;
	status = amd_comgr_get_metadata_string( md0, &size, NULL );
	std::string str_value(size-1, '\0');
	status = amd_comgr_get_metadata_string( md0, &size, &str_value[0] );
	return str_value;
      }



      class AMD_MD_parser
      {

      public:
	AMD_MD_parser(amd_comgr_metadata_node_t md):md(md)
	{
	}


	AMD_MD_parser operator[](std::string key) 
	{
	  amd_comgr_status_t status;
	  amd_comgr_metadata_kind_t kind;
	  status = amd_comgr_get_metadata_kind( md , &kind );
	  if (amd_comgr_check( "op[string]: amd_comgr_get_metadata_kind", status ))
	    {
	      QDPIO::cerr << "amd_comgr_get_metadata_kind error\n";
	      QDP_abort(1);
	    }
	  if (kind != AMD_COMGR_METADATA_KIND_MAP)
	    {
	      QDPIO::cerr << "MD is not a map\n";
	      QDP_abort(1);
	    }

	  amd_comgr_metadata_node_t value;
	  status = amd_comgr_metadata_lookup( md , key.c_str() , &value);
	  if (amd_comgr_check( "op[string]: amd_comgr_metadata_lookup", status ))
	    {
	      QDPIO::cerr << "amd_comgr_metadata_lookup error\n";
	      QDP_abort(1);
	    }

	  return AMD_MD_parser( value );
	}


	AMD_MD_parser operator[](int i) 
	{
	  amd_comgr_status_t status;
	  amd_comgr_metadata_kind_t kind;
	  status = amd_comgr_get_metadata_kind( md , &kind );
	  if (amd_comgr_check( "op[int]: amd_comgr_get_metadata_kind", status ))
	    {
	      QDPIO::cerr << "amd_comgr_get_metadata_kind error\n";
	      QDP_abort(1);
	    }
	  if (kind != AMD_COMGR_METADATA_KIND_LIST)
	    {
	      QDPIO::cerr << "MD is not a list\n";
	      QDP_abort(1);
	    }

	  size_t list_size;
	  status = amd_comgr_get_metadata_list_size( md , &list_size );
	  if (amd_comgr_check( "amd_comgr_get_metadata_list_size: ", status ))
	    {
	      QDPIO::cerr << "amd_comgr_get_metadata_list_size\n";
	      QDP_abort(1);
	    }

	  if (i >= list_size)
	    {
	      std::cout << "list size = " << list_size << ". " << i << " out of bounds\n";	  
	      QDPIO::cerr << "list size exit out of bounds\n";
	      QDP_abort(1);
	    }


	  amd_comgr_metadata_node_t list_val;

	  status = amd_comgr_index_list_metadata( md , i , &list_val);
	  if (amd_comgr_check( "amd_comgr_index_list_metadata: ", status ))
	    {
	      QDPIO::cerr << "amd_comgr_index_list_metadata\n";
	      QDP_abort(1);
	    }

	  return AMD_MD_parser(list_val);
	}


	std::string value()
	{
	  return get_metadata_string(md);
	}

	std::string kind()
	{
	  amd_comgr_status_t status;
	  amd_comgr_metadata_kind_t kind;
	  status = amd_comgr_get_metadata_kind( md , &kind );
	  if (amd_comgr_check( "print_metadata: amd_comgr_get_metadata_kind", status ))
	    {
	      QDPIO::cerr << "amd_comgr_get_metadata_kind error\n";
	      QDP_abort(1);
	    }
      
	  switch (kind) 
	    {
	    case AMD_COMGR_METADATA_KIND_STRING:
	      return "string";
	      break;
	    case AMD_COMGR_METADATA_KIND_MAP:
	      return "map";
	      break;
	    case AMD_COMGR_METADATA_KIND_LIST:
	      return "list";
	      break;
	    case AMD_COMGR_METADATA_KIND_NULL:
	      QDPIO::cout << "metadata NULL\n";
	      return "NULL";
	      break;
	    }
	}


      private:

	amd_comgr_metadata_node_t md;
      };


      void print_metadata( amd_comgr_metadata_node_t md );
      void print_metadata_string( amd_comgr_metadata_node_t md );

      amd_comgr_status_t MD_callback( amd_comgr_metadata_node_t key, amd_comgr_metadata_node_t value, void *user_data)
      {
	amd_comgr_metadata_kind_t kind;
	amd_comgr_status_t status;

	// key

	status = amd_comgr_get_metadata_kind( key , &kind );
	if (amd_comgr_check( "amd_comgr_get_metadata_kind", status ))
	  {
	    QDPIO::cerr << "X amd_comgr_get_metadata_kind error\n";
	    QDP_abort(1);
	  }
	if (kind != AMD_COMGR_METADATA_KIND_STRING)
	  {
	    QDPIO::cerr << "X key metadata is not a string\n";
	    QDP_abort(1);
	  }

	std::cout << "key = ";
	print_metadata_string( key );

	std::cout << "val = ";

	// value

	print_metadata( value );


	return AMD_COMGR_STATUS_SUCCESS;
      }

      void print_metadata_string( amd_comgr_metadata_node_t md )
      {
	amd_comgr_status_t status;
	size_t size;
	status = amd_comgr_get_metadata_string( md, &size, NULL );
	std::string str_value(size, '\0');
	status = amd_comgr_get_metadata_string( md, &size, &str_value[0] );
	std::cout << str_value << "\n";
      }

      void print_metadata_map( amd_comgr_metadata_node_t md )
      {
	amd_comgr_status_t status;

	std::cout << "begin map\n";

	status = amd_comgr_iterate_map_metadata( md, MD_callback , NULL);
	if (amd_comgr_check( "amd_comgr_iterate_map_metadata: ", status ))
	  {
	    QDPIO::cerr << "amd_comgr_iterate_map_metadata\n";
	    QDP_abort(1);
	  }

	std::cout << "end map\n";
      }



      void print_metadata_list( amd_comgr_metadata_node_t md )
      {
	amd_comgr_status_t status;
	size_t list_size;
	status = amd_comgr_get_metadata_list_size( md , &list_size );
	if (amd_comgr_check( "amd_comgr_get_metadata_list_size: ", status ))
	  {
	    QDPIO::cerr << "amd_comgr_get_metadata_list_size\n";
	    QDP_abort(1);
	  }

	std::cout << "begin list of size = " << list_size << "\n";

	for (int i = 0 ; i < list_size ; ++i )
	  {
	    amd_comgr_metadata_node_t list_val;

	    status = amd_comgr_index_list_metadata( md , i , &list_val);
	    if (amd_comgr_check( "amd_comgr_index_list_metadata: ", status ))
	      {
		QDPIO::cerr << "amd_comgr_index_list_metadata\n";
		QDP_abort(1);
	      }

	    std::cout << "list item = " << i << "\n";

	    print_metadata( list_val );
	  }

	std::cout << "end of list\n";
      }



      void print_metadata( amd_comgr_metadata_node_t md )
      {
	amd_comgr_status_t status;
	amd_comgr_metadata_kind_t kind;
	status = amd_comgr_get_metadata_kind( md , &kind );
	if (amd_comgr_check( "print_metadata: amd_comgr_get_metadata_kind", status ))
	  {
	    QDPIO::cerr << "amd_comgr_get_metadata_kind error\n";
	    QDP_abort(1);
	  }
      
	switch (kind) 
	  {
	  case AMD_COMGR_METADATA_KIND_STRING:
	    print_metadata_string( md );
	    break;
	  case AMD_COMGR_METADATA_KIND_MAP:
	    print_metadata_map( md );
	    break;
	  case AMD_COMGR_METADATA_KIND_LIST:
	    print_metadata_list( md );
	    break;
	  case AMD_COMGR_METADATA_KIND_NULL:
	    QDPIO::cout << "metadata NULL\n";
	    break;
	  }

      }


      
      void comgr( JitFunction& func , const std::string& shared , const std::string& kernel_name )
      {
	amd_comgr_data_t data;
	amd_comgr_status_t status = amd_comgr_create_data( AMD_COMGR_DATA_KIND_RELOCATABLE , &data ); //AMD_COMGR_DATA_KIND_RELOCATABLE
	if (amd_comgr_check( "amd_comgr_create_data", status ))
	  {
	    QDPIO::cerr << "amd_comgr_create_data error\n";
	    QDP_abort(1);
	  }

	

	status = amd_comgr_set_data( data , shared.size() , shared.data() );
	if (amd_comgr_check( "amd_comgr_set_data", status ))
	  {
	    QDPIO::cerr << "amd_comgr_set_data error\n";
	    QDP_abort(1);
	  }


	amd_comgr_metadata_node_t metadata;
	status = amd_comgr_get_data_metadata( data , &metadata );
	if (amd_comgr_check( "amd_comgr_get_data_metadata", status ))
	  {
	    QDPIO::cerr << "amd_comgr_get_data_metadata error\n";
	    QDP_abort(1);
	  }


	//print_metadata( metadata );

	AMD_MD_parser parser( metadata );

	int num_sgpr =       std::stoi(parser["amdhsa.kernels"][0][".sgpr_count"].value());
	int num_sgpr_spill = std::stoi(parser["amdhsa.kernels"][0][".sgpr_spill_count"].value());

	int num_vgpr =       std::stoi(parser["amdhsa.kernels"][0][".vgpr_count"].value());
	int num_vgpr_spill = std::stoi(parser["amdhsa.kernels"][0][".vgpr_spill_count"].value());

	int num_group_segment_fixed_size =   std::stoi(parser["amdhsa.kernels"][0][".group_segment_fixed_size"].value());
	int num_private_segment_fixed_size = std::stoi(parser["amdhsa.kernels"][0][".private_segment_fixed_size"].value());
	
	func.set_regs( num_sgpr );
	func.set_vregs( num_vgpr );

	func.set_spill_store( num_sgpr_spill );
	func.set_vspill_store( num_vgpr_spill );

	func.set_group_segment( num_group_segment_fixed_size );
	func.set_private_segment( num_private_segment_fixed_size );

	
	std::cout << "num_sgpr num_vgpr (spill) : "
		  << num_sgpr << "\t" 
		  << num_vgpr << "\t(" 
		  << num_sgpr_spill << "\t"
		  << num_vgpr_spill << ")\t"
		  << num_private_segment_fixed_size << "\t"
		  << num_group_segment_fixed_size << "\n";
	
      }

    } // namespace
#endif

  

  void gpu_create_events()
  {
    hipError_t res = hipEventCreate ( &evStart );
    if (res != hipSuccess)
      {
	std::cout << "error event creation start\n";
	QDP_abort(1);
      }
    res = hipEventCreate ( &evStop );
    if (res != hipSuccess)
      {
	std::cout << "error event creation stop\n";
	QDP_abort(1);
      }
  }

  void gpu_record_start()
  {
    hipError_t res = hipEventRecord ( evStart , NULL );
    if (res != hipSuccess)
      {
	QDPIO::cout << "error event record start\n";
	QDP_abort(1);
      }
  }

  void gpu_record_stop()
  {
    hipError_t res = hipEventRecord ( evStop, NULL );
    if (res != hipSuccess)
      {
	QDPIO::cout << "error event record stop\n";
	QDP_abort(1);
      }
  }

  void gpu_event_sync()
  {
    hipError_t res = hipEventSynchronize ( evStop );
    if (res != hipSuccess)
      {
	QDPIO::cout << "error event sync stop\n";
	QDP_abort(1);
      }
  }


  float gpu_get_time()
  {
    float pMilliseconds;
    hipError_t res = hipEventElapsedTime( &pMilliseconds, evStart, evStop );
    if (res != hipSuccess)
      {
	QDPIO::cout << "error event get time\n";
	QDP_abort(1);
      }
    return pMilliseconds;
  }




  

  JitResult gpu_launch_kernel( JitFunction& f, 
			       unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
			       unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
			       unsigned int  sharedMemBytes, QDPCache::KernelArgs_t kernelArgs )
  {
    // For AMD:
    // Now that they are known must copy in the actual values for the workgroup sizes
    //
    ((int*)kernelArgs.data())[0] = (int)blockDimX;
    ((int*)kernelArgs.data())[1] = (int)gridDimX;

    //std::cout << "workgroup sizes copied in: " << ((int*)kernelArgs.data())[0] << " and  " << ((int*)kernelArgs.data())[1] << "\n";
    
    auto size = kernelArgs.size();
    //std::cout << "HipLaunchKernelNoSync: kernel params size: " << size << "\n";
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernelArgs.data(),
		      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
		      HIP_LAUNCH_PARAM_END};
    
#if 1
    if (gpu_get_record_stats() && Layout::primaryNode())
      {
	gpu_record_start();
      }
#endif

    hipError_t res = hipModuleLaunchKernel((hipFunction_t)f.get_function(),  
					   gridDimX, gridDimY, gridDimZ, 
					   blockDimX, blockDimY, blockDimZ, 
					   sharedMemBytes, nullptr, nullptr, config);

#if 1
    if (gpu_get_record_stats() && Layout::primaryNode())
      {
	gpu_record_stop();
	gpu_event_sync();
	float time = gpu_get_time();
	f.add_timing( time );
      }
#endif
    
    JitResult ret;

    switch (res) {
    case hipSuccess:
      ret = JitResult::JitSuccess;
      break;
    default:
      ret = JitResult::JitError;
    }

#ifdef QDP_DEEP_LOG
    if (jit_config_deep_log())
      {
	size_t field_size = QDP_get_global_cache().getSize( f.get_dest_id() );

	void* host_ptr;

	if ( ! (host_ptr = malloc( field_size )) )
	  {
	    QDPIO::cout << "Cannot allocate host memory!" << endl;
	    QDP_abort(1);
	  }

	void* dev_ptr = QDP_get_global_cache().get_dev_ptrs( f.get_dest_id() );

	//std::cout << "d2h: start = " << f.start << "  count = " << f.count << "  size_T = " << f.size_T << "   \t";
    
	gpu_memcpy_d2h( host_ptr , dev_ptr , field_size );

	gpu_deep_logger( host_ptr , f.type_W , field_size , f.get_pretty() , f.get_is_lat() );

	free( host_ptr );
      }
#endif

    return ret;
  }


    



  void CheckError(const std::string& s,hipError_t ret) {
    if (ret != hipSuccess) {
      std::cout << s << ", hip Error code: " << (int)ret << "\n";
      exit(1);
    }
  }


  void CheckErrorWeak(const std::string& s,hipError_t ret) {
    if (ret != hipSuccess) {
      std::cout << s << ", hip Error code: " << (int)ret << "\n";
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

    hipError_t ret = hipGetDeviceCount(&deviceCount);
    CheckError("hipGetDeviceCount",ret);
    
    if (deviceCount == 0)
      { 
	std::cout << "There is no device supporting ROCm.\n"; 
	exit(1); 
      }
  }


  void gpu_done() {
  }





  void gpu_set_device(int dev)
  {
    hipError_t ret;

    ret = hipSetDevice(dev);
    CheckError("hitSetDevice",ret);

    gpu_create_events();

    gpu_auto_detect();
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
    if (!QDP_isInitialized())
      {
	//std::cout << "host free, QDP not initialized" << std::endl;
	return;
      }
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
    if (!QDP_isInitialized())
      {
	//std::cout << "GPU free, QDP not initialized" << std::endl;
	return;
      }
    hipError_t ret;
    ret = hipFree( (void*)mem );
    CheckError("hipFree",ret);
  }



  void gpu_memset( void * dest , unsigned char val , size_t N )
  {
    hipError_t ret;
    ret = hipMemsetD8( dest , val , N );
    CheckError("hipMemset",ret);
  }


  std::string gpu_get_arch()
  {
    if( gcnArch == 910 ) {
      return "gfx90a";
    }
    else { 
      return "gfx" + std::to_string(gcnArch);
    }
  }



  bool get_jitf( JitFunction& func, const std::string& shared , const std::string& kernel_name , const std::string& pretty , const std::string& str_compute )
  {
    hipModule_t module;
    hipError_t ret;

#ifdef QDP_USE_ROCM_STATS
    if (gpu_get_record_stats())
      {
	comgr( func , shared , kernel_name );
      }
#endif
    
    func.set_kernel_name( kernel_name );
    func.set_pretty( pretty );

    ret = hipModuleLoadData(&module, shared.data() );
    if (ret != hipSuccess)
      {
	CheckErrorWeak( "hipModuleLoadData" , ret );
	return false;
      }

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "shared object file loaded as hip module\n";
	QDPIO::cout << "looking for a function with name " << kernel_name << "\n";
      }
    
    hipFunction_t f;
    ret = hipModuleGetFunction( &f , module , kernel_name.c_str() );
    CheckError( "hipModuleGetFunction" , ret );

    func.set_function( f );
    
    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "Got function!\n";
      }
    return true;
  }



  void gpu_sync()
  {
    hipError_t ret;
    ret = hipStreamSynchronize(NULL);
    
    if (ret != hipSuccess)
      {
	std::cout << "error on hipCtxSynchronize\n";
	QDP_abort(1);
      }
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


