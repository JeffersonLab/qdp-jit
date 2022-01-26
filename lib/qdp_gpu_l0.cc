// -*- c++ -*-

#include "ze_api.h"

#include "qdp_config_internal.h" 
#include "qdp.h"

#include <iostream>
#include <string>
#include <iterator>
#include <unistd.h>



#define VALIDATECALL(myZeCall) \
  if (myZeCall != ZE_RESULT_SUCCESS){  \
    std::cout << "Error at "	       \
	      << #myZeCall << ": "     \
	      << __FUNCTION__ << ": "  \
	      << __LINE__ << std::endl;	      \
    std::cout << "Exit with Error Code: "     \
	      << "0x" << std::hex	      \
	      << myZeCall		      \
	      << std::dec << std::endl;	      \
    std::terminate();			      \
  }


namespace QDP {

  namespace {
    ze_driver_handle_t             driverHandle;
    ze_context_handle_t            context;
    ze_device_handle_t             device;
#if 0
    ze_command_queue_desc_t        cmdQueueDesc;
    ze_command_queue_handle_t      cmdQueue;
    ze_command_list_handle_t       cmdList;
#else
    ze_command_list_handle_t       cmdList = {};
#endif
    ze_device_properties_t         deviceProperties;
    ze_device_compute_properties_t deviceComputeProperties;

    std::vector<ze_command_queue_group_properties_t> queueProperties;

#if 0
    void gpu_cmd_Create()
    {
      std::cout << "zeCommandListCreate\n";
      ze_command_list_desc_t cmdListDesc = {};
      cmdListDesc.commandQueueGroupOrdinal = cmdQueueDesc.ordinal;    
      VALIDATECALL(zeCommandListCreate(context, device, &cmdListDesc, &cmdList));
    }

    void gpu_cmd_CloseExeDestroy()
    {
      std::cout << "zeCommandListClose\n";
      VALIDATECALL(zeCommandListClose(cmdList));

      std::cout << "zeCommandListExe\n";
      VALIDATECALL(zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, nullptr));
      
      std::cout << "zeCommandListSync\n";
      VALIDATECALL(zeCommandQueueSynchronize(cmdQueue, std::numeric_limits<uint64_t>::max())); //UINT32_MAX

      std::cout << "zeCommandListDestroy\n";
      VALIDATECALL(zeCommandListDestroy(cmdList));
    }
#else
    void gpu_cmd_Create()
    {
    }

    void gpu_cmd_CloseExeDestroy()
    {
    }
#endif

    
  }

  void gpu_auto_detect();


  void gpu_create_events()
  {
  }

  void gpu_record_start()
  {
  }

  void gpu_record_stop()
  {
  }

  void gpu_event_sync()
  {
  }


  float gpu_get_time()
  {
    return 0.;
  }


  size_t gpu_mem_free()
  {
    return 0;
  }
  
  size_t gpu_mem_total()
  {
    return 0;
  }
  
  
  int CudaGetAttributesLocalSize( JitFunction& f )
  {
    return 0;
  }


  JitResult gpu_launch_kernel( JitFunction& f, 
			       unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
			       unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
			       unsigned int  sharedMemBytes, QDPCache::KernelArgs_t kernelArgs )
  {
    JitResult ret = JitResult::JitSuccess;

    typedef std::vector< std::pair< int , void* > > KernelArgs_t;

    std::cout << "kernel args:\n";
    for ( int i = 0 ; i < kernelArgs.size() ; ++i )
      {
	std::cout << i << ": " << kernelArgs.at(i).first << " " << kernelArgs.at(i).second << "\n";
	VALIDATECALL(zeKernelSetArgumentValue( (ze_kernel_handle_t)f.get_function() , i , kernelArgs.at(i).first , kernelArgs.at(i).second ));
      }

    VALIDATECALL(zeKernelSetGroupSize( (ze_kernel_handle_t)f.get_function() , blockDimX , blockDimY , blockDimZ ));

    ze_group_count_t dispatch;
    dispatch.groupCountX = gridDimX;
    dispatch.groupCountY = gridDimY;
    dispatch.groupCountZ = 1;

    std::cout << "grid =(" << dispatch.groupCountX << "," << dispatch.groupCountY << "," << dispatch.groupCountZ << ")\n";
    std::cout << "block=(" << blockDimX << "," << blockDimY << "," << blockDimZ << ")\n";

    gpu_cmd_Create();

    // Launch kernel on the GPU
    VALIDATECALL(zeCommandListAppendLaunchKernel( cmdList , (ze_kernel_handle_t)f.get_function() , &dispatch, nullptr, 0, nullptr));

    gpu_cmd_CloseExeDestroy();

    return ret;
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


    
  void gpu_init()
  {
    std::cout << __PRETTY_FUNCTION__ << "\n";
    // Initialization
    std::cout << "zeInit\n";
    VALIDATECALL(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    std::cout << "zeDriverGet\n";
    // Get the driver
    uint32_t driverCount = 0;
    VALIDATECALL(zeDriverGet(&driverCount, nullptr));

    std::cout << "driverCount = " << driverCount << "\n";
    
    std::cout << "zeDriverGet\n";
    VALIDATECALL(zeDriverGet(&driverCount, &driverHandle));

    std::cout << "zuContextCreate\n";
    // Create the context
    ze_context_desc_t contextDescription = {};
    contextDescription.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    VALIDATECALL(zeContextCreate(driverHandle, &contextDescription, &context));
  }

  
  std::string gpu_get_arch()
  {
    return "dummy";
  }


  void gpu_set_device(int dev)
  {
    std::cout << __PRETTY_FUNCTION__ << "\n";
    unsigned int d = dev;
    
    // Level Zero device ordinals are natural numbers
    d += 1;

    std::cout << "zeDeviceGet\n";
    VALIDATECALL(zeDeviceGet(driverHandle, &d, &device));

#if 0
    std::cout << "zeDeviceGetCommandQueueGroupProperties\n";
    // Create a command queue
    uint32_t numQueueGroups = 0;
    VALIDATECALL(zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, nullptr));
    if (numQueueGroups == 0) {
        std::cout << "No queue groups found\n";
        std::terminate();
    } else {
        std::cout << "#Queue Groups: " << numQueueGroups << std::endl;
    }

    std::cout << "zeDeviceGetCommandQueueGroupProperties\n";
    queueProperties.resize(numQueueGroups);
    VALIDATECALL(zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, queueProperties.data()));

    for (uint32_t i = 0; i < numQueueGroups; i++) { 
        if (queueProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            cmdQueueDesc.ordinal = i;
        }
    }

    std::cout << "zeCommandQueueCreate\n";
    cmdQueueDesc.index = 0;
    cmdQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    VALIDATECALL(zeCommandQueueCreate(context, device, &cmdQueueDesc, &cmdQueue));
#else
    // Create an immediate command list for direct submission
    ze_command_queue_desc_t altdesc = {};
    altdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    VALIDATECALL( zeCommandListCreateImmediate(context, device, &altdesc, &cmdList) );
#endif

    gpu_auto_detect();

    std::cout << "done\n";
  }


  void gpu_get_device_props()
  {
    std::cout << __PRETTY_FUNCTION__ << "\n";
  }


  
  void gpu_auto_detect()
  {
    std::cout << __PRETTY_FUNCTION__ << "\n";

    VALIDATECALL(zeDeviceGetProperties(device, &deviceProperties));
    std::cout << "Device                         : " << deviceProperties.name << "\n" 
              << "Type                           : " << ((deviceProperties.type == ZE_DEVICE_TYPE_GPU) ? "GPU" : "FPGA") << "\n"
              << "Vendor ID                      : " << std::hex << deviceProperties.vendorId << std::dec << "\n"
	      << "Max alloc                      : " << deviceProperties.maxMemAllocSize << "\n"
	      << "Number of threads per EU       : " << deviceProperties.numThreadsPerEU << "\n"
	      << "The physical EU SIMD width     : " << deviceProperties.physicalEUSimdWidth << "\n"
	      << "Number of EUs per sub-slice    : " << deviceProperties.numEUsPerSubslice << "\n"
	      << "Number of sub-slices per slice : " << deviceProperties.numSubslicesPerSlice << "\n"
	      << "Number of slices               : " << deviceProperties.numSlices << "\n"
      ;

    VALIDATECALL(zeDeviceGetComputeProperties(device, &deviceComputeProperties));
    std::cout << "Maximum group size total       : " << deviceComputeProperties.maxTotalGroupSize << "\n" ;
    std::cout << "Maximum group size X           : " << deviceComputeProperties.maxGroupSizeX << "\n" ;
    std::cout << "Maximum group size Y           : " << deviceComputeProperties.maxGroupSizeY << "\n" ;
    std::cout << "Maximum group size Z           : " << deviceComputeProperties.maxGroupSizeZ << "\n" ;
    std::cout << "Maximum group count X          : " << deviceComputeProperties.maxGroupCountX << "\n" ;
    std::cout << "Maximum group count Y          : " << deviceComputeProperties.maxGroupCountY << "\n" ;
    std::cout << "Maximum group count Z          : " << deviceComputeProperties.maxGroupCountZ << "\n" ;
    std::cout << "Maximum shared local memory    : " << deviceComputeProperties.maxSharedLocalMemory << "\n" ;
    std::cout << "Maximum Number of subgroups    : " << deviceComputeProperties.numSubGroupSizes << "\n" ;
      
  }


  int gpu_get_device_count()
  {
    std::cout << __PRETTY_FUNCTION__ << "\n";

    // Get the device
    uint32_t deviceCount = 0;
    VALIDATECALL(zeDeviceGet(driverHandle, &deviceCount, nullptr));

    std::cout << "deviceCount = " << deviceCount << "\n";
    return deviceCount;
  }

  
  int roundUp(int numToRound, int multiple) 
  {
    assert(multiple && ((multiple & (multiple - 1)) == 0));
    return (numToRound + multiple - 1) & -multiple;
  }

  
  void gpu_host_alloc(void **mem , const size_t size)
  {
    std::cout << __PRETTY_FUNCTION__ << " size = " << size << "\n";

    ze_host_mem_alloc_desc_t hostDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    //hostDesc.flags = ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED;

    VALIDATECALL(zeMemAllocHost(context, &hostDesc, size , 1, mem));
  }


  void gpu_host_free(void *mem)
  {
    std::cout << __PRETTY_FUNCTION__ << "\n";

    VALIDATECALL(zeMemFree(context, (void*)mem));
  }





  void gpu_memcpy_h2d( void * dest , const void * src , size_t size )
  {
    //std::cout << "gpu_memcpy_h2d: dest = " << dest << ", src = " << src << ", size = " << size << "\n";

    gpu_cmd_Create();
    
    VALIDATECALL(zeCommandListAppendMemoryCopy( cmdList , dest , src , size , nullptr , 0 , nullptr ));

    gpu_cmd_CloseExeDestroy();
  }

  
  void gpu_memcpy_d2h( void * dest , const void * src , size_t size )
  {
    //std::cout << "gpu_memcpy_d2h: dest = " << dest << ", src = " << src << ", size = " << size << "\n";

    gpu_cmd_Create();

    VALIDATECALL(zeCommandListAppendMemoryCopy( cmdList , dest , src , size , nullptr , 0 , nullptr ));

    gpu_cmd_CloseExeDestroy();
  }


  bool gpu_malloc(void **mem , size_t size )
  {
    std::cout << __PRETTY_FUNCTION__ << " size = " << size << "\n";

    if ( size > deviceProperties.maxMemAllocSize )
      {
	std::cout << "L0 error: memory allocation size larger than maximum allowed: " << size << " (" << deviceProperties.maxMemAllocSize << ").\n";
	QDP_abort(1);
      }
    
    ze_device_mem_alloc_desc_t memAllocDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    //memAllocDesc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED;
    memAllocDesc.ordinal = 0;


    VALIDATECALL(zeMemAllocDevice( context, &memAllocDesc, size, 1, device, mem));
    
    return true;
  }

  
  void gpu_free(const void *mem )
  {
    std::cout << __PRETTY_FUNCTION__ << "\n";

    VALIDATECALL(zeMemFree(context, (void*)mem));
  }

  void gpu_prefetch(void *mem,  size_t size)
  {
  }


  void gpu_memset( void * dest , unsigned char val , size_t N )
  {
    //memset( dest , val , N );
  }


  
  void gpu_sync()
  {
  }




  bool get_jitf( JitFunction& func, const std::string& fname_spirv , const std::string& kernel_name , const std::string& pretty , const std::string& str_compute )
  {
    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "get_jitf enter\n";
      }

    func.set_kernel_name( kernel_name );
    func.set_pretty( pretty );

    // Module Initialization
    ze_module_handle_t module = nullptr;
    ze_kernel_handle_t kernel = nullptr;

    std::ifstream file( fname_spirv.c_str() , std::ios::binary );

    if (!file.is_open())
      {
	std::cout << "Error reading file: " << fname_spirv << "\n";
	QDP_abort(1);
      }

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "get_jitf opened file\n";
      }

    file.seekg(0, file.end);
    auto length = file.tellg();
    file.seekg(0, file.beg);

    std::unique_ptr<char[]> spirvInput(new char[length]);
    file.read(spirvInput.get(), length);

    ze_module_desc_t moduleDesc = {};
    ze_module_build_log_handle_t buildLog;
    moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    moduleDesc.pInputModule = reinterpret_cast<const uint8_t *>(spirvInput.get());
    moduleDesc.inputSize = length;
    moduleDesc.pBuildFlags = "";

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "get_jitf module create ...\n";
      }

    ze_result_t status = zeModuleCreate(context, device, &moduleDesc, &module, &buildLog);

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "get_jitf module created ...\n";
      }
    if (status != ZE_RESULT_SUCCESS) {
      // print log
      size_t szLog = 0;
      zeModuleBuildLogGetString(buildLog, &szLog, nullptr);

      char* stringLog = (char*)malloc(szLog);
      zeModuleBuildLogGetString(buildLog, &szLog, stringLog);
      std::cout << "Build log: " << stringLog << std::endl;
    }
    VALIDATECALL(zeModuleBuildLogDestroy(buildLog));

    ze_kernel_desc_t kernelDesc = {};
    kernelDesc.pKernelName = kernel_name.c_str();
  
    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "get_jitf kernel create ...\n";
      }

    VALIDATECALL(zeKernelCreate(module, &kernelDesc, &kernel));
    
    func.set_function( (void*)kernel );
    
    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "Got function!\n";
      }
    
    return true;
  }

  
  size_t gpu_getMaxGridX()  {return deviceComputeProperties.maxGroupCountX;}
  size_t gpu_getMaxGridY()  {return deviceComputeProperties.maxGroupCountY;}
  size_t gpu_getMaxGridZ()  {return deviceComputeProperties.maxGroupCountZ;}

  size_t gpu_getMaxBlockX()  {return deviceComputeProperties.maxGroupSizeX;}
  size_t gpu_getMaxBlockY()  {return deviceComputeProperties.maxGroupSizeY;}
  size_t gpu_getMaxBlockZ()  {return deviceComputeProperties.maxGroupSizeZ;}
  
  size_t gpu_getMaxSMem()  {return deviceComputeProperties.maxSharedLocalMemory;}

  unsigned gpu_getMajor() { return 0; }
  unsigned gpu_getMinor() { return 0; }

  
} // QDP


