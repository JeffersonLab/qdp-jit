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
    ze_device_handle_t             hDevice;
    ze_driver_handle_t             hDriver;
    ze_context_handle_t            hContext;
    uint32_t                       deviceCount = 0;

    ze_command_list_handle_t       cmdList;

    ze_device_properties_t         deviceProperties;
    ze_device_compute_properties_t deviceComputeProperties;

    ze_event_pool_handle_t hEventPool;
    ze_event_handle_t ev0;
    ze_event_handle_t ev1;
    
    std::vector<ze_command_queue_group_properties_t> queueProperties;

    void *globalTsStart, *globalTsEnd;
    
  }

  void gpu_auto_detect();


  void gpu_create_events()
  {
    size_t tsAllocSize = 64;

    globalTsStart = nullptr;
    globalTsEnd = nullptr;

    ze_device_mem_alloc_desc_t deviceDesc;
    deviceDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    //deviceDesc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED;
    deviceDesc.pNext = nullptr;
    deviceDesc.flags = 0;
    deviceDesc.ordinal = 0;

    VALIDATECALL(zeMemAllocDevice(hContext, &deviceDesc, tsAllocSize, 8, hDevice, &globalTsStart));
    VALIDATECALL(zeMemAllocDevice(hContext, &deviceDesc, tsAllocSize, 8, hDevice, &globalTsEnd));
  }

  
  void gpu_record_start()
  {
    VALIDATECALL(zeCommandListAppendBarrier( cmdList, nullptr, 0, nullptr));
    VALIDATECALL(zeCommandListAppendWriteGlobalTimestamp(cmdList, (uint64_t *)globalTsStart, nullptr, 0, nullptr));
  }

  
  float gpu_record_stop_sync_time()
  {
    uint64_t tsStartResult = 0, tsEndResult = 0;

    VALIDATECALL(zeCommandListAppendBarrier( cmdList, nullptr, 0, nullptr));
    VALIDATECALL(zeCommandListAppendWriteGlobalTimestamp(cmdList, (uint64_t *)globalTsEnd, nullptr, 0, nullptr));
    VALIDATECALL(zeCommandListAppendBarrier( cmdList, nullptr, 0, nullptr));

    VALIDATECALL(zeCommandListAppendMemoryCopy(cmdList, &tsStartResult, globalTsStart, sizeof(tsStartResult), nullptr, 0, nullptr));
    VALIDATECALL(zeCommandListAppendBarrier( cmdList, nullptr, 0, nullptr));
    VALIDATECALL(zeCommandListAppendMemoryCopy(cmdList, &tsEndResult, globalTsEnd, sizeof(tsEndResult), ev0 , 0, nullptr));

    VALIDATECALL(zeEventHostSynchronize( ev0 , UINT64_MAX));

    uint64_t commandDuration = tsEndResult - tsStartResult;
    uint64_t timerResolution = deviceProperties.timerResolution;

    return (float)((commandDuration * timerResolution) / 1000. / 1000.);
  }



  size_t gpu_mem_free()
  {
    // Discover memories on device
    uint32_t memCount = 0;
    zeDeviceGetMemoryProperties(hDevice, &memCount, nullptr);

    if (memCount != 1)
      {
	std::cout << "zeDeviceGetMemoryProperties: expected count of 1\n";
	std::cout << "count = " << memCount << "\n";
	exit(1);
      }

    // Allocate properties structs
    ze_device_memory_properties_t* pMemProps = (ze_device_memory_properties_t*)malloc(memCount*sizeof(ze_device_memory_properties_t));
    ze_device_memory_ext_properties_t* pExtMemProps = (ze_device_memory_ext_properties_t*)malloc(memCount*sizeof(ze_device_memory_ext_properties_t));
    
    // Make pNext in memProps point at corresponding extMemProps
    for (uint32_t i = 0; i < memCount; ++i) {
      pMemProps[i].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
      pMemProps[i].pNext = &pExtMemProps[i];
      pExtMemProps[i].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_EXT_PROPERTIES;
      pExtMemProps[i].pNext = nullptr;
    }

    // Obtain memory & extended memory properties
    //std::cout << "calling zeDeviceGetMemoryProperties\n";
    zeDeviceGetMemoryProperties(hDevice, &memCount, pMemProps);
    //std::cout << "done calling\n";
    
    //std::cout << "Memory properties:\n";
    //std::cout << "flags = " << pMemProps[0].flags << "\n";
    //std::cout << "clock = " << pMemProps[0].maxClockRate << "\n";
    //std::cout << "maxBusWidth = " << pMemProps[0].maxBusWidth << "\n";
    //std::cout << "totalSize = " << pMemProps[0].totalSize << "\n";

    return pMemProps[0].totalSize;
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
			       unsigned int  sharedMemBytes, QDPCache::KernelArgs_t kernelArgs , bool set_l0_event )
  {
    VALIDATECALL(zeCommandListAppendBarrier( cmdList, nullptr, 0, nullptr));

    JitResult ret = JitResult::JitSuccess;

    ze_group_count_t dispatch;
    dispatch.groupCountX = gridDimX;
    dispatch.groupCountY = gridDimY;
    dispatch.groupCountZ = 1;
    
    for ( int i = 0 ; i < kernelArgs.size() ; ++i )
      {
	VALIDATECALL(zeKernelSetArgumentValue( (ze_kernel_handle_t)f.get_function() , i , kernelArgs.at(i).first , kernelArgs.at(i).second ));
      }

    VALIDATECALL(zeKernelSetGroupSize( (ze_kernel_handle_t)f.get_function() , blockDimX , blockDimY , blockDimZ ));


    if (set_l0_event)
      {
	VALIDATECALL(zeCommandListAppendLaunchKernel( cmdList , (ze_kernel_handle_t)f.get_function() , &dispatch , ev1 , 0 , nullptr ));
      }
    else
      {
	VALIDATECALL(zeCommandListAppendLaunchKernel( cmdList , (ze_kernel_handle_t)f.get_function() , &dispatch , nullptr , 0 , nullptr ));
      }

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
    VALIDATECALL(zeInit(0));


    uint32_t driverCount = 0;
    VALIDATECALL(zeDriverGet(&driverCount, nullptr));

    if (driverCount < 1)
      {
	QDPIO::cerr << "Expected driver count of at least 1\n";
	QDP_abort(1);
      }

    ze_driver_handle_t* allDrivers = (ze_driver_handle_t*)malloc(driverCount * sizeof(ze_driver_handle_t));
    VALIDATECALL(zeDriverGet(&driverCount, allDrivers));

    // Always use the first driver (for now)
    hDriver = allDrivers[0];
    hDevice = nullptr;


    // Create context
    ze_context_desc_t ctxtDesc = {
      ZE_STRUCTURE_TYPE_CONTEXT_DESC,
      nullptr,
      0
    };
    VALIDATECALL(zeContextCreate(hDriver, &ctxtDesc, &hContext));

    
    VALIDATECALL(zeDeviceGet( hDriver, &deviceCount, nullptr));

    if (deviceCount < 1)
      {
	QDPIO::cerr << "Expected device count of at least 1\n";
	QDP_abort(1);
      }

    free(allDrivers);
  }

  

  void gpu_done()
  {
    VALIDATECALL(zeEventDestroy( ev0 ));
    VALIDATECALL(zeEventDestroy( ev1 ));
    VALIDATECALL(zeEventPoolDestroy(hEventPool));
  }


  
  std::string gpu_get_arch()
  {
    return "dummy";
  }

  

  void gpu_set_device(int dev)
  {
    if (deviceCount <= dev)
      {
	std::cout << "Devices found: " << deviceCount << std::endl;
	std::cout << "Trying to use device: " << dev << std::endl;
	exit(1);
      }

    ze_device_handle_t* allDevices = (ze_device_handle_t*)malloc(deviceCount * sizeof(ze_device_handle_t));
    zeDeviceGet(hDriver, &deviceCount, allDevices);

    ze_device_properties_t device_properties;
    zeDeviceGetProperties(allDevices[dev], &device_properties);
      
    if (device_properties.type != ZE_DEVICE_TYPE_GPU) {
      QDPIO::cerr << "Device " << dev << " is not of GPU type. Confused." << std::endl;
      QDP_abort(1);
    }

    hDevice = allDevices[dev];

    free(allDevices);


    // Discover all command queue groups
    uint32_t cmdqueueGroupCount = 0;
    zeDeviceGetCommandQueueGroupProperties(hDevice, &cmdqueueGroupCount, nullptr);

    //std::cout << "cmdqueueGroupCount: " << cmdqueueGroupCount << std::endl;
    
    ze_command_queue_group_properties_t* cmdqueueGroupProperties = (ze_command_queue_group_properties_t*)malloc(cmdqueueGroupCount * sizeof(ze_command_queue_group_properties_t));
    zeDeviceGetCommandQueueGroupProperties(hDevice, &cmdqueueGroupCount, cmdqueueGroupProperties);
    
    // Find a command queue type that support compute
    uint32_t computeQueueGroupOrdinal = cmdqueueGroupCount;
    for( uint32_t i = 0; i < cmdqueueGroupCount; ++i ) {
      if( cmdqueueGroupProperties[ i ].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE ) {
        computeQueueGroupOrdinal = i;
        break;
      }
    }

    
    if(computeQueueGroupOrdinal == cmdqueueGroupCount)
      {
	QDPIO::cerr << "no compute queues found" << std::endl;
	QDP_abort(1);
      }

	  
    // Create an immediate command list
    ze_command_queue_desc_t commandQueueDesc = {
      ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
      nullptr,
      computeQueueGroupOrdinal,
      0, // index
      0, // flags
      ZE_COMMAND_QUEUE_MODE_DEFAULT,
      ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    VALIDATECALL(zeCommandListCreateImmediate(hContext, hDevice, &commandQueueDesc, &cmdList));

    

    ze_event_pool_desc_t eventPoolDesc = {
      ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
      nullptr,
      ZE_EVENT_POOL_FLAG_HOST_VISIBLE, // all events in pool are visible to Host
      2 // count
    };
    VALIDATECALL(zeEventPoolCreate(hContext, &eventPoolDesc, 0, nullptr, &hEventPool));

    //ZE_EVENT_SCOPE_FLAG_DEVICE & ZE_EVENT_SCOPE_FLAG_HOST,
    //ZE_EVENT_SCOPE_FLAG_DEVICE & ZE_EVENT_SCOPE_FLAG_HOST

    ze_event_desc_t eventDesc = {
      ZE_STRUCTURE_TYPE_EVENT_DESC,
      nullptr,
      0, // index
      ZE_EVENT_SCOPE_FLAG_HOST,
      ZE_EVENT_SCOPE_FLAG_HOST
    };
    VALIDATECALL(zeEventCreate(hEventPool, &eventDesc, &ev0));


    eventDesc = {
      ZE_STRUCTURE_TYPE_EVENT_DESC,
      nullptr,
      1, // index
      ZE_EVENT_SCOPE_FLAG_HOST,
      ZE_EVENT_SCOPE_FLAG_HOST
    };
    VALIDATECALL(zeEventCreate(hEventPool, &eventDesc, &ev1));
    
    gpu_create_events();
    
    gpu_auto_detect();
  }



  
  void gpu_get_device_props()
  {
  }


  
  void gpu_auto_detect()
  {
    VALIDATECALL(zeDeviceGetProperties(hDevice, &deviceProperties));

#if 0
    QDPIO::cout << "Device                         : " << deviceProperties.name << "\n" 
              << "Type                           : " << ((deviceProperties.type == ZE_DEVICE_TYPE_GPU) ? "GPU" : "FPGA") << "\n"
              << "Vendor ID                      : " << std::hex << deviceProperties.vendorId << std::dec << "\n"
	      << "Max alloc                      : " << deviceProperties.maxMemAllocSize << "\n"
	      << "Number of threads per EU       : " << deviceProperties.numThreadsPerEU << "\n"
	      << "The physical EU SIMD width     : " << deviceProperties.physicalEUSimdWidth << "\n"
	      << "Number of EUs per sub-slice    : " << deviceProperties.numEUsPerSubslice << "\n"
	      << "Number of sub-slices per slice : " << deviceProperties.numSubslicesPerSlice << "\n"
	      << "Number of slices               : " << deviceProperties.numSlices << "\n"
      ;
#endif

    VALIDATECALL(zeDeviceGetComputeProperties(hDevice, &deviceComputeProperties));
#if 0
    QDPIO::cout << "Maximum group size total       : " << deviceComputeProperties.maxTotalGroupSize << "\n" ;
    QDPIO::cout << "Maximum group size X           : " << deviceComputeProperties.maxGroupSizeX << "\n" ;
    QDPIO::cout << "Maximum group size Y           : " << deviceComputeProperties.maxGroupSizeY << "\n" ;
    QDPIO::cout << "Maximum group size Z           : " << deviceComputeProperties.maxGroupSizeZ << "\n" ;
    QDPIO::cout << "Maximum group count X          : " << deviceComputeProperties.maxGroupCountX << "\n" ;
    QDPIO::cout << "Maximum group count Y          : " << deviceComputeProperties.maxGroupCountY << "\n" ;
    QDPIO::cout << "Maximum group count Z          : " << deviceComputeProperties.maxGroupCountZ << "\n" ;
    QDPIO::cout << "Maximum shared local memory    : " << deviceComputeProperties.maxSharedLocalMemory << "\n" ;
    QDPIO::cout << "Maximum Number of subgroups    : " << deviceComputeProperties.numSubGroupSizes << "\n" ;
#endif
  }



  int gpu_get_device_count()
  {
    return deviceCount;
  }

  
  int roundUp(int numToRound, int multiple) 
  {
    assert(multiple && ((multiple & (multiple - 1)) == 0));
    return (numToRound + multiple - 1) & -multiple;
  }

  
  void gpu_host_alloc(void **mem , const size_t size)
  {

    ze_host_mem_alloc_desc_t hostDesc;
    hostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    hostDesc.pNext = nullptr;
    hostDesc.flags = 0;
    
    VALIDATECALL(zeMemAllocHost(hContext, &hostDesc, size , jit_config_get_pool_alignment() , mem));

    //std::cout << "gpu_host_alloc size = " << size << "  addr = " << *mem << "\n";
  }


  void gpu_host_free(void *mem)
  {
    VALIDATECALL(zeMemFree(hContext, (void*)mem));
  }


  void gpu_wait_l0_event()
  {
    VALIDATECALL(zeEventHostSynchronize(ev1, UINT64_MAX));
    VALIDATECALL(zeEventHostReset( ev1 ));
  }
  

  void gpu_memcpy_h2d( void * dest , const void * src , size_t size )
  {
    VALIDATECALL(zeCommandListAppendBarrier( cmdList, nullptr, 0, nullptr));

    VALIDATECALL(zeCommandListAppendMemoryCopy( cmdList , dest , src , size , ev0 , 0 , nullptr ));

    VALIDATECALL(zeEventHostSynchronize(ev0, UINT64_MAX));
    VALIDATECALL(zeEventHostReset( ev0 ));
  }


  void gpu_memcpy_d2h( void * dest , const void * src , size_t size )
  {
    VALIDATECALL(zeCommandListAppendBarrier( cmdList, nullptr, 0, nullptr));

    VALIDATECALL(zeCommandListAppendMemoryCopy( cmdList , dest , src , size , ev0 , 0 , nullptr ));

    VALIDATECALL(zeEventHostSynchronize( ev0 , UINT64_MAX));
    VALIDATECALL(zeEventHostReset( ev0 ));
  }

  


  bool gpu_malloc(void **mem , size_t size )
  {
    //std::cout << __PRETTY_FUNCTION__ << " size = " << size << "\n";

    ze_device_mem_alloc_desc_t memAllocDesc;
    memAllocDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    memAllocDesc.pNext = nullptr;
    memAllocDesc.flags = 0;
    memAllocDesc.ordinal = 0;

    ze_relaxed_allocation_limits_exp_desc_t relaxed;
    relaxed.stype = ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC;
    relaxed.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
 
    if ( size > deviceProperties.maxMemAllocSize )
      {
	std::cout << "L0 performance warning: memory allocation size larger than maximum allowed: " << size << " (" << deviceProperties.maxMemAllocSize << ").\n";
	memAllocDesc.pNext = &relaxed;

	std::cout << "bit ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE set\n";
	memAllocDesc.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
      }

    VALIDATECALL(zeMemAllocDevice( hContext, &memAllocDesc, size, jit_config_get_pool_alignment() , hDevice, mem));

    return true;
  }


#if defined (QDP_ENABLE_MANAGED_MEMORY)
  void gpu_memcpy( void * dest , const void * src , size_t size )
  {
    VALIDATECALL(zeCommandListAppendBarrier( cmdList, nullptr, 0, nullptr));

    VALIDATECALL(zeCommandListAppendMemoryCopy( cmdList , dest , src , size , ev0 , 0 , nullptr ));

    VALIDATECALL(zeEventHostSynchronize( ev0 , UINT64_MAX));
    VALIDATECALL(zeEventHostReset( ev0 ));
  }

  
  bool gpu_malloc_managed(void **mem , size_t size )
  {
    ze_device_mem_alloc_desc_t memAllocDesc;
    memAllocDesc.flags = 0;
    memAllocDesc.pNext = nullptr;
    memAllocDesc.ordinal = 0;
    
    ze_host_mem_alloc_desc_t hostDesc;
    hostDesc.flags = 0;
    hostDesc.pNext = nullptr;

    VALIDATECALL(zeMemAllocShared( hContext, &memAllocDesc, &hostDesc , size , jit_config_get_pool_alignment() , hDevice, mem));

    return true;
  }
#endif

  
  
  void gpu_free(const void *mem )
  {
    VALIDATECALL(zeMemFree(hContext, (void*)mem));
  }


  void gpu_memset( void * dest , unsigned char val , size_t N )
  {
    VALIDATECALL(zeCommandListAppendBarrier( cmdList, nullptr, 0, nullptr));
    VALIDATECALL(zeCommandListAppendMemoryFill( cmdList , dest , &val , 1 , N , nullptr , 0 , nullptr ));
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

    ze_module_desc_t moduleDesc;
    ze_module_build_log_handle_t buildLog;

    moduleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    moduleDesc.pNext = nullptr;
    moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    moduleDesc.inputSize = length;
    moduleDesc.pInputModule = reinterpret_cast<const uint8_t *>(spirvInput.get());
    //moduleDesc.pBuildFlags = "-vc-codegen";
    moduleDesc.pBuildFlags = "";
    moduleDesc.pConstants = nullptr;

    if (jit_config_get_verbose_output())
      {
	QDPIO::cout << "get_jitf module create ...\n";
      }

    ze_result_t status = zeModuleCreate(hContext, hDevice, &moduleDesc, &module, &buildLog);

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

    ze_kernel_desc_t kernelDesc;
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

    
    if ( Layout::primaryNode() )
      {
        ze_kernel_properties_t pKernelProperties;
        pKernelProperties.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
        pKernelProperties.pNext = nullptr;
        
        VALIDATECALL(zeKernelGetProperties( kernel , &pKernelProperties ));

        func.set_regs ( pKernelProperties.privateMemSize );
        func.set_stack( pKernelProperties.spillMemSize );
        func.set_cmem ( pKernelProperties.localMemSize );
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


