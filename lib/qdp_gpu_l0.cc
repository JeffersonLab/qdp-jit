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
    ze_driver_handle_t        driverHandle;
    ze_context_handle_t       context;
    ze_device_handle_t        device;
    ze_command_queue_handle_t cmdQueue;
    ze_command_list_handle_t  cmdList;
    ze_device_properties_t    deviceProperties = {};

    std::vector<ze_command_queue_group_properties_t> queueProperties;
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
    return ret;
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

    ze_command_queue_desc_t cmdQueueDesc = {};
    for (uint32_t i = 0; i < numQueueGroups; i++) { 
        if (queueProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            cmdQueueDesc.ordinal = i;
        }
    }

    std::cout << "zeCommandQueueCreate\n";
    cmdQueueDesc.index = 0;
    cmdQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    VALIDATECALL(zeCommandQueueCreate(context, device, &cmdQueueDesc, &cmdQueue));

    std::cout << "zeCommandListCreate\n";
    // Create a command list
    ze_command_list_desc_t cmdListDesc = {};
    cmdListDesc.commandQueueGroupOrdinal = cmdQueueDesc.ordinal;    
    VALIDATECALL(zeCommandListCreate(context, device, &cmdListDesc, &cmdList));

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
    std::cout << "Device   : " << deviceProperties.name << "\n" 
              << "Type     : " << ((deviceProperties.type == ZE_DEVICE_TYPE_GPU) ? "GPU" : "FPGA") << "\n"
              << "Vendor ID: " << std::hex << deviceProperties.vendorId << std::dec << "\n"
	      << "Max alloc: " << deviceProperties.maxMemAllocSize << "\n";

    
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
    memcpy(dest,src,size);
  }

  
  void gpu_memcpy_d2h( void * dest , const void * src , size_t size )
  {
    memcpy(dest,src,size);
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




  bool get_jitf( JitFunction& func, const std::string& kernel_ptx , const std::string& kernel_name , const std::string& pretty , const std::string& str_compute )
  {
    func.set_kernel_name( kernel_name );
    func.set_pretty( pretty );


    
    return true;
  }


  size_t gpu_getMaxGridX()  {return 0;}
  size_t gpu_getMaxGridY()  {return 0;}
  size_t gpu_getMaxGridZ()  {return 0;}

  size_t gpu_getMaxBlockX()  {return 0;}
  size_t gpu_getMaxBlockY()  {return 0;}
  size_t gpu_getMaxBlockZ()  {return 0;}
  
  size_t gpu_getMaxSMem()  {return 0;}

  unsigned gpu_getMajor() { return 0; }
  unsigned gpu_getMinor() { return 0; }

  
} // QDP


