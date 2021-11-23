// -*- c++ -*-



#include "qdp_config_internal.h" 
#include "qdp.h"

#include <iostream>
#include <string>
#include <iterator>
#include <unistd.h>



namespace QDP {


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
  }

  
  std::string gpu_get_arch()
  {
    return "dummy";
  }


  void gpu_set_device(int dev)
  {
  }


  void gpu_get_device_props()
  {
  }


  
  void gpu_auto_detect()
  {
  }


  int gpu_get_device_count()
  {
    return 1;
  }

  
  int roundUp(int numToRound, int multiple) 
  {
    assert(multiple && ((multiple & (multiple - 1)) == 0));
    return (numToRound + multiple - 1) & -multiple;
  }

  
  void gpu_host_alloc(void **mem , const size_t size)
  {
    *mem = aligned_alloc( 64 , roundUp( size , 64 ) );
    //*mem = malloc( size );

    if (*mem == NULL)
      {
	std::cout << "Unable to allocate " << size << " bytes." << std::endl;
	QDP_error_exit("out of memory");
      }
  }


  void gpu_host_free(void *mem)
  {
    free( mem );
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
    gpu_host_alloc( mem , size );
    return true;
  }

  
  void gpu_free(const void *mem )
  {
    gpu_host_free( (void*)mem );
  }



  void gpu_memset( void * dest , unsigned char val , size_t N )
  {
    memset( dest , val , N );
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



  void gpu_set_default_GPU(int ngpu)
  {
  }


  int  gpu_get_default_GPU() { return 0; }
  
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


