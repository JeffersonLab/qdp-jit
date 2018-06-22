#include "qdp.h"


namespace QDP {

  void
  function_sum_convert_ind_exec( CUfunction function, 
				 int size, int threads, int blocks, int shared_mem_usage,
				 void *d_idata, void *d_odata, void *siteTable)
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";
    
    // lo <= idx < hi
    int lo = 0;
    int hi = size;
    //int do_soffset_index = 1;


    //QDP_get_global_cache().printLockSets();

    std::vector<void*> addr;

    addr.push_back( &lo );
    //std::cout << "addr lo =" << addr[0] << "\n";

    addr.push_back( &hi );
    //std::cout << "addr hi =" << addr[1] << "\n";

    //addr.push_back( &do_soffset_index );
    //std::cout << "addr do_soffset_index =" << addr[2] << " " << do_soffset_index << "\n";

    addr.push_back( &siteTable );
    //std::cout << "addr soffsetsDev =" << addr[3] << " " << soffsetsDev << "\n";

    addr.push_back( &d_idata );

    addr.push_back( &d_odata );

    kernel_geom_t now = getGeom( hi-lo , threads );

    //QDP_info("launing block=(%d,1,1)  grid=(%d,%d,1)",threads,now.Nblock_x,now.Nblock_y);

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threads,1,1,    shared_mem_usage, 0, &addr[0] , 0);
  }



  void
  function_sum_convert_exec( CUfunction function, 
			     int size, int threads, int blocks, int shared_mem_usage,
			     void *d_idata, void *d_odata)
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";
    
    // lo <= idx < hi
    int lo = 0;
    int hi = size;
    //int do_soffset_index = 1;


    //QDP_get_global_cache().printLockSets();

    std::vector<void*> addr;

    addr.push_back( &lo );
    //std::cout << "addr lo =" << addr[0] << "\n";

    addr.push_back( &hi );
    //std::cout << "addr hi =" << addr[1] << "\n";

    //addr.push_back( &do_soffset_index );
    //std::cout << "addr do_soffset_index =" << addr[2] << " " << do_soffset_index << "\n";

    //addr.push_back( &siteTable );
    //std::cout << "addr soffsetsDev =" << addr[3] << " " << soffsetsDev << "\n";

    addr.push_back( &d_idata );

    addr.push_back( &d_odata );

    kernel_geom_t now = getGeom( hi-lo , threads );

    //QDP_info("launing block=(%d,1,1)  grid=(%d,%d,1)",threads,now.Nblock_x,now.Nblock_y);

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threads,1,1,    shared_mem_usage, 0, &addr[0] , 0);
  }



  void
  function_sum_exec( CUfunction function, 
		     int size, int threads, int blocks, int shared_mem_usage,
		     void *d_idata, void *d_odata)
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

    //  QDP_info("function_sum_ind_coal_exec size=%d threads=%d blocks=%d shared_mem=%d idata=%p odata=%p ",	   size,threads,blocks,shared_mem_usage,d_idata,d_odata);

    // lo <= idx < hi
    int lo = 0;
    int hi = size;
    //int do_soffset_index = 0;
    //void *dummy_ptr = NULL;

    //QDP_get_global_cache().printLockSets();

    std::vector<void*> addr;

    addr.push_back( &lo );
    //std::cout << "addr lo =" << addr[0] << "\n";

    addr.push_back( &hi );
    //std::cout << "addr hi =" << addr[1] << "\n";

    //addr.push_back( &do_soffset_index );
    //std::cout << "addr do_soffset_index =" << addr[2] << " " << do_soffset_index << "\n";

    //addr.push_back( &dummy_ptr );
    //std::cout << "addr soffsetsDev =" << addr[3] << " " << soffsetsDev << "\n";

    addr.push_back( &d_idata );

    addr.push_back( &d_odata );

    kernel_geom_t now = getGeom( hi-lo , threads );

    //QDP_info("launing block=(%d,1,1)  grid=(%d,%d,1)",threads,now.Nblock_x,now.Nblock_y);

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threads,1,1,    shared_mem_usage, 0, &addr[0] , 0);
  }



  void function_global_max_exec( CUfunction function, 
				 int size, int threads, int blocks, int shared_mem_usage,
				 void *d_idata, void *d_odata)
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

    //QDP_info("function_global_max_exec size=%d threads=%d blocks=%d shared_mem=%d idata=%p odata=%p ",	   size,threads,blocks,shared_mem_usage,d_idata,d_odata);

    int lo = 0;
    int hi = size;

    std::vector<void*> addr;

    addr.push_back( &lo );
    //std::cout << "addr lo =" << addr[0] << "\n";

    addr.push_back( &hi );
    //std::cout << "addr hi =" << addr[1] << "\n";

    addr.push_back( &d_idata );

    addr.push_back( &d_odata );

    kernel_geom_t now = getGeom( hi-lo , threads );

    //QDP_info("launing block=(%d,1,1)  grid=(%d,%d,1)",threads,now.Nblock_x,now.Nblock_y);

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threads,1,1,    shared_mem_usage, 0, &addr[0] , 0);
  }

}

