
/*! @file
 * @brief Parscalar init routines
 * 
 * Initialization routines for parscalar implementation
 */

#include <stdlib.h>
#include <unistd.h>

#include "qdp.h"


#include "qdp_init.h"

#if defined(QDP_USE_COMM_SPLIT_INIT)
#error "Trying to use MPI comm split with scalar architecture"
#endif

namespace QDP {

namespace COUNT {
  int count = 0;
}
	

  //! Private flag for status
  static bool isInit = false;
  bool setPoolSize = false;
  bool setGeomP = false;
  bool setIOGeomP = false;
  multi1d<int> logical_geom(Nd);   // apriori logical geometry of the machine
  multi1d<int> logical_iogeom(Nd); // apriori logical 	





  void QDP_startGPU()
  {
    // Getting GPU device properties
    //gpu_auto_detect();

    // Initialize the LLVM wrapper
    llvm_backend_init();
  }

  //! Set the GPU device
  int QDP_setGPU()
  {
    int deviceCount;
    //int ret = 0;
    CudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
      QDP_error_exit("No CUDA devices found");
    }

    // Try MVapich fist
    char *rank = getenv( "MV2_COMM_WORLD_LOCAL_RANK"  );

    // Try OpenMPI
    if( ! rank ) {
       rank = getenv( "OMPI_COMM_WORLD_LOCAL_RANK" );
    } 

    int dev=0;
    if (rank) {
      int local_rank = atoi(rank);
      dev = local_rank % deviceCount;
    } else {
      if ( gpu_getDefaultGPU() == -1 )
	{
	  std::cerr << "Couldnt determine local rank. Selecting device 0. In a multi-GPU per node run this is not what one wants.\n";
	  dev = 0;
	}
      else
	{
	  dev = gpu_getDefaultGPU();
	  std::cerr << "Couldnt determine local rank. Selecting device " << dev << " as per user request.\n";
	}
#if 0
      // we don't have an initialized QMP at this point
       std::cerr << "Couldnt determine local rank. Selecting device based on global rank \n";
       std::cerr << "Please ensure that ranks increase fastest within the node for this to work \n";
       int rank_QMP = QMP_get_node_number();
       dev = rank_QMP % deviceCount;
#endif
    }

    //std::cout << "Setting CUDA device to " << dev << "\n";
    CudaSetDevice( dev );
    return dev;
  }


  void QDP_initialize(int *argc, char ***argv) 
  {
    QDP_initialize_CUDA(argc, argv);
    QDP_setGPU();
    QDP_initialize_QMP(argc, argv);
    QDP_startGPU();
  }
	
  //! Turn on the machine
  void QDP_initialize_CUDA(int *argc, char ***argv)
  {
    if (sizeof(bool) != 1)
      {
	std::cout << "Error: sizeof(bool) == " << sizeof(bool) << "   (1 is required)" << endl;
	exit(1);
      }

    if (isInit)
      {
	QDPIO::cerr << "QDP already inited" << endl;
	QDP_abort(1);
      }


    //
    // CUDA init
    //
    gpu_init();

    //
    // Process command line
    //
		
    // Look for help
    bool help_flag = false;
    for (int i=0; i<*argc; i++) 
      {
	if (strcmp((*argv)[i], "-h")==0)
	  help_flag = true;
      }
		
    setGeomP = false;
    logical_geom = 0;
		
    setIOGeomP = false;
    logical_iogeom = 0;
		
#ifdef USE_REMOTE_QIO
    int rtiP = 0;
#endif
    const int maxlen = 256;
    char rtinode[maxlen];
    strncpy(rtinode, "your_local_food_store", maxlen);
		
    // Usage
    if (Layout::primaryNode())  {
      if (help_flag) 
	{
	  fprintf(stderr,"Usage:    %s options\n",(*argv)[0]);
	  fprintf(stderr,"options:\n");
	  fprintf(stderr,"    -h        help\n");
#if defined(QDP_USE_PROFILING)   
	  fprintf(stderr,"    -p        %%d [%d] profile level\n", 
		  getProfileLevel());
#endif
				
	  // logical geometry info
	  fprintf(stderr,"    -geom     %%d");
	  for(int i=1; i < Nd; i++) 
	    fprintf(stderr," %%d");
				
	  fprintf(stderr," [-1");
	  for(int i=1; i < Nd; i++) 
	    fprintf(stderr,",-1");
	  fprintf(stderr,"] logical machine geometry\n");
				
#ifdef USE_REMOTE_QIO
	  fprintf(stderr,"    -cd       %%s [.] set working dir for QIO interface\n");
	  fprintf(stderr,"    -rti      %%d [%d] use run-time interface\n", 
		  rtiP);
	  fprintf(stderr,"    -rtinode  %%s [%s] run-time interface fileserver node\n", 
		  rtinode);
#endif
				
	  QDP_abort(1);
	}
    }


    for (int i=1; i<*argc; i++) 
      {
	else if (strcmp((*argv)[i], "-gpudirect")==0) 
	  {
	    gpu_setGPUDirect(true);
	  }
	else if (strcmp((*argv)[i], "-envvar")==0) 
	  {
	    char buffer[1024];
	    sscanf((*argv)[++i],"%s",&buffer[0]);
	    gpu_setENVVAR(buffer);
	  }
	else if (strcmp((*argv)[i], "-poolsize")==0) 
	  {
	    float f;
	    char c;
	    sscanf((*argv)[++i],"%f%c",&f,&c);
	    double mul;
	    switch (tolower(c)) {
	    case 'k': 
	      mul=1024.; 
	      break;
	    case 'm': 
	      mul=1024.*1024; 
	      break;
	    case 'g': 
	      mul=1024.*1024*1024; 
	      break;
	    case 't':
	      mul=1024.*1024*1024*1024;
	      break;
	    case '\0':
	      break;
	    default:
	      QDP_error_exit("unknown multiplication factor");
	    }
	    size_t val = (size_t)((double)(f) * mul);

	    //CUDADevicePoolAllocator::Instance().setPoolSize(val);
	    //QDP_get_global_cache().get_allocator().setPoolSize(val);
	    QDP_get_global_cache().setPoolSize(val);
	    
	    setPoolSize = true;
	  }
	else if (strcmp((*argv)[i], "-llvm-opt")==0) 
	  {
	    char tmp[1024];
	    sscanf((*argv)[++i], "%s", &tmp[0]);
	    llvm_set_opt(tmp);
	  }
	else if (strcmp((*argv)[i], "-defaultgpu")==0) 
	  {
	    int ngpu;
	    sscanf((*argv)[++i], "%d", &ngpu);
	    gpu_setDefaultGPU(ngpu);
	    std::cout << "Default GPU set to " << ngpu << "\n";
	  }
	else if (strcmp((*argv)[i], "-geom")==0) 
	  {
	    setGeomP = true;
	    for(int j=0; j < Nd; j++) 
	      {
		int uu;
		sscanf((*argv)[++i], "%d", &uu);
		logical_geom[j] = uu;
	      }
	  }
	else if (strcmp((*argv)[i], "-iogeom")==0) 
	  {
	    setIOGeomP = true;
	    for(int j=0; j < Nd; j++) 
	      {
		int uu;
		sscanf((*argv)[++i], "%d", &uu);
		logical_iogeom[j] = uu;
	      }
	  }
#ifdef USE_REMOTE_QIO
	else if (strcmp((*argv)[i], "-cd")==0) 
	  {
	    /* push the dir into the environment vars so qio.c can pick it up */
	    setenv("QHOSTDIR", (*argv)[++i], 0);
	  }
	else if (strcmp((*argv)[i], "-rti")==0) 
	  {
	    sscanf((*argv)[++i], "%d", &rtiP);
	  }
	else if (strcmp((*argv)[i], "-rtinode")==0) 
	  {
	    int n = strlen((*argv)[++i]);
	    if (n >= maxlen)
	      {
		QDPIO::cerr << __func__ << ": rtinode name too long" << endl;
		QDP_abort(1);
	      }
	    sscanf((*argv)[i], "%s", rtinode);
	  }
#endif
#if 0
	else 
	  {
	    QDPIO::cerr << __func__ << ": Unknown argument = " << (*argv)[i] << endl;
	    QDP_abort(1);
	  }
#endif
			
	if (i >= *argc) 
	  {
	    QDPIO::cerr << __func__ << ": missing argument at the end" << endl;
	    QDP_abort(1);
	  }
      }
		

    if (!setPoolSize) {
      // It'll be set later in CudaGetDeviceProps
      //QDP_error_exit("Run-time argument -poolsize <size> missing. Please consult README.");
    }

		
#if QDP_DEBUG >= 1
    // Print command line args
    for (int i=0; i<*argc; i++) 
      QDP_info("QDP_init: arg[%d] = XX%sXX",i,(*argv)[i]);
#endif
		
  }




		// -------------------------------------------------------------------------------------------

	void QDP_initialize_QMP(int *argc, char ***argv)
	{
	  Layout::init();   // setup extremely basic functionality in Layout
		
	  isInit = true;

	  // Initialize the LLVM wrapper
	  //llvm_wrapper_init();
		
	  // initialize the global streams
	  QDPIO::cin.init(&std::cin);
	  QDPIO::cout.init(&std::cout);
	  QDPIO::cerr.init(&std::cerr);
		
	  initProfile(__FILE__, __func__, __LINE__);
		
	  QDPIO::cout << "Initialize done" << std::endl;
	}


	
	//! Is the machine initialized?
	bool QDP_isInitialized() {return isInit;}
	
	//! Turn off the machine
	void QDP_finalize()
	{
		if ( ! QDP_isInitialized() )
		{
			QDPIO::cerr << "QDP is not inited" << std::endl;
			QDP_abort(1);
		}
		
		QDPIO::cout << "------------------\n";
		QDPIO::cout << "-- JIT statistics:\n";
		QDPIO::cout << "------------------\n";
		QDPIO::cout << "lattices changed to device layout:     " << get_jit_stats_lattice2dev() << "\n";
		QDPIO::cout << "lattices changed to host layout:       " << get_jit_stats_lattice2host() << "\n";
		QDPIO::cout << "functions jit-compiled:                " << get_jit_stats_jitted() << "\n";
		
		FnMapRsrcMatrix::Instance().cleanup();

#if defined(QDP_USE_HDF5)
                H5close();
#endif

		// Finalize pool allocator
		Allocator::theQDPAllocator::DestroySingleton();

		printProfile();
		
		isInit = false;
	}
	
	//! Panic button
	void QDP_abort(int status)
	{
	}
	
	//! Resumes QDP communications
	void QDP_resume() {}
	
	//! Suspends QDP communications
	void QDP_suspend() {}
	
	
} // namespace QDP;
