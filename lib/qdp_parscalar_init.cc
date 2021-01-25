
/*! @file
 * @brief Parscalar init routines
 * 
 * Initialization routines for parscalar implementation
 */

#include <stdlib.h>
#include <unistd.h>
//#include <iomanip>

#include "qdp.h"
#include "qmp.h"

#include "qdp_init.h"

#if defined(QDP_USE_COMM_SPLIT_INIT)
#include <mpi.h>
#endif

namespace QDP {

namespace COUNT {
  int count = 0;
}

  namespace {
    bool setPoolSize = false;
  }



  //! Private flag for status
  static bool isInit = false;
  bool setGeomP = false;
  bool setIOGeomP = false;
  multi1d<int> logical_geom(Nd);   // apriori logical geometry of the machine
  multi1d<int> logical_iogeom(Nd); // apriori logical 	



#if 1
  int gamma_degrand_rossi[5][4][4][2] = 
    { { {{0,0}, {0,0}, {0,0},{0,-1}},
	{{0,0}, {0,0}, {0,-1},{0,0}},
	{{0,0}, {0,1},{0,0},{0,0}},
	{{0,1},{0,0}, {0,0},{0,0}} },

      { {{0,0}, {0,0}, {0,0},{-1,0}},
	{{0,0}, {0,0}, {1,0},{0,0}},
	{{0,0}, {1,0}, {0,0},{0,0}},
	{{-1,0},{0,0}, {0,0},{0,0}} },

      { {{0,0}, {0,0}, {0,-1},{0,0}},
	{{0,0}, {0,0}, {0,0},{0,1}},
	{{0,1}, {0,0}, {0,0},{0,0}},
	{{0,0}, {0,-1}, {0,0},{0,0}} },

      { {{0,0}, {0,0}, {1,0},{0,0}},
	{{0,0}, {0,0}, {0,0},{1,0}},
	{{1,0}, {0,0}, {0,0},{0,0}},
	{{0,0}, {1,0}, {0,0},{0,0}} },

      { {{1,0}, {0,0}, {0,0},{0,0}},
	{{0,0}, {1,0}, {0,0},{0,0}},
	{{0,0}, {0,0}, {1,0},{0,0}},
	{{0,0}, {0,0}, {0,0},{1,0}} } };



  SpinMatrix QDP_Gamma_values[Ns*Ns];

  extern SpinMatrix& Gamma(int i) {
    if (i<0 || i>15)
      QDP_error_exit("Gamma(%d) value out of range",i);
    if (!isInit) {
      std::cerr << "Gamma() used before QDP_init\n";
      exit(1);
    }
    //QDP_info("++++ returning gammas[%d]",i);
    //std::cout << gammas[i] << "\n";
    return QDP_Gamma_values[i];
  }
#endif


  void qdp_jit_cache_set_poolsize()
  {
    QDPIO::cout << "qdp-jit parameters\n";
    QDPIO::cout << "  threads per block                   : " << jit_util_get_threads_per_block() << "\n";
    
    if (!setPoolSize)
      {

	size_t val = (size_t)((double)(0.90) * (double)gpu_mem_free());

	int val_in_MiB = val/1024/1024;

	if (val_in_MiB < 1)
	  {
	    std::cerr << "Less than 1 MiB device memory available. Giving up.\n";
	    QDP_abort(1);
	  }
      
	float val_min = (float)val_in_MiB;

	QDPInternal::globalMinValue( &val_min );

	if ( val_min > (float)val_in_MiB )
	  {
	    QDPIO::cerr << "Inconsistency: Global minimum " << val_min << " larger than local value " << val_in_MiB << "\n";
	    QDP_abort(1);
	  }

	if ( val_min < (float)val_in_MiB )
	  {
	    QDPIO::cout << "Global minimum " << val_min << " of available GPU memory smaller than local value " << val_in_MiB << ". Using global minimum.";
	    QDP_abort(1);
	  }
	int val_min_int = (int)val_min;

	QDPIO::cout << "  memory pool size (default)          : " << (int)val_min_int << " MB\n";
      
	QDP_get_global_cache().setPoolSize( ((size_t)val_min_int) * 1024 * 1024 );

      }
    else
      {
	QDPIO::cout << "  memory pool size (user request)     : " << (int)(QDP_get_global_cache().getPoolSize()/1024/1024) << " MB\n";
      }

  }

  

  

  void QDP_startGPU()
  {
    // Getting GPU device properties
    gpu_auto_detect();

    // Now set the pool size
    qdp_jit_cache_set_poolsize();

    // Initialize the LLVM wrapper
    llvm_backend_init();
  }


  //! Set the GPU device
  int QDP_setGPU()
  {
    int deviceCount = gpu_get_device_count();

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
      if ( gpu_get_default_GPU() == -1 )
	{
	  std::cerr << "Couldnt determine local rank. Selecting device 0. In a multi-GPU per node run this is not what one wants.\n";
	  dev = 0;
	}
      else
	{
	  dev = gpu_get_default_GPU();
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

    std::cout << "Setting GPU device to " << dev << "\n";
    gpu_set_device( dev );

    return dev;
  }

#ifdef QDP_USE_COMM_SPLIT_INIT
  int QDP_setGPUCommSplit()
  {
    char hostname[256];

    int np_global=0;
    int np_local=0;
    int rank_global=0;
    int rank_local=0;

    MPI_Comm_size(MPI_COMM_WORLD, &np_global);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_global);

    MPI_Comm nodeComm;
    MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_global,
                     MPI_INFO_NULL, &nodeComm );

    MPI_Comm_size(nodeComm, &np_local);
    MPI_Comm_rank(nodeComm, &rank_local);

    MPI_Comm_free(&nodeComm);
    gethostname(hostname, 255);

    printf("Global Rank: %d of %d Host: %s  Local Rank: %d of %d Setting CUDA Device to %d \n",
        rank_global, np_global, hostname, rank_local, np_local, rank_local);
    CudaSetDevice(rank_local);
    return rank_local;
 
  }
#endif

  void QDP_initialize(int *argc, char ***argv) 
  {
    QDP_initialize_CUDA(argc, argv);
#ifndef QDP_USE_COMM_SPLIT_INIT
    QDP_setGPU();
#endif
    QDP_initialize_QMP(argc, argv);

#ifdef QDP_USE_COMM_SPLIT_INIT
    QDP_setGPUCommSplit();
#endif
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
	std::cerr << "QDP already inited" << endl;
	QDP_abort(1);
      }

#if 1
    //QDP_info_primary("Setting gamma matrices");

    SpinMatrix dgr[5];
    for (int i=0;i<5;i++) {
      for (int s=0;s<4;s++) {
	for (int s2=0;s2<4;s2++) {
	  dgr[i].elem().elem(s,s2).elem().real() = (float)gamma_degrand_rossi[i][s2][s][0];
	  dgr[i].elem().elem(s,s2).elem().imag() = (float)gamma_degrand_rossi[i][s2][s][1];
	}
      }
      //std::cout << i << "\n" << dgr[i] << "\n";
    }
    //QDP_info_primary("Finished setting gamma matrices");
    //QDP_info_primary("Multiplying gamma matrices");

    QDP_Gamma_values[0]=dgr[4]; // Unity
    for (int i=1;i<16;i++) {
      zero_rep(QDP_Gamma_values[i]);
      bool first=true;
      //std::cout << "gamma value " << i << " ";
      for (int q=0;q<4;q++) {
	if (i&(1<<q)) {
	  //std::cout << q << " ";
	  if (first)
	    QDP_Gamma_values[i]=dgr[q];
	  else
	    QDP_Gamma_values[i]=QDP_Gamma_values[i]*dgr[q];
	  first = false;
	}
      }
      //std::cout << "\n" << QDP_Gamma_values[i] << "\n";
		  
    }
    //QDP_info_primary("Finished multiplying gamma matrices");
#endif
    
    //
    // Init CUDA
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
    int QMP_verboseP = 0;
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
	  fprintf(stderr,"    -V        %%d [%d] verbose mode for QMP\n", 
		  QMP_verboseP);
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
	if (strcmp((*argv)[i], "-V")==0) 
	  {
	    QMP_verboseP = 1;
	  }
#if defined(QDP_USE_PROFILING)   
	else if (strcmp((*argv)[i], "-p")==0) 
	  {
	    int lev;
	    sscanf((*argv)[++i], "%d", &lev);
	    setProgramProfileLevel(lev);
	  }
#endif
	else if (strcmp((*argv)[i], "-poolmemset")==0) 
	  {
	    unsigned val;
	    sscanf((*argv)[++i],"%u",&val);
	    //QDP_get_global_cache().get_allocator().enableMemset(val);
	    QDP_get_global_cache().enableMemset(val);
	  }
	else if (strcmp((*argv)[i], "-blocksize")==0)
	  {
	    unsigned val;
	    sscanf((*argv)[++i],"%u",&val);
	    jit_util_set_threads_per_block( val );
	  }
	else if (strcmp((*argv)[i], "-stats")==0) 
	  {
	    gpu_set_record_stats();
	  }
#ifdef QDP_CUDA_SPECIAL
	else if (strcmp((*argv)[i], "-cudaspecial")==0)
	  {
	    unsigned func,bsize;
	    sscanf((*argv)[++i],"%u",&func);
	    sscanf((*argv)[++i],"%u",&bsize);
	    cuda_special_set_function_blocksize(func, bsize );
	  }
#endif
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
	else if (strcmp((*argv)[i], "-ptxdb")==0) 
	  {
	    char tmp[1024];
	    sscanf((*argv)[++i], "%s", &tmp[0]);
	    llvm_set_ptxdb(tmp);
	  }
	else if (strcmp((*argv)[i], "-defaultgpu")==0) 
	  {
	    int ngpu;
	    sscanf((*argv)[++i], "%d", &ngpu);
	    gpu_set_default_GPU(ngpu);
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
	    std::cerr << __func__ << ": missing argument at the end" << endl;
	    QDP_abort(1);
	  }
      }
		

    //QDPIO::cout << "Not setting QMP verbosity level\n";
    QMP_verbose (QMP_verboseP);
		
#if QDP_DEBUG >= 1
    // Print command line args
    for (int i=0; i<*argc; i++) 
      QDP_info("QDP_init: arg[%d] = XX%sXX",i,(*argv)[i]);
#endif
    
  } // QDP_initCUDA




		// -------------------------------------------------------------------------------------------

	void QDP_initialize_QMP(int *argc, char ***argv)
	{

#if QDP_DEBUG >= 1
	  QDP_info("Now initialize QMP");
#endif
		
	  if (QMP_is_initialized() == QMP_FALSE)
	    {
	      QMP_thread_level_t prv;
	      if (QMP_init_msg_passing(argc, argv, QMP_THREAD_SINGLE, &prv) != QMP_SUCCESS)
		{
		  QDPIO::cerr << __func__ << ": QMP_init_msg_passing failed" << endl;
		  QDP_abort(1);
		}
	    }
		
#if QDP_DEBUG >= 1
	  QDP_info("QMP inititalized");
#endif
		
	  if (setGeomP)
	    if (QMP_declare_logical_topology(logical_geom.slice(), Nd) != QMP_SUCCESS)
	      {
		QDPIO::cerr << __func__ << ": QMP_declare_logical_topology failed" << endl;
		QDP_abort(1);
	      }
		
#if QDP_DEBUG >= 1
	  QDP_info("Some layout init");
#endif
		
	  Layout::init();   // setup extremely basic functionality in Layout
		
	  isInit = true;
		
#if QDP_DEBUG >= 1
	  QDP_info("Init qio");
#endif
	  // OK, I need to set up the IO geometry here...
	  // I should make it part of layout...
	  if( setIOGeomP ) { 
#if QDP_DEBUG >=1
	    std::ostringstream outbuf;
	    for(int mu=0; mu < Nd; mu++) { 
	      outbuf << " " << logical_iogeom[mu];
	    }
			
	    QDP_info("Setting IO Geometry: %s\n", outbuf.str().c_str());
#endif
			
	    Layout::setIONodeGrid(logical_iogeom);
			
	  }

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
#ifdef QDP_CUDA_SPECIAL
		for ( auto it = get_jit_stats_special_names().begin() ; it != get_jit_stats_special_names().end(); it++ )
		  {
		    QDPIO::cout << it->first << ": [" << it->second << "] = " << get_jit_stats_special( it->first ) << "\n";
		  }
#endif
		if (get_ptx_db_enabled())
		  {
		    QDPIO::cout << "PTX DB, file:                          " << get_ptx_db_fname() << "\n";
		    QDPIO::cout << "PTX DB, size (number of functions):    " << get_ptx_db_size() << "\n";
		  }
		else
		  {
		    QDPIO::cout << "PTX DB: (not used)\n";
		  }

		if ( gpu_get_record_stats() )
		  {
		    QDPIO::cout << "#" << "\t";
		    QDPIO::cout << "calls" << "\t";
		    QDPIO::cout << "stack" << "\t";
		    QDPIO::cout << "sspill" << "\t";
		    QDPIO::cout << "lspill" << "\t";
		    QDPIO::cout << "regs" << "\t";
		    QDPIO::cout << "cmem" << "\t";
		    QDPIO::cout << "sum(ms)" << "\t\t";
		    QDPIO::cout << "mean" << "\t\t";
		    QDPIO::cout << "stddev" << "\t\t";
		    QDPIO::cout << "name" << "\n";
		    
		    std::vector<JitFunction*>& all = gpu_get_functions();

#if 1
		    struct compare_t
		    {
		      inline bool operator() ( JitFunction*& lhs,  JitFunction*& rhs)
		      {
			double sum_lhs = std::accumulate( lhs->get_timings().begin(), lhs->get_timings().end(), 0.0);
			double sum_rhs = std::accumulate( rhs->get_timings().begin(), rhs->get_timings().end(), 0.0);
			return (sum_lhs > sum_rhs);
		      }
		    };

		    std::sort(all.begin(), all.end(), compare_t());
#endif

		    TextFileWriter f_stats("qdp_jit_stats.txt");

		    for ( int i = 0 ; i < all.size() ; ++i )
		      {
			QDPIO::cout << i << "\t";
			QDPIO::cout << all.at(i)->get_call_counter() << "\t";
			QDPIO::cout << all.at(i)->get_stack() << "\t";
			QDPIO::cout << all.at(i)->get_spill_store() << "\t";
			QDPIO::cout << all.at(i)->get_spill_loads() << "\t";
			QDPIO::cout << all.at(i)->get_regs() << "\t";
			QDPIO::cout << all.at(i)->get_cmem() << "\t";

			auto timings = all.at(i)->get_timings();

			double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
			double mean = sum / timings.size();
			
			double sq_sum = std::inner_product( timings.begin() , timings.end() , timings.begin() , 0.0 );
			double stdev = 0.;
			if (timings.size() > 1)
			  stdev = std::sqrt( sq_sum / timings.size() - mean * mean );

			if (Layout::primaryNode())
			  printf("%f\t%f\t%f\t",(float)sum,(float)mean,(float)stdev);
			
			QDPIO::cout << all.at(i)->get_kernel_name() << "\n";

#if 1
			//f_stats << std::fixed << std::setw( 11 );
			f_stats << i << "\t";
			f_stats << all.at(i)->get_call_counter() << "\t";
			f_stats << all.at(i)->get_stack() << "\t";
			f_stats << all.at(i)->get_spill_store() << "\t";
			f_stats << all.at(i)->get_spill_loads() << "\t";
			f_stats << all.at(i)->get_regs() << "\t";
			f_stats << all.at(i)->get_cmem() << "\t";
			f_stats << (float)sum << "\t" << (float)mean << "\t" << (float)stdev << "\t";
			f_stats << all.at(i)->get_kernel_name() << "\n";
#endif
		      }

#if 1
		    f_stats << "\n\n";
		    for ( int i = 0 ; i < all.size() ; ++i )
		      {
			f_stats << all.at(i)->get_kernel_name() << "\t";
			f_stats << all.at(i)->get_pretty() << "\n\n";
		      }
		    
		    // Close file
		    f_stats.flush();
		    f_stats.close();
#endif
		    

		  }
		
		FnMapRsrcMatrix::Instance().cleanup();

#if defined(QDP_USE_HDF5)
                H5close();
#endif

		printProfile();
		
		QMP_finalize_msg_passing();
		
		isInit = false;
	}
	
	//! Panic button
	void QDP_abort(int status)
	{
		QMP_abort(status); 
	}
	
	//! Resumes QDP communications
	void QDP_resume() {}
	
	//! Suspends QDP communications
	void QDP_suspend() {}
	
	
} // namespace QDP;
