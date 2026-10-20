#include <stdlib.h>
#include <unistd.h>
#include <numeric>

#include "qdp.h"
#include "qmp.h"
#include "qdp_init.h"

#include <sys/stat.h> // Wanna use <filesystem> but not there...

#if defined(QDP_USE_COMM_SPLIT_INIT)
#include <mpi.h>
#endif

namespace QDP {

  namespace Layout {
    void destroyLatticeCoordinate();
  }

  bool isInit = false;

  bool libdevice_path_set = false;
  bool libdevice_name_set = false;
  
  bool setGeomP = false;
  bool setIOGeomP = false;
  multi1d<int> logical_geom(Nd);   // apriori logical geometry of the machine
  multi1d<int> logical_iogeom(Nd); // apriori logical

  namespace {
    size_t parse_memory_size( const char* text )
    {
      auto invalid = [text]() {
	std::cerr << "Invalid memory size: " << text << " (expected a number with an optional k, m, g or t suffix)" << std::endl;
	exit(1);
      };

      char* suffix;
      double value = strtod( text , &suffix );
      double multiplier = 1.;

      if (suffix == text || value < 0.)
	invalid();

      if (*suffix != '\0')
	{
	  if (suffix[1] != '\0')
	    invalid();

	  switch (tolower(*suffix))
	    {
	    case 'k': multiplier = 1024.; break;
	    case 'm': multiplier = 1024.*1024.; break;
	    case 'g': multiplier = 1024.*1024.*1024.; break;
	    case 't': multiplier = 1024.*1024.*1024.*1024.; break;
	    default: invalid();
	    }
	}

      return (size_t)( value * multiplier );
    }

    void check_option_values( int argc , char** argv , int i , int n )
    {
      if ( i + n >= argc )
	{
	  std::cerr << "QDP_initialize: " << argv[i] << " needs " << n << ( n == 1 ? " value" : " values" ) << std::endl;
	  exit(1);
	}
    }

    const char* option_value( int argc , char** argv , int& i )
    {
      check_option_values( argc , argv , i , 1 );
      return argv[++i];
    }
  } 	
  

  

  void QDP_startGPU()
  {
    jit_config_print_delayed_message();
    QDP_get_global_cache().setPoolSize( jit_config_get_pool_size() );
    llvm_factory_init( Layout::primaryNode() , Layout::nodeNumber() );
  }


  int QDP_setGPU()
  {
    int deviceCount = gpu_get_device_count();
    int dev=0;
    
    if (qdp_jit_config_get_use_gpu() == -1)
      {
	// Try 
	char *rank = getenv( "PMI_RANK" );

	// Try MVapich
	if( ! rank ) {
	  rank = getenv( "MV2_COMM_WORLD_LOCAL_RANK"  );
	} 

	// Try OpenMPI
	if( ! rank ) {
	  rank = getenv( "OMPI_COMM_WORLD_LOCAL_RANK" );
	}

	// Try SLURM
	if( ! rank ) {
	  rank = getenv( "SLURM_LOCALID" );
	}

	if (rank)
	  {
	    int local_rank = atoi(rank);
	    dev = local_rank % deviceCount;
	  }
	else
	  {
	    jit_config_delayed_message("Couldnt determine local rank. Selecting device 0. In a multi-GPU per node run this is not what one wants.");
	    dev = 0;
	  }
      }
    else
      {
	dev = qdp_jit_config_get_use_gpu();
	jit_config_delayed_message("Selecting device " + std::to_string(dev) + " as per user request.");
      }


    //std::cout << "Setting GPU device to " << dev << "\n";
    gpu_set_device( dev );

    // Sync init
    jit_util_sync_init();
    
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
    gpu_set_device(rank_local);
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
    Shape::push(Shape(Ns, Nc));
      
    if (sizeof(bool) != 1)
      {
	std::cout << "Error: sizeof(bool) == " << sizeof(bool) << "   (1 is required)" << endl;
	exit(1);
      }

    if (isInit)
      {
	std::cerr << "QDP already initialized" << endl;
	QDP_abort(1);
      }

    // initialize the global streams
    QDPIO::cin.init(&std::cin);
    QDPIO::cout.init(&std::cout);
    QDPIO::cerr.init(&std::cerr);

    //
    // Init CUDA
    //
    gpu_init();

    //
    // Process command line
    //
		
    setGeomP = false;
    logical_geom = 0;
		
    setIOGeomP = false;
    logical_iogeom = 0;
		
    int QMP_verboseP = 0;
		
    
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
	    sscanf(option_value( *argc , *argv , i ), "%d", &lev);
	    setProgramProfileLevel(lev);
	  }
#endif
	else if (strcmp((*argv)[i], "-gpu-sync")==0) 
	  {
	    qdp_jit_config_set_gpu_sync(true);
	  }
	else if (strcmp((*argv)[i], "-managed")==0) 
	  {
	    qdp_jit_config_set_managed(true);
	  }
	else if (strcmp((*argv)[i], "-default-data-order-site-slow")==0) 
	  {
	    qdp_jit_config_set_default_data_order( JitDataOrder::SiteSlow );
	    jit_config_delayed_message("JIT default data order: site-slow");
	  }
	else if (strcmp((*argv)[i], "-poolsize")==0) 
	  {
	    jit_config_set_pool_size( parse_memory_size( option_value( *argc , *argv , i ) ) );
	  }
	else if (strcmp((*argv)[i], "-pool-reserve")==0) 
	  {
	    jit_config_set_pool_reserve( parse_memory_size( option_value( *argc , *argv , i ) ) );
	  }
	else if (strcmp((*argv)[i], "-poolmemset")==0) 
	  {
	    unsigned val;
	    sscanf(option_value( *argc , *argv , i ),"%u",&val);
	    QDP_get_global_cache().enableMemset(val);
	  }
	else if (strcmp((*argv)[i], "-blocksum")==0) 
	  {
	    unsigned val;
	    sscanf(option_value( *argc , *argv , i ),"%u",&val);
	    qdp_jit_config_set_blocksum( (int)val );
	  }
	else if (strcmp((*argv)[i], "-pool-max-alloc")==0) 
	  {
	    int val;
	    sscanf(option_value( *argc , *argv , i ),"%d",&val);
	    jit_config_set_max_allocation( val );
	  }
	else if (strcmp((*argv)[i], "-pool-alignment")==0) 
	  {
	    unsigned val;
	    sscanf(option_value( *argc , *argv , i ),"%u",&val);
	    if ( val == 0 || ( val & (val - 1) ) != 0 )
	      {
		std::cerr << "-pool-alignment must be a power of 2, got " << val << std::endl;
		QDP_abort(1);
	      }
	    jit_config_set_pool_alignment( val );
	  }
	else if (strcmp((*argv)[i], "-pool-stats")==0) 
	  {
	    jit_set_config_pool_stats();
	  }
	else if (strcmp((*argv)[i], "-blocksize")==0)
	  {
	    unsigned val;
	    sscanf(option_value( *argc , *argv , i ),"%u",&val);
	    jit_config_set_threads_per_block( val );
	  }
#if defined (QDP_BACKEND_ROCM)
	else if (strcmp((*argv)[i], "-workgroup-bound")==0)
	  {
	    jit_config_set_workgroup_bound( true );
	  }
	else if (strcmp((*argv)[i], "-no-workgroup-bound")==0)
	  {
	    jit_config_set_workgroup_bound( false );
	  }
	else if (strcmp((*argv)[i], "-opt-prepass")==0)
	  {
	    jit_config_set_opt_prepass( true );
	  }
	else if (strcmp((*argv)[i], "-no-opt-prepass")==0)
	  {
	    jit_config_set_opt_prepass( false );
	  }
#endif
#if defined (QDP_HOST_COMPILER)
	else if (strcmp((*argv)[i], "-enable-hostjit")==0) 
	  {
	    qdp_jit_config_set_hostjit( true );
	  }
	else if (strcmp((*argv)[i], "-disable-hostjit")==0) 
	  {
	    qdp_jit_config_set_hostjit( false );
	  }
#endif
	else if (strcmp((*argv)[i], "-stats")==0) 
	  {
	    gpu_set_record_stats();
	  }
	else if (strcmp((*argv)[i], "-timing-run")==0) 
	  {
	    jit_config_set_timing_run(true);
	  }
#ifdef QDP_DEEP_LOG
	else if (strcmp((*argv)[i], "-deep-log-create")==0)
	  {
	    char tmp[1024];
	    sscanf(option_value( *argc , *argv , i ), "%s", &tmp[0]);
	    jit_config_deep_set( tmp , true );
	    QDP_get_global_logger().setCreate();
	    QDP_get_global_logger().setName(tmp);
	  }
	else if (strcmp((*argv)[i], "-deep-log-compare")==0)
	  {
	    char tmp[1024];
	    sscanf(option_value( *argc , *argv , i ), "%s", &tmp[0]);
	    jit_config_deep_set( tmp , false );
	    QDP_get_global_logger().setCompare();
	    QDP_get_global_logger().setName(tmp);
	  }
	else if (strcmp((*argv)[i], "-deep-log-tolerance")==0)
	  {
	    double d;
	    sscanf(option_value( *argc , *argv , i ), "%lf", &d);
	    jit_config_set_tolerance(d);
	  }
	else if (strcmp((*argv)[i], "-deep-log-fuzzfactor")==0)
	  {
	    double d;
	    sscanf(option_value( *argc , *argv , i ), "%lf", &d);
	    jit_config_set_fuzzfactor(d);
	  }
	else if (strcmp((*argv)[i], "-deep-log-events")==0)
	  {
	    unsigned val;
	    sscanf(option_value( *argc , *argv , i ),"%u",&val);
	    jit_config_set_log_events( val );
	  }
	else if (strcmp((*argv)[i], "-deep-log-memset")==0)
	  {
	    unsigned val;
	    sscanf(option_value( *argc , *argv , i ),"%u",&val);
	    jit_config_set_log_events( val );
	    jit_config_set_memset_value( (unsigned char)val );
	  }
#endif
	else if (strcmp((*argv)[i], "-opt")==0)
	  {
	    unsigned val;
	    sscanf(option_value( *argc , *argv , i ),"%u",&val);
	    jit_config_set_codegen_opt( val );
	  }
#ifdef QDP_BACKEND_ROCM
	else if (strcmp((*argv)[i], "-temp-files-path")==0) { 
	  char tmp[2048];
	  sscanf(option_value( *argc , *argv , i ), "%s", &tmp[0]);
	  std::string stmp(tmp);
	  if ( stmp.front() == '-' ) {
	    // argument missing
	    QDPIO::cerr << __func__ << ": -temp-files-path: " << stmp << " starts with - \n";
	    QDP_abort(1);
	  }
	  if ( stmp.back() != '/' ) { 
	    stmp += "/";
	  }
	  // Check prepend path exists and is a directory
	  struct stat status;
	  char hostname[128];
	  gethostname(hostname, 128);
 
	  if(  stat( stmp.c_str(), &status ) != 0 ) {
	    std::cerr << hostname<<  __func__ << ": -temp-files-path: " <<  stmp << " is an invalid path...";
	    switch(errno) { 
	    case EACCES : 
	      {
		std::cerr << " search permition is denied for a component of the path \n";
		break;
	      }
	    case ENOTDIR :
	      {
		std::cerr << " a component of the path is not a directory \n";
		break;
	      }
	    default: 
	      std::cerr << " " << strerror(errno) << "\n";
	      break;
	    }
	    QDP_abort(1);
	  }
	  if( (status.st_mode & S_IFMT) != S_IFDIR ) {
	    std::cerr << hostname << __func__ << ": -temp-files-path: " << stmp << " is not a directory...\n";
	    QDP_abort(1);
	  }
	  jit_config_set_prepend_path(stmp); 				
	}
	else if (strcmp((*argv)[i], "-libdevice-extra")==0) 
	  {
	    char tmp[2048];
	    sscanf(option_value( *argc , *argv , i ), "%s", &tmp[0]);
	    jit_config_add_extra_lib(tmp);
	  }
#endif
#if defined(QDP_BACKEND_ROCM) || defined(QDP_BACKEND_L0)	
	else if (strcmp((*argv)[i], "-keep-files")==0) 
	  {
	    jit_config_set_keepfiles(true);
	  }
#endif
#ifdef QDP_BACKEND_CUDA
	else if (strcmp((*argv)[i], "-cuda-ftz")==0)
	  {
	    unsigned val;
	    sscanf(option_value( *argc , *argv , i ),"%u",&val);
	    jit_config_set_CUDA_FTZ( val );
	  }
#endif
	else if (strcmp((*argv)[i], "-gpudirect")==0) 
	  {
	    jit_config_set_gpu_direct(true);
	  }
	else if (strcmp((*argv)[i], "-ringbuffersize")==0)
	  {
	    int s;
	    sscanf(option_value( *argc , *argv , i ), "%d", &s);
	    jit_config_set_oscalar_ringbuffer_size(s);
	  }
	else if (strcmp((*argv)[i], "-libdevice-path")==0) 
	  {
	    char tmp[1024];
	    sscanf(option_value( *argc , *argv , i ), "%s", &tmp[0]);
	    llvm_set_libdevice_path(tmp);
	    libdevice_path_set = true;
	  }
	else if (strcmp((*argv)[i], "-libdevice-name")==0) 
	  {
	    char tmp[1024];
	    sscanf(option_value( *argc , *argv , i ), "%s", &tmp[0]);
	    llvm_set_libdevice_name(tmp);
	    libdevice_name_set = true;
	  }
	else if (strcmp((*argv)[i], "-cache-verbose")==0) 
	  {
	    qdp_cache_set_cache_verbose(true);
	  }
	else if (strcmp((*argv)[i], "-codegen-verbose")==0) 
	  {
	    jit_config_set_verbose_output(true);
	  }
	else if (strcmp((*argv)[i], "-debug-layout")==0) 
	  {
	    jit_config_set_debug_layout(true);
	  }
	else if (strcmp((*argv)[i], "-use-gpu")==0) 
	  {
	    int ngpu;
	    sscanf(option_value( *argc , *argv , i ), "%d", &ngpu);
	    std::cout << "usegpu " << ngpu << "\n";
	    qdp_jit_config_set_use_gpu(ngpu);
	  }
	else if (strcmp((*argv)[i], "-geom")==0) 
	  {
	    check_option_values( *argc , *argv , i , Nd );
	    setGeomP = true;
	    for(int j=0; j < Nd; j++) 
	      {
		int uu;
		sscanf(option_value( *argc , *argv , i ), "%d", &uu);
		logical_geom[j] = uu;
	      }
	  }
	else if (strcmp((*argv)[i], "-iogeom")==0) 
	  {
	    check_option_values( *argc , *argv , i , Nd );
	    setIOGeomP = true;
	    for(int j=0; j < Nd; j++) 
	      {
		int uu;
		sscanf(option_value( *argc , *argv , i ), "%d", &uu);
		logical_iogeom[j] = uu;
	      }
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



	void QDP_initialize_QMP(int *argc, char ***argv)
	{
	  if (QMP_is_initialized() == QMP_FALSE)
	    {
	      QMP_thread_level_t prv;
	      if (QMP_init_msg_passing(argc, argv, QMP_THREAD_SINGLE, &prv) != QMP_SUCCESS)
		{
		  std::cerr << __func__ << ": QMP_init_msg_passing failed" << endl;
		  QDP_abort(1);
		}
	    }
		
	  if (setGeomP)
	    if (QMP_declare_logical_topology(logical_geom.slice(), Nd) != QMP_SUCCESS)
	      {
		std::cerr << __func__ << ": QMP_declare_logical_topology failed" << endl;
		QDP_abort(1);
	      }
		
	  Layout::init();   // setup extremely basic functionality in Layout
		
	  isInit = true;
		
	  if( setIOGeomP ) { 
	    Layout::setIONodeGrid(logical_iogeom);
	  }

	  initProfile(__FILE__, __func__, __LINE__);
		
	  QDPIO::cout << "Initialize done" << std::endl;
	}


	
	//! Is the machine initialized?
	bool QDP_isInitialized() {return isInit;}
	
	//! Turn off the machine
	void QDP_finalize()
	{
#ifdef QDP_DEEP_LOG
	  //gpu_deep_logger_close();
	  QDP_get_global_logger().close();
#endif

		if ( ! QDP_isInitialized() )
		{
			QDPIO::cerr << "QDP is not inited" << std::endl;
			QDP_abort(1);
		}


		// Wait for all tasks on the GPU to finish
		gpu_sync();

		// Destroy lattice coordinates helpers
		Layout::destroyLatticeCoordinate();

		// Close RNG
		RNG::doneDefaultRNG();

		// Cleanup translation
		destroy_translation_sitetable();
		
		// Finalize ring buffer
		//
		QDP_get_global_ring_buffer().done();


		if (qdp_jit_config_get_managed())
		  {
		    QDP_get_global_alloc_cache().freeAllUnused();
		  }

		gpu_done();
		
		QDPIO::cout << "\n";
		QDPIO::cout << "        ------------------------\n";
		QDPIO::cout << "        -- qdp-jit statistics --\n";
		QDPIO::cout << "        ------------------------\n";
		QDPIO::cout << "\n";
		QDPIO::cout << "Memory cache \n";
		QDPIO::cout << "  lattice objects copied to host memory:   " << llvm_get_jit_stats_lattice2dev() << "\n";
		QDPIO::cout << "  lattice objects copied to device memory: " << llvm_get_jit_stats_lattice2host() << "\n";
		QDPIO::cout << "\n";

		if (!qdp_jit_config_get_managed())
		  {
		    QDPIO::cout << "Pool allocator \n";
		    QDPIO::cout << "  peak memory usage:                       " << QDP_get_global_cache().get_max_allocated() << "\n";
		    if (jit_config_pool_stats())
		      {
			QDPIO::cout << "  memory allocation count: \n";
			auto& count = QDP_get_global_cache().get_alloc_count();
			for ( auto i = count.begin() ; i != count.end() ; ++i )
			  {
			    QDPIO::cout << "  " << i->first << " bytes \t\t" << i->second << std::endl;
			  }
			QDPIO::cout << "\n";
		      }
		  }

		if (jit_config_get_gpu_direct())
		  {
		    const QDPCache::DirectDeviceStats& s = QDP_get_global_cache().get_direct_device_stats();
		    double v[] = { (double)s.current_bytes , (double)s.peak_bytes , (double)s.largest_allocation ,
				   (double)s.allocations , (double)s.peak_active };
		    for ( double& x : v )
		      QMP_max_double( &x );

		    QDPIO::cout << "Communication buffers outside the pool (maximum over ranks)\n";
		    QDPIO::cout << "  allocated bytes (now, peak):             " << (size_t)v[0] << ", " << (size_t)v[1] << "\n";
		    QDPIO::cout << "  largest allocation:                      " << (size_t)v[2] << "\n";
		    QDPIO::cout << "  allocations (all, peak active):          " << (size_t)v[3] << ", " << (size_t)v[4] << "\n";
		    QDPIO::cout << "\n";
		  }
		
		QDPIO::cout << "Code generator \n";
		QDPIO::cout << "  functions jit-compiled (GPU)             " << llvm_get_jit_stats_jitted(true) << "\n";
		if (qdp_jit_config_get_hostjit())
		  {
		    QDPIO::cout << "  functions jit-compiled (CPU)             " << llvm_get_jit_stats_jitted(false) << "\n";
		  }

		{
		  std::vector<JitFunction*>& all = gpu_get_functions();
		  double time_builder = 0.;
		  double time_math = 0.;
		  double time_passes = 0.;
		  double time_codegen = 0.;
		  double time_dynload = 0.;
		  double time_linking = 0.;
		  for ( int i = 0 ; i < all.size() ; ++i )
		    {
		      time_builder += all.at(i)->time_builder;
		      time_math    += all.at(i)->time_math;
		      time_passes  += all.at(i)->time_passes;
		      time_codegen += all.at(i)->time_codegen;
		      time_dynload += all.at(i)->time_dynload;
		      time_linking += all.at(i)->time_linking;
		    }
		  QDPIO::cout << "  total time for IR builder:               " << time_builder/1.e6 << " s\n";
		  QDPIO::cout << "  total time for libm IR linking:          " << time_math/1.e6 << " s\n";
		  QDPIO::cout << "  total time for IR passes:                " << time_passes/1.e6 << " s\n";
		  QDPIO::cout << "  total time for code generation:          " << time_codegen/1.e6 << " s\n";
#ifdef QDP_BACKEND_ROCM
		  QDPIO::cout << "  total time for linking:                  " << time_linking/1.e6 << " s\n";
#endif
		  QDPIO::cout << "  total time for dynamic loading:          " << time_dynload/1.e6 << " s\n";
		}

		if ( gpu_get_record_stats() )
		  {
		    QDPIO::cout << "#" << "\t";
		    QDPIO::cout << "calls" << "\t";
#ifdef QDP_USE_ROCM_STATS
		    QDPIO::cout << "sgpr" << "\t";
		    QDPIO::cout << "vgpr" << "\t";
		    QDPIO::cout << "sgpr_sp" << "\t";
		    QDPIO::cout << "vgpr_sp" << "\t";
		    QDPIO::cout << "privseg" << "\t";
		    QDPIO::cout << "grpseg" << "\t";
#else
		    QDPIO::cout << "stack" << "\t";
		    QDPIO::cout << "regs" << "\t";
		    QDPIO::cout << "cmem" << "\t";
#endif
		    QDPIO::cout << "sum(ms)" << "\t\t";
		    QDPIO::cout << "mean" << "\t\t";
		    QDPIO::cout << "stddev" << "\t\t";
		    QDPIO::cout << "name" << "\n";
		    
		    std::vector<JitFunction*>& all = gpu_get_functions();

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

		    TextFileWriter f_stats("qdp_jit_stats.txt");

		    for ( int i = 0 ; i < all.size() ; ++i )
		      {
			QDPIO::cout << i << "\t";
			QDPIO::cout << all.at(i)->get_call_counter() << "\t";
#ifdef QDP_USE_ROCM_STATS
			QDPIO::cout << all.at(i)->get_regs() << "\t";
			QDPIO::cout << all.at(i)->get_vregs() << "\t";
			QDPIO::cout << all.at(i)->get_spill_store() << "\t";
			QDPIO::cout << all.at(i)->get_vspill_store() << "\t";
			QDPIO::cout << all.at(i)->get_private_segment() << "\t";
			QDPIO::cout << all.at(i)->get_group_segment() << "\t";
#else
			QDPIO::cout << all.at(i)->get_stack() << "\t";
			QDPIO::cout << all.at(i)->get_regs() << "\t";
			QDPIO::cout << all.at(i)->get_cmem() << "\t";
#endif
			
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

			f_stats << i << "\t";
			f_stats << all.at(i)->get_call_counter() << "\t";
#ifdef QDP_USE_ROCM_STATS
			f_stats << all.at(i)->get_regs() << "\t";
			f_stats << all.at(i)->get_vregs() << "\t";
			f_stats << all.at(i)->get_spill_store() << "\t";
			f_stats << all.at(i)->get_vspill_store() << "\t";
			f_stats << all.at(i)->get_private_segment() << "\t";
			f_stats << all.at(i)->get_group_segment() << "\t";
#else
			f_stats << all.at(i)->get_stack() << "\t";
			f_stats << all.at(i)->get_spill_store() << "\t";
			f_stats << all.at(i)->get_spill_loads() << "\t";
			f_stats << all.at(i)->get_regs() << "\t";
			f_stats << all.at(i)->get_cmem() << "\t";
#endif
			
			f_stats << (float)sum << "\t" << (float)mean << "\t" << (float)stdev << "\t";
			f_stats << all.at(i)->get_kernel_name() << "\n";
		      }

		    f_stats << "\n\n";
		    for ( int i = 0 ; i < all.size() ; ++i )
		      {
			f_stats << all.at(i)->get_kernel_name() << "\t";
			f_stats << all.at(i)->get_pretty() << "\n\n";
		      }
		    
		    // Close file
		    f_stats.flush();
		    f_stats.close();
		  }

		FnMapRsrcMatrix::Instance().cleanup();

		// Sync done
		jit_util_sync_done();
		
#if defined(QDP_USE_HDF5)
                H5close();
#endif

		Allocator::theQDPAllocator::DestroySingleton();

		printProfile();

#if defined(QDP_USE_PROFILING)
		if (qdp_jit_CPU_getall().size() > 0)
		  {
		    QDPIO::cout << std::endl << std::endl;
		    QDPIO::cout << "Functions executed on host:" << std::endl;
		    for (auto const& x : qdp_jit_CPU_getall())
		      {
			char lin[80];
			sprintf(lin, " [%8d] ", x.second );
			QDPIO::cout << lin << x.first << std::endl;
		      }
		  }
#endif
		
		QMP_finalize_msg_passing();

		// Destroy layouts
		Layout::finalize();

		Shape::pop();

		isInit = false;
	}
	
	//! Panic button
	void QDP_abort(int status)
	{
	  sleep(1);
	  QMP_abort(status); 
	}
	
	//! Resumes QDP communications
	void QDP_resume() {}
	
	//! Suspends QDP communications
	void QDP_suspend() {}
	
	
} // namespace QDP;
