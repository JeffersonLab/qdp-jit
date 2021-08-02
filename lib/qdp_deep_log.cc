#include "qdp_config_internal.h" 
#include "qdp.h"

#ifdef QDP_DEEP_LOG
#include <signal.h>
#endif


namespace QDP {

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
	logger_cmp->open( jit_config_deep_log_name().c_str() , ios::out | ios::binary);
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
	logger_cmp->open( jit_config_deep_log_name().c_str() , ios::in | ios::binary);
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

} // QDP
