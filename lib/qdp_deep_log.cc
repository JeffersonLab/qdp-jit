#include "qdp_config_internal.h" 
#include<math.h>
#include<cmath>
#include "qdp.h"

#ifdef QDP_DEEP_LOG
#include <signal.h>
#endif


namespace QDP {

#ifdef QDP_DEEP_LOG
  namespace {
    std::unique_ptr<std::fstream> logger_cmp;
  }

  namespace
  {
    template<class T>
    inline
    StandardOutputStream& operator<<(StandardOutputStream& s, const multi1d<T>& d)
    {
      for(int i=0; i < d.size(); ++i)
	s << d[i] << " ";
      return s;
    }
  }


  void gpu_deep_logger_close()
  {
    if (logger_cmp)
      {
	std::cout << "closing log file" << std::endl;
	logger_cmp->close();
      }
  }


  void gpu_deep_logger_create( const void* host_ptr , std::string type_W , size_t field_size , std::string pretty , bool isLat )
  {
    if (!logger_cmp)
      {
	QDPIO::cout << "creating log object ..." << std::endl;
	logger_cmp.reset( new std::fstream );
	QDPIO::cout << "opening log file ..." << std::endl;
	logger_cmp->open( jit_config_deep_log_name().c_str() , ios::out | ios::binary);
	if(!(*logger_cmp))
	  {
	    QDPIO::cout << "Cannot open log file!" << endl;
	    QDP_abort(1);
	  }
      }
    
    size_t size_W;
	
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
	QDPIO::cerr << " size not known: " << type_W << std::endl;
	QDPIO::cerr << " pretty: " << pretty << std::endl;
	QDP_abort(1);
      }

    if (field_size % size_W)
      {
	QDPIO::cerr << " field_size not divisable by size_W" << std::endl;
	QDP_abort(1);
      }



    size_t count_word = field_size / size_W;

    const char* host_pos = (const char*)host_ptr;
    
    if (isLat)
      {
	size_t count_lat = count_word / Layout::sitesOnNode();

	std::cout << "isLat: count_word = " << count_word << ", " << "count_lat = " << count_lat << "\n";

	for( size_t lat = 0 ; lat < count_lat; ++lat )
	  {
	    //std::cout << "lat = " << lat << "\n";

	    //
	    // The start of this lattice
	    //
	    host_pos = (const char*)((char*)host_ptr + lat * size_W * Layout::sitesOnNode());

	    multi1d<int> coord(Nd);
	    coord = 0;
	    
	    for ( int dummy = 0 ; dummy < Layout::sitesOnNode() ; ++dummy )
	      {
		for ( int i = 0 ; i < Nd ; ++i )
		  {
		    if ( coord[i] >= Layout::lattSize()[i] )
		      {
			coord[i] = 0;
			
			if ( i < Nd-1 )
			  {
			    ++coord[i+1];
			  }
		      }
		  }
		
		int linear = Layout::linearSiteIndex(coord);
		//QDPIO::cout << "coord = " << coord << "   linear = " << linear << "\n";

		logger_cmp->write( host_pos + linear * size_W , size_W );

		++coord[0];
	      }
	  }
      }
    else
      {
	std::cout << "not Lat: count_word = " << count_word << "\n";

	logger_cmp->write( host_pos , size_W * count_word );
      }

  }


  namespace
  {
    bool fuzzy_cmp(float a,float b)
    {
      if (!std::isfinite(a) || !std::isfinite(b))
	return false;

      if (a==b)
	return true;

      float fuzz = std::numeric_limits<float>::epsilon();

      if (fabsf(a) <= (float)jit_config_get_fuzzfactor() * fuzz  &&  fabsf(b) <= (float)jit_config_get_fuzzfactor() * fuzz)
	return true;

      if ( fabsf(1.0 - fabsf(b/a)) > jit_config_get_tolerance() )
	return false;

      return true;
    }

    
    bool fuzzy_cmp(double a,double b)
    {
      if (!std::isfinite(a) || !std::isfinite(b))
	return false;

      if (a==b)
	return true;

      double fuzz = std::numeric_limits<double>::epsilon();

      if (fabs(a) <= jit_config_get_fuzzfactor() * fuzz  &&  fabs(b) <= jit_config_get_fuzzfactor() * fuzz)
	return true;

      if ( fabs(1.0 - fabs(b/a)) > jit_config_get_tolerance() )
	return false;

      return true;
    }


    void logger_print_lattice( std::string type_W , const char* host_pos )
    {
      for( int i = 0 ; i < Layout::sitesOnNode() ; ++i )
	{
	  if (type_W == "i")
	    {
	      QDPIO::cout << ((int*)host_pos)[i] << " ";
	    }
	  else if(type_W == "f")
	    {
	      QDPIO::cout << ((float*)host_pos)[i] << " ";
	    }
	  else if(type_W == "d")
	    {
	      QDPIO::cout << ((double*)host_pos)[i] << " ";
	    }
	  else if(type_W == "b")
	    {
	      QDPIO::cout << ((bool*)host_pos)[i] << " ";
	    }
	  else
	    {
	      QDPIO::cout << " size not known" << "\n";
	      QDP_abort(1);
	    }
	}
      QDPIO::cout << "\n";
    }

    
    bool logger_cmp_word( std::string type_W , const char* host_pos , const char* host_ref , size_t linear , size_t size_W , multi1d<int> coord , int& print_count )
    {
      if (type_W == "i")
	{
	  int cur = *(int*)host_pos;
	  int ref = *(int*)host_ref;
	  if (--print_count >= 0) QDPIO::cout << cur << " ";
	  if (cur != ref)
	    {
	      QDPIO::cout << "\nmismatch int: index = " << linear << ": cur = " << cur << "  ref = " << ref << "\n";
	      return false;
	    }
	}
      else if(type_W == "f")
	{
	  float cur = *(float*)host_pos;
	  float ref = *(float*)host_ref;
	  if (--print_count >= 0) QDPIO::cout << cur << " ";
	  if (!fuzzy_cmp(cur,ref))
	    {
	      QDPIO::cout << "\nmismatch float: index = " << linear << ": cur = " << cur << "  ref = " << ref << "\n";
	      return false;
	    }
	}
      else if(type_W == "d")
	{
	  double cur = *(double*)host_pos;
	  double ref = *(double*)host_ref;
	  if (--print_count >= 0) QDPIO::cout << cur << " ";
	  if (!fuzzy_cmp(cur,ref))
	    {
	      QDPIO::cout << "\nmismatch double: index = " << linear << ": cur = " << cur << "  ref = " << ref << "\n";
	      return false;
	    }
	}
      else if(type_W == "b")
	{
	  bool cur = *(bool*)host_pos;
	  bool ref = *(bool*)host_ref;
	  if (--print_count >= 0) QDPIO::cout << cur << " ";
	  if (cur != ref)
	    {
	      QDPIO::cout << "\nmismatch bool: index = " << linear << ": cur = " << cur << "  ref = " << ref << "\n";
	      return false;
	    }
	}
      else
	{
	  QDPIO::cout << " size not known" << "\n";
	  QDP_abort(1);
	}
      return true;
    }
    
  }
  
  
  void gpu_deep_logger_compare( const void* host_ptr , std::string type_W , size_t field_size , std::string pretty , bool isLat )
  {
    if (!logger_cmp)
      {
	QDPIO::cout << "creating log object ..." << std::endl;
	logger_cmp.reset( new std::fstream );
	QDPIO::cout << "opening log file ..." << std::endl;
	logger_cmp->open( jit_config_deep_log_name().c_str() , ios::in | ios::binary);
	if(!(*logger_cmp))
	  {
	    QDPIO::cout << "Cannot open log file!" << endl;
	    QDP_abort(1);
	  }
      }

    size_t size_W;
	
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
	QDPIO::cerr << " size not known: " << type_W << std::endl;
	QDPIO::cerr << " pretty: " << pretty << std::endl;
	QDP_abort(1);
      }

    if (field_size % size_W)
      {
	QDPIO::cerr << " field_size not divisable by size_W" << std::endl;
	QDP_abort(1);
      }

    size_t count_word = field_size / size_W;

    const char* host_pos = (const char*)host_ptr;
    
    if (isLat)
      {
	size_t count_lat = count_word / Layout::sitesOnNode();

	QDPIO::cout << "isLat: count_word = " << count_word
		    << ", count_lat = " << count_lat
		    << ", field_size = " << field_size << ": ";
	
	char* host_ref;

	if ( ! (host_ref = (char*)malloc( size_W * Layout::sitesOnNode() )) )
	  {
	    QDPIO::cout << "Cannot allocate host memory for ref!" << endl;
	    QDP_abort(1);
	  }

	int print_count = 4;
	
	for( size_t lat = 0 ; lat < count_lat; ++lat )
	  {
	    //std::cout << "lat = " << lat << "\n";
	
	    //
	    // Read in this chunk
	    //
	    //QDPIO::cout << "read in " << size_W * Layout::sitesOnNode() << " bytes\n";
	    logger_cmp->read(host_ref , size_W * Layout::sitesOnNode() );
	    
	    //
	    // The start of this lattice
	    //
	    host_pos = (const char*)((char*)host_ptr + lat * size_W * Layout::sitesOnNode());

	    multi1d<int> coord(Nd);
	    coord = 0;
	    
	    for ( int disk = 0 ; disk < Layout::sitesOnNode() ; ++disk )
	      {
		for ( int i = 0 ; i < Nd ; ++i )
		  {
		    if ( coord[i] >= Layout::lattSize()[i] )
		      {
			coord[i] = 0;
			
			if ( i < Nd-1 )
			  {
			    ++coord[i+1];
			  }
		      }
		  }
		
		size_t linear = Layout::linearSiteIndex(coord);

		if (!logger_cmp_word( type_W , host_pos + linear * size_W , host_ref + disk * size_W , linear , size_W , coord , print_count))
		  {
		    QDPIO::cout << "isLat: logger_cmp_word failed: " << pretty << "\n";
		    QDPIO::cout << "coord = " << coord << "   linear = " << linear << "\n";
		    QDPIO::cout << "ref sub lattice:\n";
		    logger_print_lattice(type_W , host_ref);
		    QDPIO::cout << "cur sub lattice:\n";
		    logger_print_lattice(type_W , host_pos);
		    sleep(1);
		    raise(SIGSEGV);
		  }
		
		++coord[0];
	      }
	  }
	QDPIO::cout << "\n";

	// Free memory for ref field
	free( host_ref );
  
      }
    else
      {
	QDPIO::cout << "not Lat: count_word = " << count_word << ": ";

	int print_count = 4;
	
	for ( int i = 0 ; i < count_word; ++i )
	  {
	    char buf[8];
	    logger_cmp->read(buf,size_W);

	    multi1d<int> coord(Nd);
	    coord=0;
	    
	    if (!logger_cmp_word( type_W , host_pos , buf , i , size_W , coord , print_count))
	      {
		QDPIO::cout << "notLat: logger_cmp_word failed: " << pretty << "\n";
		sleep(1);
		raise(SIGSEGV);
	      }
		
	    host_pos += size_W;
	  }
	QDPIO::cout << "\n";
      }
  }


  void gpu_deep_logger( const void* host_ptr , std::string type_W , size_t field_size , std::string pretty , bool isLat )
  {
    if (!jit_config_deep_log())
      return;

    if ( jit_config_deep_log_create() )
      {
	gpu_deep_logger_create( host_ptr , type_W , field_size , pretty , isLat );
      }
    else
      {
	gpu_deep_logger_compare( host_ptr , type_W , field_size , pretty , isLat );
      }
  }
#endif  

} // QDP
