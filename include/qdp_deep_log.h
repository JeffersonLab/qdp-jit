#ifndef QDP_DEEP_LOG_H
#define QDP_DEEP_LOG_H

namespace QDP {

#if 1
class JITLog
{
  std::fstream logger_cmp;
  size_t log_count = 0;
  bool compare = false;
  bool created = false;
  bool logging = false;
  std::string deep_log_name = "qdp-jit-log.dat";
  
public:
  void setCompare() { compare = true; logging = true; }
  void setCreate() { compare = false; logging = true; }

  bool fuzzy_cmp(float a,float b);
  bool fuzzy_cmp(double a,double b);
  bool fuzzy_cmp(int a,int b);
  bool fuzzy_cmp(bool a,bool b);

  void setName(const char* c)
  {
    deep_log_name = std::string(c);
  }
  
  void close() {
    std::cout << "closing log file" << std::endl;
    logger_cmp.close();
  }

  template<class T>
  void log( const OLattice<T>& dest , const Subset& s )
  {
    if (!logging) return;
    
    check_created();

    if (compare)
      log_compare(dest,s);
    else
      log_create(dest,s);
  }

  
  template<class T>
  void log_create( const OLattice<T>& dest , const Subset& s )
  {
    QDPIO::cout << "log " << log_count++ << " add lattice, sitecount = " << s.numSiteTable(dest.layout()) << endl;
    for (int i = 0 ; i < s.numSiteTable(dest.layout()) ; ++i )
      {
	dest.writeElemTo( s.siteTable(dest.layout())[i] , logger_cmp );
      }
  }

  
  template<class T>
  void log_compare( const OLattice<T>& dest , const Subset& s )
  {
    OScalar<T> disk(dest.shape());
    OScalar<T> mem(dest.shape());

    QDPIO::cout << "log: compare lattice, sitecount = " << s.numSiteTable(dest.layout()) << endl;

    for (int i = 0 ; i < s.numSiteTable(dest.layout()) ; ++i )
      {
	//QDPIO::cout << "site = " << i << std::endl;
	
	disk.readFrom( logger_cmp );

	mem.elem() = dest.elem( s.siteTable(dest.layout())[i] );

	
	auto ptr_disk = disk.get_ptr();
	auto ptr_mem = mem.get_ptr();

	for( int q = 0 ; q < disk.sizeInWords() ; ++q , ++ptr_disk, ++ptr_mem )
	  {
	    //QDPIO::cout << "index = " << q << " " << *ptr_disk << " vs " << *ptr_mem << std::endl;
	    
	    if (!fuzzy_cmp( *ptr_disk , *ptr_mem ))
	      {
		QDPIO::cout << "log mismatch lattice, subset index=" << i << " prim index=" << q << ": disk=" << *ptr_disk << " mem=" << *ptr_mem << endl;
		sleep(1);
		raise(SIGSEGV);
	      }
	    
	  }
      }
  }

  

  template<class T>
  void log( const OScalar<T>& dest )
  {
    if (!logging) return;

    check_created();

    if (compare)
      log_compare(dest);
    else
      log_create(dest);
  }

  
  template<class T>
  void log_create( const OScalar<T>& dest )
  {
    QDPIO::cout << "log " << log_count++ << " add scalar" << endl;
    dest.writeTo( logger_cmp );
  }

  
  template<class T>
  void log_compare( const OScalar<T>& mem )
  {
    OScalar<T> disk(mem.shape());

    QDPIO::cout << "log: compare scalar" << endl;

    disk.readFrom( logger_cmp );

    auto ptr_disk = disk.get_ptr();
    auto ptr_mem = mem.get_ptr();

    for( int q = 0 ; q < disk.sizeInWords() ; ++q , ++ptr_disk, ++ptr_mem )
      {
	if (!fuzzy_cmp( *ptr_disk , *ptr_mem ))
	  {
	    QDPIO::cout << "log mismatch scalar, prim index=" << q << ": disk=" << *ptr_disk << " mem=" << *ptr_mem << endl;
	    sleep(1);
	    raise(SIGSEGV);
	  }
	    
      }
  }


  
  void check_created();

};
#endif

  JITLog& QDP_get_global_logger();

  
  
#ifdef QDP_DEEP_LOG
  //void gpu_deep_logger_close();
  //void gpu_deep_logger        ( const void* host_ptr , std::string type_W , size_t field_size , std::string pretty , bool isLat );
#endif
  
}
#endif
