#ifndef QDP_FILE_DEEP_LOG
#define QDP_FILE_DEEP_LOG

#include <unistd.h>  // for sleep()
#include <csignal>   // for raise() and SIGSEGV

namespace QDP {


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

  
  // template<class T> void log_create( const OLattice<T>& dest , const Subset& s );  
  // template<class T> void log_compare( const OLattice<T>& dest , const Subset& s );

  template<class T>
  void log_create( const OLattice<T>& dest , const Subset& s )
  {
    QDPIO::cout << "log " << log_count++ << " add lattice, sitecount = " << s.numSiteTable() << endl;
    for (int i = 0 ; i < s.numSiteTable() ; ++i )
      {
	dest.writeElemTo( s.siteTable()[i] , logger_cmp );
      }
  }

  
  template<class T>
  void log_compare( const OLattice<T>& dest , const Subset& s )
  {
    OScalar<T> disk;
    OScalar<T> mem;

    QDPIO::cout << "log: compare lattice, sitecount = " << s.numSiteTable() << endl;

    for (int i = 0 ; i < s.numSiteTable() ; ++i )
      {
	//QDPIO::cout << "site = " << i << std::endl;
	
	disk.readFrom( logger_cmp );

	mem.elem() = dest.elem( s.siteTable()[i] );

	
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
    OScalar<T> disk;

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


  JITLog& QDP_get_global_logger();

  
  
  
}
#endif
