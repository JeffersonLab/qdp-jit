#ifndef QDP_PARSCALAR_SPECIFIC_H
#define QDP_PARSCALAR_SPECIFIC_H

#include "qmp.h"

namespace QDP {

  template<class RHS, class T1>
  OScalar<T1>
  peekSite(const QDPExpr<RHS,OLattice<T1> > & l, const multi1d<int>& coord)
  {
    typedef OLattice<T1> C1;
    return peekSite(C1(l), coord);
  }


  template<class T>
  void write(BinaryWriter& bin, const OScalar<T>& d)
  {
    bin.writeArray((const char *)(d.get_ptr()), 
		   sizeof(typename WordType<T>::Type_t), 
		   jit_sizeof(d) / sizeof(typename WordType<T>::Type_t));
  }


  template<class T>
  void read(BinaryReader& bin, OScalar<T>& d)
  {
    bin.readArray((char*)(d.get_ptr()), 
		  sizeof(typename WordType<T>::Type_t), 
		  jit_sizeof(d) / sizeof(typename WordType<T>::Type_t)); 
  }




  // Decompose a lexicographic site into coordinates
  multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size);

  template<class T>	 
  XMLWriter& operator<<(XMLWriter& xml, const OLattice<T>& d)
  {
    const LatticeLayout layout = d.layout();

    OScalar<T> recv_buf(d.shape());
    
    xml.openTag("OLattice");
    XMLWriterAPI::AttributeList alist;

    // Find the location of each site and send to primary node
    for(int site=0; site < layout.vol(); ++site)
      {
	multi1d<int> coord = crtesn(site, layout.lattSize());

#ifdef QDP_DEEP_LOG
      bool log_status = jit_config_deep_log();
      if (log_status)
	jit_config_deep_log_stop();
#endif
	recv_buf = peekSite( d , coord );
#ifdef QDP_DEEP_LOG
      if (log_status)
	jit_config_deep_log_start();
#endif
		
	if (layout.primaryNode())
	  {
	    std::ostringstream os;
	    os << coord[0];
	    for(int i=1; i < coord.size(); ++i)
	      os << " " << coord[i];

	    alist.clear();
	    alist.push_back(XMLWriterAPI::Attribute("site", site));
	    alist.push_back(XMLWriterAPI::Attribute("coord", os.str()));

	    xml.openTag("elem", alist);
	    xml << recv_buf;
	    xml.closeTag();
	  }
      }
    
    xml.closeTag(); // OLattice
    return xml;
  }


  void writeOLattice(BinaryWriter& bin, 
		     const char* output, size_t size, size_t nmemb);

  template<class T>
  void write(BinaryWriter& bin, const OLattice<T>& d)
  {
    OScalar<T> tmp(d.shape());
    writeOLattice(bin, (const char *)(d.get_ptr()), 
		  sizeof(typename WordType<T>::Type_t), 
		  jit_sizeof(tmp) / sizeof(typename WordType<T>::Type_t));
  }

  void writeOLattice(BinaryWriter& bin, 
		     const char* output, size_t size, size_t nmemb,
		     const multi1d<int>& coord);

  template<class T>
  void write(BinaryWriter& bin, const OLattice<T>& d, const multi1d<int>& coord)
  {
    OScalar<T> tmp(d.shape());
    writeOLattice(bin, (const char *)(d.get_ptr()), 
		  sizeof(typename WordType<T>::Type_t), 
		  jit_sizeof(tmp) / sizeof(typename WordType<T>::Type_t),
		  coord);
  }

  void writeOLattice(BinaryWriter& bin, 
		     const char* output, size_t size, size_t nmemb,
		     const Subset& sub);


  template<class T>
  void write(BinaryWriter& bin, OSubLattice<T> dd)
  {
    QDP_error_exit("write(BinaryWriter, OSubLattice) not implemented");
#if 0
    const OLattice<T>& d = dd.field();
    OScalar<T> tmp(d.shape());

    QDPIO::cout << "WARNING: code needs testing regards MG." << std::endl;
    
    writeOLattice(bin, (const char *)(dd.get_ptr())
		  sizeof(typename WordType<T>::Type_t), 
		  jit_sizeof(tmp) / sizeof(typename WordType<T>::Type_t),
		  dd.subset());
#endif
  }


  void readOLattice(BinaryReader& bin, 
		    char* input, size_t size, size_t nmemb);

  template<class T>
  void read(BinaryReader& bin, OLattice<T>& d)
  {
    OScalar<T> tmp(d.shape());
    readOLattice(bin, (char *)(d.get_ptr()), 
		 sizeof(typename WordType<T>::Type_t), 
		 jit_sizeof(tmp) / sizeof(typename WordType<T>::Type_t));
  }


  void readOLattice(BinaryReader& bin, 
		    char* input, size_t size, size_t nmemb, 
		    const multi1d<int>& coord);


  template<class T>
  void read(BinaryReader& bin, OLattice<T>& d, const multi1d<int>& coord)
  {
    OScalar<T> tmp(d.shape());
    readOLattice(bin, (char *)(d.get_ptr()), 
		 sizeof(typename WordType<T>::Type_t), 
		 jit_sizeof(tmp) / sizeof(typename WordType<T>::Type_t),
		 coord);
  }


  void readOLattice(BinaryReader& bin, 
		    char* input, size_t size, size_t nmemb, 
		    const Subset& sub);


  template<class T>
  void read(BinaryReader& bin, OSubLattice<T> d)
  {
    OScalar<T> tmp(d.shape());
    readOLattice(bin, (char *)(d.field().get_ptr()),
		 sizeof(typename WordType<T>::Type_t), 
		 jit_sizeof(tmp) / sizeof(typename WordType<T>::Type_t),
		 d.subset());
  }



  namespace LatticeTimeSliceIO 
  {
    //! Lattice time slice reader
    void readOLatticeSlice(BinaryReader& bin, char* data, 
			   size_t size, size_t nmemb,
			   int start_lexico, int stop_lexico);

    void writeOLatticeSlice(BinaryWriter& bin, const char* data, 
			    size_t size, size_t nmemb,
			    int start_lexico, int stop_lexico);


    // Read a time slice of a lattice quantity (time must be most slowly varying)
    template<class T>
    void readSlice(BinaryReader& bin, OLattice<T>& data, 
		   int start_lexico, int stop_lexico)
    {
      OScalar<T> tmp(data.shape());
      readOLatticeSlice(bin, (char *)(data.get_ptr()), 
			sizeof(typename WordType<T>::Type_t), 
			jit_sizeof(tmp) / sizeof(typename WordType<T>::Type_t),
			start_lexico, stop_lexico);
    }


    // Write a time slice of a lattice quantity (time must be most slowly varying)
    template<class T>
    void writeSlice(BinaryWriter& bin, const OLattice<T>& data, 
		    int start_lexico, int stop_lexico)
    {
      OScalar<T> tmp(data.shape());
      writeOLatticeSlice(bin, (const char *)(data.get_ptr()),
			 sizeof(typename WordType<T>::Type_t), 
			 jit_sizeof(tmp) / sizeof(typename WordType<T>::Type_t),
			 start_lexico, stop_lexico);
    }

  } // namespace LatticeTimeSliceIO


} // namespace QDP
#endif
