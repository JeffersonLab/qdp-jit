// -*- C++ -*-

/*! @file
 * @brief Outer lattice routines specific to a parallel platform with scalar layout
 */

#ifndef QDP_PARSCALAR_SPECIFIC_H
#define QDP_PARSCALAR_SPECIFIC_H

#include "qmp.h"

namespace QDP {

  

  //! Extract site element
  /*! @ingroup group1
    @param l  source to examine
    @param coord Nd lattice coordinates to examine
    @return single site object of the same primitive type
    @ingroup group1
    @relates QDPType */
  template<class T1>
  inline OScalar<typename ScalarType<T1>::Type_t>
  peekSite(const OLattice<T1>& l, const multi1d<int>& coord)
  {
    OScalar<typename ScalarType<T1>::Type_t> dest;
    int nodenum = Layout::nodeNumber(coord);

    // Find the result somewhere within the machine.
    // Then we must get it to node zero so we can broadcast it
    // out to all nodes
    if (Layout::nodeNumber() == nodenum)
      {
	dest.elem() = l.elem(Layout::linearSiteIndex(coord));
      }
    else
      {
	zero_rep(dest.elem());
      }

    // Send result to primary node via some mechanism
    QDPInternal::sendToPrimaryNode(dest, nodenum);

    // Now broadcast back out to all nodes
    QDPInternal::broadcast(dest);

    return dest;
  }

  //! Extract site element
  /*! @ingroup group1
    @param l  source to examine
    @param coord Nd lattice coordinates to examine
    @return single site object of the same primitive type
    @ingroup group1
    @relates QDPType */
  template<class RHS, class T1>
  inline OScalar<T1>
  peekSite(const QDPExpr<RHS,OLattice<T1> > & l, const multi1d<int>& coord)
  {
    // For now, simply evaluate the expression and then call the function
    typedef OLattice<T1> C1;
  
    return peekSite(C1(l), coord);
  }


  //! Insert site element
  /*! @ingroup group1
    @param l  target to update
    @param r  source to insert
    @param coord Nd lattice coordinates where to insert
    @return object of the same primitive type but of promoted lattice type
    @ingroup group1
    @relates QDPType */
  template<class T1>
  inline OLattice<T1>&
  pokeSite(OLattice<T1>& l, const OScalar<typename ScalarType<T1>::Type_t>& r, const multi1d<int>& coord)
  {
    static JitFunction function;

    if (function.empty())
      function_pokeSite_build( function , l, r);

    if (Layout::nodeNumber() == Layout::nodeNumber(coord))
      {
	function_pokeSite_exec(function, l, r, coord);
      }

    QMP_barrier();

    return l;
  }



  //-----------------------------------------------------------------------------

  //! Binary output
  /*! Assumes no inner grid */
  template<class T>
  inline
  void write(BinaryWriter& bin, const OScalar<T>& d)
  {
    bin.writeArray((const char *)&(d.elem()), 
		   sizeof(typename WordType<T>::Type_t), 
		   sizeof(T) / sizeof(typename WordType<T>::Type_t));
  }


  //! Binary input
  /*! Assumes no inner grid */
  template<class T>
  void read(BinaryReader& bin, OScalar<T>& d)
  {
    bin.readArray((char*)&(d.elem()), 
		  sizeof(typename WordType<T>::Type_t), 
		  sizeof(T) / sizeof(typename WordType<T>::Type_t)); 
  }



  // There are 2 main classes of binary/xml reader/writer methods.
  // The first is a simple/portable but inefficient method of send/recv
  // to/from the destination node.
  // The second method (the else) is a more efficient roll-around method.
  // However, this method more constrains the data layout - it must be
  // close to the original lexicographic order.
  // For now, use the direct send method

  //! Decompose a lexicographic site into coordinates
  multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size);

  //! XML output
  template<class T>	 
  XMLWriter& operator<<(XMLWriter& xml, const OLattice<T>& d)
  {
    typename ScalarType<T>::Type_t recv_buf;

    xml.openTag("OLattice");
    XMLWriterAPI::AttributeList alist;

    // Find the location of each site and send to primary node
    for(int site=0; site < Layout::vol(); ++site)
      {
	multi1d<int> coord = crtesn(site, Layout::lattSize());

	int node	 = Layout::nodeNumber(coord);
	int linear = Layout::linearSiteIndex(coord);

	// Copy to buffer: be really careful since max(linear) could vary among nodes
	if (Layout::nodeNumber() == node)
	  recv_buf = d.elem(linear);
	
	// Send result to primary node. Avoid sending prim-node sending to itself
	if (node != 0)
	  {
#if 1
	    // All nodes participate
	    QDPInternal::route((void *)&recv_buf, node, 0, sizeof(T));
#else
	    if (Layout::primaryNode())
	      QDPInternal::recvFromWait((void *)&recv_buf, node, sizeof(T));

	    if (Layout::nodeNumber() == node)
	      QDPInternal::sendToWait((void *)&recv_buf, 0, sizeof(T));
#endif
	  }

	if (Layout::primaryNode())
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


  //! Write a lattice quantity
  /*! This code assumes no inner grid */
  void writeOLattice(BinaryWriter& bin, 
		     const char* output, size_t size, size_t nmemb);

  //! Binary output
  /*! Assumes no inner grid */
  template<class T>
  void write(BinaryWriter& bin, const OLattice<T>& d)
  {
    writeOLattice(bin, (const char *)&(d.elem(0)), 
		  sizeof(typename WordType<T>::Type_t), 
		  sizeof(T) / sizeof(typename WordType<T>::Type_t));
  }

  //! Write a single site of a lattice quantity
  /*! This code assumes no inner grid */
  void writeOLattice(BinaryWriter& bin, 
		     const char* output, size_t size, size_t nmemb,
		     const multi1d<int>& coord);

  //! Write a single site of a lattice quantity
  /*! Assumes no inner grid */
  template<class T>
  void write(BinaryWriter& bin, const OLattice<T>& d, const multi1d<int>& coord)
  {
    writeOLattice(bin, (const char *)&(d.elem(0)), 
		  sizeof(typename WordType<T>::Type_t), 
		  sizeof(T) / sizeof(typename WordType<T>::Type_t),
		  coord);
  }

  //! Write a single site of a lattice quantity
  /*! This code assumes no inner grid */
  void writeOLattice(BinaryWriter& bin, 
		     const char* output, size_t size, size_t nmemb,
		     const Subset& sub);

  //! Write a single site of a lattice quantity
  /*! Assumes no inner grid */
  template<class T>
  void write(BinaryWriter& bin, OSubLattice<T> dd)
  {
    const OLattice<T>& d = dd.field();

    writeOLattice(bin, (const char *)&(d.elem(0)), 
		  sizeof(typename WordType<T>::Type_t), 
		  sizeof(T) / sizeof(typename WordType<T>::Type_t),
		  dd.subset());
  }


  //! Read a lattice quantity
  /*! This code assumes no inner grid */
  void readOLattice(BinaryReader& bin, 
		    char* input, size_t size, size_t nmemb);

  //! Binary input
  /*! Assumes no inner grid */
  template<class T>
  void read(BinaryReader& bin, OLattice<T>& d)
  {
    readOLattice(bin, (char *)&(d.elem(0)), 
		 sizeof(typename WordType<T>::Type_t), 
		 sizeof(T) / sizeof(typename WordType<T>::Type_t));
  }

  //! Read a single site of a lattice quantity
  /*! This code assumes no inner grid */
  void readOLattice(BinaryReader& bin, 
		    char* input, size_t size, size_t nmemb, 
		    const multi1d<int>& coord);

  //! Read a single site of a lattice quantity
  /*! Assumes no inner grid */
  template<class T>
  void read(BinaryReader& bin, OLattice<T>& d, const multi1d<int>& coord)
  {
    readOLattice(bin, (char *)&(d.elem(0)), 
		 sizeof(typename WordType<T>::Type_t), 
		 sizeof(T) / sizeof(typename WordType<T>::Type_t),
		 coord);
  }

  //! Read a single site of a lattice quantity
  /*! This code assumes no inner grid */
  void readOLattice(BinaryReader& bin, 
		    char* input, size_t size, size_t nmemb, 
		    const Subset& sub);

  //! Read a single site of a lattice quantity
  /*! Assumes no inner grid */
  template<class T>
  void read(BinaryReader& bin, OSubLattice<T> d)
  {
    readOLattice(bin, (char *)(d.field().getF()),
		 sizeof(typename WordType<T>::Type_t), 
		 sizeof(T) / sizeof(typename WordType<T>::Type_t),
		 d.subset());
  }



  // **************************************************************
  // Special support for slices of a lattice
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
      readOLatticeSlice(bin, (char *)&(data.elem(0)), 
			sizeof(typename WordType<T>::Type_t), 
			sizeof(T) / sizeof(typename WordType<T>::Type_t),
			start_lexico, stop_lexico);
    }


    // Write a time slice of a lattice quantity (time must be most slowly varying)
    template<class T>
    void writeSlice(BinaryWriter& bin, const OLattice<T>& data, 
		    int start_lexico, int stop_lexico)
    {
      writeOLatticeSlice(bin, (const char *)&(data.elem(0)), 
			 sizeof(typename WordType<T>::Type_t), 
			 sizeof(T) / sizeof(typename WordType<T>::Type_t),
			 start_lexico, stop_lexico);
    }

  } // namespace LatticeTimeSliceIO





} // namespace QDP
#endif
