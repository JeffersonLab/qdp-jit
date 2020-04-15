#ifndef QDP_SCALAR_SPECIFIC_H
#define QDP_SCALAR_SPECIFIC_H


namespace QDP {

  

  template<class T1>
  inline OScalar<T1>
  peekSite(const OLattice<T1>& l, const multi1d<int>& coord)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << "\n";

    OScalar<T1> dest;
    dest.elem() = l.elem(Layout::linearSiteIndex(coord));

    return dest;
  }



  template<class RHS, class T1>
  inline OScalar<T1>
  peekSite(const QDPExpr<RHS,OLattice<T1> > & l, const multi1d<int>& coord)
  {
    // For now, simply evaluate the expression and then call the function
    typedef OLattice<T1> C1;
  
    return peekSite(C1(l), coord);
  }



  template<class T1>
  inline OLattice<T1>&
  pokeSite(OLattice<T1>& l, const OScalar<T1>& r, const multi1d<int>& coord)
  {
    static CUfunction function;

    if (function.empty())
      function = function_pokeSite_build(l, r);

    function_pokeSite_exec(function, l, r, coord);

    return l;
  }




  //! Decompose a lexicographic site into coordinates
  multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size);


#ifdef QDP_USE_LIBXML2

  //! XML output
  template<class T>  
  XMLWriter& operator<<(XMLWriter& xml, const OLattice<T>& d)
  {
    xml.openTag("OLattice");

    XMLWriterAPI::AttributeList alist;

    const int vvol = Layout::vol();
    for(int site=0; site < vvol; ++site) 
      { 
	multi1d<int> coord = crtesn(site, Layout::lattSize());
	std::ostringstream os;
	os << coord[0];
	for(int i=1; i < coord.size(); ++i)
	  os << " " << coord[i];

	alist.clear();
	alist.push_back(XMLWriterAPI::Attribute("site", site));
	alist.push_back(XMLWriterAPI::Attribute("coord", os.str()));

	xml.openTag("elem", alist);
	xml << d.elem(Layout::linearSiteIndex(site));
	xml.closeTag();
      }

    xml.closeTag(); // OLattice

    return xml;
  }
#endif


  //! Binary output
  /*! Assumes no inner grid */
  template<class T>
  inline
  void write(BinaryWriter& bin, const OScalar<T>& d)
  {
    if (Layout::primaryNode()) 
      bin.writeArray((const char *)&(d.elem()), 
		     sizeof(typename WordType<T>::Type_t), 
		     sizeof(T) / sizeof(typename WordType<T>::Type_t));
  }

  //! Binary output
  /*! Assumes no inner grid */
  template<class T>  
  void write(BinaryWriter& bin, const OLattice<T>& d)
  {
    const int vvol = Layout::vol();
    for(int site=0; site < vvol; ++site) 
      {
	int i = Layout::linearSiteIndex(site);
	bin.writeArray((const char*)&(d.elem(i)), 
		       sizeof(typename WordType<T>::Type_t), 
		       sizeof(T) / sizeof(typename WordType<T>::Type_t));
      }
  }

  //! Binary output
  /*! Assumes no inner grid */
  template<class T>  
  void write(BinaryWriter& bin, OSubLattice<T> dd)
  {
    // Single node code
    const Subset& sub = dd.subset();
    const Set& set    = sub.getSet();

    const OLattice<T>& d = dd.field();

    const multi1d<int>& lat_color = set.latticeColoring();
    const int color = sub.color();

    // Choose only this color within a lexicographic loop
    const int vvol = Layout::vol();
    for(int site=0; site < vvol; ++site) 
      {
	int i = Layout::linearSiteIndex(site);
	if (lat_color[i] == color)
	  {
	    bin.writeArray((const char*)&(d.elem(i)), 
			   sizeof(typename WordType<T>::Type_t), 
			   sizeof(T) / sizeof(typename WordType<T>::Type_t));
	  }
      }
  }

  //! Write a single site of a lattice quantity at coord
  /*! Assumes no inner grid */
  template<class T>  
  void write(BinaryWriter& bin, const OLattice<T>& d, const multi1d<int>& coord)
  {
    int i = Layout::linearSiteIndex(coord);
    bin.writeArray((const char*)&(d.elem(i)), 
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

  //! Binary input
  /*! Assumes no inner grid */
  template<class T>  
  void read(BinaryReader& bin, OLattice<T>& d)
  {
    const int vvol = Layout::vol();
    for(int site=0; site < vvol; ++site) 
      {
	int i = Layout::linearSiteIndex(site);
	bin.readArray((char*)&(d.elem(i)), 
		      sizeof(typename WordType<T>::Type_t), 
		      sizeof(T) / sizeof(typename WordType<T>::Type_t));
      }
  }

  //! Read a single site and place it at coord
  /*! Assumes no inner grid */
  template<class T>  
  void read(BinaryReader& bin, OLattice<T>& d, const multi1d<int>& coord)
  {
    int i = Layout::linearSiteIndex(coord);
    bin.readArray((char*)&(d.elem(i)), 
		  sizeof(typename WordType<T>::Type_t), 
		  sizeof(T) / sizeof(typename WordType<T>::Type_t));
  }

  //! Binary input
  /*! Assumes no inner grid */
  template<class T>  
  void read(BinaryReader& bin, OSubLattice<T> dd)
  {
    // Single node code
    const Subset& sub = dd.subset();
    const Set& set    = sub.getSet();

    OLattice<T>& d = dd.field();

    const multi1d<int>& lat_color = set.latticeColoring();
    const int color = sub.color();

    // Choose only this color within a lexicographic loop
    const int vvol = Layout::vol();
    for(int site=0; site < vvol; ++site) 
      {
	int i = Layout::linearSiteIndex(site);
	if (lat_color[i] == color)
	  {
	    bin.readArray((char*)&(d.elem(i)), 
			  sizeof(typename WordType<T>::Type_t), 
			  sizeof(T) / sizeof(typename WordType<T>::Type_t));
	  }
      }
  }



  // **************************************************************
  // Special support for slices of a lattice
  namespace LatticeTimeSliceIO 
  {
    template<class T>
    void readSlice(BinaryReader& bin, OLattice<T>& data, 
		   int start_lexico, int stop_lexico)
    {
      for(int site=start_lexico; site < stop_lexico; ++site)
	{
	  int i = Layout::linearSiteIndex(site);
	  bin.readArray((char*)&(data.elem(i)), 
			sizeof(typename WordType<T>::Type_t), 
			sizeof(T) / sizeof(typename WordType<T>::Type_t));
	}
    }

    template<class T>
    void writeSlice(BinaryWriter& bin, OLattice<T>& data, 
		    int start_lexico, int stop_lexico)
    {
      for(int site=start_lexico; site < stop_lexico; ++site)
	{
	  int i = Layout::linearSiteIndex(site);
	  bin.writeArray((const char*)&(data.elem(i)), 
			 sizeof(typename WordType<T>::Type_t), 
			 sizeof(T) / sizeof(typename WordType<T>::Type_t));
	}
    }

  } // namespace LatticeTimeSliceIO




} // namespace QDP
#endif
