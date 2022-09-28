#include "qdp.h"
#include "qdp_util.h"


namespace QDP {



  //-----------------------------------------------------------------------------
  // IO routine solely for debugging. Only defined here
  template<class T>
  ostream& operator<<(ostream& s, const multi1d<T>& s1)
  {
    for(int i=0; i < s1.size(); ++i)
      s << " " << s1[i];

    return s;
  }


  void Map::make(const MapFunc& func)
  {
#if QDP_DEBUG >= 3
    QDP_info("Map::make");
#endif
    const int nodeSites = Layout::sitesOnNode();
    const int my_node = Layout::nodeNumber();
    
    multi1d<int> dstnode;
    srcnode.resize(nodeSites);
    dstnode.resize(nodeSites);

    // LAZY
    lazy_fcoord.resize(nodeSites);
    lazy_bcoord.resize(nodeSites);

    // Loop over the sites on this node
    for(int linear=0; linear < nodeSites; ++linear) 
      {
	// Get the true lattice coord of this linear site index
	multi1d<int> coord = Layout::siteCoords(my_node, linear);
	  
	// Source neighbor for this destination site
	multi1d<int> fcoord = func(coord,+1);
	  
	// Destination neighbor receiving data from this site
	// This functions as the inverse map
	multi1d<int> bcoord = func(coord,-1);

	// LAZY, save for later
	lazy_fcoord[linear] = fcoord;
	lazy_bcoord[linear] = bcoord;

      }

  } // make (not lazy part)



  void Map::make_lazy(const Subset& s) const
  {
    const int nodeSites = Layout::sitesOnNode();

    if (lazy_done.size() < 1)
      {
	goffsets.resize( MasterSet::Instance().numSubsets() );
	soffsets.resize( MasterSet::Instance().numSubsets() );
	roffsets.resize( MasterSet::Instance().numSubsets() );

	goffsetsId.resize( MasterSet::Instance().numSubsets() );
	soffsetsId.resize( MasterSet::Instance().numSubsets() );
	roffsetsId.resize( MasterSet::Instance().numSubsets() );

	srcenodes_num.resize( MasterSet::Instance().numSubsets() );
	destnodes_num.resize( MasterSet::Instance().numSubsets() );

	lazy_done.resize( MasterSet::Instance().numSubsets() );
	lazy_done = false;
      }
    
    StopWatch clock;
    clock.start();

    int s_no = s.getId();

    goffsets[s_no].resize(nodeSites);

    //--------------------------------------
    // Setup the communication index arrays

    // Loop over the sites on this node
    for(int linear=0, ri=0; linear < nodeSites; ++linear)
      {
	goffsets[s_no][linear] = Layout::linearSiteIndex(lazy_fcoord[linear]);
      }

    goffsetsId[s_no] = QDP_get_global_cache().addOwnHostMemNoPage( sizeof(int)*goffsets[s_no].size() , 
								    goffsets[s_no].slice() );

    lazy_done[s_no] = true;

  } // make_lazy

  
  


  //------------------------------------------------------------------------
  // Message passing convenience routines
  //------------------------------------------------------------------------

  namespace QDPInternal
  {
    //! Broadcast a string from primary node to all other nodes
    void broadcast_str(std::string& result)
    {
    }
  }

#if 0
  //-----------------------------------------------------------------------------
  // Write a lattice quantity
  void writeOLattice(BinaryWriter& bin, 
		     const char* output, size_t size, size_t nmemb)
  {
    const int xinc = Layout::subgridLattSize()[0];

    size_t sizemem = size*nmemb;
    size_t tot_size = sizemem*xinc;
    char *recv_buf = new(nothrow) char[tot_size];
    if( recv_buf == 0x0 ) { 
      QDP_error_exit("Unable to allocate recv_buf\n");
    }

    // Find the location of each site and send to primary node
    int old_node = 0;

    for(int site=0; site < Layout::vol(); site += xinc)
      {
	// first site in each segment uniquely identifies the node
	int node = Layout::nodeNumber(crtesn(site, Layout::lattSize()));

	// Send nodes must wait for a ready signal from the master node
	// to prevent message pileups on the master node
	if (node != old_node)
	  {
	    // On non-grid machines, use a clear-to-send like protocol
	    QDPInternal::clearToSend(recv_buf,sizeof(int),node);
	    old_node = node;
	  }
    
	// Copy to buffer: be really careful since max(linear) could vary among nodes
	if (Layout::nodeNumber() == node)
	  {
	    for(int i=0; i < xinc; ++i)
	      {
		int linear = Layout::linearSiteIndex(crtesn(site+i, Layout::lattSize()));
		memcpy(recv_buf+i*sizemem, output+linear*sizemem, sizemem);
	      }
	  }

	// Send result to primary node. Avoid sending prim-node sending to itself
	if (node != 0)
	  {
#if 1
	    // All nodes participate
	    QDPInternal::route((void *)recv_buf, node, 0, tot_size);
#else
	    if (Layout::primaryNode())
	      QDPInternal::recvFromWait((void *)recv_buf, node, tot_size);

	    if (Layout::nodeNumber() == node)
	      QDPInternal::sendToWait((void *)recv_buf, 0, tot_size);
#endif
	  }

	bin.writeArrayPrimaryNode(recv_buf, size, nmemb*xinc);
      }

    delete[] recv_buf;
  }


  // Write a single site of a lattice quantity
  void writeOLattice(BinaryWriter& bin, 
		     const char* output, size_t size, size_t nmemb,
		     const multi1d<int>& coord)
  {
    size_t tot_size = size*nmemb;
    char *recv_buf = new(nothrow) char[tot_size];
    if( recv_buf == 0x0 ) { 
      QDP_error_exit("Unable to allocate recvbuf\n");
    }


    // Send site to primary node
    int node   = Layout::nodeNumber(coord);
    int linear = Layout::linearSiteIndex(coord);

    // Send nodes must wait for a ready signal from the master node
    // to prevent message pileups on the master node
    QDPInternal::clearToSend(recv_buf,sizeof(int),node);
  
    // Copy to buffer: be really careful since max(linear) could vary among nodes
    if (Layout::nodeNumber() == node)
      memcpy(recv_buf, output+linear*tot_size, tot_size);
  
    // Send result to primary node. Avoid sending prim-node sending to itself
    if (node != 0)
      {
#if 1
	// All nodes participate
	QDPInternal::route((void *)recv_buf, node, 0, tot_size);
#else
	if (Layout::primaryNode())
	  QDPInternal::recvFromWait((void *)recv_buf, node, tot_size);

	if (Layout::nodeNumber() == node)
	  QDPInternal::sendToWait((void *)recv_buf, 0, tot_size);
#endif
      }

    bin.writeArray(recv_buf, size, nmemb);

    delete[] recv_buf;
  }


  //-----------------------------------------------------------------------------
  // Write a lattice quantity
  void writeOLattice(BinaryWriter& bin, 
		     const char* output, size_t size, size_t nmemb,
		     const Subset& sub)
  {
    // Single node code
    const Set& set    = sub.getSet();
    const multi1d<int>& lat_color = set.latticeColoring();
    const int color = sub.color();

    const int xinc = Layout::subgridLattSize()[0];

    size_t sizemem = size*nmemb;
    size_t max_tot_size = sizemem*xinc;
    char *recv_buf = new(nothrow) char[max_tot_size];
    if( recv_buf == 0x0 ) { 
      QDP_error_exit("Unable to allocate recv_buf\n");
    }

    char *recv_buf_size = new(nothrow) char[sizeof(int)];
    if( recv_buf_size == 0x0 ) { 
      QDP_error_exit("Unable to allocate recv_buf_size\n");
    }

    // Find the location of each site and send to primary node
    int old_node = 0;

    for(int site=0; site < Layout::vol(); site += xinc)
      {
	// This algorithm is cumbersome. We do not keep the coordinate function for the subset.
	// So, we have to ask the sending node how many sites are to be transferred,
	// and rely on each node to send whatever number of sites live in the desired
	// subgridLattSize strip.

	// first site in each segment uniquely identifies the node
	int node = Layout::nodeNumber(crtesn(site, Layout::lattSize()));

	// Send nodes must wait for a ready signal from the master node
	// to prevent message pileups on the master node
	if (node != old_node)
	  {
	    // On non-grid machines, use a clear-to-send like protocol
	    QDPInternal::clearToSend(recv_buf,sizeof(int),node);
	    old_node = node;
	  }
    
	// Copy to buffer: be really careful since max(linear) could vary among nodes
	int site_cnt = 0;
	if (Layout::nodeNumber() == node)
	  {
	    for(int i=0; i < xinc; ++i)
	      {
		int linear = Layout::linearSiteIndex(crtesn(site+i, Layout::lattSize()));
		if (lat_color[linear] == color)
		  {
		    memcpy(recv_buf+site_cnt*sizemem, output+linear*sizemem, sizemem);
		    site_cnt++;
		  }
	      }
	    memcpy(recv_buf_size, (void *)&site_cnt, sizeof(int));
	  }

	// Send result to primary node. Avoid sending prim-node sending to itself
	if (node != 0)
	  {
#if 0
	    // All nodes participate
	    // First send the byte size for this 
	    QDP_error_exit("Do not support route in writeOLattice(sub)");

#else
	    // We are using the point-to-point version
	    if (Layout::primaryNode())
	      {
		QDPInternal::recvFromWait((void *)recv_buf_size, node, sizeof(int));
		memcpy((void *)&site_cnt, recv_buf_size, sizeof(int));
		QDPInternal::recvFromWait((void *)recv_buf, node, site_cnt*sizemem);
	      }

	    if (Layout::nodeNumber() == node)
	      {
		QDPInternal::sendToWait((void *)recv_buf_size, 0, sizeof(int));
		QDPInternal::sendToWait((void *)recv_buf, 0, site_cnt*sizemem);
	      }
#endif
	  }

	bin.writeArrayPrimaryNode(recv_buf, size, nmemb*site_cnt);
      }

    delete[] recv_buf_size;
    delete[] recv_buf;
  }



  //! Read a lattice quantity
  /*! This code assumes no inner grid */
  void readOLattice(BinaryReader& bin, 
		    char* input, size_t size, size_t nmemb)
  {
    const int xinc = Layout::subgridLattSize()[0];

    size_t sizemem = size*nmemb;
    size_t tot_size = sizemem*xinc;
    char *recv_buf = new(nothrow) char[tot_size];
    if( recv_buf == 0x0 ) { 
      QDP_error_exit("Unable to allocate recvbuf\n");
    }

    // Find the location of each site and send to primary node
    for(int site=0; site < Layout::vol(); site += xinc)
      {
	// first site in each segment uniquely identifies the node
	int node = Layout::nodeNumber(crtesn(site, Layout::lattSize()));

	// Only on primary node read the data
	bin.readArrayPrimaryNode(recv_buf, size, nmemb*xinc);

	// Send result to destination node. Avoid sending prim-node sending to itself
	if (node != 0)
	  {
#if 1
	    // All nodes participate
	    QDPInternal::route((void *)recv_buf, 0, node, tot_size);
#else
	    if (Layout::primaryNode())
	      QDPInternal::sendToWait((void *)recv_buf, node, tot_size);

	    if (Layout::nodeNumber() == node)
	      QDPInternal::recvFromWait((void *)recv_buf, 0, tot_size);
#endif
	  }

	if (Layout::nodeNumber() == node)
	  {
	    for(int i=0; i < xinc; ++i)
	      {
		int linear = Layout::linearSiteIndex(crtesn(site+i, Layout::lattSize()));

		memcpy(input+linear*sizemem, recv_buf+i*sizemem, sizemem);
	      }
	  }
      }

    delete[] recv_buf;
  }

  //! Read a single site worth of a lattice quantity
  /*! This code assumes no inner grid */
  void readOLattice(BinaryReader& bin, 
		    char* input, size_t size, size_t nmemb,
		    const multi1d<int>& coord)
  {
    size_t tot_size = size*nmemb;
    char *recv_buf = new(nothrow) char[tot_size];
    if( recv_buf == 0x0 ) {
      QDP_error_exit("Unable to allocate recv_buf\n");
    }


    // Find the location of each site and send to primary node
    int node   = Layout::nodeNumber(coord);
    int linear = Layout::linearSiteIndex(coord);

    // Only on primary node read the data
    bin.readArrayPrimaryNode(recv_buf, size, nmemb);

    // Send result to destination node. Avoid sending prim-node sending to itself
    if (node != 0)
      {
#if 1
	// All nodes participate
	QDPInternal::route((void *)recv_buf, 0, node, tot_size);
#else
	if (Layout::primaryNode())
	  QDPInternal::sendToWait((void *)recv_buf, node, tot_size);

	if (Layout::nodeNumber() == node)
	  QDPInternal::recvFromWait((void *)recv_buf, 0, tot_size);
#endif
      }

    if (Layout::nodeNumber() == node)
      memcpy(input+linear*tot_size, recv_buf, tot_size);

    delete[] recv_buf;
  }

  //! Read a lattice quantity
  /*! This code assumes no inner grid */
  void readOLattice(BinaryReader& bin, 
		    char* input, size_t size, size_t nmemb,
		    const Subset& sub)
  {
    // Single node code
    const Set& set    = sub.getSet();
    const multi1d<int>& lat_color = set.latticeColoring();
    const int color = sub.color();

    const int xinc = Layout::subgridLattSize()[0];

    size_t sizemem = size*nmemb;
    size_t max_tot_size = sizemem*xinc;
    char *recv_buf = new(nothrow) char[max_tot_size];
    if( recv_buf == 0x0 ) { 
      QDP_error_exit("Unable to allocate recv_buf\n");
    }

    char *recv_buf_size = new(nothrow) char[sizeof(int)];
    if( recv_buf_size == 0x0 ) { 
      QDP_error_exit("Unable to allocate recv_buf_size\n");
    }

    // Find the location of each site and send to primary node
    for(int site=0; site < Layout::vol(); site += xinc)
      {
	// This algorithm is cumbersome. We do not keep the coordinate function for the subset.
	// So, we have to ask the sending node how many sites are to be transferred,
	// and rely on each node to send whatever number of sites live in the desired
	// subgridLattSize strip.

	// first site in each segment uniquely identifies the node
	int node = Layout::nodeNumber(crtesn(site, Layout::lattSize()));

	// Find the amount of data to read. Unfortunately, have to ask the remote node
	// Place the result in a send buffer
	int site_cnt = 0;
	if (Layout::nodeNumber() == node)
	  {
	    for(int i=0; i < xinc; ++i)
	      {
		int linear = Layout::linearSiteIndex(crtesn(site+i, Layout::lattSize()));
		if (lat_color[linear] == color)
		  {
		    site_cnt++;
		  }
	      }
	    memcpy(recv_buf_size, (void *)&site_cnt, sizeof(int));
	  }

	if (node != 0)
	  {
	    // Send the data size to the primary node.
	    // We are using the point-to-point version
	    if (Layout::primaryNode())
	      {
		QDPInternal::recvFromWait((void *)recv_buf_size, node, sizeof(int));
		memcpy((void *)&site_cnt, recv_buf_size, sizeof(int));
	      }

	    if (Layout::nodeNumber() == node)
	      {
		QDPInternal::sendToWait((void *)recv_buf_size, 0, sizeof(int));
	      }
	  }

	// Only on primary node read the data
	bin.readArrayPrimaryNode(recv_buf, size, nmemb*site_cnt);

	// Send result to destination node. Avoid sending prim-node sending to itself
	if (node != 0)
	  {
#if 0
	    // All nodes participate
	    // First send the byte size for this 
	    QDP_error_exit("Do not support route in readOLattice(sub)");

#else
	    // We are using the point-to-point version
	    if (Layout::primaryNode())
	      QDPInternal::sendToWait((void *)recv_buf, node, site_cnt*sizemem);

	    if (Layout::nodeNumber() == node)
	      QDPInternal::recvFromWait((void *)recv_buf, 0, site_cnt*sizemem);
#endif
	  }

	if (Layout::nodeNumber() == node)
	  {
	    for(int i=0,j=0; i < xinc; ++i)
	      {
		int linear = Layout::linearSiteIndex(crtesn(site+i, Layout::lattSize()));
		if (lat_color[linear] == color)
		  {
		    memcpy(input+linear*sizemem, recv_buf+j*sizemem, sizemem);
		    j++;
		  }
	      }
	  }
      }

    delete[] recv_buf_size;
    delete[] recv_buf;
  }


  // **************************************************************
  namespace LatticeTimeSliceIO 
  {
    void readOLatticeSlice(BinaryReader& bin, char* input, 
			   size_t size, size_t nmemb,
			   int start_lexico, int stop_lexico)
    {
      const int xinc = Layout::subgridLattSize()[0];

      if ((stop_lexico % xinc) != 0)
	{
	  QDPIO::cerr << __func__ << ": erorr: stop_lexico= " << stop_lexico << "  xinc= " << xinc << std::endl;
	  QDP_abort(1);
	}

      size_t sizemem = size*nmemb;
      size_t tot_size = sizemem*xinc;
      char *recv_buf = new(nothrow) char[tot_size];
      if( recv_buf == 0x0 ) { 
	QDP_error_exit("Unable to allocate recvbuf\n");}

      // Find the location of each site and send to primary node
      for (int site=start_lexico; site < stop_lexico; site += xinc)
	{
	  // first site in each segment uniquely identifies the node
	  int node = Layout::nodeNumber(crtesn(site, Layout::lattSize()));

	  // Only on primary node read the data
	  bin.readArrayPrimaryNode(recv_buf, size, nmemb*xinc);

	  // Send result to destination node. Avoid sending prim-node sending to itself
	  if (node != 0)
	    {
#if 1
	      // All nodes participate
	      QDPInternal::route((void *)recv_buf, 0, node, tot_size);
#else
	      if (Layout::primaryNode())
		QDPInternal::sendToWait((void *)recv_buf, node, tot_size);
	      if (Layout::nodeNumber() == node)
		QDPInternal::recvFromWait((void *)recv_buf, 0, tot_size);
#endif
	    }

	  if (Layout::nodeNumber() == node)
	    {
	      for(int i=0; i < xinc; ++i)
		{
		  int linear = Layout::linearSiteIndex(crtesn(site+i, Layout::lattSize()));
		  memcpy(input+linear*sizemem, recv_buf+i*sizemem, sizemem);
		}
	    }
	}

      delete[] recv_buf;
    }

 
    // Write a time slice of a lattice quantity (time must be most slowly varying)
    void writeOLatticeSlice(BinaryWriter& bin, const char* output, 
			    size_t size, size_t nmemb,
			    int start_lexico, int stop_lexico)
    {
      const int xinc = Layout::subgridLattSize()[0];

      if ((stop_lexico % xinc) != 0)
	{
	  QDPIO::cerr << __func__ << ": erorr: stop_lexico= " << stop_lexico << "  xinc= " << xinc << std::endl;
	  QDP_abort(1);
	}

      size_t sizemem = size*nmemb;
      size_t tot_size = sizemem*xinc;
      char *recv_buf = new(nothrow) char[tot_size];
      if( recv_buf == 0x0 ) { 
	QDP_error_exit("Unable to allocate recv_buf\n");}

      // Find the location of each site and send to primary node
      int old_node = 0;

      for (int site=start_lexico; site < stop_lexico; site += xinc)
	{
	  // first site in each segment uniquely identifies the node
	  int node = Layout::nodeNumber(crtesn(site, Layout::lattSize()));

	  // Send nodes must wait for a ready signal from the master node
	  // to prevent message pileups on the master node
	  if (node != old_node){
	    // On non-grid machines, use a clear-to-send like protocol
	    QDPInternal::clearToSend(recv_buf,sizeof(int),node);
	    old_node = node;}
    
	  // Copy to buffer: be really careful since max(linear) could vary among nodes
	  if (Layout::nodeNumber() == node){
	    for(int i=0; i < xinc; ++i){
	      int linear = Layout::linearSiteIndex(crtesn(site+i, Layout::lattSize()));
	      memcpy(recv_buf+i*sizemem, output+linear*sizemem, sizemem);
	    }
	  }

	  // Send result to primary node. Avoid sending prim-node sending to itself
	  if (node != 0)
	    {
#if 1
	      // All nodes participate
	      QDPInternal::route((void *)recv_buf, node, 0, tot_size);
#else
	      if (Layout::primaryNode())
		QDPInternal::recvFromWait((void *)recv_buf, node, tot_size);
	      if (Layout::nodeNumber() == node)
		QDPInternal::sendToWait((void *)recv_buf, 0, tot_size);
#endif
	    }

	  bin.writeArrayPrimaryNode(recv_buf, size, nmemb*xinc);
	}
      delete[] recv_buf;
    }
  }

#endif

  
  

  //-----------------------------------------------------------------------
  // Compute simple NERSC-like checksum of a gauge field
  /*
   * \ingroup io
   *
   * \param u          gauge configuration ( Read )
   *
   * \return checksum
   */    

  n_uint32_t computeChecksum(const multi1d<LatticeColorMatrix>& u,
			     int mat_size)
  {
    size_t size = sizeof(REAL32);
    size_t su3_size = size*mat_size;
    n_uint32_t checksum = 0;   // checksum

    multi1d<multi1d<ColorMatrix> > sa(Nd);   // extract gauge fields
    const int nodeSites = Layout::sitesOnNode();

    for(int dd=0; dd<Nd; dd++)        /* dir */
      {
	sa[dd].resize(nodeSites);
	QDP_extract(sa[dd], u[dd], all);
      }

    char  *chk_buf = new(nothrow) char[su3_size];
    if( chk_buf == 0x0 ) { 
      QDP_error_exit("Unable to allocate chk_buf\n");
    }

    for(int linear=0; linear < nodeSites; ++linear)
      {
	for(int dd=0; dd<Nd; dd++)        /* dir */
	  {
	    switch (mat_size)
	      {
	      case 12:
		{
		  REAL32 su3[2][3][2];

		  for(int kk=0; kk<Nc; kk++)      /* color */
		    for(int ii=0; ii<2; ii++)    /* color */
		      {
			Complex sitecomp = peekColor(sa[dd][linear],ii,kk);
			su3[ii][kk][0] = toFloat(Real(real(sitecomp)));
			su3[ii][kk][1] = toFloat(Real(imag(sitecomp)));
		      }

		  memcpy(chk_buf, &(su3[0][0][0]), su3_size);
		}
		break;

	      case 18:
		{
		  REAL32 su3[3][3][2];

		  for(int kk=0; kk<Nc; kk++)      /* color */
		    for(int ii=0; ii<Nc; ii++)    /* color */
		      {
			Complex sitecomp = peekColor(sa[dd][linear],ii,kk);
			su3[ii][kk][0] = toFloat(Real(real(sitecomp)));
			su3[ii][kk][1] = toFloat(Real(imag(sitecomp)));
		      }

		  memcpy(chk_buf, &(su3[0][0][0]), su3_size);
		}
		break;

	      default:
		QDPIO::cerr << __func__ << ": unexpected size" << endl;
		QDP_abort(1);
	      }

	    // Compute checksum
	    n_uint32_t* chk_ptr = (n_uint32_t*)chk_buf;
	    for(unsigned int i=0; i < mat_size*size/sizeof(n_uint32_t); ++i)
	      checksum += chk_ptr[i];
	  }
      }

    delete[] chk_buf;

    // Get all nodes to contribute
    QDPInternal::globalSumArray((unsigned int*)&checksum, 1);   // g++ requires me to narrow the type to unsigned int

    return checksum;
  }


  //-----------------------------------------------------------------------
  // Read a QCD archive file
  // Read a QCD (NERSC) Archive format gauge field
  /*
   * \ingroup io
   *
   * \param cfg_in     binary writer object ( Modify )
   * \param u          gauge configuration ( Modify )
   */    
  void readArchiv(BinaryReader& cfg_in, multi1d<LatticeColorMatrix>& u, 
		  uint32_t& checksum, int mat_size, int float_size)
  {
    ColorMatrix  sitefield;
    char *su3_buffer;

    REAL su3[Nc][Nc][2];
    checksum = 0;

    su3_buffer = new char[ Nc*Nc*2*float_size ];
    if( su3_buffer == 0x0 ) { 
      QDP_error_exit("Unable to allocate input buffer\n");
    }

    // Find the location of each site and send to primary node
    for(int site=0; site < Layout::vol(); ++site)
      {
	multi1d<int> coord = crtesn(site, Layout::lattSize());
  
	for(int dd=0; dd<Nd; dd++)        /* dir */
	  {
	    /* Read an fe variable and write it to the BE */
	    cfg_in.readArray(su3_buffer, float_size, mat_size);

	    if (cfg_in.fail()) {
	      QDP_error_exit("Error reading configuration");
	    }


	    // Compute checksum
	    uint32_t* chk_ptr = (uint32_t*)su3_buffer;
	    for(int i=0; i < mat_size*float_size/sizeof(uint32_t); ++i)
	      checksum += chk_ptr[i];


	    /* Transfer from input buffer to the actual su3 buffer, 
	       downcasting it to float if necessary */
	    if ( float_size == 4 ) 
	      { 
		REAL32 *su3_bufp = (REAL32 *)su3_buffer;
		REAL *su3_p = (REAL *)su3;

		for(int cp_index=0; cp_index < mat_size; cp_index++) { 
		  su3_p[cp_index] = (REAL)su3_bufp[cp_index];
		}
	      }
	    else if ( float_size == 8 ) 
	      {
		REAL64 *su3_bufp = (REAL64 *)su3_buffer;
		REAL  *su3_p = (REAL *)su3;

		for(int cp_index =0; cp_index < mat_size; cp_index++) { 
	  
		  su3_p[cp_index] = (REAL)su3_bufp[cp_index];
		}
	      }

	    /* Reconstruct the third column  if necessary */
	    if (mat_size == 12) 
	      {
		su3[2][0][0] = su3[0][1][0]*su3[1][2][0] - su3[0][1][1]*su3[1][2][1]
		  - su3[0][2][0]*su3[1][1][0] + su3[0][2][1]*su3[1][1][1];
		su3[2][0][1] = su3[0][2][0]*su3[1][1][1] + su3[0][2][1]*su3[1][1][0]
		  - su3[0][1][0]*su3[1][2][1] - su3[0][1][1]*su3[1][2][0];

		su3[2][1][0] = su3[0][2][0]*su3[1][0][0] - su3[0][2][1]*su3[1][0][1]
		  - su3[0][0][0]*su3[1][2][0] + su3[0][0][1]*su3[1][2][1];
		su3[2][1][1] = su3[0][0][0]*su3[1][2][1] + su3[0][0][1]*su3[1][2][0]
		  - su3[0][2][0]*su3[1][0][1] - su3[0][2][1]*su3[1][0][0];
          
		su3[2][2][0] = su3[0][0][0]*su3[1][1][0] - su3[0][0][1]*su3[1][1][1]
		  - su3[0][1][0]*su3[1][0][0] + su3[0][1][1]*su3[1][0][1];
		su3[2][2][1] = su3[0][1][0]*su3[1][0][1] + su3[0][1][1]*su3[1][0][0]
		  - su3[0][0][0]*su3[1][1][1] - su3[0][0][1]*su3[1][1][0];
	      }

	    /* Copy into the big array */
	    for(int kk=0; kk<Nc; kk++)      /* color */
	      {
		for(int ii=0; ii<Nc; ii++)    /* color */
		  {
		    Real re = su3[ii][kk][0];
		    Real im = su3[ii][kk][1];
		    Complex sitecomp = cmplx(re,im);
		    pokeColor(sitefield,sitecomp,ii,kk);
		  }
	      }

	    pokeSite(u[dd], sitefield, coord);
	  }
      }
    delete [] su3_buffer;
  }



  //-----------------------------------------------------------------------
  // Write a QCD archive file
  // Write a QCD (NERSC) Archive format gauge field
  /*
   * \ingroup io
   *
   * \param cfg_out    binary writer object ( Modify )
   * \param u          gauge configuration ( Read )
   */    
  void writeArchiv(BinaryWriter& cfg_out, const multi1d<LatticeColorMatrix>& u,
		   int mat_size)
  {
    ColorMatrix  sitefield;
    float su3[3][3][2];

    // Find the location of each site and send to primary node
    for(int site=0; site < Layout::vol(); ++site)
      {
	multi1d<int> coord = crtesn(site, Layout::lattSize());

	for(int dd=0; dd<Nd; dd++)        /* dir */
	  {
	    sitefield = peekSite(u[dd], coord);

	    if ( mat_size == 12 ) 
	      {
		for(int kk=0; kk < Nc; kk++)      /* color */
		  for(int ii=0; ii < Nc-1; ii++)    /* color */
		    {
		      Complex sitecomp = peekColor(sitefield,ii,kk);
		      su3[ii][kk][0] = toFloat(Real(real(sitecomp)));
		      su3[ii][kk][1] = toFloat(Real(imag(sitecomp)));
		    }
	      }
	    else
	      {
		for(int kk=0; kk < Nc; kk++)      /* color */
		  for(int ii=0; ii < Nc; ii++)    /* color */
		    {
		      Complex sitecomp = peekColor(sitefield,ii,kk);
		      su3[ii][kk][0] = toFloat(Real(real(sitecomp)));
		      su3[ii][kk][1] = toFloat(Real(imag(sitecomp)));
		    }
	      }

	    // Write a site variable
	    cfg_out.writeArray((char *)&(su3[0][0][0]),sizeof(float), mat_size);
	  }
      }

    if (cfg_out.fail())
      QDP_error_exit("Error writing configuration");
  }





  

} // namespace QDP;
