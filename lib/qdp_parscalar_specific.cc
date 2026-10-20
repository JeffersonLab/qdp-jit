#include "qdp.h"
#include "qdp_util.h"
#include "qmp.h"


namespace QDP {


  namespace{
    template<class T> inline StandardOutputStream& operator<<(StandardOutputStream& s, const multi1d<T>& d)
    {
      for(int i=0; i < d.size(); ++i)
	s << d[i] << " ";
      return s;
    }
  }

  template<class T>
  ostream& operator<<(ostream& s, const multi1d<T>& s1)
  {
    for(int i=0; i < s1.size(); ++i)
      s << " " << s1[i];

    return s;
  }


  void Map::make(const MapFunc& func)
  {
    int ref = Layout::current().index();
    
    const int nodeSites = Layout::sitesOnNode();
    const int my_node   = Layout::nodeNumber();
    
    layouts[ref].srcnode.resize(nodeSites);

    multi1d<int> dstnode;
    dstnode.resize(nodeSites);

    // LAZY
    layouts[ref].lazy_fcoord.resize(nodeSites);
    layouts[ref].lazy_bcoord.resize(nodeSites);

    // Loop over the sites on this node
    for(int linear=0; linear < nodeSites; ++linear) 
      {
	multi1d<int> coord = Layout::siteCoords(my_node, linear);
	  
	multi1d<int> fcoord = func(coord,+1);
	  
	multi1d<int> bcoord = func(coord,-1);

	layouts[ref].lazy_fcoord[linear] = fcoord;
	layouts[ref].lazy_bcoord[linear] = bcoord;

	int fnode = Layout::nodeNumber(fcoord);
	int bnode = Layout::nodeNumber(bcoord);

	layouts[ref].srcnode[linear]  = fnode;
	dstnode[linear]  = bnode;
      }

    multi1d<int> srcenodes_tmp = uniquify_list(layouts[ref].srcnode);
    multi1d<int> destnodes_tmp = uniquify_list(dstnode);

    int cnt_srcenodes = 0;
    for(int i=0; i < srcenodes_tmp.size(); ++i)
      if (srcenodes_tmp[i] != my_node)
	++cnt_srcenodes;

    int cnt_destnodes = 0;
    for(int i=0; i < destnodes_tmp.size(); ++i)
      if (destnodes_tmp[i] != my_node)
	++cnt_destnodes;

    if (cnt_srcenodes > 0 && cnt_destnodes == 0)
      QDP_error_exit("Map: some bizarre error - no dest nodes but have srce nodes");

    if (cnt_srcenodes == 0 && cnt_destnodes > 0)
      QDP_error_exit("Map: some bizarre error - no srce nodes but have dest nodes");

    if (cnt_srcenodes > 1)
      QDP_error_exit("Map detected more than one src node (not supported)");

    if (cnt_destnodes > 1)
      QDP_error_exit("Map detected more than one dest node (not supported)");

    offnodeP = (cnt_srcenodes > 0) ? true : false;

    if (! offnodeP)
      {
#if QDP_DEBUG >= 3
	QDP_info("no off-node communications: exiting Map::make");
#endif
	return;
      }

    for(int i=0; i < srcenodes_tmp.size(); ++i)
      if (srcenodes_tmp[i] != my_node)
	layouts[ref].srcenode = srcenodes_tmp[i];

    for(int i=0; i < destnodes_tmp.size(); ++i)
      if (destnodes_tmp[i] != my_node)
	layouts[ref].destnode = destnodes_tmp[i];

    layouts[ref].lazy_destnode_fcoord.resize( nodeSites );
    for(int i=0; i < nodeSites; ++i)
      {
	multi1d<int> coord = Layout::siteCoords(layouts[ref].destnode, i);
	layouts[ref].lazy_destnode_fcoord[i] = func(coord,+1);
      }

  }



  Map::~Map()
  {
    if (myId > 0)
      MasterMap::Instance().unregister_map(myId);

    for ( auto& [ref, l] : layouts )
      for ( subset_t& sub : l.subsets )
	if (sub.done)
	  signoff_lazy( sub );
  }


  void Map::signoff_lazy( subset_t& sub ) const
  {
    for ( int id : { sub.goffsetsId , sub.roffsetsId , sub.soffsetsId } )
      if (id >= 0)
	QDP_get_global_cache().signoff( id );

    sub = subset_t();
  }


  void Map::make_lazy(const Subset& s, int ref) const
  {

    if (lazy_done_s(s,ref))
      return;
    
    const LatticeLayout lattice_layout = Layout::from_index( ref );
    const int nodeSites = lattice_layout.sitesOnNode();
    const int my_node   = lattice_layout.nodeNumber();
    const int s_no      = s.getId();

    layout( ref , "make_lazy" );
    map_t& l = layouts[ref];

    if ((int)l.subsets.size() <= s_no)
      l.subsets.resize( s_no + 1 );

    subset_t& sub = l.subsets[s_no];

    if (sub.done)
      signoff_lazy( sub );

    sub.goffsets.resize(nodeSites);

    for(int linear=0, ri=0; linear < nodeSites; ++linear)
      {
	if (l.srcnode[linear] == my_node)
	  {
	    sub.goffsets[linear] = lattice_layout.linearSiteIndex( l.lazy_fcoord[linear] );
	  }
	else
	  {
	    if ( s.isElement(lattice_layout,linear) )
	      {
		sub.goffsets[linear] = -(ri++)-1;
	      }
	  }
      }

    sub.goffsetsId = QDP_get_global_cache().addOwnHostMemNoPage( sizeof(int) * sub.goffsets.size() , sub.goffsets.slice() );

    if (offnodeP && myId < 0)
      {
	myId = MasterMap::Instance().register_map(*this);
      }

    sub.done       = true;
    sub.generation = MasterSet::Instance().generation(s_no);

    if (offnodeP)
      {
	sub.srcenode_num = 0;
	sub.destnode_num = 0;

	for(int linear=0; linear < nodeSites; ++linear)
	  {
	    if ( s.isElement(lattice_layout,linear) )
	      {
		int this_node = l.srcnode[linear];
		if (this_node != my_node)
		  {
		    if (l.srcenode == this_node)
		      {
			sub.srcenode_num++;
			sub.destnode_num++;
		      }
		  }
	      }
	  }

	sub.soffsets.resize( sub.destnode_num );
	sub.roffsets.resize( sub.srcenode_num );

	int ri=0;
	int si=0;
	for(int i=0; i < nodeSites; ++i)
	  {
	    multi1d<int> fcoord = l.lazy_destnode_fcoord[i];
	    
	    int fnode = lattice_layout.nodeNumber(fcoord);
	    int fline = lattice_layout.linearSiteIndex(fcoord);

	    if ( s.isElement(lattice_layout,i)  &&  fnode == my_node )
	      {
		sub.soffsets[si++] = fline;
	      }

	    if ( s.isElement(lattice_layout,i)  &&  l.srcnode[i] != my_node )
	      sub.roffsets[ri++] = i;
	  }

	if ( ri != sub.srcenode_num )
	  QDP_error_exit("internal error: ri != srcenodes_num[0]");

	if ( si != sub.destnode_num )
	  QDP_error_exit("internal error: si != destnodes_num[0]");

	if (sub.goffsets.size()==0)
	  QDP_error_exit("gsoffsets empty");

	sub.roffsetsId = QDP_get_global_cache().addOwnHostMem( sizeof(int) * sub.roffsets.size() , sub.roffsets.slice() );
	sub.soffsetsId = QDP_get_global_cache().addOwnHostMem( sizeof(int) * sub.soffsets.size() , sub.soffsets.slice() );
      } // offnode
  } // make_lazy

  
  

  namespace QDPInternal
  {
    void broadcast_str(std::string& result)
    {
      char *dd_tmp;
      int lleng;

      if (Layout::primaryNode()) 
	{
	  lleng = result.length();
	}

      QDPInternal::broadcast(lleng);

      dd_tmp = new(nothrow) char[lleng];
      if( dd_tmp == 0x0 ) { 
	QDP_error_exit("Unable to allocate dd_tmp\n");
      }

      if (Layout::primaryNode())
	memcpy(dd_tmp, result.c_str(), lleng);
  
      QDPInternal::broadcast((void *)dd_tmp, lleng);

      if (! Layout::primaryNode())
	{
	  result.assign(dd_tmp, lleng);
	}

      delete[] dd_tmp;
    }

    bool gridArch()
    { 
      return (QMP_get_msg_passing_type() == QMP_GRID) ? true : false;
    }

    void clearToSend(void *buffer, int count, int node)
    { 
      if (! QDPInternal::gridArch())
	{
	  if(Layout::nodeNumber() == 0 && node != 0)
	    QDPInternal::sendToWait(buffer, node, count);
	  if(Layout::nodeNumber() == node && node != 0)
	    QDPInternal::recvFromWait(buffer, 0, count);
	}
    }

    void route(void *buffer, int srce_node, int dest_node, int count)
    { 
      DML_route_bytes((char*)buffer, count, srce_node, dest_node);
    }

    void sendToWait(void *send_buf, int dest_node, int count)
    {
      QMP_msgmem_t request_msg = QMP_declare_msgmem(send_buf, count);
      QMP_msghandle_t request_mh = QMP_declare_send_to(request_msg, dest_node, 0);

      if (QMP_start(request_mh) != QMP_SUCCESS)
	QDP_error_exit("sendToWait failed\n");

      QMP_wait(request_mh);

      QMP_free_msghandle(request_mh);
      QMP_free_msgmem(request_msg);
    }

    void 
    recvFromWait(void *recv_buf, int srce_node, int count)
    {
      QMP_msgmem_t request_msg = QMP_declare_msgmem(recv_buf, count);
      QMP_msghandle_t request_mh = QMP_declare_receive_from(request_msg, srce_node, 0);

      if (QMP_start(request_mh) != QMP_SUCCESS)
	QDP_error_exit("recvFromWait failed\n");

      QMP_wait(request_mh);

      QMP_free_msghandle(request_mh);
      QMP_free_msgmem(request_msg);
    }

  }


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

    int old_node = 0;

    for(int site=0; site < Layout::vol(); site += xinc)
      {
	int node = Layout::nodeNumber(crtesn(site, Layout::lattSize()));

	if (node != old_node)
	  {
	    QDPInternal::clearToSend(recv_buf,sizeof(int),node);
	    old_node = node;
	  }
    
	if (Layout::nodeNumber() == node)
	  {
	    for(int i=0; i < xinc; ++i)
	      {
		int linear = Layout::linearSiteIndex(crtesn(site+i, Layout::lattSize()));
		memcpy(recv_buf+i*sizemem, output+linear*sizemem, sizemem);
	      }
	  }

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
    const multi1d<int>& lat_color = set.latticeColoring(Layout::current());
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
    const multi1d<int>& lat_color = set.latticeColoring(Layout::current());
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
		  n_uint32_t& checksum, int mat_size, int float_size)
  {
    Shape shape = u[0].shape();
    const LatticeLayout layout = u[0].layout();
    size_t sites_on_node = layout.sitesOnNode();
	
    size_t size = float_size;
    size_t su3_size = size*mat_size;
    size_t tot_size = su3_size*Nd;
    const int nodeSites = sites_on_node;

    char  *input = new(nothrow) char[tot_size*nodeSites];  // keep another copy in input buffers
    if( input == 0x0 ) { 
      QDP_error_exit("Unable to allocate input\n");
    }

    char  *recv_buf = new(nothrow) char[tot_size];
    if( recv_buf == 0x0 ) { 
      QDP_error_exit("Unable to allocate recv_buf\n");
    }

    checksum = 0;

    // Find the location of each site and send to primary node
    for(int site=0; site < layout.vol(); ++site)
      {
	multi1d<int> coord = crtesn(site, Layout::lattSize());

	int node   = layout.nodeNumber(coord);
	int linear = layout.linearSiteIndex(coord);

	// Only on primary node read the data
	cfg_in.readArrayPrimaryNode(recv_buf, size, mat_size*Nd);

	if (layout.primaryNode()) 
	  {
	    // Compute checksum
	    n_uint32_t* chk_ptr = (n_uint32_t*)recv_buf;
	    for(unsigned int i=0; i < mat_size*Nd*size/sizeof(n_uint32_t); ++i)
	      checksum += chk_ptr[i];
	  }

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
      }

    delete[] recv_buf;

    QDPInternal::broadcast(checksum);

    // Reconstruct the gauge field
    ColorMatrix  sitefield(shape);
    REAL su3[3][3][2];

    for(int linear=0; linear < nodeSites; ++linear)
      {
	for(int dd=0; dd<Nd; dd++)        /* dir */
	  {
	    // Transfer the data from input into SU3
	    if (float_size == 4) 
	      {
		REAL* su3_p = (REAL *)su3;
		REAL32* input_p = (REAL32 *)( input+su3_size*(dd+Nd*linear) );
		for(int cp_index=0; cp_index < mat_size; cp_index++) {
		  su3_p[cp_index] = (REAL)(input_p[cp_index]);
		}
	      }
	    else if (float_size == 8) 
	      {
		// IEEE64BIT case
		REAL *su3_p = (REAL *)su3;
		REAL64 *input_p = (REAL64 *)( input+su3_size*(dd+Nd*linear) );
		for(int cp_index=0; cp_index < mat_size; cp_index++) { 
		  su3_p[cp_index] = (REAL)input_p[cp_index];
		}
	      }
	    else { 
	      QDPIO::cerr << __func__ << ": Unknown mat size" << endl;
	      QDP_abort(1);
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
		    Complex sitecomp = cmplx(Real(su3[ii][kk][0]), Real(su3[ii][kk][1]));
		    pokeColor(sitefield,sitecomp,ii,kk);
		  }
	      }
      
	    u[dd].elem(linear) = sitefield.elem();
	  }
      }
  
    delete[] input;
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
    Shape shape = u[0].shape();
    const LatticeLayout layout = u[0].layout();
    size_t sites_on_node = layout.sitesOnNode();

    size_t size = sizeof(REAL32);
    size_t su3_size = size*mat_size;
    size_t tot_size = su3_size*Nd;
    char *recv_buf = new(nothrow) char[tot_size];
    if( recv_buf == 0x0 ) { 
      QDP_error_exit("Unable to allocate recv_buf\n");
    }

    const int nodeSites = sites_on_node;

    multi1d<multi1d<ColorMatrix> > sa(Nd);   // extract gauge fields

    for(int dd=0; dd<Nd; dd++)        /* dir */
      {
	sa[dd].resize(nodeSites);
	QDP_extract(sa[dd], u[dd], all);
      }

    // Find the location of each site and send to primary node
    for(int site=0; site < layout.vol(); ++site)
      {
	multi1d<int> coord = crtesn(site, layout.lattSize());

	int node   = layout.nodeNumber(coord);
	int linear = layout.linearSiteIndex(coord);

	// Copy to buffer: be really careful since max(linear) could vary among nodes
	if (layout.nodeNumber() == node)
	  {
	    char *recv_buf_tmp = recv_buf;

	    for(int dd=0; dd<Nd; dd++)        /* dir */
	      {
		if ( mat_size == 12 ) 
		  {
		    REAL32 su3[2][3][2];

		    for(int kk=0; kk<Nc; kk++)      /* color */
		      for(int ii=0; ii<2; ii++)    /* color */
			{
			  Complex sitecomp = peekColor(sa[dd][linear],ii,kk);
			  su3[ii][kk][0] = toFloat(Real(real(sitecomp)));
			  su3[ii][kk][1] = toFloat(Real(imag(sitecomp)));
			}

		    memcpy(recv_buf_tmp, &(su3[0][0][0]), su3_size);
		  }
		else
		  {
		    REAL32 su3[3][3][2];

		    for(int kk=0; kk<Nc; kk++)      /* color */
		      for(int ii=0; ii<Nc; ii++)    /* color */
			{
			  Complex sitecomp = peekColor(sa[dd][linear],ii,kk);
			  su3[ii][kk][0] = toFloat(Real(real(sitecomp)));
			  su3[ii][kk][1] = toFloat(Real(imag(sitecomp)));
			}

		    memcpy(recv_buf_tmp, &(su3[0][0][0]), su3_size);
		  }

		recv_buf_tmp += su3_size;
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

	cfg_out.writeArrayPrimaryNode(recv_buf, size, mat_size*Nd);
      }

    delete[] recv_buf;

    if (cfg_out.fail())
      {
	QDPIO::cerr << __func__ << ": error writing configuration" << endl;
	QDP_abort(1);
      }
  }

  

} // namespace QDP;
