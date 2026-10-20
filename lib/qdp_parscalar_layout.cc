
/*! @file
 * @brief Parscalar layout routines
 * 
 * Layout routines for parscalar implementation
 * QDP data parallel interface
 *
 * Layout
 *
 * This routine provides various layouts, including
 *    lexicographic
 *    2-checkerboard  (even/odd-checkerboarding of sites)
 *    32-style checkerboard (even/odd-checkerboarding of hypercubes)
 */

#include "qdp.h"
#include "qdp_util.h"

#include "qmp.h"


namespace QDP 
{

#if 1
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
#endif

#if 0
//-----------------------------------------------------------------------------
// IO routine solely for debugging. Only defined here
  template<class T>
  ostream& operator<<(ostream& s, const multi1d<T>& s1)
  {
    for(int i=0; i < s1.size(); ++i)
      s << " " << s1[i];

    return s;
  }
#endif



  bool LayoutData::qmp_layout_done = false;

  namespace Layout {

    bool fits_node_grid( const multi1d<int>& nrow )
    {
      if (nrow.size() != Nd)
	return false;

      const multi1d<int> min_dim = LayoutData::minimalLayoutMapping();
      const int* nodes = LayoutData::qmp_layout_done ? QMP_get_logical_dimensions() : nullptr;

      for(int i=0; i < Nd; ++i)
	{
	  const int n = nodes ? nodes[i] : 1;
	  if (nrow[i] <= 0 || nrow[i] % n != 0)
	    return false;
	  if ((nrow[i] / n) % min_dim[i] != 0)
	    return false;
	}
      return true;
    }
  }
  
//-----------------------------------------------------------------------------
    //-----------------------------------------------------
    //! Local data specific to all architectures
    /*! 
     * NOTE: the disadvantage to using a struct to keep things together is
     * that subsequent groupings of namespaces can not just add onto the
     * current namespace.
     */


    //-----------------------------------------------------
    // Functions

    //! Layout initialized ?
    bool LayoutData::initialized() {return _initialized;}
    
    //! Main destruction routine
    void LayoutData::destroy() {}

    //! Set virtual grid (problem grid) lattice size
    void LayoutData::setLattSize(const multi1d<int>& nrows) {nrow = nrows; }

    //! Virtual grid (problem grid) lattice size
    const multi1d<int>& LayoutData::lattSize() {return nrow;}

    //! Total lattice volume
    size_t LayoutData::vol() {return _vol;}

    //! Subgrid lattice volume
    int LayoutData::sitesOnNode() {return subgrid_vol;}

    //! Returns whether this is the primary node
    bool LayoutData::primaryNode() {return (node_rank == 0) ? true : false;}

    //! Subgrid (grid on each node) lattice size
    const multi1d<int>& LayoutData::subgridLattSize() {return subgrid_nrow;}

    //! Returns the node number of this node
    int LayoutData::nodeNumber() {return node_rank;}

    //! Returns the number of nodes
    int LayoutData::numNodes() {return num_nodes;}

    //! Returns the logical node coordinates for this node
    const multi1d<int>& LayoutData::nodeCoord() {return logical_coord;}

    //! Returns the logical size of this machine
    const multi1d<int>& LayoutData::logicalSize() {return logical_size;}

    //! Returns the node number given some logical node coordinate
    /*! This is not meant to be speedy */
    int LayoutData::getNodeNumberFrom(const multi1d<int>& node_coord) 
    {
      return QMP_get_node_number_from(node_coord.slice());
    }

    //! Returns the logical node coordinates given some node number
    /*! This is not meant to be speedy */
    multi1d<int> LayoutData::getLogicalCoordFrom(int node) 
    {
      multi1d<int> node_coord(Nd);
      const int* node_crd = QMP_get_logical_coordinates_from(node);  // QMP mallocs here

      for(int i=0; i < Nd; ++i)
	node_coord[i] = node_crd[i];

      // Hackery as free cannot take a const void*, so grab
      // node_crd with a non const pointer
      int* non_const_node_crd = const_cast<int*>(node_crd);
      free(non_const_node_crd);   // free up QMP's memory
      return node_coord;
    }

    //! Return the smallest lattice size per node allowed
  //multi1d<int> LayoutData::minimalLayoutMapping();

    //! Initializer for layout
    void LayoutData::init(int _pos)
    {
      num_nodes = QMP_get_number_of_nodes();
      node_rank = QMP_get_node_number();
		
      // Default should be current behaviour (as of 7/4/12)
      // No iogrid is defined -> each node is its own I/O node
      iogrid_defined = false;  
      num_iogrid = num_nodes;

      pos = _pos;
    }


    int LayoutData::ref() const
    {
      if ( pos < 0 )
	{
	  QDPIO::cout << "LayoutData::ref(), pos not initialized." << std::endl;
	  sleep(1);
	  QDP_abort(1);
	}
      return pos;
    }

    //! The linearized site index for the corresponding lexicographic site
    int LayoutData::linearSiteIndex(int site)
    { 
      multi1d<int> coord = crtesn(site, lattSize());
    
      return linearSiteIndex(coord);
    }


	  
	  //! check if I/O grid is defined
	  /*! Always defined for scalar node: it is 1x1x1x1 */
	  bool LayoutData::isIOGridDefined(void) 
	  { 
	    return iogrid_defined; 
	  }
	  
	  //! number of I/O nodes
	  int LayoutData::numIONodeGrid(void)
	  {
	    return num_iogrid;	  
	  }
	  
	  //! Set the I/O Node grid -- satisfy interface
	  void LayoutData::setIONodeGrid(const multi1d<int>& io_grid) 
	  {
		iogrid.resize(Nd);	
		for(int mu=0; mu < Nd; mu++) {
		  iogrid[mu] = io_grid[mu];
		}
		num_iogrid = io_grid[0];
		for(int mu=1; mu < Nd; mu++) { 
		  num_iogrid *= io_grid[mu];
		}
		iogrid_defined = true;
	  }
	  
	  //! Get the I/O Node grid
	  const multi1d<int>& LayoutData::getIONodeGrid()
	  {
		  return iogrid;
	  }
	  
    //! Initializer for all the layout defaults
    void LayoutData::initDefaults(bool verbose)
    {
      if (verbose)
	jit_config_print();
      
#if QDP_DEBUG >= 2
      QDP_info("Create default subsets");
#endif

      if (verbose)
	QDPIO::cout << "initDefaultSets()" << endl;
      // Default set and subsets
      initDefaultSets();

      if (verbose)
	QDPIO::cout << "initDefaultMaps()" << endl;
      // Default maps
      initDefaultMaps();
      
      if (verbose)
	QDPIO::cout << "initDefaultRNG()" << endl;
      // Initialize RNG
      RNG::initDefaultRNG();
      if (verbose)
	QDPIO::cout << "Finished init of RNG" << endl;
      
      // Set default profile level
      //setProfileLevel(getProgramProfileLevel());
    }


    //! Main lattice creation routine
    void LayoutData::create()
    {
      if ( ! QDP_isInitialized() )
	QDP_error_exit("QDP is not initialized");

      if (nrow.size() != Nd)
	QDP_error_exit("dimension of lattice size not the same as the default");

      _vol=1;
      for(int i=0; i < Nd; ++i) 
	_vol *= nrow[i];
  
#if QDP_DEBUG >= 2
      QDP_info("vol=%d",_layout._vol);
#endif

#if QDP_DEBUG >= 2
      QDP_info("Initialize layout");
#endif

      // Simple check - we insist here that the total volume is divisible
      // by the number of processors. Will also insist that the problem
      // size is regular on each node
      if (_vol % numNodes() != 0)
	QDP_error_exit("LayoutData::create - problem size not divisible by number of processors");

    
      // Return the smallest lattice size per node allowed
      multi1d<int> min_dim = minimalLayoutMapping();

      //QDPIO::cout << "minimal layout mapping = " << min_dim << std::endl;
      
      // Another sanity check - if the specified lattice size is not a multiple
      // of the minimal size, then barf
      for(int i=0; i < Nd; ++i)
	if (nrow[i] % min_dim[i] != 0)
	  QDP_error_exit("LayoutData::create - Error: problem size not multiple of smallest size allowed for this type of layout");


      // Now, layout the machine. Note, if the logical_machine size was set previously
      // it will be used
      multi1d<int> nrow_layout(Nd);
      for(int i=0; i < Nd; ++i)
	nrow_layout[i] = nrow[i] / min_dim[i];

      const bool verbose = !qmp_layout_done || jit_config_get_debug_layout();

      if (!qmp_layout_done)
	{
	  QDPIO::cout << "calling QMP_layout_grid" << std::endl;
	  int* nrow_slice=const_cast<int*>(nrow_layout.slice());
	  QMP_layout_grid(nrow_slice, Nd);
	  qmp_layout_done = true;
	}
      else
	{
	  if (verbose)
	    QDPIO::cout << "skip calling QMP_layout_grid" << std::endl;
	}
      
      // Pull out useful stuff
      const int* phys_size = QMP_get_logical_dimensions();
      const int* phys_coord = QMP_get_logical_coordinates();

      if (!Layout::fits_node_grid(nrow))
	{
	  QDPIO::cerr << "LayoutData::create - problem size " << nrow << "does not fit the node grid";
	  for(int i=0; i < Nd; ++i)
	    QDPIO::cerr << " " << phys_size[i];
	  QDPIO::cerr << " with a subgrid that is a multiple of " << min_dim << std::endl;
	  QDP_abort(1);
	}

      subgrid_nrow.resize(Nd);
      logical_coord.resize(Nd);
      logical_size.resize(Nd);

      subgrid_vol = 1;

      for(int i=0; i < Nd; ++i)
      {
	logical_coord[i] = phys_coord[i];
	logical_size[i] = phys_size[i];
	subgrid_nrow[i] = nrow[i] / logical_size[i];

	subgrid_vol *= subgrid_nrow[i];
      }

      // Diagnostics
      if (verbose)
	{
	  QDPIO::cout << "Lattice initialized:\n";
	  QDPIO::cout << "  problem size =";
	  for(int i=0; i < Nd; ++i)
	    QDPIO::cout << " " << nrow[i];
	  QDPIO::cout << endl;

	  QDPIO::cout << "  layout size =";
	  for(int i=0; i < Nd; ++i)
	    QDPIO::cout << " " << nrow_layout[i];
	  QDPIO::cout << endl;

	  QDPIO::cout << "  logical machine size =";
	  for(int i=0; i < Nd; ++i)
	    QDPIO::cout << " " << logical_size[i];
	  QDPIO::cout << endl;

	  QDPIO::cout << "  subgrid size =";
	  for(int i=0; i < Nd; ++i)
	    QDPIO::cout << " " << subgrid_nrow[i];
	  QDPIO::cout << endl;

	  QDPIO::cout << "  total number of nodes = " << numNodes() << endl;
	  QDPIO::cout << "  total volume = " << _vol << endl;
	  QDPIO::cout << "  subgrid volume = " << subgrid_vol << endl;
	  if ( iogrid_defined ) {
	    QDPIO::cout << "  Number of IO nodes = " << num_iogrid << endl;
	    QDPIO::cout << "  IO grid size =";
	    for(int i=0; i < Nd; i++)
	      QDPIO::cout << " " << iogrid[i];
	    QDPIO::cout << endl;
	  }
	}

      // Sanity check - check the QMP node number functions
      for(int node=0; node < numNodes(); ++node)
      { 
	multi1d<int> coord = getLogicalCoordFrom(node);
	int node2 = getNodeNumberFrom(coord);

	if (node != node2)
	  QDP_error_exit("LayoutData::create - Layout problems, the QMP logical to physical node map functions do not work correctly with this lattice size");
      }

      // Sanity check - check the layout functions make sense
#if QDP_DEBUG >= 2
	// BJ: Put this into a debug loop as it can take a serious amount of time for a really
        // large lattice
      for(int site=0; site < vol(); ++site) 
      {
	multi1d<int> coord1 = crtesn(site, lattSize());

	int linear = linearSiteIndex(coord1);
	int node   = nodeNumber(coord1);

	multi1d<int> coord2 = siteCoords(node,linear);
	int j = local_site(coord2, lattSize());
      
#if QDP_DEBUG >= 3
	QDP_info("site= %d   coord= %d %d %d %d   linear= %d node=%d   crd=%d %d %d %d   j= %d",
		 site,coord1[0],coord1[1],coord1[2],coord1[3],
		 linear,node,
		 coord2[0],coord2[1],coord2[2],coord2[3],
		 j);
#endif
	if (site != j)
	  QDP_error_exit("LayoutData::create - Layout problems, the layout functions do not work correctly with this lattice size");
      }
#endif
      // Initialize various defaults
      initDefaults(verbose);

      // Set layout init to 'true'
      _initialized = true;
      
      if (verbose)
	QDPIO::cout << "Finished lattice layout" << endl;

#if QDP_DEBUG >= 2
      QDP_info("Layout successfully initialized");
#endif
    }


//-----------------------------------------------------------------------------
#if QDP_USE_LEXICO_LAYOUT == 1

    //! The linearized site index for the corresponding coordinate
    /*! This layout is a simple lexicographic lattice ordering */
    int LayoutData::linearSiteIndex(const multi1d<int>& coord)
    {
      multi1d<int> tmp_coord(Nd);

      for(int i=0; i < coord.size(); ++i)
	tmp_coord[i] = coord[i] % LayoutData::subgridLattSize()[i];
    
      return local_site(tmp_coord, LayoutData::subgridLattSize());
    }


    //! The node number for the corresponding lattice coordinate
    /*! This layout is a simple lexicographic lattice ordering */
    int LayoutData::nodeNumber(const multi1d<int>& coord)
    {
      multi1d<int> tmp_coord(Nd);

      for(int i=0; i < coord.size(); ++i)
	tmp_coord[i] = coord[i] / LayoutData::subgridLattSize()[i];
    
      return LayoutData::getNodeNumberFrom(tmp_coord);
    }


    //! Returns the lattice site for some input node and linear index
    /*! This layout is a simple lexicographic lattice ordering */
    multi1d<int> LayoutData::siteCoords(int node, int linear)
    {
      multi1d<int> coord = getLogicalCoordFrom(node);

      // Get the base (origins) of the absolute lattice coord
      coord *= LayoutData::subgridLattSize();
    
      // Find the coordinate within a node and accumulate
      // This is a lexicographic ordering
      coord += crtesn(linear, LayoutData::subgridLattSize());

      return coord;
    }


    //! Return the smallest lattice size per node allowed
    /*! This layout is a simple lexicographic lattice ordering */
    multi1d<int> LayoutData::minimalLayoutMapping()
    {
      multi1d<int> dim(Nd);
      dim = 1;

      return dim;
    }

//-----------------------------------------------------------------------------

#elif QDP_USE_CB2_LAYOUT == 1


    //! The linearized site index for the corresponding coordinate
    /*! This layout is appropriate for a 2 checkerboard (red/black) lattice */
    int LayoutData::linearSiteIndex(const multi1d<int>& coord)
    {
      int subgrid_vol_cb = sitesOnNode() >> 1;
      multi1d<int> subgrid_cb_nrow = subgridLattSize();
      subgrid_cb_nrow[0] >>= 1;

      int cb = 0;
      for(int m=0; m < Nd; ++m)
	cb += coord[m];
      cb &= 1;

      multi1d<int> subgrid_cb_coord(Nd);
      subgrid_cb_coord[0] = (coord[0] >> 1) % subgrid_cb_nrow[0];
      for(int i=1; i < Nd; ++i)
	subgrid_cb_coord[i] = coord[i] % subgrid_cb_nrow[i];
    
      return local_site(subgrid_cb_coord, subgrid_cb_nrow) + cb*subgrid_vol_cb;
    }


    //! The node number for the corresponding lattice coordinate
    /*! 
     * This layout is appropriate for a 2 checkerboard (red/black) lattice,
     * but to find the nodeNumber this function resembles a simple lexicographic 
     * layout
     */
    int LayoutData::nodeNumber(const multi1d<int>& coord)
    {
      multi1d<int> tmp_coord(Nd);

      for(int i=0; i < coord.size(); ++i)
	tmp_coord[i] = coord[i] / subgridLattSize()[i];
    
      return getNodeNumberFrom(tmp_coord);
    }


    //! Reconstruct the lattice coordinate from the node and site number
    /*! 
     * This is the inverse of the nodeNumber and linearSiteIndex functions.
     * The API requires this function to be here.
     */
    multi1d<int> LayoutData::siteCoords(int node, int linearsite) // ignore node
    {
      int subgrid_vol_cb = sitesOnNode() >> 1;
      multi1d<int> subgrid_cb_nrow = subgridLattSize();
      subgrid_cb_nrow[0] >>= 1;

      // Get the base (origins) of the absolute lattice coord
      multi1d<int> coord = getLogicalCoordFrom(node);
      coord *= subgridLattSize();
    
      int cb = linearsite / subgrid_vol_cb;
      multi1d<int> tmp_coord = crtesn(linearsite % subgrid_vol_cb, subgrid_cb_nrow);

      // Add on position within the node
      // NOTE: the cb for the x-coord is not yet determined
      coord[0] += 2*tmp_coord[0];
      for(int m=1; m < Nd; ++m)
	coord[m] += tmp_coord[m];

      // Determine cb including global node cb
      int cbb = cb;
      for(int m=1; m < Nd; ++m)
	cbb += coord[m];
      coord[0] += (cbb & 1);

      return coord;
    }


    //! Return the smallest lattice size per node allowed
    /*! This layout is appropriate for a 2 checkerboard (red/black) lattice */
    multi1d<int> LayoutData::minimalLayoutMapping()
    {
      multi1d<int> dim(Nd);
      dim = 1;
      dim[0] = 2;       // must have multiple length 2 for cb

      return dim;
    }

//-----------------------------------------------------------------------------

#elif QDP_USE_CB3D_LAYOUT == 1


    //! The linearized site index for the corresponding coordinate
    /*! This layout is appropriate for a 2 checkerboard (red/black) lattice */
    int LayoutData::linearSiteIndex(const multi1d<int>& coord)
    {
      int subgrid_vol_cb = LayoutData::sitesOnNode() / 2;
      multi1d<int> subgrid_cb_nrow = LayoutData::subgridLattSize();
      subgrid_cb_nrow[0] /= 2;

      int cb = 0;
      for(int m=0; m < Nd-1; ++m) { // Nd-1 checkerboard
	cb += coord[m];
      }
      cb &= 1;

      multi1d<int> subgrid_cb_coord(Nd);
      subgrid_cb_coord[0] = (coord[0] / 2) % subgrid_cb_nrow[0];
      for(int i=1; i < Nd; ++i)
	subgrid_cb_coord[i] = coord[i] % subgrid_cb_nrow[i];
    
      return local_site(subgrid_cb_coord, subgrid_cb_nrow) + cb*subgrid_vol_cb;
    }


    //! The node number for the corresponding lattice coordinate
    /*! 
     * This layout is appropriate for a 2 checkerboard (red/black) lattice,
     * but to find the nodeNumber this function resembles a simple lexicographic 
     * layout
     */
    int LayoutData::nodeNumber(const multi1d<int>& coord)
    {
      multi1d<int> tmp_coord(Nd);

      for(int i=0; i < coord.size(); ++i)
	tmp_coord[i] = coord[i] / LayoutData::subgridLattSize()[i];
    
      return LayoutData::getNodeNumberFrom(tmp_coord);
    }


    //! Reconstruct the lattice coordinate from the node and site number
    /*! 
     * This is the inverse of the nodeNumber and linearSiteIndex functions.
     * The API requires this function to be here.
     */
    multi1d<int> LayoutData::siteCoords(int node, int linearsite) // ignore node
    {
      int subgrid_vol_cb = LayoutData::sitesOnNode() / 2;
      multi1d<int> subgrid_cb_nrow = LayoutData::subgridLattSize();
      subgrid_cb_nrow[0] /= 2;


      // Get the base (origins) of the absolute lattice coord
      multi1d<int> coord = getLogicalCoordFrom(node);
      coord *= LayoutData::subgridLattSize();
    
      int cb = linearsite / subgrid_vol_cb;
      multi1d<int> tmp_coord = crtesn(linearsite % subgrid_vol_cb, subgrid_cb_nrow);

      

      // Add on position within the node
      // NOTE: the cb for the x-coord is not yet determined
      coord[0] += 2*tmp_coord[0];
      for(int m=1; m < Nd; ++m)
	coord[m] += tmp_coord[m];



      // Determine cb including global node cb
      int cbb = cb;
      for(int m=1; m < Nd-1; ++m)
	cbb += coord[m];

   
      coord[0] += (cbb & 1);

      return coord;
    }


    //! Return the smallest lattice size per node allowed
    /*! This layout is appropriate for a 2 checkerboard (red/black) lattice */
    multi1d<int> LayoutData::minimalLayoutMapping()
    {
      multi1d<int> dim(Nd);
      dim = 1;
      dim[0] = 2;       // must have multiple length 2 for cb

      return dim;
    }


//-----------------------------------------------------------------------------

#elif QDP_USE_CB32_LAYOUT == 1

#warning "Using a 32 checkerboard layout"

#error "THIS BIT STILL UNDER CONSTRUCTION"

  namespace Layout
  {
    //! The linearized site index for the corresponding coordinate
    /*! This layout is appropriate for a 32-style checkerboard lattice */
    int linearSiteIndex(const multi1d<int>& coord)
    {
      int subgrid_vol_cb = LayoutData::sitesOnNode() >> (Nd+1);
      multi1d<int> subgrid_cb_nrow = LayoutData::subgridLattSize();
      subgrid_cb_nrow[0] >>= 2;
      for(int i=1; i < Nd; ++i) 
	subgrid_cb_nrow[i] >>= 1;

      int subl = coord[Nd-1] & 1;
      for(int m=Nd-2; m >= 0; --m)
	subl = (subl << 1) + (coord[m] & 1);

      int cb = 0;
      for(int m=0; m < Nd; ++m)
	cb += coord[m] >> 1;

      subl += (cb & 1) << Nd;   // Final color or checkerboard

      // Construct the checkerboard lattice coord
      multi1d<int> subgrid_cb_coord(Nd);

      subgrid_cb_coord[0] = (coord[0] >> 2) % subgrid_cb_nrow[0];
      for(int i=1; i < Nd; ++i)
	subgrid_cb_coord[i] = (coord[i] >> 1) % subgrid_cb_nrow[i];
    
      return local_site(subgrid_cb_coord, subgrid_cb_nrow) + subl*subgrid_vol_cb;
    }


    //! The node number for the corresponding lattice coordinate
    /*! 
     * This layout is appropriate for a 32 checkerboard (red/black) lattice,
     * but to find the nodeNumber this function resembles a simple lexicographic 
     * layout
     */
    int nodeNumber(const multi1d<int>& coord)
    {
      multi1d<int> tmp_coord(Nd);

      for(int i=0; i < coord.size(); ++i)
	tmp_coord[i] = coord[i] / LayoutData::subgridLattSize()[i];
    
      return LayoutData::getNodeNumberFrom(tmp_coord);
    }


    //! Reconstruct the lattice coordinate from the node and site number
    /*! 
     * This is the inverse of the nodeNumber and linearSiteIndex functions.
     * The API requires this function to be here.
     */
    multi1d<int> siteCoords(int node, int linearsite) // ignore node
    {
      int subgrid_vol_cb = LayoutData::sitesOnNode() >> (Nd+1);
      multi1d<int> subgrid_cb_nrow = LayoutData::subgridLattSize();
      subgrid_cb_nrow[0] >>= 2;
      for(int i=1; i < Nd; ++i) 
	subgrid_cb_nrow[i] >>= 1;

      // Get the base (origins) of the absolute lattice coord
      multi1d<int> coord = LayoutData::getLogicalCoordFrom(node);
      coord *= LayoutData::subgridLattSize();
    
      int subl = linearsite / subgrid_vol_cb;
      multi1d<int> tmp_coord = crtesn(linearsite % subgrid_vol_cb, subgrid_cb_nrow);

      // Add on position within the node
      // NOTE: the cb for the x-coord is not yet determined
      coord[0] += tmp_coord[0] << 2;
      for(int m=1; m < Nd; ++m)
	coord[m] += tmp_coord[m] << 1;

      int cb = 0;
      for(int m=1; m<Nd; ++m)
	cb += coord[m];
      cb &= 1;

      subl ^= (cb << Nd);
      for(int m=0; m<Nd; ++m)
	coord[m] ^= (subl & (1 << m)) >> m;
      coord[0] ^= (subl & (1 << Nd)) >> (Nd-1);   // this gets the hypercube cb

      return coord;
    }


    //! Return the smallest lattice size per node allowed
    /*! This layout is appropriate for a 32-style checkerboard lattice */
    multi1d<int> minimalLayoutMapping()
    {
      multi1d<int> dim(Nd);

      dim[0] = 4;       // must have multiple length 2 for cb and hypercube
      for(int m=1; m < Nd; ++m)
	dim[m] = 2;     //  must have multiple length 2 for hypercube

      return dim;
    }
  }

#else

#error "no appropriate layout defined"

#endif

//-----------------------------------------------------------------------------


} // namespace QDP;
