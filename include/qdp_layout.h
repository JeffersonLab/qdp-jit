#ifndef QDP_LAYOUT_H
#define QDP_LAYOUT_H

/*! @file
 * @brief Lattice layout
 *
 * Lattice layout namespace and operations
 */

namespace QDP {

namespace Layout
{
  void init();
  void setLattSize(const multi1d<int>& nrows);
  void setLattSize(const std::vector<int>& nrows);
  void create();

  LatticeLayout current();
  LatticeLayout find_or_create( const multi1d<int>& nrow );
  void push( LatticeLayout layout );
  void pop();
  
  //! Returns the logical node number for the corresponding lattice coordinate
  /*! The API requires this function to be here */
  int nodeNumber(const multi1d<int>& coord) ;

  //! The linearized site index within a node for the corresponding lattice coordinate
  /*! The API requires this function to be here */
  int linearSiteIndex(const multi1d<int>& coord) ;

  //! Reconstruct the lattice coordinate from the node and site number
  /*! 
   * This is the inverse of the nodeNumber and linearSiteIndex functions.
   * The API requires this function to be here.
   */
  multi1d<int> siteCoords(int node, int index) ;

#if 0
  extern "C" { 
    /* Export this to "C" */
    void QDPXX_getSiteCoords(int coord[], int node, int linear) ;
    int QDPXX_getLinearSiteIndex(const int coord[]);
    int QDPXX_nodeNumber(const int coord[]);

  }
#endif
  
  //! Returns the node number of this node
  int nodeNumber() ;

  //! Returns the number of nodes
  int numNodes() ;
	

  //! Virtual grid (problem grid) lattice size
  const multi1d<int>& lattSize() ;

  //! Total lattice volume
  size_t vol() ;

  //! Number of sites on node
  int sitesOnNode() ;

  //! Returns whether this is the primary node
  bool primaryNode() ;

  //! The linearized site index for the corresponding lexicographic site
  int linearSiteIndex(int lexicosite) ;

  //! Returns the logical node coordinates for this node
  const multi1d<int>& nodeCoord() ;

  //! Returns the logical node coordinates for the corresponding lattice coordinate
  multi1d<int> nodeCoord(const multi1d<int>& coord);

  //! Subgrid (grid on each node) lattice size
  const multi1d<int>& subgridLattSize() ;

  //! Returns the logical size of this machine
  const multi1d<int>& logicalSize() ;

  //! Returns the node number given some logical node coordinate
  /*! This is not meant to be speedy */
  int getNodeNumberFrom(const multi1d<int>& node_coord);

  //! Returns the logical node coordinates given some node number
  /*! This is not meant to be speedy */
  multi1d<int> getLogicalCoordFrom(int node);


  //! Check if I/O grid is defined
  bool isIOGridDefined(void) ;

  //! set the IO node grid
  void setIONodeGrid(const multi1d<int>& io_grid);

  //! number of I/O nodes
  int numIONodeGrid(void) ;
	
  //! Get the I/O Node grid
  const multi1d<int>& getIONodeGrid() ;



} // Layout


  class LayoutScope
  {
  public:
    explicit LayoutScope( LatticeLayout layout ) { Layout::push( layout ); }
    ~LayoutScope() { close(); }

    void close()
    {
      if (active)
	{
	  Layout::pop();
	  active = false;
	}
    }

    LayoutScope( const LayoutScope& ) = delete;
    LayoutScope& operator=( const LayoutScope& ) = delete;

  private:
    bool active = true;
  };


  class LayoutShapeScope
  {
  public:
    LayoutShapeScope( LatticeLayout layout , int ns , int nc )
    {
      Layout::push( layout );
      Shape::push( Shape( ns , nc ) );
    }

    template<class T>
    explicit LayoutShapeScope( const OLattice<T>& f ) : LayoutShapeScope( f.layout() , f.shape().ns() , f.shape().nc() ) {}

    ~LayoutShapeScope() { close(); }

    void close()
    {
      if (active)
	{
	  Shape::pop();
	  Layout::pop();
	  active = false;
	}
    }

    LayoutShapeScope( const LayoutShapeScope& ) = delete;
    LayoutShapeScope& operator=( const LayoutShapeScope& ) = delete;

  private:
    bool active = true;
  };


  template<class RHS, class T1>
  LatticeLayout get_layout_from_rhs(const QDPExpr<RHS, OLattice<T1> >& rhs)
  {
    std::vector<int> layouts;
    CollectLayouts cl(layouts);
    int dummy = forEach(rhs, cl , BitOrCombine());

    if (layouts.size() < 1)
      {
	QDPIO::cout << "Weird sum.\n";
	sleep(1);
	QDP_abort(1);
      }

    LatticeLayout finest = Layout::from_index( layouts[0] );
    for( int i : layouts )
      {
	const LatticeLayout l = Layout::from_index( i );
	if ( l.vol() > finest.vol() )
	  finest = l;
      }
    return finest;
  }


  template<class RHS, class T1>
  LatticeLayout get_layout_from_rhs(const QDPExpr<RHS, OScalar<T1> >& rhs)
  {
    return LatticeLayout();
  }

  

  template<class T> 
  LatticeLayout get_layout_from_rhs(const QDPType< T , OLattice<T> >& rhs)
  {
    return rhs.layout();
  }
      

  template<class T> 
  LatticeLayout get_layout_from_rhs(const QDPType< T , OScalar<T> >& rhs)
  {
    return LatticeLayout();
  }
  



  
//! Declaration of shift function object
extern ArrayBiDirectionalMap  shift;


/*! @} */   // end of group layout

} // QDP

#endif
