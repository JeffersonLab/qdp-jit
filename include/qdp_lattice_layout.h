#ifndef QDP_LATTICE_LAYOUT_H
#define QDP_LATTICE_LAYOUT_H


namespace QDP {

class LatticeLayout;

namespace Layout {
  LatticeLayout from_index( int index );
}

class LatticeLayout
{
public:
  LatticeLayout() = default;

  int index() const { return idx; }

  const multi1d<int>& lattSize() const;
  const multi1d<int>& subgridLattSize() const;
  size_t vol() const;
  int sitesOnNode() const;

  int nodeNumber() const;
  int numNodes() const;
  bool primaryNode() const;
  const multi1d<int>& logicalSize() const;

  int nodeNumber( const multi1d<int>& coord ) const;
  int linearSiteIndex( const multi1d<int>& coord ) const;
  multi1d<int> siteCoords( int node , int index ) const;

  LatticeLayout coarsen( const multi1d<int>& blocking ) const;

  friend bool operator==( LatticeLayout a , LatticeLayout b ) { return a.idx == b.idx; }
  friend bool operator!=( LatticeLayout a , LatticeLayout b ) { return a.idx != b.idx; }
  friend bool operator< ( LatticeLayout a , LatticeLayout b ) { return a.idx <  b.idx; }

private:
  explicit LatticeLayout( int index ) : idx(index) {}
  friend LatticeLayout Layout::from_index( int index );

  int idx = -1;
};

} // QDP

template<>
struct std::hash< QDP::LatticeLayout >
{
  size_t operator()( QDP::LatticeLayout l ) const noexcept { return std::hash<int>()( l.index() ); }
};

namespace QDP {

  class LayoutData;

namespace Layout {

  bool fits_node_grid( const multi1d<int>& nrow );
  LayoutData& data( int index );
  void finalize();

}

  class LayoutData
  {
  public:
    ~LayoutData() {
      //printf("~LayoutData\n");
    }
    
    //! Initialize some fundamental pieces of the layout
    /*! This routine is used to boostrap the   create   function below */
    void init(int pos);

    //! Main lattice creation routine
    void create();

    //! Main destruction routine
    void destroy();

    //! Set lattice size -- problem size
    void setLattSize(const multi1d<int>& nrows);

    // Is the Layout initialized ?
    bool initialized();
  
    //! Returns the logical node number for the corresponding lattice coordinate
    /*! The API requires this function to be here */
    int nodeNumber(const multi1d<int>& coord);

    //! The linearized site index within a node for the corresponding lattice coordinate
    /*! The API requires this function to be here */
    int linearSiteIndex(const multi1d<int>& coord);

    //! Reconstruct the lattice coordinate from the node and site number
    /*! 
     * This is the inverse of the nodeNumber and linearSiteIndex functions.
     * The API requires this function to be here.
     */
    multi1d<int> siteCoords(int node, int index);

#if 0
    extern "C" { 
      /* Export this to "C" */
      void QDPXX_getSiteCoords(int coord[], int node, int linear);
      int QDPXX_getLinearSiteIndex(const int coord[]);
      int QDPXX_nodeNumber(const int coord[]);
    }
#endif
    
    //! Returns the node number of this node
    int nodeNumber();

    //! Returns the number of nodes
    int numNodes();
	

    //! Virtual grid (problem grid) lattice size
    const multi1d<int>& lattSize();

    //! Total lattice volume
    size_t vol();

    //! Number of sites on node
    int sitesOnNode();

    //! Returns whether this is the primary node
    bool primaryNode();

    //! The linearized site index for the corresponding lexicographic site
    int linearSiteIndex(int lexicosite);

    //! Returns the logical node coordinates for this node
    const multi1d<int>& nodeCoord();

    //! Returns the logical node coordinates for the corresponding lattice coordinate
    multi1d<int> nodeCoord(const multi1d<int>& coord);

    //! Subgrid (grid on each node) lattice size
    const multi1d<int>& subgridLattSize();

    //! Returns the logical size of this machine
    const multi1d<int>& logicalSize();

    //! Returns the node number given some logical node coordinate
    /*! This is not meant to be speedy */
    int getNodeNumberFrom(const multi1d<int>& node_coord);

    //! Returns the logical node coordinates given some node number
    /*! This is not meant to be speedy */
    multi1d<int> getLogicalCoordFrom(int node);

    //! Check if I/O grid is defined
    bool isIOGridDefined(void);

    //! set the IO node grid
    void setIONodeGrid(const multi1d<int>& io_grid);

    //! number of I/O nodes
    int numIONodeGrid(void);
	
    //! Get the I/O Node grid
    const multi1d<int>& getIONodeGrid();

    int ref() const;


  private:
    void initDefaults(bool verbose);

    static multi1d<int> minimalLayoutMapping();

    //! Total lattice volume
    size_t _vol;

    //! Lattice size
    multi1d<int> nrow;

    //! Subgrid lattice volume
    int subgrid_vol;

    //! Subgrid lattice size
    multi1d<int> subgrid_nrow;

    //! Logical node coordinates
    multi1d<int> logical_coord;

    //! Logical system size
    multi1d<int> logical_size;

    //! Node rank
    int node_rank;

    //! Total number of nodes
    int num_nodes;

    bool iogrid_defined;
    int  num_iogrid;
    multi1d<int> iogrid;

    bool _initialized = false;

    int pos = -1;
    
    static bool qmp_layout_done;

    friend bool Layout::fits_node_grid( const multi1d<int>& nrow );
  };

} // QDP
#endif
