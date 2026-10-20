#include "qdp.h"
#include "qdp_util.h"

namespace QDP {

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

  const multi1d<int>& LatticeLayout::lattSize() const
  {
    return Layout::data(idx).lattSize();
  }

  const multi1d<int>& LatticeLayout::subgridLattSize() const
  {
    return Layout::data(idx).subgridLattSize();
  }

  size_t LatticeLayout::vol() const
  {
    return Layout::data(idx).vol();
  }

  int LatticeLayout::sitesOnNode() const
  {
    return Layout::data(idx).sitesOnNode();
  }

  int LatticeLayout::nodeNumber() const
  {
    return Layout::data(idx).nodeNumber();
  }

  int LatticeLayout::numNodes() const
  {
    return Layout::data(idx).numNodes();
  }

  bool LatticeLayout::primaryNode() const
  {
    return Layout::data(idx).primaryNode();
  }

  const multi1d<int>& LatticeLayout::logicalSize() const
  {
    return Layout::data(idx).logicalSize();
  }

  int LatticeLayout::nodeNumber( const multi1d<int>& coord ) const
  {
    return Layout::data(idx).nodeNumber( coord );
  }

  int LatticeLayout::linearSiteIndex( const multi1d<int>& coord ) const
  {
    return Layout::data(idx).linearSiteIndex( coord );
  }

  multi1d<int> LatticeLayout::siteCoords( int node , int index ) const
  {
    return Layout::data(idx).siteCoords( node , index );
  }

  LatticeLayout LatticeLayout::coarsen( const multi1d<int>& blocking ) const
  {
    const multi1d<int>& latt = lattSize();

    bool divides = blocking.size() == latt.size();
    for ( int i = 0 ; divides && i < blocking.size() ; ++i )
      divides = blocking[i] > 0 && latt[i] % blocking[i] == 0;

    if (!divides)
      {
	QDPIO::cerr << "LatticeLayout::coarsen: blocking " << blocking << "does not divide the lattice " << latt << endl;
	QDP_abort(1);
      }

    multi1d<int> nrow(blocking.size());
    for ( int i = 0 ; i < blocking.size() ; ++i )
      nrow[i] = latt[i] / blocking[i];

    return Layout::find_or_create( nrow );
  }


namespace Layout {

  namespace
  {
    LayoutData& current_data()
    {
      return data( current().index() );
    }
  }

  void setLattSize(const multi1d<int>& nrow)
  {
    current_data().setLattSize( nrow );
  }
  
  void setLattSize(const std::vector<int>& _nrow)
  {
    multi1d<int> nrow(_nrow.size());
    for ( int i = 0 ; i < _nrow.size() ; ++i )
      nrow[i] = _nrow[i];
    setLattSize(nrow);
  }

  void create()
  {
    current_data().create();
  }

  
  
  
  int nodeNumber(const multi1d<int>& coord)
  {
    return current_data().nodeNumber(coord);
  }

  int linearSiteIndex(const multi1d<int>& coord) 
  {
    return current_data().linearSiteIndex(coord);
  }

  multi1d<int> siteCoords(int node, int index) 
  {
    return current_data().siteCoords(node,index);
  }

  int nodeNumber() 
  {
    return current_data().nodeNumber();
  }

  int numNodes() 
  {
    return current_data().numNodes();
  }

  const multi1d<int>& lattSize() 
  {
    return current_data().lattSize();
  }

  size_t vol() 
  {
    return current_data().vol();
  }

  int sitesOnNode() 
  {
    return current_data().sitesOnNode();
  }

  bool primaryNode()
  {
    return current_data().primaryNode();
  }

  int linearSiteIndex(int lexicosite)
  {
    return current_data().linearSiteIndex(lexicosite);
  }

  const multi1d<int>& nodeCoord()
  {
    return current_data().nodeCoord();
  }

  multi1d<int> nodeCoord(const multi1d<int>& coord)
  {
    return current_data().nodeCoord(coord);
  }
  
  const multi1d<int>& subgridLattSize()
  {
    return current_data().subgridLattSize();
  }
  
  const multi1d<int>& logicalSize()
  {
    return current_data().logicalSize();
  }

  int getNodeNumberFrom(const multi1d<int>& node_coord)
  {
    return current_data().getNodeNumberFrom(node_coord);
  }

  multi1d<int> getLogicalCoordFrom(int node)
  {
    return current_data().getLogicalCoordFrom(node);
  }

  bool isIOGridDefined()
  {
    return current_data().isIOGridDefined();
  }

  void setIONodeGrid(const multi1d<int>& io_grid)
  {
    return current_data().setIONodeGrid(io_grid);
  }
  
  int numIONodeGrid()
  {
    return current_data().numIONodeGrid();
  }
	
  const multi1d<int>& getIONodeGrid()
  {
    return current_data().getIONodeGrid();
  }

  
} // Layout

} // QDP
