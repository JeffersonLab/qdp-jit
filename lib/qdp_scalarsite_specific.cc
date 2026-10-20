#include "qdp.h"
#include "qdp_util.h"

namespace QDP {


namespace Layout
{

  typedef std::map< std::pair< int , int > , LatticeInteger > QDPLatCoord_t;

  namespace {
    QDPLatCoord_t* _latCoord = nullptr;
  }

  QDPLatCoord_t& qdp_jit_latCoord()
  {
    if (!_latCoord)
      _latCoord = new QDPLatCoord_t;
    return *_latCoord;
  }
    

  void destroyLatticeCoordinate()
  {
    qdp_jit_latCoord().clear();
  }

  
  
  LatticeInteger latticeCoordinate(int mu)
  {
    if (mu < 0 || mu >= Nd)
      QDP_error_exit("dimension out of bounds");

    int ref = Layout::current().index();
    auto key = std::make_pair( ref , mu );

    if ( ! qdp_jit_latCoord().count( key ) )
      {
	LatticeInteger d;
	const int nodeSites = Layout::sitesOnNode();
	const int nodeNumber = Layout::nodeNumber();
	for(int i=0; i < nodeSites; ++i) 
	  {
	    Integer cc = Layout::siteCoords(nodeNumber,i)[mu];
	    multi1d<int> coord = Layout::siteCoords(nodeNumber,i);
	    pokeSite( d , cc , coord );
	  }
	qdp_jit_latCoord()[ key ] = d;
      }

    return qdp_jit_latCoord()[ key ];
  }
}


//-----------------------------------------------------------------------------
// IO routine solely for debugging. Only defined here
template<class T>
ostream& operator<<(ostream& s, const multi1d<T>& s1)
{
  for(int i=0; i < s1.size(); ++i)
    s << " " << s1[i];
  return s;
}


//-----------------------------------------------------------------------------
//! Constructor from a function object
void Set::make(const SetFunc& fun)
{
  if (set_index < 0)
    {
      set_index = set_counter++;
    }
  
  int ref = Layout::current().index();

  if (registered && layout_done(ref))
    {
      for (int i = 0 ; i < sub.size() ; ++i)
	MasterMap::Instance().drop_subset( sub[i].getId() );
      MasterSet::Instance().remade( *this );
    }

  int nsubset_indices = fun.numSubsets();
  const int nodeSites = Layout::sitesOnNode();
  const int nodeNumber = Layout::nodeNumber();

#if QDP_DEBUG >= 2
  QDP_info("Set a subset: nsubset = %d",nsubset_indices);
#endif

  if (registered && sub.size() != nsubset_indices)
    {
      QDPIO::cerr << "Set::make: remaking a set with a different number of subsets ("
		  << sub.size() << " -> " << nsubset_indices << ") is not supported" << std::endl;
      QDP_abort(1);
    }

  // This actually allocates the subsets
  if (sub.size() != nsubset_indices)
    {
      sub.resize(nsubset_indices);
    }

  // Create the space of the colorings of the lattice
  if (lat_color[ref].size() != nodeSites)
    lat_color[ref].resize(nodeSites);

  // Create the array holding the array of sitetable info
  if (sitetables[ref].size() != nsubset_indices)
    sitetables[ref].resize(nsubset_indices);

  // Create the array holding the array of membertable info
  if (membertables[ref].size() != nsubset_indices)
    membertables[ref].resize(nsubset_indices);

  for (int i = 0 ; i < idSiteTable[ref].size() ; ++i)
    if (idSiteTable[ref][i] != -1)
      QDP_get_global_cache().signoff( idSiteTable[ref][i] );

  if (idSiteTable[ref].size() != nsubset_indices)
    idSiteTable[ref].resize(nsubset_indices);

  
  // Loop over linear sites determining their color
  for(int linear=0; linear < nodeSites; ++linear)
  {
    multi1d<int> coord = Layout::siteCoords(nodeNumber, linear);

    int node   = Layout::nodeNumber(coord);
    int lin    = Layout::linearSiteIndex(coord);
    int icolor = fun(coord);

#if QDP_DEBUG >= 3
    cerr<<"linear="<<linear<<" coord="<<coord<<" node="<<node<<" col="<<icolor << endl;
#endif

    // Sanity checks
    if (node != nodeNumber)
      QDP_error_exit("Set: found site with node outside current node!");

    if (lin != linear)
      QDP_error_exit("Set: inconsistent linear sites");

    if (icolor < 0 || icolor >= nsubset_indices)
      QDP_error_exit("Set: coloring is outside legal range: color[%d]=%d",linear,icolor);

    // The coloring of this linear site
    lat_color[ref][linear] = icolor;
  }


  for(int cb=0; cb < nsubset_indices; ++cb)
  {
    multi1d<bool>& membertable = membertables[ref][cb];
    membertable.resize(nodeSites);

    int num_sitetable = 0;
    for(int linear=0; linear < nodeSites; ++linear)
      if (lat_color[ref][linear] == cb) {
	++num_sitetable;
	membertable[linear] = true;
      } else {
	membertable[linear] = false;
      }
    
    multi1d<int>& sitetable = sitetables[ref][cb];
    sitetable.resize(num_sitetable);

    bool ordRep;
    int start, end;

    if (num_sitetable > 0)
    {
      for(int i=0; i < num_sitetable; ++i)
	sitetable[i] = -1;

      for(int linear=0, j=0; linear < nodeSites; ++linear)
	if (lat_color[ref][linear] == cb)
	  sitetable[j++] = linear;
    }


    idSiteTable[ref][cb] = sitetable.size()   > 0 ? QDP_get_global_cache().addOwnHostMemNoPage( sitetable.size()   * sizeof(int)  , sitetable.slice() ) : -1 ;

    sub[cb].make( ref , &sitetables[ref][cb], &idSiteTable[ref][cb], cb, this, &membertables[ref][cb] ); // -1 for the masterset id which is still unknown

#if QDP_DEBUG >= 2
    QDP_info("Subset(%d)",cb);
#endif
  } // cb

  if (!registered)
    {
      MasterSet::Instance().add( *this );
    }
  
  registered=true;
}
	  

} // namespace QDP;
