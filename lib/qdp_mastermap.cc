// -*- C++ -*-

#include "qdp.h"
#include "qdp_util.h"

namespace QDP {


  MasterMap& MasterMap::Instance()
  {
    static MasterMap singleton;
    return singleton;
  }


  void MasterMap::remove_neg(multi1d<int>& out, const multi1d<int>& orig) const {
    multi1d<int> c(Layout::sitesOnNode());
    int num=0;
    for(int i=0 ; i<Layout::sitesOnNode() ; ++i) 
      if (orig[i] >= 0)
	c[num++]=i;
    out.resize( num );
    for(int i=0; i < num; ++i)
      out[i] = c[i];
  }



  void MasterMap::remove_neg_in_subset(multi1d<int>& out, const multi1d<int>& orig, int s_no ) const 
  {
    const Subset& subset = MasterSet::Instance().getSubset(s_no);

    multi1d<int> c(Layout::sitesOnNode());

    int num=0;
    for(int i=0 ; i<Layout::sitesOnNode() ; ++i) 
      if (orig[i] >= 0 && subset.isElement(i))
	c[num++]=i;

    out.resize( num );

    for(int i=0; i < num; ++i)
      out[i] = c[i];
  }



  void MasterMap::uniquify_list_inplace(multi1d<int>& out , const multi1d<int>& ll) const
  {
    multi1d<int> d(ll.size());

    // Enter the first element as unique to prime the search
    int ipos = 0;
    int num = 0;
    int prev_node;
  
    d[num++] = prev_node = ll[ipos++];

    // Find the unique source nodes
    while (ipos < ll.size())
      {
	int this_node = ll[ipos++];

	if (this_node != prev_node)
	  {
	    // Has this node occured before?
	    bool found = false;
	    for(int i=0; i < num; ++i)
	      if (d[i] == this_node)
		{
		  found = true;
		  break;
		}

	    // If this is the first time this value has occurred, enter it
	    if (! found)
	      d[num++] = this_node;
	  }

	prev_node = this_node;
      }

    // Copy into a compact size array
    out.resize(num);
    for(int i=0; i < num; ++i) {
      out[i] = d[i];
    }

  }





  int MasterMap::registrate(const Map& map) {
    //QDP_info("Map registered id=%d (total=%d)",1 << vecPMap.size(),vecPMap.size()+1 );
    int id = 1 << vecPMap.size();
    vecPMap.push_back(&map);

    

    for (int s_no = 0 ; s_no < MasterSet::Instance().numSubsets() ; ++s_no ) 
      {
	const Subset& subset = MasterSet::Instance().getSubset(s_no);

	//QDP_info("Resizing power set to %d", id << 1 );
	powerSet[s_no].resize( id << 1 );
	powerSetC[s_no].resize( id << 1 );

	for (int i = 0 ; i < id ; ++i ) {

	  multi1d<int> ct(Layout::sitesOnNode()); // complement, inner region
	  multi1d<int> pt(Layout::sitesOnNode()); // positive, union of receive sites
	  for(int q=0 ; q<Layout::sitesOnNode() ; ++q) {
	    ct[q]=q;
	    pt[q]=-1;
	  }

	  for (int q = 0 ; q < powerSet[s_no][i]->size() ; ++q ) {
	    ct[ (*powerSet[s_no][i])[q] ] = -1;
	    pt[ (*powerSet[s_no][i])[q] ] = (*powerSet[s_no][i])[q];
	  }

	  for (int q = 0; q < map.roffset( subset ).size() ; ++q ) {
	    ct[ map.roffset( subset )[q] ] = -1;
	    pt[ map.roffset( subset )[q] ] = map.roffset( subset )[q];
	  }

	  powerSet[s_no][i|id] = new multi1d<int>;
	  powerSetC[s_no][i|id]= new multi1d<int>;

	  // remove_neg( *powerSetC[i|id] , ct );
	  // remove_neg( *powerSet[i|id] , pt );

	  remove_neg_in_subset( *powerSetC[s_no][i|id] , ct , s_no );
	  remove_neg_in_subset( *powerSet[s_no][i|id] , pt , s_no );
	}
      }
    return id;
  }

  const multi1d<int>& MasterMap::getInnerSites(const Subset& s,int bitmask) const {
    assert( s.getId() >= 0 && s.getId() < powerSet.size() && "subset Id out of range");
    if ( bitmask < 1 || bitmask > powerSetC.size()-1 )
      QDP_error_exit("internal error: get count inner %d",bitmask);
    return *powerSetC[s.getId()][bitmask];
  }
  const multi1d<int>& MasterMap::getFaceSites(const Subset& s,int bitmask) const {
    assert( s.getId() >= 0 && s.getId() < powerSet.size() && "subset Id out of range");
    if ( bitmask < 1 || bitmask > powerSet.size()-1 )
      QDP_error_exit("internal error: get count face %d",bitmask);
    return *powerSet[s.getId()][bitmask]; 
  }

  int MasterMap::getCountInner(const Subset& s,int bitmask) const {
    assert( s.getId() >= 0 && s.getId() < powerSet.size() && "subset Id out of range");
    if ( bitmask < 1 || bitmask > powerSetC.size()-1 )
      QDP_error_exit("internal error: get count inner %d",bitmask);
    return powerSetC[s.getId()][bitmask]->size(); 
  }
  int MasterMap::getCountFace(const Subset& s,int bitmask) const {
    assert( s.getId() >= 0 && s.getId() < powerSet.size() && "subset Id out of range");
    if ( bitmask < 1 || bitmask > powerSet.size()-1 )
      QDP_error_exit("internal error: get count face %d",bitmask);
    return powerSet[s.getId()][bitmask]->size(); 
  }


} // namespace QDP


