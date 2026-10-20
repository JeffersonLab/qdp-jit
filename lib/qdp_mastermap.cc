#include "qdp.h"

#include <algorithm>
#include <bit>

namespace QDP {

  MasterMap& MasterMap::Instance()
  {
    static MasterMap* singleton = new MasterMap;
    return *singleton;
  }


  int MasterMap::register_map(const Map& map)
  {
    auto free = std::find( maps.begin() , maps.end() , nullptr );
    int bit = free - maps.begin();

    if (bit >= 8 * (int)sizeof(int) - 1)
      {
	QDPIO::cerr << "MasterMap: too many maps with off-node communication (at most "
		    << 8 * sizeof(int) - 1 << " at a time are supported)" << std::endl;
	QDP_abort(1);
      }

    if (free == maps.end())
      maps.push_back(&map);
    else
      *free = &map;

    return 1 << bit;
  }


  template<class Pred>
  void MasterMap::drop_tables(Pred pred)
  {
    for ( auto it = tables.begin() ; it != tables.end() ; )
      {
	if (pred( it->first ))
	  {
	    for ( int id : { it->second.id_inner , it->second.id_face } )
	      if (id >= 0)
		QDP_get_global_cache().signoff( id );
	    it = tables.erase( it );
	  }
	else
	  ++it;
      }
  }


  void MasterMap::unregister_map(int id)
  {
    int bit = std::countr_zero( (unsigned)id );
    assert( id > 0 && bit < (int)maps.size() && maps[bit] );
    maps[bit] = nullptr;

    drop_tables( [id](const auto& key) { return std::get<2>(key) & id; } );
  }


  void MasterMap::drop_subset(int subset_id)
  {
    drop_tables( [subset_id](const auto& key) { return std::get<1>(key) == subset_id; } );
  }


  void MasterMap::generate_tables(Storage& t, const Subset& subset, LatticeLayout layout, int mask) const
  {
    std::vector<char> is_face( layout.sitesOnNode() , 0 );

    for( int bit = 0 ; bit < (int)maps.size() ; ++bit )
      {
	if ( (1 << bit) & mask )
	  {
	    const Map& map = *maps[ bit ];

	    map.make_lazy( subset , layout.index() );

	    const multi1d<int>& roffset = map.roffset( subset , layout.index() );
	    for (int q = 0 ; q < roffset.size() ; ++q )
	      is_face[ roffset[q] ] = 1;
	  }
      }

    const multi1d<int>& sites = subset.siteTable(layout);
    for (int i = 0 ; i < sites.size() ; ++i )
      {
	if (is_face[ sites[i] ])
	  t.face.push_back( sites[i] );
	else
	  t.inner.push_back( sites[i] );
      }

    t.id_inner = QDP_get_global_cache().addOwnHostMemNoPage( t.inner.size() * sizeof(int) , t.inner.data() );
    t.id_face  = QDP_get_global_cache().addOwnHostMemNoPage( t.face.size()  * sizeof(int) , t.face.data() );
  }


  MasterMap::Tables MasterMap::get(const Subset& s, LatticeLayout layout, int mask)
  {
    if (!mask)
      return { s.numSiteTable(layout) , s.getIdSiteTable(layout) , s.siteTable(layout).slice() , 0 , -1 , nullptr };

    auto [it, inserted] = tables.try_emplace( std::make_tuple( layout.index() , s.getId() , mask ) );
    Storage& t = it->second;

    if (inserted)
      generate_tables( t , s , layout , mask );

    return { (int)t.inner.size() , t.id_inner , t.inner.data() ,
	     (int)t.face.size()  , t.id_face  , t.face.data() };
  }

} // namespace QDP
