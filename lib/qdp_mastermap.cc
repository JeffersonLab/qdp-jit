// -*- C++ -*-

#include "qdp.h"
#include "qdp_util.h"

namespace QDP {


  MasterMap& MasterMap::Instance()
  {
    static MasterMap singleton;
    return singleton;
  }




  namespace {
    void remove_neg_in_subset(std::vector<int>& out, const std::vector<int>& orig, int s_no )
    {
      const Subset& subset = MasterSet::Instance().getSubset(s_no);

      out.resize(0);

      for(int i=0 ; i<Layout::sitesOnNode() ; ++i) 
	if (orig[i] >= 0 && subset.isElement(i))
	  out.push_back( i );
    }
  }




  
  void MasterMap::generate_tables(const Subset& subset, int bitmask)
  {
    int s_no = subset.getId();
    auto key = make_pair(s_no,bitmask);

    //QDPIO::cout << "generate_tables  subset = " << s_no << "  bitmask = " << bitmask << std::endl;
    
    //
    // check if (subset,map) already generated
    //
    if (mapTables.count( key ))
      return;

    //QDPIO::cout << "generating tables for subset_id = " << s_no << " bitmask = " << bitmask << std::endl;

    tables t;

    std::vector<int> t_roffset( Layout::sitesOnNode() , -1 );
#ifdef QDP_CODEGEN_VECTOR
    std::vector<int> t_loffset( Layout::sitesOnNode() , -1 );
    std::vector<int> t_innerVNodeSIMD( Layout::sitesOnNode() , -1 );
#endif
    
    std::vector<int> t_innerScalar( Layout::sitesOnNode());
    for( int q = 0 ; q < Layout::sitesOnNode() ; ++q )
      {
	t_innerScalar[ q ] = q ;
      }

#ifdef QDP_CODEGEN_VECTOR
    auto& vnode = *mapTables.at( make_pair(s_no,0) ).innerVNodeSIMD;
    for( int q = 0 ; q < vnode.size() ; ++q )
      {
	t_innerVNodeSIMD[ vnode[q] ] = vnode[q];
      }
#endif
    
    int bit = 1;
    //QDPIO::cout << "adding masks.. " << std::endl;
    while( (1 << (bit-1)) <= bitmask )
      {
	if ( (1 << (bit-1)) & bitmask )
	  {
	    //QDPIO::cout << "adding mask: " << bit << std::endl;
	
	    const Map& map(*vecPMap.at( bit-1 ));

	    // roffset
	    //
	    map.getRoffsetsId( subset ); // make sure the lazy part was computed!
#ifdef QDP_CODEGEN_VECTOR
	    for (int q = 0; q < map.loffset( subset ).size() ; ++q )
	      {
		t_loffset       [ map.loffset( subset )[q] ] = map.loffset( subset )[q];
		t_innerVNodeSIMD[ map.loffset( subset )[q] ] = -1;
	      }
#endif
	
	    for (int q = 0; q < map.roffset( subset ).size() ; ++q )
	      {
		t_roffset       [ map.roffset( subset )[q] ] = map.roffset( subset )[q];
		t_innerScalar   [ map.roffset( subset )[q] ] = -1;
#ifdef QDP_CODEGEN_VECTOR
		t_loffset       [ map.roffset( subset )[q] ] = -1;  // remove the roffset from the loffset
		t_innerVNodeSIMD[ map.roffset( subset )[q] ] = -1;
#endif
	      }
	  }
	bit++;
      }

    remove_neg_in_subset( *t.face             , t_roffset        , s_no );
    remove_neg_in_subset( *t.innerScalar      , t_innerScalar    , s_no );
#ifdef QDP_CODEGEN_VECTOR
    remove_neg_in_subset( *t.innerVNodeScalar , t_loffset        , s_no );
    remove_neg_in_subset( *t.innerVNodeSIMD   , t_innerVNodeSIMD , s_no );
#endif
    
    // QDPIO::cout << "face count             = " << t.face->size() << std::endl;
    // QDPIO::cout << "innerScalar count      = " << t.innerScalar->size() << std::endl;
#ifdef QDP_CODEGEN_VECTOR
    // QDPIO::cout << "innerVNodeScalar count = " << t.innerVNodeScalar->size() << std::endl;
    // QDPIO::cout << "innerVNodeSIMD count   = " << t.innerVNodeSIMD->size() << std::endl;
#endif
    
    t.id_face             = QDP_get_global_cache().addOwnHostMemNoPage( t.face->size() * sizeof(int)             , t.face->data() );
    t.id_innerScalar      = QDP_get_global_cache().addOwnHostMemNoPage( t.innerScalar->size() * sizeof(int)      , t.innerScalar->data() );
#ifdef QDP_CODEGEN_VECTOR
    t.id_innerVNodeScalar = QDP_get_global_cache().addOwnHostMemNoPage( t.innerVNodeScalar->size() * sizeof(int) , t.innerVNodeScalar->data() );
    t.id_innerVNodeSIMD   = QDP_get_global_cache().addOwnHostMemNoPage( t.innerVNodeSIMD->size() * sizeof(int)   , t.innerVNodeSIMD->data() );
#endif
    
    // Now insert the whole thing
    //
    mapTables[ key ] = t;
  }


  

  int MasterMap::register_justid(const Map& map)
  {
    int id = 1 << vecPMap.size();
    vecPMap.push_back(&map);
    return id;
  }

  
  // ----------------------------------
#ifdef QDP_CODEGEN_VECTOR
  int MasterMap::getCountVNodeInnerSIMD  (const Subset& s,int bitmask) 
  {
    generate_tables(s,bitmask);
    auto key = make_pair(s.getId(),bitmask);
    return mapTables.at(key).innerVNodeSIMD->size();
  }
  
  int MasterMap::getCountVNodeInnerScalar(const Subset& s,int bitmask) 
  {
    generate_tables(s,bitmask);
    auto key = make_pair(s.getId(),bitmask);
    return mapTables.at(key).innerVNodeScalar->size();
  }
#endif
  
  int MasterMap::getCountInnerScalar     (const Subset& s,int bitmask) 
  {
    generate_tables(s,bitmask);
    auto key = make_pair(s.getId(),bitmask);
    return mapTables.at(key).innerScalar->size();
  }
  
  int MasterMap::getCountFace            (const Subset& s,int bitmask) 
  {
    generate_tables(s,bitmask);
    auto key = make_pair(s.getId(),bitmask);
    return mapTables.at(key).face->size();
  }


  // ---------------------

  
#ifdef QDP_CODEGEN_VECTOR
  int MasterMap::getIdVNodeInnerSIMD  (const Subset& s,int bitmask) 
  {
    generate_tables(s,bitmask);
    auto key = make_pair(s.getId(),bitmask);
    return mapTables.at(key).id_innerVNodeSIMD;
  }
  
  int MasterMap::getIdVNodeInnerScalar(const Subset& s,int bitmask) 
  {
    generate_tables(s,bitmask);
    auto key = make_pair(s.getId(),bitmask);
    return mapTables.at(key).id_innerVNodeScalar;
  }
#endif
  
  int MasterMap::getIdInnerScalar     (const Subset& s,int bitmask) 
  {
    generate_tables(s,bitmask);
    auto key = make_pair(s.getId(),bitmask);
    return mapTables.at(key).id_innerScalar;
  }
  
  int MasterMap::getIdFace            (const Subset& s,int bitmask) 
  {
    generate_tables(s,bitmask);
    auto key = make_pair(s.getId(),bitmask);
    return mapTables.at(key).id_face;
  }

  



} // namespace QDP


