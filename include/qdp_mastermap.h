// -*- C++ -*-

#ifndef QDP_MASTERMAP_H
#define QDP_MASTERMAP_H

namespace QDP {

  class MasterMap {
  public:
    static MasterMap& Instance();

    int  register_justid(const Map& map);
    void register_work  (const Map& map, const Subset& subset);
    
#ifdef QDP_CODEGEN_VECTOR
    int getCountVNodeInnerSIMD  (const Subset& s,int bitmask) ;
    int getCountVNodeInnerScalar(const Subset& s,int bitmask) ;
#endif
    int getCountInnerScalar     (const Subset& s,int bitmask) ;
    int getCountFace            (const Subset& s,int bitmask) ;

#ifdef QDP_CODEGEN_VECTOR
    int getIdVNodeInnerSIMD  (const Subset& s,int bitmask) ;
    int getIdVNodeInnerScalar(const Subset& s,int bitmask) ;
#endif
    int getIdInnerScalar     (const Subset& s,int bitmask) ;
    int getIdFace            (const Subset& s,int bitmask) ;

  private:
    void generate_tables(const Subset& subset, int bitmask);

    struct tables
    {
#ifdef QDP_CODEGEN_VECTOR
      std::shared_ptr< std::vector<int> > innerVNodeSIMD;
      std::shared_ptr< std::vector<int> > innerVNodeScalar;
#endif
      std::shared_ptr< std::vector<int> > innerScalar;
      std::shared_ptr< std::vector<int> > face;

#ifdef QDP_CODEGEN_VECTOR
      int id_innerVNodeSIMD;
      int id_innerVNodeScalar;
#endif
      int id_innerScalar;
      int id_face;

      tables() {
#ifdef QDP_CODEGEN_VECTOR
	innerVNodeSIMD   = std::make_shared< std::vector<int> >();
	innerVNodeScalar = std::make_shared< std::vector<int> >();
#endif
	innerScalar      = std::make_shared< std::vector<int> >();
	face             = std::make_shared< std::vector<int> >();
      }
    };

    MasterMap()
    {
      //QDPIO::cout << "constructing master map with " << MasterSet::Instance().numSubsets() << " subsets\n";
      const int my_node   = Layout::nodeNumber();

      // QDPIO::cout << "vnode latt size = ("
      // 		  << Layout::virtualNodeSubgridLattSize()[0] << " "
      // 		  << Layout::virtualNodeSubgridLattSize()[1] << " "
      // 		  << Layout::virtualNodeSubgridLattSize()[2] << " "
      // 		  << Layout::virtualNodeSubgridLattSize()[3] << ")" << std::endl;
	
      // Make the subset without any shifts
      //
      for (int s_no = 0 ; s_no < MasterSet::Instance().numSubsets() ; ++s_no )
	{
	  const Subset& subset( MasterSet::Instance().getSubset(s_no) );

	  tables t;

	  //QDPIO::cout << "******* Subset " << s_no << "  has " << subset.numSiteTable() << std::endl;

	  std::vector<int> coverSIMD;
	  
	  for (int i = 0 ; i < subset.numSiteTable() ; ++i )
	    {
	      int j = subset.siteTable()[ i ];
	      multi1d<int> coord = Layout::siteCoords( my_node , j );

	      // Always add this site to the Scalar sitetable
	      //
	      t.innerScalar->push_back(j);

	      //QDPIO::cout << i << "  coord (" << coord[0] << " " <<  coord[1] << " " << coord[2] << " " << coord[3] << ")" << std::endl;

#ifdef QDP_CODEGEN_VECTOR
	      bool isSIMD = false;
	      
	      bool firstvnode = true;
	      for ( int d = 0 ; d < Nd ; ++d )
		{
		  if ( coord[d] % Layout::subgridLattSize()[d] >= Layout::virtualNodeSubgridLattSize()[d] )
		    firstvnode = false;
		}

	      std::vector<int> tmp;
	      if (firstvnode)
		{
		  //QDPIO::cout << t.innerVNodeSIMD.size() << " = " << j << std::endl;

		  //
		  // Check that all vector components lie within this subset
		  bool all_in = true;
		  
		  auto& vn_coords = Layout::virtualNodeCoords();
		  for ( int c = 0 ; c < vn_coords.size() ; c++ )
		    {
		      auto vec_coord = coord + vn_coords[c];

		      int lin = Layout::linearSiteIndex( vec_coord );
		      all_in = all_in && subset.isElement( lin );

		      tmp.push_back(lin);
		    }

		  if (all_in)
		    {
		      isSIMD = true;
		    }
		}

	      if (isSIMD)
		{
		  t.innerVNodeSIMD->push_back(j);
		  coverSIMD.insert( coverSIMD.end() , tmp.begin() , tmp.end() );
		}
#endif
	    } // i


#ifdef QDP_CODEGEN_VECTOR
	  // Next round, do the VNodeScalar respecting coverSIMD
	  //
	  for (int i = 0 ; i < subset.numSiteTable() ; ++i )
	    {
	      int j = subset.siteTable()[ i ];

	      if ( std::find( coverSIMD.begin(), coverSIMD.end(), j) == coverSIMD.end() )
		{
		  t.innerVNodeScalar->push_back(j);
		}
	    }
#endif
	  
	  //QDPIO::cout << "Scalar    site count = " << t.innerScalar->size() << std::endl;
#ifdef QDP_CODEGEN_VECTOR
	  //QDPIO::cout << "VN SIMD   site count = " << t.innerVNodeSIMD->size() << std::endl;
	  //QDPIO::cout << "VN Scalar site count = " << t.innerVNodeScalar->size() << std::endl;
#endif
	  
	  t.id_face             = -1;
	  t.id_innerScalar      = QDP_get_global_cache().addOwnHostMemNoPage( t.innerScalar->size() * sizeof(int)      , t.innerScalar->data()       );
#ifdef QDP_CODEGEN_VECTOR
	  t.id_innerVNodeScalar = QDP_get_global_cache().addOwnHostMemNoPage( t.innerVNodeScalar->size() * sizeof(int) , t.innerVNodeScalar->data()  );
	  t.id_innerVNodeSIMD   = QDP_get_global_cache().addOwnHostMemNoPage( t.innerVNodeSIMD->size() * sizeof(int)   , t.innerVNodeSIMD->data()    );
#endif
	  
	  // Store in map
	  //
	  mapTables[ make_pair(s_no,0) ] = t;
	}
    }

    std::vector<const Map*> vecPMap;

    std::map< std::pair<int,int> , tables > mapTables;
    
  };

} // namespace QDP

#endif
