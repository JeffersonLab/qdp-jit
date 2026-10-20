#include "qdp.h"


namespace QDP
{

  namespace {

    std::map<
      std::pair<int,int> ,
      std::pair<int , multi1d<int>* > > __sitetable; // [dest,from layout]cache id, sitetable

    std::map< int , std::pair<int , multi1d<int>* > > __lexico_to; // [layout] translate lexico to configured data layout

    typedef std::pair< int , std::vector<int> > __blocksum_key_t;
    std::map< __blocksum_key_t , std::pair<int , multi1d<int>* > > __blocksum_rhs;
    std::map< __blocksum_key_t , std::pair<int , multi1d<int>* > > __blocksum_dest;

    
    typedef std::pair< multi1d<int> , Set* > __set_t;
    std::vector< __set_t >  __blocking_sets;
  }



  multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size);
  int local_site(const multi1d<int>& coord, const multi1d<int>& latt_size);


  class SetBlockingFunc : public SetFunc
  {
    multi1d<int> blocking;
    multi1d<int> coarse_dim;
    int num_subsets;
  public:
    SetBlockingFunc(const multi1d<int>& blocking_): blocking(blocking_)
    {
      coarse_dim.resize(blocking.size());

      num_subsets = 1;
      for(int m=0; m < blocking.size(); ++m)
	{
	  coarse_dim[m] = Layout::subgridLattSize()[m] / blocking[m];
	  num_subsets *= coarse_dim[m];
	}
      if ( jit_config_get_debug_layout() )
	QDPIO::cout << "Blocking function: num_subsets = " << num_subsets << endl;
    }

    
    int operator() (const multi1d<int>& coordinate) const
    {
      multi1d<int> tmp(coordinate);

      for( int i = 0 ; i < tmp.size() ; ++i )
	tmp[i] %= Layout::subgridLattSize()[i];

      for( int i = 0 ; i < tmp.size() ; ++i )
	tmp[i] /= blocking[i];
      
      int col = local_site( tmp , coarse_dim );

      assert(col >= 0);
      assert(col < num_subsets );
      
      return col;
    }

    int numSubsets() const {return num_subsets;}
  };



  
  Set& get_blocking_set( const multi1d<int>& blocking )
  {
    bool found = false;
    Set* s;

    for ( int i = 0 ; i < __blocking_sets.size() ; ++i )
      {
	if ( __blocking_sets[i].first == blocking )
	  {
	    if ( jit_config_get_debug_layout() )
	      QDPIO::cout << "blocking set found." << endl;
	    s = __blocking_sets[i].second;
	    found = true;
	  }
      }

    if ( ! found )
      {
	if ( jit_config_get_debug_layout() )
	  QDPIO::cout << "Creating new blocking set." << endl;
	s = new Set;
	__blocking_sets.push_back( make_pair( blocking , s ));
      }

    if ( ! s->layout_done( Layout::current().index() ) )
      {
	if ( jit_config_get_debug_layout() )
	  QDPIO::cout << "Making new blocking set." << endl;
	s->make(SetBlockingFunc(blocking));
      }

    return *s;
  }


  
  void destroy_translation_sitetable()
  {
    const bool verbose = jit_config_get_debug_layout();

    if ( verbose )
      QDPIO::cout << "Translation sitetable cleanup" << endl;
    for ( const auto &both : __sitetable )
      {
	if ( verbose )
	  QDPIO::cout << both.first.first << " " << both.first.second << " delete " << endl;
	QDP_get_global_cache().signoff( both.second.first );
	delete both.second.second;
      }

    if ( verbose )
      QDPIO::cout << "Lexico sitetables cleanup" << endl;
    for ( const auto &both : __lexico_to )
      {
	if ( verbose )
	  QDPIO::cout << both.first << " delete " << endl;
	QDP_get_global_cache().signoff( both.second.first );
	delete both.second.second;
      }

    if ( verbose )
      QDPIO::cout << "Blocking sets cleanup" << endl;
    for ( int i = 0 ; i < __blocking_sets.size() ; ++i )
      {
	delete __blocking_sets[i].second;
      }

    if ( verbose )
      QDPIO::cout << "Blocksum rhs sitetables cleanup" << endl;
    for ( const auto &both : __blocksum_rhs )
      {
	if ( verbose )
	  QDPIO::cout << both.first.first << " delete " << endl;
	QDP_get_global_cache().signoff( both.second.first );
	delete both.second.second;
      }

    if ( verbose )
      QDPIO::cout << "Blocksum dest sitetables cleanup" << endl;
    for ( const auto &both : __blocksum_dest )
      {
	if ( verbose )
	  QDPIO::cout << both.first.first << " delete " << endl;
	QDP_get_global_cache().signoff( both.second.first );
	delete both.second.second;
      }

  }
  

  int get_translation_sitetable_id( int dest , int from )
  {
    std::pair<int,int> key({dest,from});
    
    if ( __sitetable.find(key) != __sitetable.end())
      {
	return __sitetable[key].first;
      }

    
    int nodeNumber = Layout::nodeNumber();
    int local_from_sites = Layout::from_index( from ).sitesOnNode();
    int local_dest_sites = Layout::from_index( dest ).sitesOnNode();

    multi1d<int> latt_from = Layout::data(from).lattSize();
    multi1d<int> latt_dest = Layout::data(dest).lattSize();
    multi1d<int> subgrid_latt_from = Layout::data(from).subgridLattSize();

    for ( int mu = 0 ; mu < latt_from.size() ; ++mu )
      if ( latt_from[mu] % latt_dest[mu] != 0 )
	{
	  std::ostringstream os;
	  os << "mixed-layout expression: the destination lattice";
	  for ( int nu = 0 ; nu < latt_from.size() ; ++nu )
	    os << " " << latt_from[nu];
	  os << " is not a multiple of the operand lattice";
	  for ( int nu = 0 ; nu < latt_dest.size() ; ++nu )
	    os << " " << latt_dest[nu];
	  QDP_error_exit( "%s" , os.str().c_str() );
	}

    const bool verbose = jit_config_get_debug_layout();

    if ( verbose )
      {
	QDPIO::cout << "latt_from: ";
	QDPIO::cout << latt_from[0] << " "
		    << latt_from[1] << " "
		    << latt_from[2] << " "
		    << latt_from[3] << endl;

	QDPIO::cout << "latt_dest: ";
	QDPIO::cout << latt_dest[0] << " "
		    << latt_dest[1] << " "
		    << latt_dest[2] << " "
		    << latt_dest[3] << endl;

	QDPIO::cout << "subgrid_latt_from: ";
	QDPIO::cout << subgrid_latt_from[0] << " "
		    << subgrid_latt_from[1] << " "
		    << subgrid_latt_from[2] << " "
		    << subgrid_latt_from[3] << endl;
      }

    
    __sitetable[key].second = new multi1d<int>(local_from_sites);

    
    // reconstruct blocking
    multi1d<int> blocking = latt_from / latt_dest;

    assert( blocking.size() == 4 );
    if ( verbose )
      {
	QDPIO::cout << "Building translation table with blocking: ";
	QDPIO::cout << blocking[0] << " "
		    << blocking[1] << " "
		    << blocking[2] << " "
		    << blocking[3] << endl;
      }
    assert( blocking[0] != 0 );
    assert( blocking[1] != 0 );
    assert( blocking[2] != 0 );
    assert( blocking[3] != 0 );    
    
    for(int linear=0; linear < local_from_sites; ++linear)
      {
	multi1d<int> coord = Layout::data(from).siteCoords(nodeNumber, linear);

	multi1d<int> node_coord = Layout::data(from).nodeCoord();

	
	multi1d<int> tmp(node_coord);
	tmp *= subgrid_latt_from;

	multi1d<int> tmp2(coord);
	tmp2 -= tmp;

	assert( tmp2[0] >= 0 );
	assert( tmp2[1] >= 0 );
	assert( tmp2[2] >= 0 );
	assert( tmp2[3] >= 0 );

	 assert( tmp2[0] < subgrid_latt_from[0] );
	 assert( tmp2[1] < subgrid_latt_from[1] );
	 assert( tmp2[2] < subgrid_latt_from[2] );
	 assert( tmp2[3] < subgrid_latt_from[3] );
	
	tmp2 /= blocking;

	// now translate the dest site

	multi1d<int> node_coord_dest = Layout::data(dest).nodeCoord();

	node_coord_dest *= Layout::data(dest).subgridLattSize();

	tmp2 += node_coord_dest;
	
	int dest_site = Layout::data(dest).linearSiteIndex(tmp2);

	assert( dest_site >= 0 );
	assert( dest_site < local_dest_sites );

	(*__sitetable[key].second)[ linear ] = dest_site;
      }

    __sitetable[key].first = QDP_get_global_cache().addOwnHostMem( __sitetable[key].second->size() * sizeof(int) , __sitetable[key].second->slice() );

    return __sitetable[key].first;
  }




  void check_shift_layout( int shifted , int expression )
  {
    if ( shifted == expression )
      return;

    get_translation_sitetable_id( shifted , expression );

    multi1d<int> latt_shifted    = Layout::data(shifted).lattSize();
    multi1d<int> latt_expression = Layout::data(expression).lattSize();

    std::ostringstream os;
    os << "shift in a mixed-layout expression: the shifted field is on the lattice";
    for ( int mu = 0 ; mu < latt_shifted.size() ; ++mu )
      os << " " << latt_shifted[mu];
    os << ", the expression is evaluated on the lattice";
    for ( int mu = 0 ; mu < latt_expression.size() ; ++mu )
      os << " " << latt_expression[mu];
    os << ". Shift the field on its own layout first, or assign it to a field on the expression's layout and shift that";
    QDP_error_exit( "%s" , os.str().c_str() );
  }



  int get_lexico_sitetable_id( int layout )
  {
    int key=layout;
    
    if ( __lexico_to.find(key) != __lexico_to.end())
      {
	return __lexico_to[key].first;
      }

    const bool verbose = jit_config_get_debug_layout();

    if ( verbose )
      QDPIO::cout << "creating lexico to current layout sitetable: " << layout << endl;
    
    int nodeNumber  = Layout::nodeNumber();
    int local_sites = Layout::from_index( layout ).sitesOnNode();

    multi1d<int> latt         = Layout::data(layout).lattSize();
    multi1d<int> subgrid_latt = Layout::data(layout).subgridLattSize();

    if ( verbose )
      {
	QDPIO::cout << "latt: ";
	QDPIO::cout << latt[0] << " "
		    << latt[1] << " "
		    << latt[2] << " "
		    << latt[3] << endl;

	QDPIO::cout << "subgrid_latt: ";
	QDPIO::cout << subgrid_latt[0] << " "
		    << subgrid_latt[1] << " "
		    << subgrid_latt[2] << " "
		    << subgrid_latt[3] << endl;
      }

    
    __lexico_to[key].second = new multi1d<int>(local_sites);

    
    for(int linear=0; linear < local_sites; ++linear)
      {
	multi1d<int> coord = crtesn( linear , subgrid_latt );


	// now translate the dest site

	multi1d<int> node_coord = Layout::data(layout).nodeCoord();

	node_coord *= Layout::data(layout).subgridLattSize();

	coord += node_coord;


	int site = Layout::data(layout).linearSiteIndex( coord );

	assert( site >= 0 );
	assert( site < local_sites );

	if ( verbose )
	  QDPIO::cout << "lexico_to[ " << linear << " ] = " << site << endl;
	
	(*__lexico_to[key].second)[ linear ] = site;
      }
	
    __lexico_to[key].first = QDP_get_global_cache().addOwnHostMem( __lexico_to[key].second->size() * sizeof(int) , __lexico_to[key].second->slice() );

    return __lexico_to[key].first;
  }






  void get_blocksum_sitetable_id( int& dest_table , int& rhs_table, int layout , const multi1d<int>& blocking )
  {
    const __blocksum_key_t key( layout , std::vector<int>( blocking.slice() , blocking.slice() + blocking.size() ) );
    
    if ( __blocksum_rhs.find(key) != __blocksum_rhs.end())
      {
	rhs_table  = __blocksum_rhs [key].first;
	dest_table = __blocksum_dest[key].first;
	return;
      }

    const bool verbose = jit_config_get_debug_layout();

    if ( verbose )
      QDPIO::cout << "creating blocksum to current layout sitetable: " << layout << endl;
    
    int nodeNumber  = Layout::nodeNumber();

    int dest_ref = Layout::from_index(layout).coarsen(blocking).index();
    
    int rhs_sites   = Layout::from_index( layout ).sitesOnNode();
    int dest_sites  = Layout::from_index( dest_ref ).sitesOnNode();

    if ( verbose )
      {
	QDPIO::cout << "local sites: " << rhs_sites << endl;
	QDPIO::cout << "dest sites:  " << dest_sites << endl;
      }

    multi1d<int> latt         = Layout::data(layout).lattSize();
    multi1d<int> subgrid_latt = Layout::data(layout).subgridLattSize();

    multi1d<int> dest_latt(latt);
    multi1d<int> dest_subgrid_latt(subgrid_latt);

    dest_latt         /= blocking;
    dest_subgrid_latt /= blocking;
    
    if ( verbose )
      {
	QDPIO::cout << "latt: ";
	QDPIO::cout << latt[0] << " "
		    << latt[1] << " "
		    << latt[2] << " "
		    << latt[3] << endl;

	QDPIO::cout << "subgrid_latt: ";
	QDPIO::cout << subgrid_latt[0] << " "
		    << subgrid_latt[1] << " "
		    << subgrid_latt[2] << " "
		    << subgrid_latt[3] << endl;
      }

    
    __blocksum_rhs[key].second  = new multi1d<int>(rhs_sites);
    __blocksum_dest[key].second = new multi1d<int>(dest_sites);

    
    int block_n = 1;
    for( int i = 0 ; i < blocking.size() ; ++i )
      block_n *= blocking[i];

    if ( verbose )
      QDPIO::cout << "block sites:  " << block_n << endl;
    
    multi1d<int> node_origin = Layout::data(layout).nodeCoord();
    node_origin *= Layout::data(layout).subgridLattSize();

    multi1d<int> node_origin_dest(node_origin);
    node_origin_dest /= blocking;
    
    for(int linear = 0; linear < dest_sites; ++linear)
      {
	multi1d<int> tmp = crtesn( linear , dest_subgrid_latt );
	multi1d<int> coord_block_origin(tmp);
	coord_block_origin *= blocking;
	
	for( int block_i = 0 ; block_i < block_n ; ++block_i )
	  {
	    multi1d<int> coord_block(coord_block_origin);
	    multi1d<int> tmp2 = crtesn( block_i , blocking );
	    coord_block += tmp2;
	    coord_block += node_origin;

	    int site = Layout::data(layout).linearSiteIndex( coord_block );

	    assert( site >= 0 );
	    assert( site < rhs_sites );
	
	    (*__blocksum_rhs[key].second)[ linear * block_n + block_i ] = site;
	  }

	multi1d<int> coord_block_origin_dest(tmp);
	coord_block_origin_dest += node_origin_dest;
	
	int site_dest = Layout::data( dest_ref ).linearSiteIndex( coord_block_origin_dest );

	assert( site_dest >= 0 );
	assert( site_dest < dest_sites );

	(*__blocksum_dest[key].second)[ linear ] = site_dest;
      }
	
    __blocksum_rhs[key].first  = QDP_get_global_cache().addOwnHostMem( __blocksum_rhs [key].second->size() * sizeof(int) , __blocksum_rhs [key].second->slice() );
    __blocksum_dest[key].first = QDP_get_global_cache().addOwnHostMem( __blocksum_dest[key].second->size() * sizeof(int) , __blocksum_dest[key].second->slice() );

    rhs_table  = __blocksum_rhs [key].first;
    dest_table = __blocksum_dest[key].first;

    return;
  }


} // QDP
