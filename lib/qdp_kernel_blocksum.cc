#include "qdp.h"


namespace QDP {
namespace BlockSum {

  namespace {

    unsigned next_power_of_2( unsigned n )
    {
      unsigned p = 1;
      while ( p < n )
	p <<= 1;
      return p;
    }

    unsigned floor_power_of_2( unsigned n )
    {
      unsigned p = 1;
      while ( 2 * p <= n )
	p <<= 1;
      return p;
    }

  } // namespace


  Geometry geometry( int tile_size , int tiles , size_t value_size )
  {
    const unsigned max_threads = std::min( floor_power_of_2( gpu_getMaxSMem() / value_size ) ,
					   floor_power_of_2( gpu_getMaxBlockX() >> 1 ) );

    unsigned group_size;
    if ( qdp_jit_config_get_blocksum() > 0 )
      {
	group_size = qdp_jit_config_get_blocksum();
	if ( group_size & ( group_size - 1 ) )
	  {
	    QDPIO::cerr << "blockSum: -blocksum " << group_size << " is not a power of 2" << std::endl;
	    QDP_abort(1);
	  }
      }
    else
      {
	const size_t enough_threads = value_size <= 32 ? ( 1 << 19 ) : ( 1 << 12 );
	const unsigned max_group = std::min( next_power_of_2( tile_size ) , 32u );

	group_size = 1;
	while ( group_size < max_group && (size_t)tiles * group_size < enough_threads )
	  group_size <<= 1;
      }

    Geometry geom;
    geom.group_size        = std::min( group_size , max_threads );
    geom.threads_per_block = std::max( std::min( (unsigned)jit_config_get_threads_per_block() , max_threads ) ,
				       (unsigned)geom.group_size );
    return geom;
  }



  namespace {

    bool exec( JitFunction& function , const ExprPlan& dest_plan , const ExprPlan& rhs_plan ,
	       LatticeLayout layout_dest , LatticeLayout layout_rhs , const multi1d<int>& blocking , const Geometry& geom , size_t value_size )
    {
      int tile_size = 1;
      for ( int i = 0 ; i < blocking.size() ; ++i )
	tile_size *= blocking[i];

      const int tiles = layout_dest.sitesOnNode();

      int id_table_dest;
      int id_table_rhs;
      get_blocksum_sitetable_id( id_table_dest , id_table_rhs , layout_rhs.index() , blocking );

      LaunchArgs args;
      args.add_int( tiles );
      args.add_int( tile_size );
      args.add_int( geom.group_size );
      args.add_id( id_table_rhs );
      args.add_id( id_table_dest );
      args.add_int( layout_rhs.sitesOnNode() );
      args.add_plan( rhs_plan , all );
      args.add_int( tiles );
      args.add_plan( dest_plan , all );

      kernel_geom_t kgeom;
      kgeom.threads_per_block = geom.threads_per_block;
      kgeom.Nblock_x = ( (size_t)tiles * geom.group_size + geom.threads_per_block - 1 ) / geom.threads_per_block;
      kgeom.Nblock_y = 1;

      if ( (size_t)kgeom.Nblock_x > gpu_getMaxGridX() )
	{
	  QDPIO::cerr << "blockSum: " << kgeom.Nblock_x << " blocks exceed the maximum grid size " << gpu_getMaxGridX() << std::endl;
	  QDP_abort(1);
	}

      const size_t shared = geom.group_size == 1 ? 0 : geom.threads_per_block * value_size;
      return args.launch( function , kgeom , shared );
    }

  } // namespace


  void launch( const Kernel& kernel , const ExprPlan& dest_plan , const ExprPlan& rhs_plan ,
	       LatticeLayout layout_dest , LatticeLayout layout_rhs , const multi1d<int>& blocking , size_t value_size )
  {
    int tile_size = 1;
    for ( int i = 0 ; i < blocking.size() ; ++i )
      tile_size *= blocking[i];

    Geometry geom = geometry( tile_size , layout_dest.sitesOnNode() , value_size );

    while ( !exec( kernel( geom.group_size == 1 ) , dest_plan , rhs_plan , layout_dest , layout_rhs , blocking , geom , value_size ) )
      {
	geom.threads_per_block >>= 1;
	if ( geom.threads_per_block == 0 )
	  {
	    QDPIO::cerr << "blockSum: no block size fits the register limit" << std::endl;
	    QDP_abort(1);
	  }
	geom.group_size = std::min( geom.group_size , geom.threads_per_block );
      }
  }

}
} // namespace QDP
