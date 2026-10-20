#include "qdp.h"


namespace QDP {
namespace Reduce {

  Segments::Segments( const Subset& s , LatticeLayout layout ):
    is_multi( false ), sizes( 1 ), tables( 1 )
  {
    sizes[0]  = s.numSiteTable( layout );
    tables[0] = s.getIdSiteTable( layout );
  }


  Segments::Segments( const Set& ss , LatticeLayout layout ):
    is_multi( true ), sizes( ss.numSubsets() ), tables( ss.numSubsets() )
  {
    for ( int i = 0 ; i < ss.numSubsets() ; ++i )
      {
	sizes[i]  = ss[i].numSiteTable( layout );
	tables[i] = ss[i].getIdSiteTable( layout );
      }
  }


  int Segments::max_size() const
  {
    int m = 0;
    for ( int i = 0 ; i < sizes.size() ; ++i )
      m = std::max( m , sizes[i] );
    return m;
  }



  bool exec_expr( JitFunction& function , const ExprPlan& plan , const Segments& segments ,
		  int latsize , const kernel_geom_t& geom , size_t shared , int out_id , int out_offset )
  {
    LaunchArgs args( plan );

    auto launch = [&]() {
      args.add_int( latsize );
      args.add_plan( plan , all );
      args.add_int( out_offset );
      args.add_id( out_id );
      return args.launch( function , geom , shared );
    };

    if ( !segments.multi() )
      {
	args.add_int( segments.size(0) );
	args.add_id( segments.table_ids()[0] );
	return launch();
      }

    const multi1d<int>& sizes = segments.all_sizes();
    int sizes_id = QDP_get_global_cache().addOwnHostMem( sizes.size() * sizeof(int) , sizes.slice() );
    DeviceMulti tables( segments.table_ids() );

    args.add_id( sizes_id );
    args.add_id( tables.get_id() );
    bool ret = launch();

    QDP_get_global_cache().signoff( sizes_id );
    return ret;
  }


  bool exec_partials( JitFunction& function , const kernel_geom_t& geom , size_t shared , int n , int in_id , int out_id )
  {
    LaunchArgs args;
    args.add_int( n );
    args.add_id( in_id );
    args.add_id( out_id );

    return args.launch( function , geom , shared );
  }



  namespace {

    unsigned next_power_of_2( unsigned n )
    {
      unsigned p = 1;
      while ( p < n )
	p <<= 1;
      return p;
    }


    unsigned threads_for( unsigned max_threads , unsigned n , size_t value_size )
    {
      unsigned threads = max_threads;
      while ( threads > 1 && ( threads * value_size > gpu_getMaxSMem() || threads > next_power_of_2( n ) ) )
	threads >>= 1;

      if ( threads * value_size > gpu_getMaxSMem() )
	{
	  QDPIO::cerr << "Reduce: a value of " << value_size << " bytes doesn't fit into shared memory ("
		      << gpu_getMaxSMem() << " bytes)" << std::endl;
	  QDP_abort(1);
	}
      return threads;
    }


    class Scratch
    {
      int ids[2] = { -1 , -1 };

    public:
      Scratch() = default;
      Scratch( const Scratch& ) = delete;
      Scratch& operator=( const Scratch& ) = delete;

      ~Scratch()
      {
	for ( int id : ids )
	  if ( id >= 0 )
	    QDP_get_global_cache().signoff( id );
      }

      int get( int pass , size_t size )
      {
	if ( ids[0] < 0 )
	  {
	    ids[0] = QDP_get_global_cache().add( size );
	    ids[1] = QDP_get_global_cache().add( size );
	  }
	return ids[ pass % 2 ];
      }
    };


    bool passes( unsigned max_threads , int count , int batch , int max_size , size_t value_size , size_t bytes_first , int result_id ,
		 const FirstPass& first , const LaterPass& later )
    {
      Scratch scratch;

      int n = max_size;
      int in_id = -1;

      for ( int pass = 0 ; ; ++pass )
	{
	  unsigned threads = threads_for( max_threads , n , value_size );

	  const size_t bytes = pass == 0 ? bytes_first : value_size;
	  const size_t per_thread = std::max( (size_t)1 , std::min( 64 / std::max( bytes , (size_t)1 ) , (size_t)n / 16384 ) );
	  unsigned blocks = ( n + (size_t)threads * per_thread - 1 ) / ( (size_t)threads * per_thread );

	  if ( blocks > gpu_getMaxGridX() )
	    {
	      QDPIO::cerr << "Reduce: " << blocks << " blocks exceed the maximum grid size " << gpu_getMaxGridX() << std::endl;
	      QDP_abort(1);
	    }

	  const int results = pass == 0 ? blocks * batch : blocks;

	  int out_id = results == 1 ? result_id : scratch.get( pass , (size_t)results * count * value_size );

	  kernel_geom_t geom;
	  geom.threads_per_block = threads;
	  geom.Nblock_x = blocks;
	  geom.Nblock_y = count;

	  const size_t shared = threads * value_size;

	  bool good = true;
	  if ( pass == 0 )
	    for ( int k = 0 ; k < batch && good ; ++k )
	      good = first( geom , shared , out_id , k , k * blocks );
	  else
	    good = later( geom , shared , n , in_id , out_id );

	  if ( !good )
	    return false;

	  if ( results == 1 )
	    return true;

	  n = results;
	  in_id = out_id;
	}
    }

  } // namespace


  size_t bytes_per_site( const ExprPlan& plan )
  {
    size_t bytes = 0;
    for ( const PlanEntry& e : plan.entries )
      if ( e.kind == PlanEntry::Kind::Lattice && e.cache_id >= 0 )
	bytes += QDP_get_global_cache().getSize( e.cache_id ) / Layout::from_index( e.layout_ref ).sitesOnNode();
    return bytes;
  }


  void run_passes( int count , int batch , int max_size , size_t value_size , size_t bytes_first , void* result ,
		   const FirstPass& first , const LaterPass& later )
  {
    assert( batch == 1 || count == 1 );

    if ( max_size == 0 || batch == 0 )
      return;

    if ( (size_t)count > gpu_getMaxGridY() )
      {
	QDPIO::cerr << "Reduce: " << count << " subsets exceed the maximum grid size in y " << gpu_getMaxGridY() << std::endl;
	QDP_abort(1);
      }

    int result_id = QDP_get_global_cache().addOwnHostMem( count * value_size , result );

    unsigned max_threads = gpu_getMaxBlockX() >> 1;
    while ( !passes( max_threads , count , batch , max_size , value_size , bytes_first , result_id , first , later ) )
      {
	max_threads >>= 1;
	if ( max_threads == 0 )
	  {
	    QDPIO::cerr << "Reduce: no block size fits the register limit" << std::endl;
	    QDP_abort(1);
	  }
      }

    QDP_get_global_cache().assureOnHost( result_id );
    QDP_get_global_cache().signoff( result_id );
  }

}
} // namespace QDP
