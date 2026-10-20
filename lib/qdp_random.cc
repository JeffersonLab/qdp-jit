#include "qdp.h"


namespace QDP {

  namespace Layout {
  LatticeInteger latticeCoordinate(int mu);
  }

  
namespace RNG
{
  std::map< int , RNG_Internals_t > RNG_Internals;

  RNG_Internals_t& get_RNG_Internals(LatticeLayout layout)
  {
    if ( !RNG_Internals.count( layout.index() ) )
      {
	QDPIO::cerr << "get_RNG_Internals: layout not found " << layout.index() << std::endl;
	sleep(1);
	QDP_abort(1);
      }
    return RNG_Internals[ layout.index() ];
  }
  
  int numbits(size_t x)
  {
    int num = 1;
    size_t iceiling = 2;
    while (iceiling <= x)
    {
      num++;
      iceiling *= 2;
    }

    return num;
  }


  void initDefaultRNG()
  {
    RNG::initRNG();

    Seed seed = 11;
    RNG::setrn(seed);
  }

  void doneDefaultRNG()
  {
    RNG_Internals.clear();
  }

  
  void initRNG()
  {
    int ref = Layout::current().index();

    int old_profile_level = setProfileLevel(0);

    Seed seed_tmp3;
    Seed seed_tmp2;
    Seed seed_tmp1;
    Seed seed_tmp0;

    seed_tmp3 = 1222;
    seed_tmp2 = (seed_tmp3 << 12) | 1498;
    seed_tmp1 = (seed_tmp2 << 12) | 712;
    seed_tmp0 = (seed_tmp1 << 12) | 1645;

    RNG_Internals[ref].ran_mult = seed_tmp0;

    int nbits = numbits(Layout::vol());

    /* Get the lattice coordinate of each site (note the origin is 0) and
     *   build up a lexicographic ordering for the lattice. The definition
     *   here is totally arbitrary and only this routine needs to worry
     *   about it.  The lexicographic value of site K is
     *
     *     lexoc(k) = sum_{i = 1, ndim} x(k,i)*L^i     +   1
     */
    LatticeInteger lexoc;
    lexoc = Layout::latticeCoordinate(Nd-1);

    for(int m=Nd-2; m>=0; --m)
    {
      lexoc *= Layout::lattSize()[m];
      lexoc += Layout::latticeCoordinate(m);
    }

    lexoc += 1;

    LatticeSeed laa;
    laa = RNG_Internals[ref].ran_mult;
    
    LatticeSeed lattice_ran_mult_tmp;
    lattice_ran_mult_tmp = 1;

    LatticeSeed laamult;
    LatticeBoolean lbit;

    for(int i=0; i<nbits; ++i)
    {
      lbit = (lexoc & 1) > 0;

      laamult = lattice_ran_mult_tmp * laa;

      copymask(lattice_ran_mult_tmp,lbit,laamult);

      lexoc >>= 1;
      laamult = laa * laa;
      laa = laamult;
    }

    bool bit;
    Seed aa;
    Seed aamult;

    size_t ibit = Layout::vol();
    aa = RNG_Internals[ref].ran_mult;
    RNG_Internals[ref].ran_mult_n = RNG_Internals[ref].ran_mult;   // produces def  ran_mult_n = ran_mult^{vol+1}

    for(int i=0; i<nbits; ++i)
    {
      bit = (ibit & 1) > 0;

      aamult = RNG_Internals[ref].ran_mult_n * aa;
      if (bit)
	RNG_Internals[ref].ran_mult_n = aamult;

      ibit >>= 1;
      aamult = aa * aa;
      aa = aamult;
    }

    RNG_Internals[ref].lattice_ran_mult = lattice_ran_mult_tmp;

    for (int node = 0 ; node < Layout::numNodes() ; ++node )
      {
	multi1d<int> coord = Layout::siteCoords( node , 0 );
	pokeSite( RNG_Internals[ref].lat_ran_seed , RNG_Internals[ref].ran_seed , coord );
      }
    Random::run_seedinit( RNG_Internals[ref].lat_ran_seed , RNG_Internals[ref].ran_mult_n );
    
    RNG_Internals[ref].lat_ran_mult_n = RNG_Internals[ref].ran_mult_n;

    setProfileLevel(old_profile_level);
  }


  void setrn(const Seed& seed)
  {
    int ref = Layout::current().index();
    RNG_Internals[ref].ran_seed = seed;
  }


  void savern(Seed& seed)
  {
    int ref = Layout::current().index();
    seed = RNG_Internals[ref].ran_seed;
  }
  
}


} // namespace QDP;



