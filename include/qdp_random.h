#ifndef QDP_RANDOM_H
#define QDP_RANDOM_H


namespace QDP {

  namespace RNG
  {
    struct RNG_Internals_t
    {
      Seed ran_seed;
      Seed ran_mult;
      Seed ran_mult_n;
      LatticeSeed lattice_ran_mult;
      LatticeSeed lat_ran_mult_n;
      LatticeSeed lat_ran_seed;
    };

    RNG_Internals_t& get_RNG_Internals( LatticeLayout layout = Layout::current() );
  }


//! Random number generator namespace
/*!
 * A collection of routines and data for supporting random numbers
 * 
 * It is a linear congruential with modulus m = 2**47, increment c = 0,
 * and multiplier a = (2**36)*m3 + (2**24)*m2 + (2**12)*m1 + m0.  
 */

namespace RNG
{
  void initDefaultRNG(void);
  void doneDefaultRNG(void);
  void initRNG(void);
  void setrn(const Seed& lseed);
  void savern(Seed& lseed);
}

  
} // namespace QDP

#endif
