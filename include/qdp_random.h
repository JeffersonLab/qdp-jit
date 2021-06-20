// -*- C++ -*-
//
// QDP data parallel interface
//
// Random number support

#ifndef QDP_RANDOM_H
#define QDP_RANDOM_H

namespace QDP {

  namespace RNG
  {
    struct RNG_Internals_t
    {
      //! Global (current) seed
      Seed ran_seed;
      //! RNG multiplier
      Seed ran_mult;
      //! RNG multiplier raised to the volume+1
      Seed ran_mult_n;
      //! The lattice of skewed RNG multipliers
      LatticeSeed lattice_ran_mult;

      LatticeSeed lat_ran_mult_n;
      LatticeSeed lat_ran_seed;
    };

    std::shared_ptr<RNG_Internals_t> get_RNG_Internals();

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
  //! Default initialization of the RNG
  /*! Uses arbitrary internal seed to initialize the RNG */
  void initDefaultRNG(void);

  // Finalization of RNG
  void doneDefaultRNG(void);

  //! Initialize the internals of the RNG
  void initRNG(void);

  //! Initialize the RNG seed
  /*!
   * Seeds are big-ints
   */
  void setrn(const Seed& lseed);

  //! Recover the current RNG seed
  /*!
   * Seeds are big-ints
   */
  void savern(Seed& lseed);


  //! Internal seed multiplier
  float sranf(Seed& seed, Seed&, const Seed&);

  //! Internal seed multiplier
  void sranf(float* d, int N, Seed& seed, ILatticeSeed&, const Seed&);
}

  
} // namespace QDP

#endif
