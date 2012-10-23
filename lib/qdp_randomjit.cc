//
// Random number generator support


#include "qdp.h"

#if 1

namespace QDP {

// Random number generator namespace
/* 
 * A collection of routines and data for supporting random numbers
 * 
 * It is a linear congruential with modulus m = 2**47, increment c = 0,
 * and multiplier a = (2**36)*m3 + (2**24)*m2 + (2**12)*m1 + m0.  
 */



namespace RNG
{

#if 0
    //! Find the number of bits required to represent x.
  int numbits(int x)
  {
    int num = 1;
    int iceiling = 2;
    while (iceiling <= x)
    {
      num++;
      iceiling *= 2;
    }

    return num;
  }

  //! Initialize the random number generator seed
  void setrn(const Seed& seed)
  {
    ran_seed = seed;
  }


  //! Return a copy of the random number seed
  void savern(Seed& seed)
  {
    seed = ran_seed;
  }
#endif

  //! Scalar random number generator. Done on the front end. */
  /*! 
   * It is linear congruential with modulus m = 2**47, increment c = 0,
   * and multiplier a = (2**36)*m3 + (2**24)*m2 + (2**12)*m1 + m0.  
   */
  float sranf(typename JITContainerType<Seed>::Type_t& seed, 
	      typename JITContainerType<Seed>::Type_t& skewed_seed, 
	      const typename JITContainerType<Seed>::Type_t& seed_mult)
  {
    std::cout << __PRETTY_FUNCTION__ << "\n";
#if 0
    Real _sranf;
    float _ssranf;
    Seed ran_tmp;

    _sranf = seedToFloat(skewed_seed);
    cast_rep(_ssranf, _sranf);

    ran_tmp = seed * seed_mult;
    seed = ran_tmp;

    ran_tmp = skewed_seed * seed_mult;
    skewed_seed = ran_tmp;

    return _ssranf;
#endif
  }

#if 0
  //! Scalar random number generator. Done on the front end. */
  /*! 
   * It is linear congruential with modulus m = 2**47, increment c = 0,
   * and multiplier a = (2**36)*m3 + (2**24)*m2 + (2**12)*m1 + m0.  
   */
  void sranf(float* d, int N, & seed, ILatticeSeed& skewed_seed, const Seed& seed_mult)
  {
    /* Calculate the random number and update the seed according to the
     * following algorithm
     *
     * FILL(twom11,TWOM11);
     * FILL(twom12,TWOM12);
     * i3 = ran_seed(3)*ran_mult(0) + ran_seed(2)*ran_mult(1)
     *    + ran_seed(1)*ran_mult(2) + ran_seed(0)*ran_mult(3);
     * i2 = ran_seed(2)*ran_mult(0) + ran_seed(1)*ran_mult(1)
     *    + ran_seed(0)*ran_mult(2);
     * i1 = ran_seed(1)*ran_mult(0) + ran_seed(0)*ran_mult(1);
     * i0 = ran_seed(0)*ran_mult(0);
     *
     * ran_seed(0) = mod(i0, 4096);
     * i1          = i1 + i0/4096;
     * ran_seed(1) = mod(i1, 4096);
     * i2          = i2 + i1/4096;
     * ran_seed(2) = mod(i2, 4096);
     * ran_seed(3) = mod(i3 + i2/4096, 2048);
     *
     * sranf = twom11*(TO_REAL32(VALUE(ran_seed(3)))
     *       + twom12*(TO_REAL32(VALUE(ran_seed(2)))
     *       + twom12*(TO_REAL32(VALUE(ran_seed(1)))
     *       + twom12*(TO_REAL32(VALUE(ran_seed(0)))))));
     */
    ILatticeReal _sranf;
    Seed ran_tmp1;
    ILatticeSeed ran_tmp2;

    _sranf = seedToFloat(skewed_seed);
    for(int i=0; i < N; ++i)
    {
      cast_rep(d[i], getSite(_sranf,i));
    }

    ran_tmp1 = seed * seed_mult;
    seed = ran_tmp1;

    ran_tmp2 = skewed_seed * seed_mult;
    skewed_seed = ran_tmp2;
  }
#endif

};


} // namespace QDP;

#endif

