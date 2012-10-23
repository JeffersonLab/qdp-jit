// -*- C++ -*-
//
// QDP data parallel interface
//
// Random number support

#ifndef QDP_RANDOMJIT_H
#define QDP_RANDOMJIT_H

namespace QDP {


namespace RNG
{
#if 0
  void setrn(const Seed& lseed);
  void savern(Seed& lseed);
  void sranf(float* d, int N, Seed& seed, ILatticeSeed&, const Seed&);
#endif
  float sranf(typename JITContainerType<Seed>::Type_t& seed, 
	      typename JITContainerType<Seed>::Type_t& skewed_seed, 
	      const typename JITContainerType<Seed>::Type_t& seed_mult);
}


} // namespace QDP

#endif
