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

  // WordJIT<T> sranf(typename JITContainerType<Seed>::Type_t& seed, 
  // 		   typename JITContainerType<Seed>::Type_t& skewed_seed, 
  // 		   const typename JITContainerType<Seed>::Type_t& seed_mult)

  template<class T>
  void sranf(WordJIT<T>& dest,
	     OLatticeJIT<PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > >& seed, 
	     OLatticeJIT<PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > >& skewed_seed, 
	     const OLatticeJIT<PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > >& seed_mult)
  {
    //std::cout << __PRETTY_FUNCTION__ << "\n";

    dest = seedToFloat( skewed_seed.elem(0) ).elem().elem().elem();

    seed.elem(0)        = seed.elem(0)        * seed_mult.elem(0);
    skewed_seed.elem(0) = skewed_seed.elem(0) * seed_mult.elem(0);

#if 0
    PScalarJIT<PScalarJIT<RScalarJIT<WordJIT<float> > > > _sranf(seed.func());
    _sranf = seedToFloat(skewed_seed.elem());                                // this is to be returned!!

    PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > ran_tmp(seed.func());

    ran_tmp = seed.elem() * seed_mult.elem();
    seed.elem() = ran_tmp;

    ran_tmp = skewed_seed.elem() * seed_mult.elem();
    skewed_seed.elem() = ran_tmp;

    return _sranf.elem().elem().elem();
#endif
  }


}


} // namespace QDP

#endif
