// -*- C++ -*-
//
// QDP data parallel interface
//
// Random number support

#ifndef QDP_RANDOMJIT_H
#define QDP_RANDOMJIT_H

#if 0
namespace QDP {


namespace RNG
{

  // WordJIT<T> sranf(typename JITType<Seed>::Type_t& seed, 
  // 		   typename JITType<Seed>::Type_t& skewed_seed, 
  // 		   const typename JITType<Seed>::Type_t& seed_mult)

  template<class T>
  void sranf(WordJIT<T>& dest,
	     PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > & seed, 
	     PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > >& skewed_seed, 
	     const OScalarJIT<PScalarJIT<PSeedJIT<RScalarJIT<WordJIT<int> > > > >& seed_mult)
  {
    //std::cout << __PRETTY_FUNCTION__ << "\n";

    //dest.func().insert_label("before_seedToFloat");    
    dest = seedToFloat( skewed_seed ).elem().elem().elem();

    //dest.func().insert_label("before_seed");    
    seed        = seed        * seed_mult.elem(0);

    //dest.func().insert_label("before_skewed_seed");    
    skewed_seed = skewed_seed * seed_mult.elem(0);

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

#endif
