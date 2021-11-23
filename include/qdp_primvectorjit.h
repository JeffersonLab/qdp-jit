// -*- C++ -*-

/*! \file
 * \brief Primitive Vector
 */


#ifndef QDP_PRIMVECTORJIT_H
#define QDP_PRIMVECTORJIT_H

namespace QDP {



template <class T, int N, template<class,int> class C> class PVectorJIT: public BaseJIT<T,N>
{
public:
  typedef C<T,N>  CC;

public:
        T& elem(int i)       {return this->arrayF(i);}
  const T& elem(int i) const {return this->arrayF(i);}
};




//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

// Underlying word type
template<class T1, int N, template<class,int> class C>
struct WordType<PVectorJIT<T1,N,C> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

template<class T1, int N, template<class, int> class C> 
struct SinglePrecType< PVectorJIT<T1,N,C> >
{
  typedef PVectorJIT< typename SinglePrecType<T1>::Type_t, N, C> Type_t;
};

template<class T1, int N, template<class, int> class C> 
struct DoublePrecType< PVectorJIT<T1,N,C> >
{
  typedef PVectorJIT< typename DoublePrecType<T1>::Type_t, N, C> Type_t;
};

// Internally used scalars
template<class T, int N, template<class,int> class C>
struct InternalScalar<PVectorJIT<T,N,C> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive scalar leaving grid alone
template<class T, int N, template<class,int> class C>
struct PrimitiveScalar<PVectorJIT<T,N,C> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T, int N, template<class,int> class C>
struct LatticeScalar<PVectorJIT<T,N,C> > {
  typedef C<typename LatticeScalar<T>::Type_t, N>  Type_t;
};



//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------
  

//! dest = 0
template<class T, int N, template<class,int> class C> 
inline void 
zero_rep(PVectorJIT<T,N,C>& dest) 
{
  for(int i=0; i < N; ++i)
    zero_rep(dest.elem(i));
}


//! dest  = random  
  template<class T, int N, template<class,int> class C, class T1, class T2, class T3>
inline void
fill_random_jit(PVectorJIT<T,N,C>& d, T1 seed, T2 skewed_seed, const T3& seed_mult)
{
  // Loop over rows the slowest
  for(int i=0; i < N; ++i)
    fill_random_jit(d.elem(i), seed, skewed_seed, seed_mult);
}


//! dest  = gaussian
template<class T,class T2, int N, template<class,int> class C, template<class,int> class C2>
inline void
fill_gaussian(PVectorJIT<T,N,C>& d, PVectorREG<T2,N,C2>& r1, PVectorREG<T2,N,C2>& r2)
{
  for(int i=0; i < N; ++i)
    fill_gaussian(d.elem(i), r1.elem(i), r2.elem(i));
}


/*! @} */  // end of group primvector

} // namespace QDP

#endif
