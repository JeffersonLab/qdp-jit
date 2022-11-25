#ifndef QDP_VIEWLEAF
#define QDP_VIEWLEAF

namespace QDP {


  
template<class T>
struct LeafFunctor<OScalarJIT<T>, ViewLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT<T> & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup( s.elem() );
    return reg;
  }
};



template<class T>
struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< T > > > > >, ViewLeaf>
{
  typedef typename REGType< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< T > > > > >::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< T > > > > > & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup_value( s.elem() );
    return reg;
  }
};




template<class T>
struct LeafFunctor<OLatticeJIT<T>, ViewLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const OLatticeJIT<T> & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup( s.elem( v.getLayout() , v.getIndex() ) );
    return reg;
  }
};









template<class T>
struct LeafFunctor<OLatticeJIT<PScalarJIT<T> >, ViewSpinLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const OLatticeJIT<PScalarJIT<T> > & s, const ViewSpinLeaf& v)
  {
    return s.elem( v.getLayout() , v.getIndex() ).getRegElem();
  }
};


template<class T,int N>
struct LeafFunctor<OLatticeJIT<PSpinMatrixJIT<T,N> >, ViewSpinLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const OLatticeJIT<PSpinMatrixJIT<T,N> > & s, const ViewSpinLeaf& v)
  {
    if (v.indices.size() != 2)
      {
	QDPIO::cout << "at spinmat leaf but not 2 indices provided, instead = " << v.indices.size() << std::endl;
	QDP_abort(1);
      }

    return s.elem( v.getLayout() , v.getIndex() ).getRegElem( v.indices[ 0 ] , v.indices[ 1 ] );
  }
};


template<class T,int N>
struct LeafFunctor<OLatticeJIT<PSpinVectorJIT<T,N> >, ViewSpinLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const OLatticeJIT<PSpinVectorJIT<T,N> > & s, const ViewSpinLeaf& v)
  {
    if (v.indices.size() != 1)
      {
	QDPIO::cout << "at spinvec leaf but not 1 index provided" << std::endl;
	QDP_abort(1);
      }

    return s.elem( v.getLayout() , v.getIndex() ).getRegElem( v.indices[ 0 ] );
  }
};

  
  

template<class T>
struct LeafFunctor<OScalarJIT< PScalarJIT<T> >, ViewSpinLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT<T> > & s, const ViewSpinLeaf& v)
  {
    return s.elem().getRegElem();
  }
};


template<class T,int N>
struct LeafFunctor<OScalarJIT<PSpinMatrixJIT<T,N> >, ViewSpinLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT<PSpinMatrixJIT<T,N> > & s, const ViewSpinLeaf& v)
  {
    if (v.indices.size() != 2)
      {
	QDPIO::cout << "at OSca<spinmat> leaf but not 2 indices provided" << std::endl;
	QDP_abort(1);
      }

    return s.elem().getRegElem( v.indices[ 0 ] , v.indices[ 1 ] );
  }
};



template<class T>
struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< T > > > > >, ViewSpinLeaf>
{
  typedef typename REGType< PScalarJIT < RScalarJIT< WordJIT< T > > > >::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< T > > > > > & s, const ViewSpinLeaf& v)
  {
    Type_t reg;
    reg.setup_value( s.elem().elem() );
    return reg;
  }
};




template<class T>
struct LeafFunctor<OScalarJIT<T>, JIT2BASE>
{
  typedef OScalar< typename BASEType<T>::Type_t > Type_t;
  inline static
  Type_t apply(const OScalarJIT<T> & s, const JIT2BASE& v)
  {
    Type_t r;
    return r;
  }
};

template<class T>
struct LeafFunctor<OLatticeJIT<T>, JIT2BASE>
{
  typedef OLattice< typename BASEType<T>::Type_t >  Type_t;
  //typedef int Type_t;
  inline static
  Type_t apply(const OLatticeJIT<T> & s, const JIT2BASE& v)
  {
    Type_t r;
    return r;
  }
};



template<int N>
struct LeafFunctor<GammaType<N>, JIT2BASE>
{
  typedef GammaType<N> Type_t;
  inline static
  Type_t apply(const GammaType<N> & s, const JIT2BASE& v)
  {
    Type_t r(s.elem());
    return r;
  }
};


template<int N>
struct LeafFunctor<GammaTypeDP<N>, JIT2BASE>
{
  typedef GammaTypeDP<N> Type_t;
  inline static
  Type_t apply(const GammaTypeDP<N> & s, const JIT2BASE& v)
  {
    Type_t r(s.elem());
    return r;
  }
};



}

#endif
