#ifndef QDP_VIEWLEAF
#define QDP_VIEWLEAF

namespace QDP {

template<class T>
struct LeafFunctor<QDPTypeJIT<T,OLatticeJIT<T> >, ViewLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const QDPTypeJIT<T,OLatticeJIT<T> > & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup( s.elem( v.getLayout() , v.getIndex() ) );
    return reg;
  }
};


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



template<>
struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< float > > > > >, ViewLeaf>
{
  typedef typename REGType< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< float > > > > >::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< float > > > > > & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup_value( s.elem() );
    return reg;
  }
};

template<>
struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< double > > > > >, ViewLeaf>
{
  typedef typename REGType< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< double > > > > >::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< double > > > > > & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup_value( s.elem() );
    return reg;
  }
};

template<>
struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< int > > > > >, ViewLeaf>
{
  typedef typename REGType< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< int > > > > >::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< int > > > > > & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup_value( s.elem() );
    return reg;
  }
};

template<>
struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< bool > > > > >, ViewLeaf>
{
  typedef typename REGType< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< bool > > > > >::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< bool > > > > > & s, const ViewLeaf& v)
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

  


}

#endif
