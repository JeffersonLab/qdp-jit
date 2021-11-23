#ifndef QDP_VIEWLEAF
#define QDP_VIEWLEAF

namespace QDP {



  // --------------------------------
  // ViewLeaf

  
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



  // --------------------------------
  // ViewLeafScalar
  
// template<class T>
// struct LeafFunctor<OScalarJIT<T>, ViewLeafScalar>
// {
//   typedef typename REGType<T>::Type_t Type_t;
//   inline static
//   Type_t apply(const OScalarJIT<T> & s, const ViewLeafScalar& v)
//   {
//     Type_t reg;
//     reg.setup( s.elem() );
//     return reg;
//   }
// };



// template<>
// struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< float > > > > >, ViewLeafScalar>
// {
//   typedef typename REGType< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< float > > > > >::Type_t Type_t;
//   inline static
//   Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< float > > > > > & s, const ViewLeafScalar& v)
//   {
//     Type_t reg;
//     reg.setup_value( s.elem() );
//     return reg;
//   }
// };

// template<>
// struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< double > > > > >, ViewLeafScalar>
// {
//   typedef typename REGType< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< double > > > > >::Type_t Type_t;
//   inline static
//   Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< double > > > > > & s, const ViewLeafScalar& v)
//   {
//     Type_t reg;
//     reg.setup_value( s.elem() );
//     return reg;
//   }
// };

// template<>
// struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< int > > > > >, ViewLeafScalar>
// {
//   typedef typename REGType< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< int > > > > >::Type_t Type_t;
//   inline static
//   Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< int > > > > > & s, const ViewLeafScalar& v)
//   {
//     Type_t reg;
//     reg.setup_value( s.elem() );
//     return reg;
//   }
// };

// template<>
// struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< bool > > > > >, ViewLeafScalar>
// {
//   typedef typename REGType< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< bool > > > > >::Type_t Type_t;
//   inline static
//   Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< bool > > > > > & s, const ViewLeafScalar& v)
//   {
//     Type_t reg;
//     reg.setup_value( s.elem() );
//     return reg;
//   }
// };



  // --------------------------------
  

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


// template<class T>
// struct LeafFunctor<OLatticeJIT<T>, ViewLeafScalar>
// {
//   typedef typename REGType<typename ScalarType<T>::Type_t >::Type_t Type_t;
//   inline static
//   Type_t apply(const OLatticeJIT<T> & s, const ViewLeafScalar& v)
//   {
//     Type_t reg;
//     reg.setup( s.elemScalar( v.getLayout() , v.getIndex() ) );
//     return reg;
//   }
// };








  

  //
  // ViewSpinLeaf
  //

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



template<>
struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< float > > > > >, ViewSpinLeaf>
{
  typedef typename REGType< PScalarJIT < RScalarJIT< WordJIT< float > > > >::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< float > > > > > & s, const ViewSpinLeaf& v)
  {
    Type_t reg;
    reg.setup_value( s.elem().elem() );
    return reg;
  }
};

template<>
struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< double > > > > >, ViewSpinLeaf>
{
  typedef typename REGType< PScalarJIT < RScalarJIT< WordJIT< double > > > >::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< double > > > > > & s, const ViewSpinLeaf& v)
  {
    Type_t reg;
    reg.setup_value( s.elem().elem() );
    return reg;
  }
};

template<>
struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< int > > > > >, ViewSpinLeaf>
{
  typedef typename REGType< PScalarJIT < RScalarJIT< WordJIT< int > > > >::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< int > > > > > & s, const ViewSpinLeaf& v)
  {
    Type_t reg;
    reg.setup_value( s.elem().elem() );
    return reg;
  }
};

template<>
struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< bool > > > > >, ViewSpinLeaf>
{
  typedef typename REGType< PScalarJIT < RScalarJIT< WordJIT< bool > > > >::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< bool > > > > > & s, const ViewSpinLeaf& v)
  {
    Type_t reg;
    reg.setup_value( s.elem().elem() );
    return reg;
  }
};



  //
  // ViewSpinLeafScalar
  //


// template<class T>
// struct LeafFunctor<OLatticeJIT<PScalarJIT<T> >, ViewSpinLeafScalar>
// {
//   typedef typename REGType<typename ScalarType<T>::Type_t>::Type_t Type_t;
//   inline static
//   Type_t apply(const OLatticeJIT<PScalarJIT<T> > & s, const ViewSpinLeafScalar& v)
//   {
//     return s.elemScalar( v.getLayout() , v.getIndex() ).getRegElem();
//   }
// };


// template<class T,int N>
// struct LeafFunctor<OLatticeJIT<PSpinMatrixJIT<T,N> >, ViewSpinLeafScalar>
// {
//   typedef typename REGType<typename ScalarType<T>::Type_t>::Type_t Type_t;
//   inline static
//   Type_t apply(const OLatticeJIT<PSpinMatrixJIT<T,N> > & s, const ViewSpinLeafScalar& v)
//   {
//     if (v.indices.size() != 2)
//       {
// 	QDPIO::cout << "at spinmat leaf but not 2 indices provided, instead = " << v.indices.size() << std::endl;
// 	QDP_abort(1);
//       }

//     return s.elemScalar( v.getLayout() , v.getIndex() ).getRegElem( v.indices[ 0 ] , v.indices[ 1 ] );
//   }
// };


// template<class T,int N>
// struct LeafFunctor<OLatticeJIT<PSpinVectorJIT<T,N> >, ViewSpinLeafScalar>
// {
//   typedef typename REGType<typename ScalarType<T>::Type_t>::Type_t Type_t;
//   inline static
//   Type_t apply(const OLatticeJIT<PSpinVectorJIT<T,N> > & s, const ViewSpinLeafScalar& v)
//   {
//     if (v.indices.size() != 1)
//       {
// 	QDPIO::cout << "at spinvec leaf but not 1 index provided" << std::endl;
// 	QDP_abort(1);
//       }

//     return s.elemScalar( v.getLayout() , v.getIndex() ).getRegElem( v.indices[ 0 ] );
//   }
// };

  
  

// template<class T>
// struct LeafFunctor<OScalarJIT< PScalarJIT<T> >, ViewSpinLeafScalar>
// {
//   typedef typename REGType<typename ScalarType<T>::Type_t>::Type_t Type_t;
//   inline static
//   Type_t apply(const OScalarJIT< PScalarJIT<T> > & s, const ViewSpinLeafScalar& v)
//   {
//     return s.elem().getRegElem();
//   }
// };


// template<class T,int N>
// struct LeafFunctor<OScalarJIT<PSpinMatrixJIT<T,N> >, ViewSpinLeafScalar>
// {
//   typedef typename REGType<typename ScalarType<T>::Type_t>::Type_t Type_t;
//   inline static
//   Type_t apply(const OScalarJIT<PSpinMatrixJIT<T,N> > & s, const ViewSpinLeafScalar& v)
//   {
//     if (v.indices.size() != 2)
//       {
// 	QDPIO::cout << "at OSca<spinmat> leaf but not 2 indices provided" << std::endl;
// 	QDP_abort(1);
//       }

//     return s.elem().getRegElem( v.indices[ 0 ] , v.indices[ 1 ] );
//   }
// };



// template<class T,int N>
// struct LeafFunctor<OScalarJIT<PSpinVectorJIT<T,N> >, ViewSpinLeafScalar>
// {
//   typedef typename REGType<typename ScalarType<T>::Type_t>::Type_t Type_t;
//   inline static
//   Type_t apply(const OScalarJIT<PSpinVectorJIT<T,N> > & s, const ViewSpinLeafScalar& v)
//   {
//     if (v.indices.size() != 1)
//       {
// 	QDPIO::cout << "at OSca<spinvec> leaf but not 1 indices provided" << std::endl;
// 	QDP_abort(1);
//       }

//     return s.elem().getRegElem( v.indices[ 0 ] );
//   }
// };


// template<>
// struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< float > > > > >, ViewSpinLeafScalar>
// {
//   typedef typename REGType< PScalarJIT < RScalarJIT< WordJIT< float > > > >::Type_t Type_t;
//   inline static
//   Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< float > > > > > & s, const ViewSpinLeafScalar& v)
//   {
//     Type_t reg;
//     reg.setup_value( s.elem().elem() );
//     return reg;
//   }
// };

// template<>
// struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< double > > > > >, ViewSpinLeafScalar>
// {
//   typedef typename REGType< PScalarJIT < RScalarJIT< WordJIT< double > > > >::Type_t Type_t;
//   inline static
//   Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< double > > > > > & s, const ViewSpinLeafScalar& v)
//   {
//     Type_t reg;
//     reg.setup_value( s.elem().elem() );
//     return reg;
//   }
// };

// template<>
// struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< int > > > > >, ViewSpinLeafScalar>
// {
//   typedef typename REGType< PScalarJIT < RScalarJIT< WordJIT< int > > > >::Type_t Type_t;
//   inline static
//   Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< int > > > > > & s, const ViewSpinLeafScalar& v)
//   {
//     Type_t reg;
//     reg.setup_value( s.elem().elem() );
//     return reg;
//   }
// };

// template<>
// struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< bool > > > > >, ViewSpinLeafScalar>
// {
//   typedef typename REGType< PScalarJIT < RScalarJIT< WordJIT< bool > > > >::Type_t Type_t;
//   inline static
//   Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< bool > > > > > & s, const ViewSpinLeafScalar& v)
//   {
//     Type_t reg;
//     reg.setup_value( s.elem().elem() );
//     return reg;
//   }
// };

///////////////////////////////////////////////////


  

/// JIT2BASE

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



}

#endif
