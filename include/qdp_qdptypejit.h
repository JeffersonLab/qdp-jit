// -*- C++ -*-

#ifndef QDP_QDPTYPEJIT_H
#define QDP_QDPTYPEJIT_H

namespace QDP {

  



  template<class T, class C> 
  class QDPTypeJIT: public QDPTypeJITBase
  {
  public:
    //! Type of the first argument
    typedef T Subtype_t;

    //! Type of the container class
    typedef C Container_t;


    QDPTypeJIT( jit_function_t func_ , 
		jit_value_t base_ ,
		jit_value_t index_ ): func_m(func_), base_m(base_), index_m(index_) {
      std::cout << "QDPTypeJIT 3er ctor\n";
      assert(func_m);
      assert(base_m);
      assert(index_m);
    }

    QDPTypeJIT( jit_function_t func_ , 
		jit_value_t base_ ): func_m(func_), base_m(base_) {
      std::cout << "QDPTypeJIT 2er ctor\n";
      assert(func_m);
      assert(base_m);
    }

    QDPTypeJIT(const QDPTypeJIT& a) : func_m(a.func_m), base_m(a.base_m), index_m(a.index_m) { }

    ~QDPTypeJIT(){}


    jit_value_t getThreadedBase( DeviceLayout lay ) const {

      std::cout << "getThreadedBase should use mul_wide\n";
      assert(index_m);
      assert(base_m);

      jit_value_t wordsize = jit_val_create_const_int( sizeof(typename WordType<T>::Type_t) );
      jit_value_t ret0 = jit_ins_mul( wordsize , index_m );
      if (lay != DeviceLayout::Coalesced) {
	jit_value_t tsize = jit_val_create_const_int( T::Size_t );
	jit_value_t ret1 = jit_ins_mul( ret0 , tsize );
	jit_value_t ret2 = jit_ins_add( ret1 , base_m );
	return ret2;
      }
      jit_value_t ret1 = jit_ins_add( ret0 , base_m );
      return ret1;
    }



    T& elem(DeviceLayout lay) {
      int innerSites = lay == DeviceLayout::Coalesced ? Layout::sitesOnNode() : 1;
      F.setup(func_m,getThreadedBase(lay),innerSites,0);
      return F;
    }

    const T& elem(DeviceLayout lay) const {
      int innerSites = lay == DeviceLayout::Coalesced ? Layout::sitesOnNode() : 1;
      F.setup(func_m,getThreadedBase(lay),innerSites,0);
      return F;
    }

    T& elem() {
      F.setup(func_m,getThreadedBase(DeviceLayout::Scalar),1,0);
      return F;
    }

    const T& elem() const {
      F.setup(func_m,getThreadedBase(DeviceLayout::Scalar),1,0);
      return F;
    }

    //jit_function_t& getFunc() const { return function; }
    //int getAddr() const { return r_base; }
    //jit_function_t getFunc() const { return func; }


  private:
    jit_function_t func_m;
    jit_value_t    base_m;
    jit_value_t    index_m;
    mutable T F;


  };
  

  //-----------------------------------------------------------------------------
  // Traits classes to support return types
  //-----------------------------------------------------------------------------

#if 0
  // Default unary(QDPTypeJIT) -> QDPTypeJIT
    template<class T1, class C1, class Op>
    struct UnaryReturn<QDPTypeJIT<T1,C1>, Op> {
      typedef QDPTypeJIT<typename UnaryReturn<T1, Op>::Type_t,
			 typename UnaryReturn<C1, Op>::Type_t>  Type_t;
};

// Default binary(QDPTypeJIT,QDPTypeJIT) -> QDPTypeJIT
template<class T1, class C1, class T2, class C2, class Op>
struct BinaryReturn<QDPTypeJIT<T1,C1>, QDPTypeJIT<T2,C2>, Op> {
  typedef QDPTypeJIT<typename BinaryReturn<T1, T2, Op>::Type_t,
		     typename BinaryReturn<C1, C2, Op>::Type_t>  Type_t;
};

  // Currently, the only trinary operator is ``where'', so return 
  // based on T2 and T3
  // Default trinary(QDPTypeJIT,QDPTypeJIT,QDPTypeJIT) -> QDPTypeJIT
  template<class T1, class C1, class T2, class C2, class T3, class C3, class Op>
  struct TrinaryReturn<QDPTypeJIT<T1,C1>, QDPTypeJIT<T2,C2>, QDPTypeJIT<T3,C3>, Op> {
    typedef QDPTypeJIT<typename BinaryReturn<T2, T3, Op>::Type_t,
		       typename BinaryReturn<C2, C3, Op>::Type_t>  Type_t;
};


//-----------------------------------------------------------------------------
// We need to specialize CreateLeaf<T> for our class, so that operators
// know what to stick in the leaves of the expression tree.
//-----------------------------------------------------------------------------

template<class T, class C>
struct CreateLeaf<QDPTypeJIT<T,C> >
{
  typedef QDPTypeJIT<T,C> Inp_t;
  typedef Reference<Inp_t> Leaf_t;
  //  typedef Inp_t Leaf_t;
  PETE_DEVICE
  inline static
  Leaf_t make(const Inp_t &a) { return Leaf_t(a); }
};


//-----------------------------------------------------------------------------
// Specialization of LeafFunctor class for applying the EvalLeaf1
// tag to a QDPTypeJIT. The apply method simply returns the array
// evaluated at the point.
//-----------------------------------------------------------------------------

template<class T, class C>
struct LeafFunctor<QDPTypeJIT<T,C>, ElemLeaf>
{
  typedef Reference<T> Type_t;
  //  typedef T Type_t;
  PETE_DEVICE
  inline static Type_t apply(const QDPTypeJIT<T,C> &a, const ElemLeaf &f)
  { 
    return Type_t(a.elem());
  }
};

template<class T, class C>
struct LeafFunctor<QDPTypeJIT<T,C>, EvalLeaf1>
{
  typedef Reference<T> Type_t;
  //  typedef T Type_t;
  PETE_DEVICE
  inline static Type_t apply(const QDPTypeJIT<T,C> &a, const EvalLeaf1 &f)
  { 
    return Type_t(a.elem(f.val1()));
  }
};


#endif


#if 0
template<class T>
struct LeafFunctor<QDPTypeJIT<T,OLattice<T> >, ParamLeaf>
{
  typedef QDPTypeJIT<T,typename JITType<OLattice<T> >::Type_t>  TypeA_t;
  //typedef typename JITType< OLattice<T> >::Type_t  TypeA_t;
  typedef TypeA_t  Type_t;
  inline static Type_t apply(const QDPTypeJIT<T,OLattice<T> > &a, const ParamLeaf& p)
  {
    return Type_t( p.getFunc() , jit_add_param( p.getFunc() , jit_ptx_type::u64 ) );
  }
};

template<class T>
struct LeafFunctor<QDPTypeJIT<T,OScalar<T> >, ParamLeaf>
{
  typedef QDPTypeJIT<T,typename JITType<OScalar<T> >::Type_t>  TypeA_t;
  //typedef typename JITType< OScalar<T> >::Type_t  TypeA_t;
  typedef TypeA_t  Type_t;
  inline static Type_t apply(const QDPTypeJIT<T,OScalar<T> > &a, const ParamLeaf& p)
  {
    return Type_t( p.getFunc() , jit_add_param( p.getFunc() , jit_ptx_type::u64 ) );
  }
};
#endif


// template<class T, class C>
// struct LeafFunctor<QDPTypeJIT<T,C>, ViewLeaf>
// {
//   typedef T Type_t;
//   inline static
//   Type_t apply(const QDPTypeJIT<T,C>& s, const ViewLeaf& v)
//   { 
//     return s.elem( v.getLayout() );
//   }
// };





} // namespace QDP

#endif

