// -*- C++ -*-
//
//
// QDP data parallel interface
//

#ifndef QDP_PRIMGAMMA_H
#define QDP_PRIMGAMMA_H

namespace QDP {

//-------------------------------------------------------------------------------------
//! Gamma matrices
//! Simple interface for gamma matrices. These are run-time constructable
template<int N> class GammaType
{
public:
  //! Destructor
  ~GammaType() {}

  //! Index  from an integer
  GammaType(int mm) : m(mm) {}

public:
  //! The integer representation for which product of gamma matrices
  int elem() const {return m;}  

private:
  //! Main constructor 
  GammaType() {}
  //! Representation
  /*! 
   * The integer in the range 0 to Ns*Ns-1 that represents which product
   * gamma matrices
   */
  const int m;
};


template<int N>
struct LeafFunctor<GammaType<N>, ViewSpinLeaf>
{
  typedef GammaType<N> Type_t;
  inline static
  Type_t apply(const GammaType<N>& g, const ViewSpinLeaf& v)
  {
    return Type_t(g.elem());
  }
};

  

template<int N>
struct FirstWord<GammaType<N> >
{
  static int get(const GammaType<N>& a)
  {
    return 0;
  }
};



//
// GammaType Trait classes for code generation
//


template<int N>
struct LeafFunctor<GammaType<N>, ParamLeaf>
{
  typedef GammaType<N> Type_t;
  inline static
  Type_t apply(const GammaType<N>& g, const ParamLeaf& p) 
  {
    Type_t ret(g.elem());
    return ret;
  }
};


template<int N>
struct LeafFunctor<GammaType<N>, ParamLeafScalar>
{
  typedef GammaType<N> Type_t;
  inline static
  Type_t apply(const GammaType<N>& g, const ParamLeafScalar& p) 
  {
    Type_t ret(g.elem());
    return ret;
  }
};

  
template<int N>
struct LeafFunctor<GammaType<N>, ShiftPhase1>
{
  typedef int Type_t;
  static int apply(const GammaType<N> &s, const ShiftPhase1 &f) { return 0; }
};

template<int N>
struct LeafFunctor<GammaType<N>, ShiftPhase2>
{
  typedef int Type_t;
  static int apply(const GammaType<N> &s, const ShiftPhase2 &f) { return 0; }
};

template<int N>
struct LeafFunctor<GammaType<N>, ViewLeaf>
{
  typedef GammaType<N> Type_t;
  inline static
  Type_t apply(const GammaType<N>& g, const ViewLeaf& v)
  {
    Type_t ret(g.elem());
    return ret;
  }
};

template<int N>
struct LeafFunctor<GammaType<N>, AddressLeaf>
{
  typedef int Type_t;
  inline static
  Type_t apply(const GammaType<N>& g, const AddressLeaf& v)
  {
    return 0;
  }
};

template<int N>
struct LeafFunctor<GammaType<N>, DynKeyTag>
{
  typedef bool Type_t;
  inline static
  Type_t apply(const GammaType<N>& g, const DynKeyTag& v)
  {
    v.key.add( g.elem() );
    return false;
  }
};

  

//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

template<class T2, int N>
struct BinaryReturn<GammaType<N>, PScalar<T2>, OpGammaTypeMultiply> {
  typedef PScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N>
struct BinaryReturn<PScalar<T2>, GammaType<N>, OpMultiplyGammaType> {
  typedef PScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

 

  // ------- DP
#if 1

template<int N> class GammaTypeDP
{
public:
  //! Destructor
  ~GammaTypeDP() {}

  //! Index  from an integer
  GammaTypeDP(int mm) : m(mm) {}

public:
  //! The integer representation for which product of gamma matrices
  int elem() const {return m;}  

private:
  //! Main constructor 
  GammaTypeDP() {}
  //! Representation
  /*! 
   * The integer in the range 0 to Ns*Ns-1 that represents which product
   * gamma matrices
   */
  const int m;
};


template<int N>
struct LeafFunctor<GammaTypeDP<N>, ViewSpinLeaf>
{
  typedef GammaTypeDP<N> Type_t;
  inline static
  Type_t apply(const GammaTypeDP<N>& g, const ViewSpinLeaf& v)
  {
    return Type_t(g.elem());
  }
};


template<int N>
struct FirstWord<GammaTypeDP<N> >
{
  static int get(const GammaTypeDP<N>& a)
  {
    return 0;
  }
};


template<int N>
struct LeafFunctor<GammaTypeDP<N>, ParamLeaf>
{
  typedef GammaTypeDP<N> Type_t;
  inline static
  Type_t apply(const GammaTypeDP<N>& g, const ParamLeaf& p) 
  {
    Type_t ret(g.elem());
    return ret;
  }
};

template<int N>
struct LeafFunctor<GammaTypeDP<N>, ShiftPhase1>
{
  typedef int Type_t;
  static int apply(const GammaTypeDP<N> &s, const ShiftPhase1 &f) { return 0; }
};

template<int N>
struct LeafFunctor<GammaTypeDP<N>, ShiftPhase2>
{
  typedef int Type_t;
  static int apply(const GammaTypeDP<N> &s, const ShiftPhase2 &f) { return 0; }
};

template<int N>
struct LeafFunctor<GammaTypeDP<N>, ViewLeaf>
{
  typedef GammaTypeDP<N> Type_t;
  inline static
  Type_t apply(const GammaTypeDP<N>& g, const ViewLeaf& v)
  {
    Type_t ret(g.elem());
    return ret;
  }
};

template<int N>
struct LeafFunctor<GammaTypeDP<N>, AddressLeaf>
{
  typedef int Type_t;
  inline static
  Type_t apply(const GammaTypeDP<N>& g, const AddressLeaf& v)
  {
    return 0;
  }
};

template<int N>
struct LeafFunctor<GammaTypeDP<N>, DynKeyTag>
{
  typedef bool Type_t;
  inline static
  Type_t apply(const GammaTypeDP<N>& g, const DynKeyTag& v)
  {
    v.key.add( g.elem() );
    return false;
  }
};

//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

template<class T2, int N>
struct BinaryReturn<GammaTypeDP<N>, PScalar<T2>, OpGammaTypeMultiply> {
  typedef PScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N>
struct BinaryReturn<PScalar<T2>, GammaTypeDP<N>, OpMultiplyGammaType> {
  typedef PScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

#endif


} // namespace QDP

#endif
