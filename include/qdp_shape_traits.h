#ifndef QDP_SHAPE_TRAITS_H
#define QDP_SHAPE_TRAITS_H


namespace QDP {

template<class T>
struct ShapeAxes
{
  static constexpr bool has_spin  = false;
  static constexpr bool has_color = false;
};


template<class T>
struct ShapeAxes<PScalarJIT<T>>
{
  static constexpr bool has_spin =
      ShapeAxes<T>::has_spin;

  static constexpr bool has_color =
      ShapeAxes<T>::has_color;
};


template<class T>
struct ShapeAxes<RScalarJIT<T>>
{
  static constexpr bool has_spin  = false;
  static constexpr bool has_color = false;
};


template<class T>
struct ShapeAxes<RComplexJIT<T>>
{
  static constexpr bool has_spin  = false;
  static constexpr bool has_color = false;
};


template<class T>
struct ShapeAxes<PSpinVectorJIT<T>>
{
  static constexpr bool has_spin = true;

  static constexpr bool has_color =
      ShapeAxes<T>::has_color;
};


template<class T>
struct ShapeAxes<PSpinMatrixJIT<T>>
{
  static constexpr bool has_spin = true;

  static constexpr bool has_color =
      ShapeAxes<T>::has_color;
};


template<class T>
struct ShapeAxes<PColorVectorJIT<T>>
{
  static constexpr bool has_spin =
      ShapeAxes<T>::has_spin;

  static constexpr bool has_color = true;
};


template<class T>
struct ShapeAxes<PColorMatrixJIT<T>>
{
  static constexpr bool has_spin =
      ShapeAxes<T>::has_spin;

  static constexpr bool has_color = true;
};



  template<class Op, class A>
  struct UnaryShapeTransform
  {
    static void apply(Shape&, const A&)
    {
      // Most operations do not transform an extent.
    }
  };
  

  template<class Op, class LHS, class RHS>
  struct BinaryShapeTransform
  {
    static void apply(Shape&, const LHS&, const RHS&)
    {
      // Most binary operations do not transform extents.
    }
  };


  template<class Op, class A, class B, class C>
  struct TrinaryShapeTransform
  {
    static void apply(Shape&, const A&, const B&, const C&)
    {
      // Most binary operations do not transform extents.
    }
  };



  template<class A>
  struct SpinProjectShapeTransform
  {
    static void apply(Shape& shape, const A&)
    {
      if (shape.ns() != 4)
	{
	  QDP_error_exit(
			 "Spin projection requires Ns=4; got Ns=%d",
			 shape.ns());
	}

      shape.set_ns(2);
    }
  };

  template<class A>
  struct SpinReconstructShapeTransform
  {
    static void apply(Shape& shape, const A&)
    {
      if (shape.ns() != 2)
	{
	  QDP_error_exit(
			 "Spin reconstruction requires Ns=2; got Ns=%d",
			 shape.ns());
	}

      shape.set_ns(4);
    }
  };

  

  template<class A> struct UnaryShapeTransform<FnSpinProjectDir0Plus ,A>: public SpinProjectShapeTransform<A>{};
  template<class A> struct UnaryShapeTransform<FnSpinProjectDir0Minus,A>: public SpinProjectShapeTransform<A>{};
  template<class A> struct UnaryShapeTransform<FnSpinProjectDir1Plus ,A>: public SpinProjectShapeTransform<A>{};
  template<class A> struct UnaryShapeTransform<FnSpinProjectDir1Minus,A>: public SpinProjectShapeTransform<A>{};
  template<class A> struct UnaryShapeTransform<FnSpinProjectDir2Plus ,A>: public SpinProjectShapeTransform<A>{};
  template<class A> struct UnaryShapeTransform<FnSpinProjectDir2Minus,A>: public SpinProjectShapeTransform<A>{};
  template<class A> struct UnaryShapeTransform<FnSpinProjectDir3Plus ,A>: public SpinProjectShapeTransform<A>{};
  template<class A> struct UnaryShapeTransform<FnSpinProjectDir3Minus,A>: public SpinProjectShapeTransform<A>{};

  template<class A> struct UnaryShapeTransform<FnSpinReconstructDir0Plus ,A>: public SpinReconstructShapeTransform<A>{};
  template<class A> struct UnaryShapeTransform<FnSpinReconstructDir0Minus,A>: public SpinReconstructShapeTransform<A>{};
  template<class A> struct UnaryShapeTransform<FnSpinReconstructDir1Plus ,A>: public SpinReconstructShapeTransform<A>{};
  template<class A> struct UnaryShapeTransform<FnSpinReconstructDir1Minus,A>: public SpinReconstructShapeTransform<A>{};
  template<class A> struct UnaryShapeTransform<FnSpinReconstructDir2Plus ,A>: public SpinReconstructShapeTransform<A>{};
  template<class A> struct UnaryShapeTransform<FnSpinReconstructDir2Minus,A>: public SpinReconstructShapeTransform<A>{};
  template<class A> struct UnaryShapeTransform<FnSpinReconstructDir3Plus ,A>: public SpinReconstructShapeTransform<A>{};
  template<class A> struct UnaryShapeTransform<FnSpinReconstructDir3Minus,A>: public SpinReconstructShapeTransform<A>{};


  
  template<class A, class B, class C>
  struct TrinaryShapeTransform<FnColorContract,A,B,C> {
    static void apply(Shape& shape, const A&, const B&, const C&)
    {
      shape.set_nc( 1 );
    }
  };

  
  
} //namespace
#endif
