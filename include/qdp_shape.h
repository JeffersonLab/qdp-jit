#ifndef QDP_SHAPE_H
#define QDP_SHAPE_H

#include <cassert>
#include <csignal>


namespace QDP
{
  template<class T> struct JITType;

  class Shape
  {
    int ns_;
    int nc_;
  public:
    enum class Axis { Spin , Color };

    Shape():ns_(1),nc_(1)
    {
    }

    explicit Shape(int ns,int nc):ns_(ns),nc_(nc){} 
    
    int      nc() const {return nc_;}

    int      ns() const {return ns_;}

    
    int extent( Axis axis ) const
    {
      if ( axis == Axis::Spin )
	return ns_;
      else
	return nc_;
    }

    void set_nc(int n) { nc_=n; }
    void set_ns(int n) { ns_=n; }

    static void push( const Shape& shape );
    static void pop();
    static Shape current();
  };

  inline bool operator==( const Shape& l , const Shape& r )
  {
    return l.nc() == r.nc() && l.ns() == r.ns();
  }

  inline bool operator!=( const Shape& l , const Shape& r )
  {
    return !(l == r);
  }
  
  
  std::ostream& operator<< (std::ostream& out, const Shape& v);

  
  class ShapeScope
  {
  public:
    explicit ShapeScope( const Shape& shape ) { Shape::push( shape ); }

    ~ShapeScope() { close(); }

    void close()
    {
      if (active)
	{
	  Shape::pop();
	  active = false;
	}
    }

    ShapeScope( const ShapeScope& ) = delete;
    ShapeScope& operator=( const ShapeScope& ) = delete;

  private:
    bool active = true;
  };

  int merge_extent(int lhs, int rhs);


  int merge_axis_extent(int lhs, bool lhs_has_axis, int rhs, bool rhs_has_axis);

  inline bool extent_accepts(int lhs, int rhs, bool rhs_has_axis)
  {
    return rhs == lhs || (rhs == 1 && !rhs_has_axis);
  }

  template<class RJ>
  bool shape_accepts_jit(const Shape& lhs, const Shape& rhs)
  {
    return extent_accepts(lhs.ns(), rhs.ns(), ShapeAxes<RJ>::has_spin) &&
      extent_accepts(lhs.nc(), rhs.nc(), ShapeAxes<RJ>::has_color);
  }

  void incompatible_shapes_exit(const char* who, const Shape& lhs, const Shape& rhs);

  inline Shape merge_shapes(const Shape& lhs, const Shape& rhs)
  {
    return Shape{
      merge_extent(lhs.ns(), rhs.ns()),
      merge_extent(lhs.nc(), rhs.nc())
    };
  }


  inline Shape merge_shapes(const Shape& a, const Shape& b, const Shape& c)
  {
    return merge_shapes(merge_shapes(a, b), c);
  }


  template<class A, class B>
  Shape merge_operand_shapes(const A& a, const B& b)
  {
    return Shape{
      merge_axis_extent(a.shape().ns(), ShapeAxes<A>::has_spin, b.shape().ns(), ShapeAxes<B>::has_spin),
      merge_axis_extent(a.shape().nc(), ShapeAxes<A>::has_color, b.shape().nc(), ShapeAxes<B>::has_color)
    };
  }

  template<class A, class B, class C>
  Shape merge_operand_shapes(const A& a, const B& b, const C& c)
  {
    const bool ab_spin  = ShapeAxes<A>::has_spin  || ShapeAxes<B>::has_spin;
    const bool ab_color = ShapeAxes<A>::has_color || ShapeAxes<B>::has_color;
    const Shape ab = merge_operand_shapes(a, b);
    return Shape{
      merge_axis_extent(ab.ns(), ab_spin, c.shape().ns(), ShapeAxes<C>::has_spin),
      merge_axis_extent(ab.nc(), ab_color, c.shape().nc(), ShapeAxes<C>::has_color)
    };
  }

  

  template<class J>
  Shape fit_shape_jit(Shape q, Shape fallback = Shape())
  {
    if constexpr (ShapeAxes<J>::has_spin)
      {
	if (q.ns() == 1)
	  q.set_ns(fallback.ns());
      }
    else
      q.set_ns(1);

    if constexpr (ShapeAxes<J>::has_color)
      {
	if (q.nc() == 1)
	  q.set_nc(fallback.nc());
      }
    else
      q.set_nc(1);

    return q;
  }


  template<class T>
  Shape fit_shape(Shape q, Shape fallback = Shape())
  {
    return fit_shape_jit<typename JITType<T>::Type_t>(q, fallback);
  }



  template<class J, class RJ>
  Shape fit_shape_from_jit(Shape q, Shape fallback)
  {
    if constexpr (ShapeAxes<J>::has_spin)
      {
	if (!ShapeAxes<RJ>::has_spin && q.ns() == 1)
	  q.set_ns(fallback.ns());
      }
    else
      q.set_ns(1);

    if constexpr (ShapeAxes<J>::has_color)
      {
	if (!ShapeAxes<RJ>::has_color && q.nc() == 1)
	  q.set_nc(fallback.nc());
      }
    else
      q.set_nc(1);

    return q;
  }


  template<class T, class T1>
  Shape fit_shape_from(Shape q, Shape fallback)
  {
    return fit_shape_from_jit<typename JITType<T>::Type_t, typename JITType<T1>::Type_t>(q, fallback);
  }

  

} // QDP
#endif
