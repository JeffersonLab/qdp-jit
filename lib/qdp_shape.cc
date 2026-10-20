#include "qdp.h"
#include <stack>


namespace QDP
{
  namespace
  {
    std::stack<Shape>* shape_stack = nullptr;

    std::stack<Shape>& get_shape_stack()
    {
      if (!shape_stack)
	shape_stack = new std::stack<Shape>;
      return *shape_stack;
    }
  }


  void Shape::push( const Shape& shape )
  {
    get_shape_stack().push(shape);
  }

  void Shape::pop()
  {
    if (get_shape_stack().empty())
      QDP_error_exit("Shape::pop: Shape stack is empty");
    get_shape_stack().pop();
  }

  Shape Shape::current()
  {
    if (get_shape_stack().empty())
      QDP_error_exit("Shape::current: Shape stack is empty");
    return get_shape_stack().top();
  }


  int merge_extent(int lhs, int rhs)
  {
    assert(lhs != 0 && rhs != 0);

    if (lhs == rhs)
      return lhs;

    if (lhs == 1)
      return rhs;

    if (rhs == 1)
      return lhs;

    QDP_error_exit("Cannot combine incompatible extents %d and %d", lhs, rhs);
    return 1;
  }



  
  int merge_axis_extent(int lhs, bool lhs_has_axis, int rhs, bool rhs_has_axis)
  {
    assert(lhs != 0 && rhs != 0);

    if (lhs == 1 && !lhs_has_axis)
      return rhs;

    if (rhs == 1 && !rhs_has_axis)
      return lhs;

    if (lhs != rhs)
      QDP_error_exit("Cannot combine incompatible extents %d and %d", lhs, rhs);

    return lhs;
  }


  void incompatible_shapes_exit(const char* who, const Shape& lhs, const Shape& rhs)
  {
    QDP_error_exit("%s: Incompatible tensor dimensions: "
		   "lhs=(Nc=%d,Ns=%d), rhs=(Nc=%d,Ns=%d)",
		   who, lhs.nc(), lhs.ns(), rhs.nc(), rhs.ns());
  }

  std::ostream& operator<< (std::ostream& out, const Shape& v) {
    out << "Shape(ns=" << v.ns() << ", nc=" << v.nc() << ")";
    return out;
  }




}
