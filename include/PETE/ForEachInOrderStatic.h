// -*- C++ -*-
// ACL:license
// ----------------------------------------------------------------------
// This software and ancillary information (herein called "SOFTWARE")
// called PETE (Portable Expression Template Engine) is
// made available under the terms described here.  The SOFTWARE has been
// approved for release with associated LA-CC Number LA-CC-99-5.
// 
// Unless otherwise indicated, this SOFTWARE has been authored by an
// employee or employees of the University of California, operator of the
// Los Alamos National Laboratory under Contract No.  W-7405-ENG-36 with
// the U.S. Department of Energy.  The U.S. Government has rights to use,
// reproduce, and distribute this SOFTWARE. The public may copy, distribute,
// prepare derivative works and publicly display this SOFTWARE without 
// charge, provided that this Notice and any statement of authorship are 
// reproduced on all copies.  Neither the Government nor the University 
// makes any warranty, express or implied, or assumes any liability or 
// responsibility for the use of this SOFTWARE.
// 
// If SOFTWARE is modified to produce derivative works, such modified
// SOFTWARE should be clearly marked, so as not to confuse it with the
// version available from LANL.
// 
// For more information about PETE, send e-mail to pete@acl.lanl.gov,
// or visit the PETE web page at http://www.acl.lanl.gov/pete/.
// ----------------------------------------------------------------------
// ACL:license

//-----------------------------------------------------------------------------
// Classes:
// ForEachInOrder
// TagVisitor
//-----------------------------------------------------------------------------

#ifndef POOMA_PETE_FOREACHINORDERSTATIC_H
#define POOMA_PETE_FOREACHINORDERSTATIC_H

//-----------------------------------------------------------------------------
// Overview: 
//
//   ForEachInOrderStatic is like ForEach except that it traverses the parse
//   tree "in order", meaning it visits the parts of a TBTree as follows:
//
//         visit left child
//         visit value
//         visit right child
//
//   In addition, it calls a start() function on the visit tag before
//   visit(left) and a finish() function after visit(right). This
//   additional bit of generality allows special actions to be taken,
//   in essence, when the ForEachInOrderStatic::apply moves down and back up
//   the edges of the parse tree (such as printing parentheses).
//
//   An "in order" traversal is not what one does for evaluating
//   expressions, so this may not be useful for much, but I wanted to
//   do it to gain some more experience with PETE.
//
//   This first cut will only do TBTrees.
//
//   The TagFunctor and TagCombine structs from ForEach.h can be reused. 
//
//   TagVisitor is a new class that visits the "value" node prior,
//   between, and after the left and right children are visited.
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Typedefs:
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Includes:
//-----------------------------------------------------------------------------

#include "PETE/PETE.h"

//-----------------------------------------------------------------------------
// Forward Declarations:
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// Full Description:
//
// The ForEachInOrderStatic struct implements a template meta-program
// traversal of a PETE Expression parse-tree. As explained above, this 
// is done "in order" rather than the "post order" traversal done by
// ForEach. 
//
// The ForEachInOrderStatic struct defines:
//
//   typename ForEachInOrderStatic<Expr,FTag,VTag,CTag>::Type_t
//
//   Type_t::apply(Expr& expr, FTag f, VTag v, CTag c) {...};
//
// where
//
//   Expr is the type of an expression tree.
//   FTag is the type of a functor tag.
//   VTag is the type of a visitor tag.
//   CTag is the type of a combiner tag.
//
// Details:
//
//   Type_t::apply(Expr &e, FTag f, VTag v, CTag c) 
//
// function that traverses the expression tree defined by e, and for
// each binary-tree node it does:
//
//     TagVisitor<...>::start(e.value_m,v);
//
//     left_val = ForEachInOrderStatic<...>::apply(e.left_m,f,v,c),
//
//     TagVisitor<...>::visit(e.value_m,v);
//
//     right_val = ForEachInOrderStatic<...>::apply(e.right_m,f,v,c),
//
//     retval = TagCombineInOrdere<...>::
//               apply(left_val, right_val, e.value_m, c)
//
//     TagVisitor<...>::finish(e.value_m,v);
//
//     return retval;
//
// The TagFunctor is specialized to perform a specific action at the
// leaf nodes. 
//
// The TagVisitor is specialized to perform specific actions both at
// the start and finish of a new TBTree node, and to perform a
// specific operation when it visits the "value" node of the parse
// tree (i.e. this can be specialized to perform specific operations
// for every type of operator). Note that the value returned by
// TagVisitor::apply must be of the type Op. This usually means
// that TagVisitor::apply will just return e.value_m after it does
// its calculation.
//
// The TagCombiner is specialized to combine the results of visiting
// the left, right, and value fields.
//
// The type of object returned is given by: 
//
//    typename ForEachInOrderStatic<Expr,FTag,VTag,CTag>::Type_t 
//
//-----------------------------------------------------------------------------

//
// struct TagVisitor
//
// "Visitor" functor whose apply() method is applied to the value_m
// field of an expression between the left-traversal and the
// right-traversal.
//
// Default is "null" behavior. Just return the op. This should make
// this equivalent to ForEach. This should probably always return the
// unless it is ignored by everything else. But it can take other
// actions as well.
//
// Also includes start() and finish() functions that are called when
// the traversal moves down and back up an edge, respectively.
//

template <class Op, class VTag>
struct TagVisitor 
{
  static void start(VTag) { }
  static void center(VTag) { }
  static void visit(VTag) { }
  static void finish(VTag) { }
};  


// 
// struct ForEachInOrderStatic
//
// Template meta-program for traversing the parse tree.
//
// Default behaviour assumes you're at a leaf, in which case
// it just applies the FTag functor
//

template<class Expr, class FTag, class VTag, class CTag>
struct ForEachInOrderStatic
{
  typedef LeafFunctor<Expr,FTag> Tag_t;
  typedef typename Tag_t::Type_t Type_t;

  static Type_t apply(const FTag &f, const VTag &v, 
		      const CTag &c) 
  {
    return Tag_t::apply(f);
  }
};

//
// The Refernce case needs to apply the functor to the wrapped object.
//

template<class T, class FTag, class VTag, class CTag>
struct ForEachInOrderStatic<Reference<T>,FTag,VTag,CTag>
{
  typedef LeafFunctor<T,FTag> Tag_t;
  typedef typename Tag_t::Type_t Type_t;

  static Type_t apply(const FTag &f,
		      const VTag &v, const CTag &c) 
  {
    return Tag_t::apply(f);
  }
};

//
// struct ForEachInOrderStatic
//
// Specialization for a TBTree. This just performs the recursive
// traversal described above.
//

template<class Op, class A, class FTag, class VTag, 
  class CTag>
struct ForEachInOrderStatic<UnaryNode<Op, A>, FTag, VTag, CTag>
{
  typedef ForEachInOrderStatic<A, FTag, VTag, CTag> ForEachA_t;
  typedef TagVisitor<Op, VTag>          Visitor_t;

  static void apply(const FTag &f, 
		    const VTag &v, const CTag &c)
  {
    Visitor_t::visit(v);

    Visitor_t::start(v);

    ForEachA_t::apply(f, v, c);
        
    Visitor_t::finish(v);
  }
};


/*!
 * struct ForEachInOrderStatic for BinaryNode
 */

template<class Op, class A, class B, class FTag, class VTag, 
  class CTag>
struct ForEachInOrderStatic<BinaryNode<Op, A, B>, FTag, VTag, CTag>
{
  typedef ForEachInOrderStatic<A, FTag, VTag, CTag> ForEachA_t;
  typedef ForEachInOrderStatic<B, FTag, VTag, CTag> ForEachB_t;
  typedef TagVisitor<Op, VTag>                Visitor_t;

  static void apply(const FTag &f, 
		    const VTag &v, const CTag &c) 
  {
    Visitor_t::visit(v);

    Visitor_t::start(v);

    ForEachA_t::apply(f, v, c);

    Visitor_t::center(v);

    ForEachB_t::apply(f, v, c);

    Visitor_t::finish(v);
  }
};


/*!
 * struct ForEachInOrderStatic for BinaryNode
 */

template<class Op, class A, class B, class C, class FTag, class VTag, 
  class CTag>
struct ForEachInOrderStatic<TrinaryNode<Op, A, B, C>, FTag, VTag, CTag>
{
  typedef ForEachInOrderStatic<A, FTag, VTag, CTag> ForEachA_t;
  typedef ForEachInOrderStatic<B, FTag, VTag, CTag> ForEachB_t;
  typedef ForEachInOrderStatic<C, FTag, VTag, CTag> ForEachC_t;
  typedef TagVisitor<Op, VTag>                Visitor_t;

  static void apply(const FTag &f, 
		    const VTag &v, const CTag &c) 
  {
    Visitor_t::visit(v);

    Visitor_t::start(v);

    ForEachA_t::apply(f, v, c);

    Visitor_t::center(v);

    ForEachB_t::apply(f, v, c);

    Visitor_t::center(v);

    ForEachC_t::apply(f, v, c);

    Visitor_t::finish(v);
  }
};


struct NullTag {};

template<class A,class B,class Op>
struct Combine2<A, B, Op, NullTag>
{
  typedef int Type_t;
  static Type_t combine(A, B, Op, NullTag)
    { return 0; }
};



#endif  // PETE_PETE_FOREACHINORDERSTATIC_H

// ACL:rcsinfo
// ----------------------------------------------------------------------
// $RCSfile: ForEachInOrderStatic.h,v $   $Author: edwards $
// $Revision: 1.2 $   $Date: 2004-07-27 05:24:35 $
// ----------------------------------------------------------------------
// ACL:rcsinfo
