#ifndef QDP_GLOBALFUNCS_SUBTYPE_H
#define QDP_GLOBALFUNCS_SUBTYPE_H

namespace QDP
{


  template<class T1, class C1, class T2, class C2>
  inline typename BinaryReturn<C1, C2, FnInnerProduct>::Type_t
  innerProduct(const QDPSubType<T1,C1>& s1, const QDPType<T2,C2>& s2)
  {
    return sum(localInnerProduct(s1,s2));
  }
  template<class T1, class C1, class T2, class C2>
  inline typename BinaryReturn<C1, C2, FnInnerProduct>::Type_t
  innerProduct(const QDPType<T1,C1>& s1, const QDPSubType<T2,C2>& s2)
  {
    return sum(localInnerProduct(s1,s2));
  }
#if 0
  template<class T1, class C1, class T2, class C2>
  inline typename BinaryReturn<C1, C2, FnInnerProduct>::Type_t
  innerProduct(const QDPSubType<T1,C1>& s1, const QDPSubType<T2,C2>& s2)
  {
    return sum(localInnerProduct(s1,s2));
  }
#endif

  template<class T1,class C1,class T2,class C2>
  typename QDPSubTypeTrait< typename BinaryReturn<C1,C2,FnLocalInnerProduct>::Type_t >::Type_t
  localInnerProduct(const QDPSubType<T1,C1> & l,const QDPType<T2,C2> & r)
  {
    QDP_error_exit("localInnerProduct with subtype: not implemented");
    __builtin_unreachable();
  }



  template<class T1,class C1,class T2,class C2>
  typename QDPSubTypeTrait< typename BinaryReturn<C1,C2,FnLocalInnerProduct>::Type_t >::Type_t
  localInnerProduct(const QDPType<T1,C1> & l,const QDPSubType<T2,C2> & r)
  {
    QDP_error_exit("localInnerProduct with subtype: not implemented");
    __builtin_unreachable();
  }




} // namespace

#endif
