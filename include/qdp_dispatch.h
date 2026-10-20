#ifndef QDP_DISPATCH_H
#define QDP_DISPATCH_H


namespace QDP {

  inline
  int qdpNumThreads()
  {
    return 1;
  }

  template<class Arg>
  void dispatch_to_threads(int numSiteTable, Arg a, void (*func)(int,int,int,Arg*))
  {
    func(0, numSiteTable, 0, &a);
  }

} // QDP

#endif
