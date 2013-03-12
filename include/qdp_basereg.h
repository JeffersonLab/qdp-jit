#ifndef QDP_PRIMREGBASE
#define QDP_PRIMREGBASE

// This code is not used
#if 0
namespace QDP {

  template<class T, int N, class P >
  class BaseREG {
    T F[N];

  public:
    BaseREG() {}

    T& arrayF(int i) {       
      return F[i]; 
    }
    const T& arrayF(int i) const { 
      return F[i]; 
    }

    void setup( const typename JITType<P>::Type_t& j ) {
      for (int i = 0 ; i < N ; i++ ) 
	F[i].setup( j.arrayF(i) );
    }
  };

}
#endif

#endif
