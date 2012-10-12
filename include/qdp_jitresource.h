#ifndef QDP_JITR_H
#define QDP_JITR_H

#include<array>

namespace QDP {



template <int... Is>
struct indices {};

template <int N, int... Is>
struct build_indices
  : build_indices<N-1, N-1, Is...> {};

template <int... Is>
struct build_indices<0, Is...> : indices<Is...> {};








template<class T,int N>
class JV {
public:
  enum { ThisSize = N };                 // Size in T's
  enum { Size_t = ThisSize * T::Size_t}; // Size in registers

  JV(const JV& a): JV( a.jit , a.r_addr , a.off_full , a.off_level ) {
    std::cout << "JV::JV() copy ctor " << __PRETTY_FUNCTION__ << " " << (void*)this << " " << (void*)&a.jit << "\n";
  }


  JV(Jit& j) : JV(j, build_indices<N>{}) {}
  template<int... Is>
  JV(Jit& j ,  indices<Is...>) : 
    jit(j),  F{{(void(Is),j)...}} {
    std::cout << "JV::JV() new regs " << (void*)this << " " << (void*)&j << "\n";
  }


  JV(Jit& j, int r , int of , int ol): JV(j,r,of,ol,build_indices<N>{}) {}
  template<int... Indices>
  JV(Jit& j, int r , int of , int ol, indices<Indices...>)
    : jit(j), r_addr(r), off_full(of), off_level(ol), F { { {j,r,of*N,ol+of*Indices}... } }
  {
    std::cout << "JV::JV() global view " << (void*)this << " " << (void*)&j << "\n";
  }

  Jit& func() const {
    std::cout << "JV::func() " << (void*)this << " returns=" << (void*)&jit << "\n";

    return jit;
  }

  const std::array<T,N>& getF() const { return F; }
  std::array<T,N>& getF() { return F; }
 
private:
  Jit& jit;
  int off_full;
  int off_level;
  int r_addr;
  std::array<T,N> F;
};




}

#endif
