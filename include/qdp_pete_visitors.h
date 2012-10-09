#ifndef QDP_PETE_VIS_H
#define QDP_PETE_VIS_H


namespace QDP {


struct ViewLeaf
{
  int i1_m;
  inline ViewLeaf(int i1) : i1_m(i1) { }
  inline int val1() const { return i1_m; }
};
  



struct ParamLeaf
{
  Jit& func;  // Function we are building

  //mutable int cnt;      // parameter count
  //ParamLeaf(const Jit& func_, int cnt_) : func(func_), cnt(cnt_) { }

  ParamLeaf(Jit& func_) : func(func_) {}

  Jit& getFunc() const {return func;}
  int getParamLattice( int wordSize ) const {
    return func.addParamLatticeBaseAddr( wordSize );
  }
  int getParamScalar() const {
    return func.addParamScalarBaseAddr();
  }
};



struct AddressLeaf
{
  mutable std::vector<void*> addr;
  void setAddr(void* p) const {
    addr.push_back(p);
  }
};

  
}

#endif
