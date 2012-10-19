#ifndef QDP_PETE_VIS_H
#define QDP_PETE_VIS_H


namespace QDP {


struct ShiftPhase1
{
};

struct ShiftPhase2
{
};



struct ViewLeaf
{
  int i1_m;
  inline ViewLeaf(int i1) : i1_m(i1) { }
  inline int val1() const { return i1_m; }
};
  



struct ParamLeaf
{
  Jit& func;  // Function we are building
  int r_idx;

  //mutable int cnt;      // parameter count
  //ParamLeaf(const Jit& func_, int cnt_) : func(func_), cnt(cnt_) { }

  ParamLeaf(Jit& func_,int r_idx) : func(func_),r_idx(r_idx) {}

  Jit& getFunc() const {return func;}
  int getRegIdx() const {return r_idx;}

  int getParamLattice( int wordSize ) const {
    return func.addParamLatticeBaseAddr( r_idx , wordSize );
  }
  int getParamScalar() const {
    return func.addParamScalarBaseAddr();
  }
  int getParamIndexField() const {
    return func.addParamIndexField();
  }
};



struct AddressLeaf
{
  ~AddressLeaf() {
    QDPCache::Instance().releasePrevLockSet();
    QDPCache::Instance().beginNewLockSet();
  }

  mutable std::vector<void*> addr;
  void setAddr(void* p) const {
    //std::cout << "AddressLeaf::setAddr " << p << "\n";
    addr.push_back(p);
  }
};

  
}

#endif
