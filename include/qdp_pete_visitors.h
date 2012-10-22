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
  Jit::LatticeLayout layout_m;
  inline ViewLeaf(Jit::LatticeLayout l,int i1) : i1_m(i1), layout_m(l) { }
  inline int val1() const { return i1_m; }
  inline Jit::LatticeLayout layout() const { return layout_m; }
};
  



struct ParamLeaf
{
  Jit& func;  // Function we are building
  int r_idx;
  Jit::LatticeLayout layout;

  //mutable int cnt;      // parameter count
  //ParamLeaf(const Jit& func_, int cnt_) : func(func_), cnt(cnt_) { }

  ParamLeaf(Jit& func_,int r_idx,Jit::LatticeLayout lay) : func(func_),r_idx(r_idx),layout(lay) {}

  bool isCoal() const {
    return layout == Jit::LatticeLayout::COAL; 
  }

  Jit& getFunc() const {return func;}
  int getRegIdx() const {return r_idx;}

  int getParamLattice( int idx_multiplier ) const {
    return func.addParamLatticeBaseAddr( r_idx , idx_multiplier );
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
