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
  mutable int r_idx;
  Jit::LatticeLayout layout;

  //mutable int cnt;      // parameter count
  //ParamLeaf(const Jit& func_, int cnt_) : func(func_), cnt(cnt_) { }

  ParamLeaf(Jit& func_,int r_idx,Jit::LatticeLayout lay) : func(func_),r_idx(r_idx),layout(lay) {}

#if 1
  bool isCoal() const {
    return layout == Jit::LatticeLayout::COAL; 
  }
#endif

  Jit::LatticeLayout getLayout() const { return layout; }

  Jit& getFunc() const {return func;}
  int getRegIdx() const {return r_idx;}

  int getParamLattice( int idx_multiplier ) const {
    return func.addParamLatticeBaseAddr( r_idx , idx_multiplier );
  }
  int getParamScalar() const {
    return func.addParamScalarBaseAddr();
  }
  int getParamIndexFieldAndOption() const {
    return r_idx = func.addParamIndexFieldAndOption();
  }
};



struct AddressLeaf
{
  union Types {
    void * ptr;
    float  fl;
    int    in;
    double db;
    bool   bl;
  };

  mutable std::vector<Types> addr;
  void setAddr(void* p) const {
    //std::cout << "AddressLeaf::setAddr " << p << "\n";
    Types t;
    t.ptr = p;
    addr.push_back(t);
  }
  void setLit( float f ) const {
    //std::cout << "AddressLeaf::setLit float " << f << "\n";
    Types t;
    t.fl = f;
    addr.push_back(t);
  }
  void setLit( double d ) const {
    //std::cout << "AddressLeaf::setLit double " << d << "\n";
    Types t;
    t.db = d;
    addr.push_back(t);
  }
  void setLit( int i ) const {
    //std::cout << "AddressLeaf::setLit int " << i << "\n";
    Types t;
    t.in = i;
    addr.push_back(t);
  }
  void setLit( bool b ) const {
    //std::cout << "AddressLeaf::setLit bool " << b << "\n";
    Types t;
    t.bl = b;
    addr.push_back(t);
  }
};


  template<class LeafType, class LeafTag>
  struct AddOpParam
  { };

  template<class LeafType>
  struct AddOpParam<LeafType,ParamLeaf>
  { 
    static LeafType apply(const LeafType&, const ParamLeaf& p) { return LeafType(); }
  };

  template<class LeafType, class LeafTag>
  struct AddOpAddress
  { };

  template<class LeafType>
  struct AddOpAddress<LeafType,AddressLeaf>
  { 
    static void apply(const LeafType&, const AddressLeaf& p) {}
  };



  
}

#endif
