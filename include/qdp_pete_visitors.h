#ifndef QDP_PETE_VIS_H
#define QDP_PETE_VIS_H


namespace QDP {







struct ShiftPhase1
{
};

struct ShiftPhase2
{
};



class ViewLeaf
{
private:
  JitDeviceLayout::LayoutEnum layout_m;
  IndexDomainVector index_vec_m;
public:
  ViewLeaf( JitDeviceLayout::LayoutEnum layout , IndexDomainVector index_vec ) : layout_m(layout), index_vec_m( index_vec ) { }

  JitDeviceLayout::LayoutEnum getLayout() const { return layout_m; }

  IndexDomainVector getIndexVec() const  { 
    return index_vec_m;
  }

  llvm::Value    *getIndex() const  { 
    return get_index_from_index_vector( index_vec_m );
  }
};
  



struct ParamLeaf {};


  // int getParamLattice( int idx_multiplier ) const {
  //   return func.addParamLatticeBaseAddr( r_idx , idx_multiplier );
  // }
  // int getParamScalar() const {
  //   return func.addParamScalarBaseAddr();
  // }
  // int getParamIndexFieldAndOption() const {
  //   return r_idx = func.addParamIndexFieldAndOption();
  // }




struct AddressLeaf
{
  union Types {
    void * ptr;
    float  fl;
    int    in;
    int64_t in64;
    double db;
    bool   bl;
  };

  AddressLeaf() {    
    //std::cout << "AddressLeaf default ctor\n";
  }

  AddressLeaf(const AddressLeaf& cp) {
    addr = cp.addr;
    //std::cout << "AddressLeaf copy ctor my_size = " << addr.size() << "\n";
  }

  AddressLeaf& operator=(const AddressLeaf& cp) {
    addr = cp.addr;
    //std::cout << "AddressLeaf assignment my_size = " << addr.size() << "\n";
  }

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
  void setLit( int64_t i ) const {
    //std::cout << "AddressLeaf::setLit int64_t " << i << "\n";
    Types t;
    t.in64 = i;
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
