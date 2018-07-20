#ifndef QDP_PETE_VIS_H
#define QDP_PETE_VIS_H


namespace QDP {







struct ShiftPhase1
{
  ShiftPhase1(const Subset& _s):subset(_s) {}
  const Subset& subset;
};

struct ShiftPhase2
{
};



struct ViewLeaf
{
  JitDeviceLayout layout_m;
  llvm::Value * index_m;
  ViewLeaf( JitDeviceLayout layout , llvm::Value * index ) : layout_m(layout), index_m(index) { }
  JitDeviceLayout getLayout() const { return layout_m; }
  llvm::Value    *getIndex() const  { return index_m; }
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
  ~AddressLeaf() {
    //QDPIO::cout << "signing off: ";
    for (auto i : ids_signoff) {
      //QDPIO::cout << i << ", ";
      QDP_get_global_cache().signoff(i);
    }
    //QDPIO::cout << "\n";
  }
  
  AddressLeaf(const Subset& s): subset(s) {    
    //std::cout << "AddressLeaf default ctor\n";
  }

#if 0
  AddressLeaf(const AddressLeaf& cp): subset(cp.subset) {
    addr = cp.addr;
    //std::cout << "AddressLeaf copy ctor my_size = " << addr.size() << "\n";
  }
#endif

  AddressLeaf& operator=(const AddressLeaf& cp) = delete;
  AddressLeaf(const AddressLeaf& cp) = delete;

  
  mutable std::vector<int> ids;
  mutable std::vector<int> ids_signoff;
  const Subset& subset;

  void setId( int id ) const {
    ids.push_back( id );
  }
  void setLit( float f ) const {
    ids.push_back( QDP_get_global_cache().addJitParamFloat(f) );
    ids_signoff.push_back( ids.back() );
  }
  void setLit( double d ) const {
    ids.push_back( QDP_get_global_cache().addJitParamDouble(d) );
    ids_signoff.push_back( ids.back() );
  }
  void setLit( int i ) const {
    ids.push_back( QDP_get_global_cache().addJitParamInt(i) );
    ids_signoff.push_back( ids.back() );
  }
  void setLit( int64_t i ) const {
    ids.push_back( QDP_get_global_cache().addJitParamInt64(i) );
    ids_signoff.push_back( ids.back() );
  }
  void setLit( bool b ) const {
    ids.push_back( QDP_get_global_cache().addJitParamBool(b) );
    ids_signoff.push_back( ids.back() );
  }
};


  template<class LeafType, class LeafTag>
  struct AddOpParam
  { };

  template<class LeafType>
  struct AddOpParam<LeafType,ParamLeaf>
  { 
    typedef LeafType Type_t;
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
