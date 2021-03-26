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





struct AddressLeaf
{
  ~AddressLeaf() {
    for (auto i : ids_signoff) {
      QDP_get_global_cache().signoff(i);
    }
  }
  
  AddressLeaf(const Subset& s): subset(s) {}
  AddressLeaf(const AddressLeaf& cp) = delete;

  AddressLeaf& operator=(const AddressLeaf& cp) = delete;

  
  mutable std::vector<QDPCache::ArgKey> ids;
  mutable std::vector<int> ids_signoff;
  const Subset& subset;

  void setId( int id ) const {
    ids.push_back( QDPCache::ArgKey(id) );
  }
  void setIdElem( int id , int elem ) const {
    ids.push_back( QDPCache::ArgKey(id,elem) );
  }
  
  template<class T> void setLit( T f ) const;

  
};

  template<>
  void AddressLeaf::setLit<float>( float f ) const;
  
  template<>
  void AddressLeaf::setLit<double>( double d ) const;
  
  template<>
  void AddressLeaf::setLit<int>( int i ) const;
  
  template<>
  void AddressLeaf::setLit<int64_t>( int64_t i ) const;
  
  template<>
  void AddressLeaf::setLit<bool>( bool b ) const;


  
  

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
