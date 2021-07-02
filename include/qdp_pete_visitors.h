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
  ViewLeaf( JitDeviceLayout layout , llvm::Value * index ) : layout_m(layout), index_m(index) {}
  JitDeviceLayout getLayout() const { return layout_m; }
  llvm::Value    *getIndex() const  { return index_m; }
};


struct ViewSpinLeaf
{
  JitDeviceLayout layout_m;
  llvm::Value * index_m;
  mutable std::vector< JitForLoop > loops;
  ViewSpinLeaf( JitDeviceLayout layout , const std::vector< JitForLoop >& l, llvm::Value * index  ) : layout_m(layout), index_m(index), loops(l) {}
  ViewSpinLeaf( JitDeviceLayout layout , llvm::Value * index  ) : layout_m(layout), index_m(index) {}
  ViewSpinLeaf( const ViewSpinLeaf& rhs, JitForLoop l1 )                 : layout_m(rhs.layout_m), index_m(rhs.index_m), loops({l1}) {}
  ViewSpinLeaf( const ViewSpinLeaf& rhs, JitForLoop l1 , JitForLoop l2 ) : layout_m(rhs.layout_m), index_m(rhs.index_m), loops({l1,l2}) {}
  const std::vector< JitForLoop >& getLoops() const { return loops; }
  JitDeviceLayout getLayout() const { return layout_m; }
  llvm::Value    *getIndex() const  { return index_m; }
  JitForLoop index_first() const { return loops.at(0); }
  JitForLoop index_second() const { return loops.at(1); }
};



struct ParamLeaf {};

struct JIT2BASE {};


struct JitCreateLoopsLeaf
{
  mutable std::vector< JitForLoop > loops;
};


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

  
  mutable std::vector<int> ids;
  mutable std::vector<int> ids_signoff;
  const Subset& subset;

  void setId( int id ) const {
    ids.push_back( id );
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






  struct DynKeyTag
  {
    const DynKey& key;
  
    DynKeyTag(const DynKey& k): key(k) {}

    DynKeyTag(const DynKeyTag& cp) = delete;
    DynKeyTag& operator=(const DynKeyTag& cp) = delete;
  };


  template<class T>
  struct LeafFunctor< T , DynKeyTag >
  {
    typedef bool Type_t;
    inline static
    Type_t apply(const T &s, const DynKeyTag &) 
    {
      return false;
    }
  };



  
}

#endif
