#ifndef QDP_PETE_VISITORS_H
#define QDP_PETE_VISITORS_H


namespace QDP {

  struct ShiftPhase1
  {
    ShiftPhase1(const Subset& _s, int ref):subset(_s), layout_ref(ref) {}
    const Subset& subset;
    int layout_ref;
    bool in_shift = false;
  };

  struct ShiftPhase2
  {
  };

  
  struct EvalVar {};

  struct DummyJitLeaf {};

  
  struct SelfAssignTag
  {
    SelfAssignTag( int id_ ): id(id_) {}
    int id;
    int shift_depth = 0;
  };

  template<class T>
  struct LeafFunctor< T , SelfAssignTag >
  {
    typedef int Type_t;
    inline static
    Type_t apply(const T &s, SelfAssignTag &) 
    {
      return 0;
    }
  };

  
  struct CollectLayouts
  {
    CollectLayouts( std::vector<int>& _layouts ): layouts(_layouts) {}
    std::vector<int>& layouts;
  };


  template<class T>
  struct LeafFunctor< T , CollectLayouts >
  {
    typedef int Type_t;
    inline static
    Type_t apply(const T &s, CollectLayouts &) 
    {
      return 0;
    }
  };


  template<class LeafType, class LeafTag>
  struct AddOpParam {};
  
  
}

#endif
