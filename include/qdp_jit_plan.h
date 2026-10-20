#ifndef QDP_JIT_PLAN_H
#define QDP_JIT_PLAN_H

#include <variant>

namespace QDP {

  struct FnMap;

  struct PlanEntry
  {
    enum class Kind {
      Lattice,
      Scalar,
      Literal,
      Gamma,
      OpParam,   // int parameter of a peek/poke operation
      Shift
    };

    typedef std::variant<float,double,int,int64_t,bool> LiteralValue;

    enum class ScalarMem { Ring, OwnHost, OwnHostUndef };

    Kind kind;

    Shape            shape;
    JitDataFormat data_format;
    int           layout_slot = -1;
    int           gamma = 0;
    int           offnode = 0;

    int           layout_ref = -1;
    int           cache_id = -1;
    const void*   host_ptr = nullptr;
    size_t        bytes = 0;
    ScalarMem     mem = ScalarMem::Ring;               // Scalar
    bool          written = false;
    LiteralValue  literal;
    int           op_param = 0;
    FnMap*        fnmap = nullptr;
  };


  struct ExprPlan
  {
    std::vector<PlanEntry> entries;
    std::vector<int>       layout_of_slot;

    bool dest_overlap = false;

    int slot_of_layout( int layout_ref )
    {
      for ( int slot = 0 ; slot < (int)layout_of_slot.size() ; ++slot )
	if ( layout_of_slot[slot] == layout_ref )
	  return slot;
      layout_of_slot.push_back( layout_ref );
      return layout_of_slot.size() - 1;
    }

    DynKey key() const
    {
      DynKey k;
      k.reserve( 6 * entries.size() + 1 );
      for ( size_t i = 1 ; i < entries.size() ; ++i )
	add_to_key( k , entries[i] );
      if ( !entries.empty() )
	add_to_key( k , entries[0] );
      k.add( dest_overlap ? 1 : 0 );
      return k;
    }

    std::vector<int> structure() const
    {
      std::vector<int> s;
      s.reserve( 5 * entries.size() );
      for ( const PlanEntry& e : entries )
	{
	  s.push_back( (int)e.kind );
	  switch (e.kind)
	    {
	    case PlanEntry::Kind::Lattice:
	      s.push_back( e.shape.ns() );
	      s.push_back( e.shape.nc() );
	      s.push_back( (int)e.data_format.order );
	      s.push_back( (int)e.data_format.storage );
	      s.push_back( e.layout_slot );
	      break;
	    case PlanEntry::Kind::Scalar:
	    case PlanEntry::Kind::Literal:
	      s.push_back( e.shape.ns() );
	      s.push_back( e.shape.nc() );
	      break;
	    case PlanEntry::Kind::Gamma:
	      s.push_back( e.gamma );
	      break;
	    case PlanEntry::Kind::OpParam:
	      break;
	    case PlanEntry::Kind::Shift:
	      s.push_back( e.offnode );
	      break;
	    }
	}
      s.push_back( dest_overlap ? 1 : 0 );
      return s;
    }

    int num_leaf_params() const
    {
      int n = 0;
      std::vector<bool> slot_seen( layout_of_slot.size() , false );
      for ( const PlanEntry& e : entries )
	{
	  switch (e.kind)
	    {
	    case PlanEntry::Kind::Lattice:
	      n += 1;
	      if ( e.layout_slot != 0 && !slot_seen[e.layout_slot] )
		n += 2;
	      slot_seen[e.layout_slot] = true;
	      break;
	    case PlanEntry::Kind::Scalar:
	    case PlanEntry::Kind::Literal:
	    case PlanEntry::Kind::OpParam:
	      n += 1;
	      break;
	    case PlanEntry::Kind::Gamma:
	      break;
	    case PlanEntry::Kind::Shift:
	      n += 2;
	      break;
	    }
	}
      return n;
    }

    int num_params() const
    {
      return 3 + num_leaf_params();
    }

  private:
    static void add_to_key( DynKey& k , const PlanEntry& e )
    {
      switch (e.kind)
	{
	case PlanEntry::Kind::Lattice:
	  k.add( e.shape );
	  k.add( e.data_format );
	  k.add( e.layout_slot );
	  break;
	case PlanEntry::Kind::Scalar:
	case PlanEntry::Kind::Literal:
	  k.add( e.shape );
	  break;
	case PlanEntry::Kind::Gamma:
	  k.add( e.gamma );
	  break;
	case PlanEntry::Kind::OpParam:
	  break;
	case PlanEntry::Kind::Shift:
	  k.add( e.offnode );
	  break;
	}
    }
  };


  //
  class IterationSpace
  {
    ParamRef     p_th_count;
    ParamRef     p_latsize_dest;
    ParamRef     p_sitetable_dest = -1;
    llvm::Value* idx_thread = nullptr;
    llvm::Value* idx_dest = nullptr;

  public:
    IterationSpace( bool with_sitetable = true )
    {
      p_th_count       = llvm_add_param<int> ();
      p_latsize_dest   = llvm_add_param<int> ();
      if ( with_sitetable )
	p_sitetable_dest = llvm_add_param<int*>();
    }

    ParamRef latsize_dest() const { return p_latsize_dest; }

    llvm::Value* get_idx_thread()
    {
      if ( !idx_thread )
	{
	  idx_thread = llvm_thread_idx();
	  llvm_cond_exit( llvm_ge( idx_thread , llvm_derefParam( p_th_count ) ) );
	}
      return idx_thread;
    }

    llvm::Value* get_idx_dest()
    {
      assert( p_sitetable_dest >= 0 );
      if ( !idx_dest )
	idx_dest = llvm_array_type_indirection<int>( p_sitetable_dest , get_idx_thread() );
      return idx_dest;
    }
  };


  //
  //
  class MGParamLeaf
  {
    struct dummy
    {
      ParamRef     p_latsize;
      ParamRef     p_sitetable;
      llvm::Value* idx;
      bool         idx_set;
    };
    std::map< int , dummy > map_;

    const ExprPlan& plan;
    size_t       cursor = 0;
    ParamRef     dest_base_ = -1;

    ParamRef     latsize_slot0;

  public:

    MGParamLeaf( const ExprPlan& plan_ , ParamRef latsize_slot0_ ):
      plan(plan_), latsize_slot0(latsize_slot0_) {}

    ParamRef dest_base() const { return dest_base_; }

    template<class W>
    ParamRef lattice_base()
    {
      ParamRef r = llvm_add_param< W* >();
      if ( cursor == 1 )
	dest_base_ = r;
      return r;
    }

    const PlanEntry& next( PlanEntry::Kind kind )
    {
      assert( cursor < plan.entries.size() && plan.entries[cursor].kind == kind );
      return plan.entries[cursor++];
    }

    bool all_entries_used() const { return cursor == plan.entries.size(); }


    
    ParamRef getLatSizeParamRef( int slot )
    {
      if ( slot == 0 )
	return latsize_slot0;
      
      if ( map_.find( slot ) == map_.end() )
	{
	  map_[ slot ].p_latsize   = llvm_add_param<int> ();
	  map_[ slot ].p_sitetable = llvm_add_param<int*>();
	  map_[ slot ].idx_set     = false;
	}
      return map_.at( slot ).p_latsize;
    }



    
    llvm::Value* getIdx( int slot , llvm::Value* r_idx_dest_ )
    {
      if ( slot == 0 )
	{
	  return r_idx_dest_;
	}
      
      if ( map_.find( slot ) == map_.end() )
	{
	  QDPIO::cerr << "MGParamLeaf::getIdx, layout slot not found " << slot << endl;
	  raise(SIGSEGV);
	  QDP_abort(1);
	}

      if ( ! map_.at( slot ).idx_set )
	{
	  map_.at( slot ).idx = llvm_array_type_indirection<int>( map_.at( slot ).p_sitetable , r_idx_dest_ );
	  map_.at( slot ).idx_set = true;
	}
      
      return map_.at( slot ).idx;
    }

  };


  class MGViewLeaf
  {
    llvm::Value* idx_dest;
    MGParamLeaf& p;
  public:
    MGViewLeaf( llvm::Value * idx_dest_ , MGParamLeaf& p_ ) : idx_dest(idx_dest_), p(p_) {}
    MGParamLeaf& param_leaf() const { return p; }
    llvm::Value*    getIndex( int slot ) const
    {
      return p.getIdx( slot , idx_dest );
    }    
  };


  template<class LeafType>
  struct AddOpParam<LeafType,MGParamLeaf>
  { 
    typedef LeafType Type_t;
    static LeafType apply(const LeafType&, const MGParamLeaf& p) { return LeafType(); }
  };

  class JitFunctionCache
  {
    std::map< DynKey , JitFunction > functions;

  public:
    template<class Build>
    JitFunction& get( const ExprPlan& plan , const Build& build )
    {
      JitFunction& function = functions[ plan.key() ];
      if ( function.empty() )
	{
	  build( function );
	  function.set_plan_structure( plan.structure() );
	}
      assert( function.get_plan_structure() == plan.structure() );
      return function;
    }

    template<class Structure, class Build>
    JitFunction& get( const DynKey& key , const Structure& structure , const Build& build )
    {
      JitFunction& function = functions[ key ];
      if ( function.empty() )
	{
	  build( function );
	  function.set_plan_structure( structure() );
	}
      assert( function.get_plan_structure() == structure() );
      return function;
    }
  };


}

#endif
