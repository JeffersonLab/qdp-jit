#ifndef QDP_JIT_FUNCTION_H
#define QDP_JIT_FUNCTION_H

#include <cassert>
#include<signal.h>

namespace QDP {


  void gpu_set_record_stats();
  bool gpu_get_record_stats();

  
  class DynKey
  {
    mutable std::vector<int> keys;
    mutable bool offnode_comms = false;
    
  public:
    void add( int i ) const
    {
      keys.push_back( i );
    }

    void set_offnode_comms( bool s ) const
    {
      offnode_comms = s;
    }

    bool get_offnode_comms() const
    {
      return offnode_comms;
    }

  
    bool operator <(const DynKey& rhs) const
    {
      if ( rhs.keys.size() != keys.size() )
	{
	  std::cout << " DynKey: some weird error" << std::endl;
	  QDP_abort(1);
	}

      for ( int i = 0 ; i < keys.size() ; ++i )
	{
	  if ( keys.at(i) > rhs.keys.at(i) )
	    {
	      return false;
	    }
	  if ( keys.at(i) < rhs.keys.at(i) )
	    {
	      return true;
	    }
	}
      return false;
    }

    friend std::ostream& operator<<(std::ostream &os, const DynKey& rhs);

    friend StandardOutputStream& operator<<(StandardOutputStream& s, const DynKey& rhs);
  };





  
  
  class JitFunction
  {
  public:
    JitFunction();
    
    JitFunction (const JitFunction&) = delete;
    JitFunction& operator= (const JitFunction&) = delete;

    
    typedef void* Func_t;
    bool empty() { return isEmpty; }
    
    void set_function( Func_t f ) { isEmpty = false; function = f; }
    Func_t get_function() {
      check_empty();
      return function;
    }

    void setMaxWG(int wg) { max_wg = wg; }
    int  getMaxWG() { return max_wg; }

    void inc_call_counter();
    int get_call_counter();
    
    void set_stack( int s ) { stack = s; }
    void set_spill_store( int s ) { spill_store = s; }   // ROCm: sgpr spills
    void set_vspill_store( int s ) { vspill_store = s; } // ROCm: vgpr spills 
    void set_spill_loads( int s ) { spill_loads = s; }
    void set_regs( int s ) { regs = s; }
    void set_vregs( int s ) { vregs = s; }
    void set_cmem( int s ) { cmem = s; }
    void set_group_segment( int s ) { group_segment = s; }
    void set_private_segment( int s ) { private_segment = s; }
    void set_kernel_name( const std::string& s ) { kernel_name = s; }
    void set_pretty( const std::string& s ) { pretty = s; }
    void add_timing( float t ) { timings.push_back(t); }
    
    int get_stack() { return stack; }
    int get_spill_store() { return spill_store; }
    int get_vspill_store() { return vspill_store; }
    int get_spill_loads() { return spill_loads; }
    int get_regs() { return regs; }
    int get_vregs() { return vregs; }
    int get_cmem() { return cmem; }
    int get_group_segment() { return group_segment; }
    int get_private_segment() { return private_segment; }
    std::string get_kernel_name() { return kernel_name; }
    std::string get_pretty() { return pretty; }
    const std::vector<float>& get_timings() const { return timings; }

    void set_dest_id( int i ) { dest_id = i; }
    int  get_dest_id() { return dest_id; }

    
    void set_enable_tuning() { enable_tuning = true; }
    bool get_enable_tuning() { return enable_tuning; }
    
    void set_threads_per_block( int i ) { threads_per_block = i; }
    int  get_threads_per_block() { return threads_per_block; }

#ifdef QDP_DEEP_LOG
    void set_is_lat( bool lat ) { isLat = lat ? 1 : 0; }
    bool get_is_lat()
    {
      if (isLat < 0)
	{
	  raise(SIGSEGV);
	}
      return isLat == 1;
    }
    // int start;
    // int count;   // number of elements of size T
    // int size_T;
    std::string type_W;
  private:
    int isLat = -1;
  public:
#endif

    double time_builder = 0.;
    double time_math = 0.;
    double time_passes = 0.;
    double time_codegen = 0.;
    double time_dynload = 0.;
#ifdef QDP_BACKEND_ROCM
    double time_linking = 0.;
#endif
    
  private:
    void check_empty();
    
    Func_t function;
    bool isEmpty;
    int max_wg;
    int called;
    
    int stack;
    int spill_store;
    int vspill_store;
    int spill_loads;
    int regs;
    int vregs;
    int cmem;
    int group_segment;
    int private_segment;
    
    std::string kernel_name;
    std::string pretty;

    std::vector<float> timings;

    // for tuning
    int dest_id  = -1;
    bool enable_tuning = false;
    
    // Best configuration resulting from tuning
    int  threads_per_block = -1;
    bool tuned = false;

  };


  std::vector<JitFunction*>& gpu_get_functions();

#if defined (QDP_CODEGEN_VECTOR)
  typedef std::map< DynKey , std::array<JitFunction,2> >  JitFunctionMap;
#else
  typedef std::map< DynKey , JitFunction >  JitFunctionMap;
#endif
  
} // namespace


#endif


