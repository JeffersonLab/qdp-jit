#ifndef QDP_JIT_FUNCTION_H
#define QDP_JIT_FUNCTION_H


namespace QDP {

  class JitFunction;

  void gpu_set_record_stats();
  bool gpu_get_record_stats();
  std::vector<JitFunction*>& gpu_get_functions();
  
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
    void set_spill_store( int s ) { spill_store = s; }
    void set_spill_loads( int s ) { spill_loads = s; }
    void set_regs( int s ) { regs = s; }
    void set_cmem( int s ) { cmem = s; }
    void set_kernel_name( const std::string& s ) { kernel_name = s; }
    void set_pretty( const std::string& s ) { pretty = s; }
    void add_timing( float t ) { timings.push_back(t); }
    
    int get_stack() { return stack; }
    int get_spill_store() { return spill_store; }
    int get_spill_loads() { return spill_loads; }
    int get_regs() { return regs; }
    int get_cmem() { return cmem; }
    std::string get_kernel_name() { return kernel_name; }
    std::string get_pretty() { return pretty; }
    const std::vector<float>& get_timings() const { return timings; }
    
  private:
    void check_empty();
    
    Func_t function;
    bool isEmpty;
    int max_wg;
    int called;
    
    int stack;
    int spill_store;
    int spill_loads;
    int regs;
    int cmem;
    
    std::string kernel_name;
    std::string pretty;

    std::vector<float> timings;
  };


  
} // namespace


#endif


