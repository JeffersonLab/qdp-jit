#ifndef QDP_DATALAYOUT_H
#define QDP_DATALAYOUT_H


namespace QDP {

  struct IndexRet
  {
    ParamRef p_multi_index;  // if neg. -> recv. buffer, otherwise it's the local index
    ParamRef p_recv_buf;
    Shape shape;
    bool offnode = true;
  };


  class NestedIndex
  {
  public:
    class Extent
    {
      llvm::Value* val;
      ParamRef     p_ref;
      bool         isVal;
    public:
      Extent(llvm::Value* v):val(v),isVal(true){}
      Extent(ParamRef p):p_ref(p),isVal(false){}

      llvm::Value* get_val() const
      {
	if (!isVal)
	  return llvm_derefParam( p_ref );
	else
	  return val;
      }
    };

    struct Level
    {
      Extent       extent;
      llvm::Value* index;
    };

    void push_back( ParamRef extent , llvm::Value* index )
    {
      levels.push_back( Level{ Extent(extent) , index } );
    }

    void push_back( llvm::Value* extent , llvm::Value* index )
    {
      levels.push_back( Level{ Extent(extent) , index } );
    }

    void push_back( int extent , int index )
    {
      levels.push_back( Level{ Extent(llvm_create_value(extent)) , llvm_create_value(index) } );
    }

    size_t size() const { return levels.size(); }

    auto begin() { return levels.begin(); }
    auto end() { return levels.end(); }
    auto cbegin() const { return levels.cbegin(); }
    auto cend() const { return levels.cend(); }
    auto begin() const { return levels.begin(); }
    auto end() const { return levels.end(); }

  private:
    std::vector< Level > levels;
  };


  llvm::Value * datalayout( JitDataOrder lay , NestedIndex a );

} // QDP

#endif
