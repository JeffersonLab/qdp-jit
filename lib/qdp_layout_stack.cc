#include "qdp.h"

#include<vector>

namespace QDP {

  namespace
  {
    template<class T>
    inline
    StandardOutputStream& operator<<(StandardOutputStream& s, const multi1d<T>& d)
    {
      for(int i=0; i < d.size(); ++i)
	s << d[i] << " ";
      return s;
    }
  }
  
namespace Layout
{
  
  namespace
  {
    std::vector< std::shared_ptr<LayoutData> > layout;
    std::stack< int > layout_stack;
  }

  void finalize()
  {
    layout.clear();
    layout_stack = std::stack< int >();
  }

  LatticeLayout current()
  {
    if ( layout_stack.size() == 0 )
      {
	std::cerr << "Layout::current called with an empty layout stack" << std::endl;
	QDP_abort(1);
      }
    return from_index( layout_stack.top() );
  }

  LayoutData& data( int index )
  {
    if ( index < 0 || index >= layout.size() )
      {
	QDPIO::cerr << "Layout::data called with invalid index " << index << endl;
	QDP_abort(1);
      }
    return *layout.at(index);
  }

  LatticeLayout from_index( int index )
  {
    data( index );
    return LatticeLayout( index );
  }

  
  void init()
  {
    layout.push_back( std::make_shared<LayoutData>() );

    int cur = layout.size() - 1;
    layout.back()->init(cur);

    layout_stack.push( cur );
  }

  
  void push( LatticeLayout l )
  {
    if ( l.index() < 0 || l.index() >= layout.size() )
      {
	QDPIO::cerr << "Layout::push called with invalid layout " << l.index() << endl;
	QDP_abort(1);
      }
    layout_stack.push( l.index() );
  }
  

  LatticeLayout find_or_create( const multi1d<int>& nrow )
  {
    for ( int i = 0 ; i < layout.size() ; ++i )
      if ( layout[i]->lattSize() == nrow )
	return from_index( i );

    layout.push_back( std::make_shared<LayoutData>() );
    int cur = layout.size() - 1;

    layout_stack.push( cur );

    layout.back()->init(cur);
    layout.back()->setLattSize( nrow );
    layout.back()->create();

    layout_stack.pop();

    return from_index( cur );
  }

  
  void pop()
  {
    if ( layout_stack.empty() )
      {
	QDPIO::cerr << "Layout::pop called with empty layout stack" << endl;
	QDP_abort(1);
      }

    layout_stack.pop();
  }


} // Layout

} // QDP
