#include "qdp.h"
#include "qdp_util.h"

namespace QDP 
{
  Set set_all;
  Subset all;
  Set rb3;
  Set rb;
  Set mcb;
  Subset even;
  Subset odd;

  void Set::signOffTables()
  {
    for (auto& t : idSiteTable )
      {
	for( int i = 0 ; i < t.second.size() ; ++i )
	  {
	    if (t.second[i] != -1)
	      {
		QDP_get_global_cache().signoff( t.second[i] );
	      }
	  }
      }
  }

  
  Set::~Set() {
    for (int i = 0 ; i < sub.size() ; ++i)
      if (sub[i].getId() >= 0)
	MasterMap::Instance().drop_subset( sub[i].getId() );
    signOffTables();
  }


  int Set::set_counter;

  std::pair<int,int> Subset::getIndexPair() const
  {
    return { set->getIndex() , sub_index };
  }


  Subset::Subset() {
    id = -1;

    layouts = std::make_shared< std::map< int , subset_t > >();
  }


  Set::Set(): registered(false) {
  }


  Set::Set(const SetFunc& fn): registered(false) {
    make(fn);    
  }

  class SetAllFunc : public SetFunc
  {
  public:
    int operator() (const multi1d<int>& coordinate) const {return 0;}
    int numSubsets() const {return 1;}
  };

  class SetRBFunc : public SetFunc
  {
  public:
    int operator() (const multi1d<int>& coordinate) const
      {
	int sum = 0;
	for(int m=0; m < coordinate.size(); ++m)
	  sum += coordinate[m];

	return sum & 1;
      }

    int numSubsets() const {return 2;}
  };

  class SetRB3Func : public SetFunc
  {
  public:
    int operator() (const multi1d<int>& coordinate) const
      {
	if (coordinate.size() < 3) { 
	  QDPIO::cerr << "Need at least 3d for 3d checkerboarding" << endl;
	  QDP_abort(1);
	}
	int sum = 0;
	for(int m=0; m < 3; ++m)
	  sum += coordinate[m];

	return sum & 1;
      }

    int numSubsets() const {return 2;}
  };

  
  class Set32CBFunc : public SetFunc
  {
  public:
    int operator() (const multi1d<int>& coordinate) const
      {
	int initial_color = 0;
	for(int m=Nd-1; m >= 0; --m)
	  initial_color = (initial_color << 1) + (coordinate[m] & 1);

	int cb = 0;
	for(int m=0; m < Nd; ++m)
	  cb += coordinate[m] >> 1;

	cb &= 1;
	return initial_color + (cb << Nd);
      }

    int numSubsets() const {return 1 << (Nd+1);}
  };


  void initDefaultSets()
  {
    // Initialize the red/black checkerboard
    rb.make(SetRBFunc());

    // Initialize the 3d red/black checkerboard.
    rb3.make(SetRB3Func());

    // Initialize the 32-style checkerboard
    mcb.make(Set32CBFunc());

    // The all set
    set_all.make(SetAllFunc());

    // The all subset
    //all.make(set_all[0]);
    all = set_all[0];

    // COPY the rb[0] to the even subset
    even = rb[0];

    // COPY the rb[1] to the odd subset
    odd = rb[1];
  }

	  
  void Subset::make(int _layout_ref , multi1d<int>* ind, int* ind_id, int cb, Set* _set, multi1d<bool>* _memb )
  {
    (*layouts)[ _layout_ref ].sitetable = ind;
    (*layouts)[ _layout_ref ].idSiteTable = ind_id;
    (*layouts)[ _layout_ref ].membertable = _memb;

    sub_index  = cb;
    set        = _set;
  }

  void Subset::make(const Subset& s)
  {
    layouts    = s.layouts;
    set        = s.set;
    id         = s.id;
    sub_index  = s.sub_index;
  }

  Subset& Subset::operator=(const Subset& s)
  {
    make(s);
    return *this;
  }
  
  Subset::Subset(const Subset& s)
  {
    make(s);
  }

} // namespace QDP;
