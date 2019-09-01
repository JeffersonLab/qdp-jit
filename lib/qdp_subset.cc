/*! @file
 * @brief Sets and subsets
 */

#include "qdp.h"
#include "qdp_util.h"

namespace QDP 
{

  //---------------------------------------------------------------------
  //! Default all set
  Set set_all;

  //! Default all subset
  Subset all;

  //! Default rb3 subset -- Always unordered
  Set rb3;

  //! Default 2-checkerboard (red/black) set
  Set rb;

  //! Default 2^{Nd+1}-checkerboard set. Useful for pure gauge updating.
  Set mcb;

  //! Even subset
  Subset even;

  //! Odd subset
  Subset odd;


  void Set::signOffTables() {
    for( int i = 0 ; i < idSiteTable.size() ; ++i ) {
      if (idSiteTable[i] != -1) {
	QDP_get_global_cache().signoff( idSiteTable[i] );
      }
    }
    for( int i = 0 ; i < idMemberTable.size() ; ++i ) {
      if (idMemberTable[i] != -1) {
	QDP_get_global_cache().signoff( idMemberTable[i] );
      }
    }
  }
  
  Set::~Set() {
    signOffTables();
  }



  Subset::Subset() {
    id = -1;
  }




  Set::Set(): registered(false) {
  }



  //! Constructor from a function object
  Set::Set(const SetFunc& fn): registered(false) {
    make(fn);    
  }



  //! Function object used for constructing the all subset
  class SetAllFunc : public SetFunc
  {
  public:
    int operator() (const multi1d<int>& coordinate) const {return 0;}
    int numSubsets() const {return 1;}
  };

  
  //! Function object used for constructing red-black (2) checkerboard */
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

  //! Function object used for constructing red-black (2) checkerboard in 3d
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

  
  //! Function object used for constructing 32 checkerboard. */
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


  //! Initializer for sets
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
    all.make(set_all[0]);

    // COPY the rb[0] to the even subset
    even = rb[0];

    // COPY the rb[1] to the odd subset
    odd = rb[1];
  }

	  
  //-----------------------------------------------------------------------------
  //! Simple constructor called to produce a Subset from inside a Set
  void Subset::make(bool _rep, int _start, int _end, multi1d<int>* ind, int* ind_id, int cb, Set* _set, multi1d<bool>* _memb, int* memb_id , int _id )
  {
#ifdef GPU_DEBUG  
    QDP_debug("Subset::make(...) Will reserve device memory now...");
#endif    
    ordRep    = _rep;
    startSite = _start;
    endSite   = _end;
    sub_index = cb;
    sitetable = ind;
    set       = _set;
    membertable = _memb;
    id        = _id;     // masterset

    idSiteTable = ind_id;
    idMemberTable = memb_id;

  }


  //! Simple constructor called to produce a Subset from inside a Set
  void Subset::make(const Subset& s)
  {
    ordRep    = s.ordRep;
    startSite = s.startSite;
    endSite   = s.endSite;
    sub_index = s.sub_index;
    set       = s.set;
    membertable   = s.membertable;
    idMemberTable = s.idMemberTable;
    id            = s.id;
    sitetable     = s.sitetable;
    idSiteTable   = s.idSiteTable;
  }



  //! Simple constructor called to produce a Subset from inside a Set
  Subset& Subset::operator=(const Subset& s)
  {
    make(s);
    return *this;
  }


  Subset::Subset(const Subset& s)
  {
    make(s);
  }


  //-----------------------------------------------------------------------------
  // = operator
  Set& Set::operator=(const Set& s)
  {
    lat_color = s.lat_color;
    sitetables = s.sitetables;
    membertables = s.membertables;

    int nsubset_indices = s.numSubsets();

    signOffTables();
    idSiteTable.resize(nsubset_indices);
    idMemberTable.resize(nsubset_indices);

    sub.resize(nsubset_indices);

    for(int cb=0; cb < nsubset_indices; ++cb)
      {
	idSiteTable[cb]   = sitetables[cb].size()   > 0 ? QDP_get_global_cache().registrateOwnHostMem( sitetables[cb].size()   * sizeof(int)  , sitetables[cb].slice()   , NULL ) : -1 ;
	idMemberTable[cb] = membertables[cb].size() > 0 ? QDP_get_global_cache().registrateOwnHostMem( membertables[cb].size() * sizeof(bool) , membertables[cb].slice() , NULL ) : -1 ;

	sub[cb].make( s[cb].ordRep, s[cb].startSite, s[cb].endSite,
		      &sitetables[cb], &idSiteTable[cb],
		      cb, this,
		      &membertables[cb], &idMemberTable[cb] ,
		      s[cb].id );
      }
    
    return *this;
  }

  //-----------------------------------------------------------------------------

} // namespace QDP;
