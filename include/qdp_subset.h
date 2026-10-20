#ifndef QDP_SUBSET_H
#define QDP_SUBSET_H


namespace QDP {

class SetFunc
{
public:
  virtual ~SetFunc() {}
  virtual int operator() (const multi1d<int>& coordinate) const = 0;
  virtual int numSubsets() const = 0;
};

class Set;

class Subset 
{
  friend class Set;
public:
  Subset();
  Subset(const Subset& s);
  void make(const Subset& s);
  int getIdSiteTable(LatticeLayout layout) const
  {
    return *table( layout.index() , "getIdSiteTable" ).idSiteTable;
  }
  int getIdSiteTable() const { return getIdSiteTable( Layout::current() ); }

  Subset& operator=(const Subset& s);
  int color() const {return sub_index;}

  multi1d<bool>& getIsElement(LatticeLayout layout) const
  {
    return *table( layout.index() , "getIsElement" ).membertable;
  }
  multi1d<bool>& getIsElement() const { return getIsElement( Layout::current() ); }

  bool isElement(LatticeLayout layout, int index) const
  {
    return (*table( layout.index() , "isElement" ).membertable)[index];
  }
  bool isElement(int index) const { return isElement( Layout::current() , index ); }

  void setId(int _id) { id = _id; }
  int getId() const { return id; }

  const multi1d<int>& siteTable(LatticeLayout layout) const
  {
    return *table( layout.index() , "siteTable" ).sitetable;
  }
  const multi1d<int>& siteTable() const { return siteTable( Layout::current() ); }

  int numSiteTable(LatticeLayout layout) const
  {
    return table( layout.index() , "numSiteTable" ).sitetable->size();
  }
  int numSiteTable() const { return numSiteTable( Layout::current() ); }

  const Set& getSet() const
  {
     return *set;
  }


  std::pair<int,int> getIndexPair() const;
  

protected:
  void make(int layout_ref, multi1d<int>* ind, int* ind_id, int cb, Set* set, multi1d<bool>* memb );

private:
  struct subset_t;

  const subset_t& table( int ref , const char* m ) const
  {
    auto it = layouts->find( ref );
    if ( it == layouts->end() )
      {
	QDPIO::cerr << "Subset " << m << ": no site tables for layout " << ref << std::endl;
	QDP_abort(1);
      }
    return it->second;
  }

  
  struct subset_t
  {
    multi1d<int>* sitetable;
    int* idSiteTable;
    multi1d<bool>* membertable;
  };

  std::shared_ptr< std::map< int , subset_t > > layouts;
  
  Set *set;
  int sub_index;

  int id;
};


class Set 
{
public:
  Set();
  Set(const SetFunc& fn);
  void make(const SetFunc& fn);
  const Subset& operator[](int subset_index) const {return sub[subset_index];}
        Subset& operator[](int subset_index)       {return sub[subset_index];}
  int numSubsets() const {return sub.size();}
  ~Set();
  void signOffTables();

  Set & operator=(const Set&) = delete;
  Set(const Set&) = delete;

  bool layout_done( int ref ) const
  {
    for(auto const& it: lat_color )
      {
	if ( it.first == ref )
	  return true;
      }
    return false;
  }
  
  int getIndex() const
  {
    assert(set_index >= 0);
    return set_index;
  }

  
protected:
  multi1d<Subset> sub;
  std::map< int , multi1d<int> > lat_color;
  std::map< int , multi1d<multi1d<int> > > sitetables;
  std::map< int , multi1d<multi1d<bool> > > membertables;
  std::map< int , multi1d<int> > idSiteTable;
  bool registered;
  static int set_counter;
  int set_index = -1;
  
public:
  const multi1d<int>& latticeColoring(LatticeLayout layout) const
  {
    if ( lat_color.count( layout.index() ) < 1 )
      {
	QDPIO::cout << "Set::latticeColoring() ref (" << layout.index() << ") not initialized." << std::endl;
	sleep(1);
	QDP_abort(1);
      }

    return lat_color.at(layout.index());
  }
};




extern Subset all;
extern Set rb3;
extern Set rb;
extern Set mcb;
extern Subset even;
extern Subset odd;


} // namespace QDP

#endif
