#include "qdp.h"
#include "qdp_util.h"

namespace QDP {

  FnMap::FnMap(const Map& m): map(m)
  {
  }
  
  FnMap::FnMap(const FnMap& f) : map(f.map) , pRsrc(f.pRsrc)
  {
  }

  void FnMap::init_wrapper(int layout_ref)
  {
    pRsrc = make_shared<RsrcWrapper>( map.get_destnode(layout_ref) , map.get_srcenode(layout_ref) );
  }


ArrayBiDirectionalMap  shift;

struct NearestNeighborMapFunc : public ArrayMapFunc
{
  NearestNeighborMapFunc() {}

  virtual ~NearestNeighborMapFunc() {} 

  virtual multi1d<int> operator() (const multi1d<int>& coord, int sign, int dir) const
    {
      multi1d<int> lc = coord;

      const multi1d<int>& nrow = Layout::lattSize();
      lc[dir] = (coord[dir] + sgnum(sign) + 4*nrow[dir]) % nrow[dir];

      return lc;
    }

  virtual int numArray() const {return Nd;}

private:
  int sgnum(int x) const {return (x > 0) ? 1 : -1;}
}; 


void initDefaultMaps()
{
  NearestNeighborMapFunc bbb;

  shift.make(bbb);
}


// This class is is used for binding the direction index of an ArrayMapFunc
// so as to construct a MapFunc
struct PackageArrayMapFunc : public MapFunc
{
  PackageArrayMapFunc(const ArrayMapFunc& mm, int dd): pmap(mm), dir(dd) {}

  // Virtual Destructor - no cleanup needed
  virtual ~PackageArrayMapFunc() {}
  virtual multi1d<int> operator() (const multi1d<int>& coord, int isign) const
    {
      return pmap(coord, isign, dir);
    }

private:
  const ArrayMapFunc& pmap;
  const int dir;
}; 


void ArrayMap::make(const ArrayMapFunc& func)
{
  // We are allowed to declare a mapsa, but not allocate one.
  // There is an empty constructor for Map. Hence, the resize will
  // actually allocate the space.
  mapsa.resize(func.numArray());

  // Loop over each direction making the Map
  for(int dir=0; dir < func.numArray(); ++dir)
  {
    PackageArrayMapFunc  my_local_map(func,dir);

    mapsa[dir].make(my_local_map);
  }
}


struct PackageBiDirectionalMapFunc : public MapFunc
{
  PackageBiDirectionalMapFunc(const MapFunc& mm, int mmult): pmap(mm), mult(mmult) {}

  virtual ~PackageBiDirectionalMapFunc() {}
  virtual multi1d<int> operator() (const multi1d<int>& coord, int isign) const
    {
      return pmap(coord, mult*isign);
    }

private:
  const MapFunc& pmap;
  const int mult;
}; 


void BiDirectionalMap::make(const MapFunc& func)
{
  bimaps.resize(2);

  PackageBiDirectionalMapFunc  my_neg_map(func,-1);
  bimaps[0].make(my_neg_map);

  PackageBiDirectionalMapFunc  my_pos_map(func,+1);
  bimaps[1].make(my_pos_map);
}


struct PackageArrayBiDirectionalMapFunc : public MapFunc
{
  PackageArrayBiDirectionalMapFunc(const ArrayMapFunc& mm, int mmult, int dd) : 
    pmap(mm), mult(mmult), dir(dd) {}

  virtual multi1d<int> operator() (const multi1d<int>& coord, int isign) const
    {
      return pmap(coord, mult*isign, dir);
    }

private:
  const ArrayMapFunc& pmap;
  const int mult;
  const int dir;
}; 


void ArrayBiDirectionalMap::make(const ArrayMapFunc& func)
{
  if (bimapsa.size2() != 2 || bimapsa.size1() != func.numArray() )
    bimapsa.resize(2,func.numArray());

  for(int dir=0; dir < func.numArray(); ++dir)
  {
    PackageArrayBiDirectionalMapFunc  my_neg_map(func,-1,dir);
    bimapsa(0,dir).make(my_neg_map);

    PackageArrayBiDirectionalMapFunc  my_pos_map(func,+1,dir);
    bimapsa(1,dir).make(my_pos_map);
  }
}



} // namespace QDP;
