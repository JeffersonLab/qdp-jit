#ifndef QDP_MASTERMAP_H
#define QDP_MASTERMAP_H

namespace QDP {

  class MasterMap {
  public:
    static MasterMap& Instance();

    int  register_map(const Map& map);

    void unregister_map(int id);

    void drop_subset(int subset_id);

    int  num_tables() const { return tables.size(); }

    struct Tables
    {
      int        num_inner;
      int        id_inner;
      const int* inner;

      int        num_face;
      int        id_face;
      const int* face;
    };

    Tables get(const Subset& s, LatticeLayout layout, int mask);

  private:
    MasterMap() {}

    struct Storage
    {
      std::vector<int> inner;
      std::vector<int> face;

      int id_inner;
      int id_face;
    };

    void generate_tables(Storage& t, const Subset& subset, LatticeLayout layout, int mask) const;

    template<class Pred> void drop_tables(Pred pred);

    std::vector<const Map*> maps;

    std::map< std::tuple<int,int,int> , Storage > tables;
  };

} // namespace QDP

#endif
