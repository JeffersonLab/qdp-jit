#ifndef QDP_MASTERSET_H
#define QDP_MASTERSET_H

namespace QDP {

  class MasterSet {
  public:
    static MasterSet& Instance();
    void add(Set& set);
    int numSubsets() const;

    void remade(const Set& set);

    int generation(int subset_id) const { return generations[subset_id]; }

  private:

    MasterSet() {
    }

    std::vector<int> generations;
  };

} // namespace QDP

#endif
