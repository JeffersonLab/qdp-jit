// -*- C++ -*-

#ifndef QDP_MASTERSET_H
#define QDP_MASTERSET_H

namespace QDP {

  class MasterSet {
  public:
    static MasterSet& Instance();
    void registrate(Set& set);
    const Subset& getSubset(int id);
    int numSubsets() const;

  private:

    MasterSet() {
    }

    std::vector<const Subset*> vecSubset;
  };

} // namespace QDP

#endif
