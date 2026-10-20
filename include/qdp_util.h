#ifndef QDP_UTIL_H
#define QDP_UTIL_H


namespace QDP {

multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size);
int local_site(const multi1d<int>& coord, const multi1d<int>& latt_size);
multi1d<int> uniquify_list(const multi1d<int>& ll);
void initDefaultSets();
void initDefaultMaps();

} // namespace QDP

#endif
