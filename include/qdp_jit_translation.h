#ifndef QDP_JIT_TRANSLATION_H
#define QDP_JIT_TRANSLATION_H

namespace QDP {

  int get_translation_sitetable_id( int dest , int from );
  int get_lexico_sitetable_id( int layout );

  void check_shift_layout( int shifted , int expression );

  void destroy_translation_sitetable();

  Set& get_blocking_set( const multi1d<int>& blocking );

  void get_blocksum_sitetable_id( int& dest_table , int& rhs_table, int layout , const multi1d<int>& blocking );

}
#endif
