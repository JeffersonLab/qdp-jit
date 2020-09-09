#ifndef cuda_special__
#define cuda_special__

void cuda_special_set_maxgridx( int maxx );
void cuda_special_set_function_blocksize(int func, int bsize );
int64_t cuda_special_get_maxgridx();
std::map<int,int>& cuda_special_get_blocksize();
std::pair<int,int> getBlockDim( int __num_sites , int threads_per_block );

//void evaluate_special_test( int th_count, std::vector<void*> args );
//void evaluate_special_test2( int th_count, std::vector<void*> args );
void evaluate_special_quarkContract13( int N, std::vector<void*> args );
void evaluate_special_quarkContract14( int N, std::vector<void*> args );
void evaluate_special_quarkContract23( int N, std::vector<void*> args );
void evaluate_special_quarkContract24( int N, std::vector<void*> args );

void jumper_jit_stats_special(int i);

#endif
