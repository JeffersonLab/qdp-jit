#include<vector>
#include<iostream>
#include<utility>
#include<algorithm>
#include<map>

#include "qdp.h"

namespace {

  int64_t cuda_special_maxgridx = -1;
  std::map<int,int> cuda_special_blocksize;

} // namespace


std::pair<int,int> getBlockDim( int __num_sites , int threads_per_block )
{
  int64_t num_sites = __num_sites;
  
  int64_t M = cuda_special_maxgridx * threads_per_block;
  int64_t Nblock_y = (num_sites + M-1) / M;

  int64_t P = threads_per_block;
  int64_t Nblock_x = (num_sites + P-1) / P;
    
  return std::make_pair(Nblock_x,Nblock_y);
}


int64_t cuda_special_get_maxgridx()
{
  return cuda_special_maxgridx;
}

void jumper_jit_stats_special(int i)
{
  jit_stats_special(i);
}

std::map<int,int>& cuda_special_get_blocksize()
{
  return cuda_special_blocksize;
}



void cuda_special_set_function_blocksize(int func, int bsize )
{
  cuda_special_blocksize[func]=bsize;
  std::cout << "CUDA special function " << func << " block size set to " << bsize << "\n";
}


void cuda_special_set_maxgridx( int maxx )
{
  cuda_special_maxgridx = maxx;
}

