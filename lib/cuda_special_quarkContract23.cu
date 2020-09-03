#include<vector>
#include<iostream>
#include<utility>
#include<algorithm>
#include<map>

#include <cuda_special.h>

namespace {

  __device__ int getGlobalIdx_2D_1D()
  {
    int blockId = blockIdx.y* gridDim.x+ blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
  }




  class cuComplex
  {
  public:
  
    __device__ cuComplex() {}
  
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}

    __device__ float real() const { return r; }

    __device__ float imag() const { return i; }

    __device__ cuComplex& operator+=(const cuComplex& a) {
      r = r + a.real();
      i = i + a.imag();
      return *this;
    }

    __device__ cuComplex& operator=(const cuComplex& a) {
      r = a.real();
      i = a.imag();
      return *this;
    }

  
  private:
    float   r;
    float   i;
  };

  __device__ cuComplex operator*( const cuComplex& a , const cuComplex& b )
  {
    return cuComplex(a.real()*b.real() - a.imag()*b.imag(), a.imag()*b.real() + a.real()*b.imag());
  }

  __device__ cuComplex operator+( const cuComplex& a , const cuComplex& b )
  {
    return cuComplex( a.real() + b.real() , a.imag() + b.imag() );
  }

  __device__ cuComplex operator-( const cuComplex& a , const cuComplex& b )
  {
    return cuComplex( a.real() - b.real() , a.imag() - b.imag() );
  }


  // slowest to fastest index:
  // complex,color,spin,lattice
  //
  __device__ int isc( int N , int si , int sj , int ci, int cj )
  {
    return N * ( ( ( ci * 3 + cj ) * 4 + si ) * 4 + sj );
  }


} // namespace

  
__device__ void quarkContractXX_23_l( int i , int j ,
				    cuComplex (&A)[4][4][3][3],
				    cuComplex (&B)[4][4][3][3],
				    cuComplex (&C)[4][4][3][3] )
{
  A[i][j][0][0]
    = B[i][0][1][1] * C[0][j][2][2]
    - B[i][0][1][2] * C[0][j][2][1]
    - B[i][0][2][1] * C[0][j][1][2]
    + B[i][0][2][2] * C[0][j][1][1];

  
  A[i][j][0][1]
    = B[i][0][2][1] * C[0][j][0][2] 
    - B[i][0][2][2] * C[0][j][0][1] 
    - B[i][0][0][1] * C[0][j][2][2] 
    + B[i][0][0][2] * C[0][j][2][1];

  A[i][j][0][2]
    = B[i][0][0][1] * C[0][j][1][2] 
    - B[i][0][0][2] * C[0][j][1][1] 
    - B[i][0][1][1] * C[0][j][0][2] 
    + B[i][0][1][2] * C[0][j][0][1];

  A[i][j][1][0]
    = B[i][0][1][2] * C[0][j][2][0] 
    - B[i][0][1][0] * C[0][j][2][2] 
    - B[i][0][2][2] * C[0][j][1][0] 
    + B[i][0][2][0] * C[0][j][1][2];

  A[i][j][1][1]
    = B[i][0][2][2] * C[0][j][0][0] 
    - B[i][0][2][0] * C[0][j][0][2] 
    - B[i][0][0][2] * C[0][j][2][0] 
    + B[i][0][0][0] * C[0][j][2][2];

  A[i][j][1][2]
    = B[i][0][0][2] * C[0][j][1][0] 
    - B[i][0][0][0] * C[0][j][1][2] 
    - B[i][0][1][2] * C[0][j][0][0] 
    + B[i][0][1][0] * C[0][j][0][2];

  A[i][j][2][0]
    = B[i][0][1][0] * C[0][j][2][1] 
    - B[i][0][1][1] * C[0][j][2][0] 
    - B[i][0][2][0] * C[0][j][1][1] 
    + B[i][0][2][1] * C[0][j][1][0];

  A[i][j][2][1]
    = B[i][0][2][0] * C[0][j][0][1] 
    - B[i][0][2][1] * C[0][j][0][0] 
    - B[i][0][0][0] * C[0][j][2][1] 
    + B[i][0][0][1] * C[0][j][2][0];

  A[i][j][2][2]
    = B[i][0][0][0] * C[0][j][1][1] 
    - B[i][0][0][1] * C[0][j][1][0] 
    - B[i][0][1][0] * C[0][j][0][1] 
    + B[i][0][1][1] * C[0][j][0][0];

}


__device__ void quarkContractXX_23_accum_l( int i , int j , int k ,
					    cuComplex (&A)[4][4][3][3],
					    cuComplex (&B)[4][4][3][3],
					    cuComplex (&C)[4][4][3][3] )
{
  A[i][j][0][0]
    += B[i][k][1][1] * C[k][j][2][2]
    - B[i][k][1][2] * C[k][j][2][1]
    - B[i][k][2][1] * C[k][j][1][2]
    + B[i][k][2][2] * C[k][j][1][1];

  
  A[i][j][0][1]
    += B[i][k][2][1] * C[k][j][0][2] 
    - B[i][k][2][2] * C[k][j][0][1] 
    - B[i][k][0][1] * C[k][j][2][2] 
    + B[i][k][0][2] * C[k][j][2][1];

  A[i][j][0][2]
    += B[i][k][0][1] * C[k][j][1][2] 
    - B[i][k][0][2] * C[k][j][1][1] 
    - B[i][k][1][1] * C[k][j][0][2] 
    + B[i][k][1][2] * C[k][j][0][1];

  A[i][j][1][0]
    += B[i][k][1][2] * C[k][j][2][0] 
    - B[i][k][1][0] * C[k][j][2][2] 
    - B[i][k][2][2] * C[k][j][1][0] 
    + B[i][k][2][0] * C[k][j][1][2];

  A[i][j][1][1]
    += B[i][k][2][2] * C[k][j][0][0] 
    - B[i][k][2][0] * C[k][j][0][2] 
    - B[i][k][0][2] * C[k][j][2][0] 
    + B[i][k][0][0] * C[k][j][2][2];

  A[i][j][1][2]
    += B[i][k][0][2] * C[k][j][1][0] 
    - B[i][k][0][0] * C[k][j][1][2] 
    - B[i][k][1][2] * C[k][j][0][0] 
    + B[i][k][1][0] * C[k][j][0][2];

  A[i][j][2][0]
    += B[i][k][1][0] * C[k][j][2][1] 
    - B[i][k][1][1] * C[k][j][2][0] 
    - B[i][k][2][0] * C[k][j][1][1] 
    + B[i][k][2][1] * C[k][j][1][0];

  A[i][j][2][1]
    += B[i][k][2][0] * C[k][j][0][1] 
    - B[i][k][2][1] * C[k][j][0][0] 
    - B[i][k][0][0] * C[k][j][2][1] 
    + B[i][k][0][1] * C[k][j][2][0];

  A[i][j][2][2]
    += B[i][k][0][0] * C[k][j][1][1] 
    - B[i][k][0][1] * C[k][j][1][0] 
    - B[i][k][1][0] * C[k][j][0][1] 
    + B[i][k][1][1] * C[k][j][0][0];

}



__global__ void quarkContract23_l( int N , float* A, float* B, float* C)
{
  cuComplex Al[4][4][3][3];
  cuComplex Bl[4][4][3][3];
  cuComplex Cl[4][4][3][3];

  int idx = getGlobalIdx_2D_1D();
  if (idx >= N)
    return;

  const int im = N * 9 * 16;

  for(int j=0; j < 4; ++j)
    for(int i=0; i < 4; ++i)
      for(int cj=0; cj < 3; ++cj)
	for(int ci=0; ci < 3; ++ci)
	  {
	    Bl[i][j][ci][cj] = cuComplex( B[ isc(N,i,j,ci,cj) + idx ] , B[ isc(N,i,j,ci,cj) + idx + im ] );
	    Cl[i][j][ci][cj] = cuComplex( C[ isc(N,i,j,ci,cj) + idx ] , C[ isc(N,i,j,ci,cj) + idx + im ] );
	  }
  


  for(int j=0; j < 4; ++j)
    for(int i=0; i < 4; ++i)
      {
	quarkContractXX_23_l( i , j , Al, Bl, Cl);

	for(int k=1; k < 4; ++k)
	  quarkContractXX_23_accum_l( i , j , k , Al , Bl , Cl );
      }


  for(int j=0; j < 4; ++j)
    for(int i=0; i < 4; ++i)
      for(int cj=0; cj < 3; ++cj)
	for(int ci=0; ci < 3; ++ci)
	  {
	    A[ isc(N,i,j,ci,cj) + idx ] = Al[i][j][ci][cj].real();
	    A[ isc(N,i,j,ci,cj) + idx + im ] = Al[i][j][ci][cj].imag();
	  }

  
}







void evaluate_special_quarkContract23( int N, std::vector<void*> args )
{
  //
  // Unique function identifier
  //
  const int func_num = 2;
  const int default_blocksize = 128;
  
  if (cuda_special_get_maxgridx() == -1)
    {
      std::cerr << "evaluate_special_test3, cuda_special_maxgridx not set\n";
      exit(1);
    }
  
  int threads_per_block = default_blocksize;
  if ( cuda_special_get_blocksize().count( func_num ) > 0 )
    threads_per_block = cuda_special_get_blocksize()[ func_num ];
  
  std::pair<int,int> size = getBlockDim( N , threads_per_block );
  dim3 grid(  size.first , size.second , 1 );
  dim3 block( threads_per_block , 1 , 1 );

  // std::cout << "launching : grid( " << size.first << " , " << size.second << " , 1)   ";
  // std::cout << "launching : block( " << threads_per_block << " , 1 , 1 )\n";

  quarkContract23_l<<< grid , block >>>( N , (float*)args[0] , (float*)args[1] , (float*)args[2] );
}
