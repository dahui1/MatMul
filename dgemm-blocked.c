#include <emmintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 72
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double *restrict C)
{
  /*
   * copy optimization on this subblock?
   * try just for A?
  */


  /* For each column j of B */ 
  for (int j = 0; j < N; ++j) 
    {
    for (int k = 0; k < K; ++k)
    {
      __m128d bvec = _mm_load1_pd(&B[k+j*lda]);
    /* For each row i of A */
    for (int i = 0; i < M/8*8; i+=8)
      {
        /* Compute C(i,j) */
        __m128d cvec = _mm_loadu_pd(&C[i+j*lda]);
        __m128d avec = _mm_loadu_pd(&A[i+k*lda]);
        __m128d cvec1 = _mm_loadu_pd(&C[i+j*lda+2]);
        __m128d avec1 = _mm_loadu_pd(&A[i+k*lda+2]);
        __m128d cvec2 = _mm_loadu_pd(&C[i+j*lda+4]);
        __m128d avec2 = _mm_loadu_pd(&A[i+k*lda+4]);
        __m128d cvec3 = _mm_loadu_pd(&C[i+j*lda+6]);
        __m128d avec3 = _mm_loadu_pd(&A[i+k*lda+6]);

        cvec = _mm_add_pd(cvec, _mm_mul_pd(avec,bvec));
        cvec1 = _mm_add_pd(cvec1, _mm_mul_pd(avec1,bvec));
        cvec2 = _mm_add_pd(cvec2, _mm_mul_pd(avec2,bvec));
        cvec3 = _mm_add_pd(cvec3, _mm_mul_pd(avec3,bvec));

        _mm_storeu_pd(&C[i+j*lda],cvec);
        _mm_storeu_pd(&C[i+j*lda+2],cvec1);
        _mm_storeu_pd(&C[i+j*lda+4],cvec2);
        _mm_storeu_pd(&C[i+j*lda+6],cvec3);
      }
    for (int i=M/8*8; i< M; i ++) 
      {
        double cij = C[i+j*lda];
        cij += A[i+k*lda] * B[k+j*lda];
        C[i+j*lda] = cij;
      }
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double *restrict C)
{
  /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      {
      int N = min (BLOCK_SIZE, lda-j);
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
          int K = min (BLOCK_SIZE, lda-k);
        /* For each block-row of A */ 
        for (int i = 0; i < lda; i += BLOCK_SIZE)
          {
          /* Correct block dimensions if block "goes off edge of" the matrix */
          int M = min (BLOCK_SIZE, lda-i);
          /* Perform individual block dgemm */
          do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }
}
