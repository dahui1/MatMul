const char* dgemm_desc = "Simple new dgemm.";

#include <immintrin.h>

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 72
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each column of B */
  for (int j = 0; j < N; ++j)
  {
    /* For each two columns of B */
    for (int i = 0; i < M; ++i) 
    {
      int step1 = 2;
      /* For each two rows of A */
      __m128d cinter = {0, 0};
      for (int k = 0; k < K; k += step1) {
        step1 = min (K - k, 2);
        __m128d aik = {0, 0};
        __m128d bkj = {0, 0};
        if (step1 == 2) {
          aik = _mm_loadu_pd(&A[k + i * lda]);
          bkj = _mm_loadu_pd(&B[k + j * lda]);
        } else {
          aik[0] = A[k + j * lda];
          bkj[0] = B[k + j * lda];
        }

        __m128d mul = _mm_mul_pd(aik, bkj);

        cinter = _mm_add_pd (mul, cinter);
      }
      C[i+j*lda] += cinter[0] + cinter[1];
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  int n = lda;

  double* newA = (double*) malloc (n * n * sizeof(double));

  /* Change A into Row Major matrix */
  for (int i = 0; i < n * n; ++i) {
    int row = i % n;
    int column = i / n;
    newA[row * n + column] = A[i];
  }

  A = newA;

  for (int j = 0; j < lda; j += BLOCK_SIZE)
    for (int k = 0; k < lda; k += BLOCK_SIZE)
      for (int i = 0; i < lda; i += BLOCK_SIZE)
      {
      	int M = min (BLOCK_SIZE, lda-i);
      	int N = min (BLOCK_SIZE, lda-j);
      	int K = min (BLOCK_SIZE, lda-k);

      	do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}

