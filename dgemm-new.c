const char* dgemm_desc = "Simple new dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 72
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  for (int j = 0; j < N; ++j)
  {
    int step1 = 4;
    for (int k = 0; k < K; k += step1) 
    {
      step1 = min ( K - k, step1 );
      double bkj = 0, bk1j = 0, bk2j = 0, bk3j = 0;
      switch (step1) {
        case 4:
          bk3j = B[k + 3 + j * lda];
        case 3:
          bk2j = B[k + 2 + j * lda];
        case 2:
          bk1j = B[k + 1 + j * lda];
        case 1:
          bkj = B[k + j * lda];
          break;
      }

      for (int i = 0; i < M; ++i) {
      	C[i+j*lda] += A[i+k*lda] * bkj;
        C[i+j*lda] += A[i+(k + 1)*lda] * bk1j;
        C[i+j*lda] += A[i+(k + 2)*lda] * bk2j;
        C[i+j*lda] += A[i+(k + 3)*lda] * bk3j;
      }
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  for (int j = 0; j < lda; j += BLOCK_SIZE)
    for (int k = 0; k < lda; k += BLOCK_SIZE)
      for (int i = 0; i < lda; i += BLOCK_SIZE)
      {
      	int M = min (BLOCK_SIZE, lda-i);
	      int N = min (BLOCK_SIZE, lda-j);
	      int K = min (BLOCK_SIZE, lda-k);

	      /* Perform individual block dgemm */
	      do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}

