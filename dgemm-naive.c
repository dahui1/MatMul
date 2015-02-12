#include <emmintrin.h>

const char* dgemm_desc = "Naive, three-loop dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int n, double* A, double* B, double* C)
{
  /* For each column j of B */
  for (int j = 0; j < n; ++j) 
  {
    for( int k = 0; k < n; k++ )
    {
      __m128d bvec = _mm_load1_pd(&B[k+j*n]);
      /* For each row i of A */
      for (int i = 0; i < n/8*8; i += 8)
      {
        /* Compute C(i,j) */
        __m128d cvec = _mm_loadu_pd(&C[i+j*n]);
        __m128d avec = _mm_loadu_pd(&A[i+k*n]);
        cvec = _mm_add_pd(cvec, _mm_mul_pd(avec,bvec));
        _mm_storeu_pd(&C[i+j*n],cvec);
        cvec = _mm_loadu_pd(&C[i+j*n+2]);
        avec = _mm_loadu_pd(&A[i+k*n+2]);
        cvec = _mm_add_pd(cvec, _mm_mul_pd(avec,bvec));
        _mm_storeu_pd(&C[i+j*n+2],cvec);
        cvec = _mm_loadu_pd(&C[i+j*n+4]);
        avec = _mm_loadu_pd(&A[i+k*n+4]);
        cvec = _mm_add_pd(cvec, _mm_mul_pd(avec,bvec));
        _mm_storeu_pd(&C[i+j*n+4],cvec);
        cvec = _mm_loadu_pd(&C[i+j*n+6]);
        avec = _mm_loadu_pd(&A[i+k*n+6]);
        cvec = _mm_add_pd(cvec, _mm_mul_pd(avec,bvec));
        _mm_storeu_pd(&C[i+j*n+6],cvec);

      }
      for (int i = n/8*8; i < n; i ++) 
      {
        double cij = C[i+j*n];
        cij += A[i+k*n] * B[k+j*n];
        C[i+j*n] = cij;
      }
    }
  }
}
