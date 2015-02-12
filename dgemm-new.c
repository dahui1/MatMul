#include <emmintrin.h>

const char* dgemm_desc = "tutorial based dgemm.";

void add_dot(int lda, int stride, double* A, double* B, double* C)
{
  for (int k=0; k<lda;k++)
  {
    *C += A[k*stride] * B[k];
            /* Compute C(i,j) */
        // double cij = C[i+j*n];
        // cij += A[i+k*n] * B[k+j*n];
        // C[i+j*n] = cij;
  }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each column j of B */
  for (int j = 0; j < lda; ++j) // columns of C
  {
    /* For each row i of A */
    for (int i = 0; i < lda; i ++) // rows of C
    {
      //AddDot( k, &A( i,0 ), lda, &B( 0,j ), &C( i,j ) );
      add_dot(lda,lda,A+i,B+j*lda,C+i+j*lda);
    }
  }
}
