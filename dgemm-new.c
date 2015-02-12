#include <emmintrin.h>

const char* dgemm_desc = "tutorial based dgemm.";

void add_dot(int lda, int stride, double* A, double* B, double* C)
{
  for (int k=0; k<lda;k++)
  {
    *C += A[k*stride] * B[k];
  }
}

void add_dot_1x4(int lda, double* A, double* B, double* C) 
{
  add_dot(lda,lda,A,B,C);
  add_dot(lda,lda,A,B+lda,C+lda);
  add_dot(lda,lda,A,B+2*lda,C+2*lda);
  add_dot(lda,lda,A,B+3*lda,C+2*lda);
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each column j of B */
  for (int j = 0; j < lda; j+=4) // columns of C
  {
    /* For each row i of A */
    for (int i = 0; i < lda; i ++) // rows of C
    {
      //AddDot( k, &A( i,0 ), lda, &B( 0,j ), &C( i,j ) );
      add_dot_1x4(lda, A+i, B+(j*lda),C+i+(j*lda));
    }
  }
}
