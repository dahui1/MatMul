#include <emmintrin.h>

const char* dgemm_desc = "tutorial based dgemm.";

// calculate 1 value of C with one row of A and one column of B
void add_dot(int lda, int stride, double* A, double* B, double* C)
{
  for (int k=0; k<lda;k++)
  {
    *C += A[k*stride] * B[k];
  }
}

/*
  AddDot( k, &A( 0, 0 ), lda, &B( 0, 0 ), &C( 0, 0 ) );
  AddDot( k, &A( 0, 0 ), lda, &B( 0, 1 ), &C( 0, 1 ) );
  AddDot( k, &A( 0, 0 ), lda, &B( 0, 2 ), &C( 0, 2 ) );
  AddDot( k, &A( 0, 0 ), lda, &B( 0, 3 ), &C( 0, 3 ) );
  */


// calculate 4 values of C, lying on a row
void add_dot_1x4(int lda, int stride, double* A, double* B, double* C) 
{
  //add_dot(lda,stride,A,B,C);
  for (int k=0; k<lda;k++)
  {
    C[0] += A[k*stride] * B[k];
  }
  //add_dot(lda,stride,A,B+lda,C+lda);
  for (int k=0; k<lda;k++)
  {
    C[lda] += A[k*stride] * B[lda+k];
  }
  //add_dot(lda,stride,A,B+2*lda,C+2*lda);
  for (int k=0; k<lda;k++)
  {
    C[2*lda] += A[k*stride] * B[k+2*lda];
  }
  //add_dot(lda,stride,A,B+3*lda,C+3*lda;
  for (int k=0; k<lda;k++)
  {
    C[3*lda] += A[k*stride] * B[k+3*lda];
  }
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
      //add_dot(lda,lda,A+i,B+j*lda,C+i+j*lda);
      add_dot_1x4(lda,lda,A+i,B+j*lda,C+i+j*lda);
    }
  }
}
