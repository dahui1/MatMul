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



// calculate 4 values of C, lying on a row
void add_dot_1x4(int lda, int stride, double* A, double* B, double* C) 
{
  double A_row;
  double c0, c1, c2, c3;
  double *bcol0, *bcol1, *bcol2, *bcol3;
  c0 = 0.0;
  c1 = 0.0;
  c2 = 0.0;
  c3 = 0.0;
  bcol0 = &B[0];
  bcol1 = &B[lda];
  bcol2 = &B[lda*2];
  bcol3 = &B[lda*3];
  for (int k=0; k<lda/4*4;k+=4)
  {
    A_row = A[k*stride];
    c0 += A_row * *bcol0++;
    c1 += A_row * *bcol1++;
    c2 += A_row * *bcol2++;
    c3 += A_row * *bcol3++;

    A_row = A[(k+1)*stride];
    c0 += A_row * *bcol0++;
    c1 += A_row * *bcol1++;
    c2 += A_row * *bcol2++;
    c3 += A_row * *bcol3++;

    A_row = A[(k+2)*stride];
    c0 += A_row * *bcol0++;
    c1 += A_row * *bcol1++;
    c2 += A_row * *bcol2++;
    c3 += A_row * *bcol3++;

    A_row = A[(k+3)*stride];
    c0 += A_row * *bcol0++;
    c1 += A_row * *bcol1++;
    c2 += A_row * *bcol2++;
    c3 += A_row * *bcol3++;

  }
  for (int k=lda/4*4;k<lda;k++)
  {
    A_row = A[k*stride];
    c0 += A_row * *bcol0++;
    c1 += A_row * *bcol1++;
    c2 += A_row * *bcol2++;
    c3 += A_row * *bcol3++;
  }
  C[0] += c0;
  C[lda] += c1;
  C[2*lda] += c2;
  C[3*lda] += c3;
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
      add_dot_1x4(lda,lda,A+i,B+j*lda,C+i+j*lda);
    }
  }
}
