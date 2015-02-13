#include <emmintrin.h>

const char* dgemm_desc = "tutorial based dgemm.";

void add_dot_1x4(int, int, double*, double*, double*);

// calculate 1 value of C with one row of A and one column of B
void add_dot(int lda, int stride, double* A, double* B, double* C)
{
  for (int k=0; k<lda;k++)
  {
    *C += A[k*stride] * B[k];
  }
}

// compute 4 x 4 matrix of C
void add_dot_4x4(int lda, int stride, double* A, double* B, double* C)
{
  // first row
  //add_dot(lda,stride,A,B,C);
  for (int k=0; k<lda;k++)
  {
    double a_row0, a_row1, a_row2, a_row3;
    double c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33;
    double *bcol0, *bcol1, *bcol2, *bcol3;

    a_row0 = A[k*stride];
    a_row1 = A[1+k*stride];
    a_row2 = A[2+k*stride];
    a_row3 = A[3+k*stride];

    bcol0 = &B[k];
    bcol1 = &B[lda+k];
    bcol2 = &B[2*lda+k];
    bcol3 = &B[3*lda+k];

    c00 = 0.0;  c01 = 0.0; c02 = 0.0; c03 = 0.0;
    c10 = 0.0;  c11 = 0.0; c12 = 0.0; c13 = 0.0;
    c20 = 0.0;  c21 = 0.0; c22 = 0.0; c23 = 0.0;
    c30 = 0.0;  c31 = 0.0; c32 = 0.0; c33 = 0.0;
    // first row
    c00 += a_row0 * *bcol0;
    c01 += a_row0 * *bcol1;
    c02 += a_row0 * *bcol2;
    c03 += a_row0 * *bcol3;
    // second row
    c10 += a_row1 * *bcol0;
    c11 += a_row1 * *bcol1;
    c12 += a_row1 * *bcol2;
    c13 += a_row1 * *bcol3;  
    // third row
    c20 += a_row2 * *bcol0;
    c21 += a_row2 * *bcol1;
    c22 += a_row2 * *bcol2;
    c23 += a_row2 * *bcol3;
    // fourth row
    c30 += a_row3 * *bcol0++;
    c31 += a_row3 * *bcol1++;
    c32 += a_row3 * *bcol2++;
    c33 += a_row3 * *bcol3++;

    C[0] += c00;  C[0+lda] += c01; C[0+2*lda] += c02; C[0+3*lda] += c03;
    C[1] += c10;  C[1+lda] += c11; C[1+2*lda] += c12; C[1+3*lda] += c13;
    C[2] += c20;  C[2+lda] += c21; C[2+2*lda] += c22; C[2+3*lda] += c23;
    C[3] += c30;  C[3+lda] += c31; C[3+2*lda] += c23; C[3+3*lda] += c33;
  }

}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each column j of B */
  for (int j = 0; j < lda/4*4; j+=4) // columns of C
  {
    /* For each row i of A */
    for (int i = 0; i < lda/4*4; i+=4) // rows of C
    {
      add_dot_4x4(lda,lda,A+i,B+j*lda,C+i+j*lda);
    }
    for (int i=lda/4*4; i<lda; i++) 
    {
      add_dot_1x4(lda,lda,A+i,B+j*lda,C+i+j*lda);
    }
  }
  for (int j=lda/4*4; j<lda; j++)
  {
    for (int i=0;i<lda;i++)
    {
      add_dot(lda,lda,A+i,B+j*lda,C+i+j*lda);
    }
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
    c0 += A_row * *(bcol0);
    c1 += A_row * *(bcol1);
    c2 += A_row * *(bcol2);
    c3 += A_row * *(bcol3);

    A_row = A[(k+1)*stride];
    c0 += A_row * *(bcol0+1);
    c1 += A_row * *(bcol1+1);
    c2 += A_row * *(bcol2+1);
    c3 += A_row * *(bcol3+1);

    A_row = A[(k+2)*stride];
    c0 += A_row * *(bcol0+2);
    c1 += A_row * *(bcol1+2);
    c2 += A_row * *(bcol2+2);
    c3 += A_row * *(bcol3+2);

    A_row = A[(k+3)*stride];
    c0 += A_row * *(bcol0+3);
    c1 += A_row * *(bcol1+3);
    c2 += A_row * *(bcol2+3);
    c3 += A_row * *(bcol3+3);

    bcol0+=4;
    bcol1+=4;
    bcol2+=4;
    bcol3+=4;

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
