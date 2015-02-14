#include <emmintrin.h>

#define block1 128
#define block2 256

#define min(a,b) (((a)<(b))?(a):(b))

const char* dgemm_desc = "final version for hw1";

void add_dot_1x4(int, int, double*, double*, double*);

// calculate 1 value of C with one row of A and one column of B
void add_dot(int lda, int stride, double* A, double* B, double* C)
{
  for (int k=0; k<stride;k++)
  {
    *C += A[k*lda] * B[k];
  }
}

// compute 4 x 4 matrix of C
void add_dot_4x4(int lda, int stride, double* A, double* B, double* C)
{
  __m128d a01, a23, b0, b1, b2, b3, c0, c1, c2, c3, c4, c5, c6, c7;
  double *bcol0, *bcol1, *bcol2, *bcol3;

  bcol0 = &B[0];
  bcol1 = &B[lda];
  bcol2 = &B[lda*2];
  bcol3 = &B[lda*3];

  c0 = _mm_setzero_pd(); c1 = _mm_setzero_pd();
  c2 = _mm_setzero_pd(); c3 = _mm_setzero_pd();
  c4 = _mm_setzero_pd(); c5 = _mm_setzero_pd();
  c6 = _mm_setzero_pd(); c7 = _mm_setzero_pd();

  for (int k=0; k<stride;k++)
  {

    a01 = _mm_loadu_pd(A);
    a23 = _mm_loadu_pd(A+2);
    A += 4;

    b0 = _mm_load1_pd(bcol0++);
    b1 = _mm_load1_pd(bcol1++);
    b2 = _mm_load1_pd(bcol2++);
    b3 = _mm_load1_pd(bcol3++);

    c0 = _mm_add_pd(c0, _mm_mul_pd(a01,b0));
    c1 = _mm_add_pd(c1, _mm_mul_pd(a01,b1));
    c2 = _mm_add_pd(c2, _mm_mul_pd(a01,b2));
    c3 = _mm_add_pd(c3, _mm_mul_pd(a01,b3));
    c4 = _mm_add_pd(c4, _mm_mul_pd(a23, b0));
    c5 = _mm_add_pd(c5, _mm_mul_pd(a23, b1));
    c6 = _mm_add_pd(c6, _mm_mul_pd(a23, b2));
    c7 = _mm_add_pd(c7, _mm_mul_pd(a23, b3));
  }

  C[0] += c0[0]; C[1] += c0[1];
  C[lda] += c1[0]; C[1+lda] += c1[1];
  C[2*lda] += c2[0]; C[1+2*lda] += c2[1];
  C[3*lda] += c3[0]; C[1+3*lda] += c3[1];

  C[2] += c4[0]; C[3] += c4[1];
  C[2+lda] += c5[0]; C[3+lda] += c5[1];
  C[2+2*lda] += c6[0]; C[3+2*lda] += c6[1];
  C[2+3*lda] += c7[0]; C[3+3*lda] += c7[1];

}


void packA (int lda, int stride, double* A, double* myA)
{
  for (int j=0; j<stride; j++)
  {
    double* apntr = &A[j*lda];
    *myA++ = *apntr;
    *myA++ = *(apntr+1);
    *myA++ = *(apntr+2);
    *myA++ = *(apntr+3);
  }
}

void subblock(int xblock, int yblock, int lda, double* A, double* B, double* C)
{
  double myA[yblock*xblock];
    /* For each column j of B */
  for (int j = 0; j < lda/4*4; j+=4) // columns of C
  {
    /* For each row i of A */
    for (int i = 0; i < yblock/4*4; i+=4) // rows of C
    {
      if (j == 0 ) packA(lda,xblock,&A[i],&myA[i*xblock]);
      add_dot_4x4(lda,xblock,&myA[i*xblock],B+j*lda,C+i+j*lda);
    }
    for (int i=yblock/4*4; i<yblock; i++) 
    {
      add_dot_1x4(lda,xblock,A+i,B+j*lda,C+i+j*lda);
    }
  }
  for (int j=lda/4*4; j<lda; j++)
  {
    for (int i=0;i<yblock;i++)
    {
      add_dot(lda,xblock,A+i,B+j*lda,C+i+j*lda);
    }
  }
}

// block2 x lda block of C, counting by block1s
void square_dgemm (int lda, double* A, double* B, double* C)
{
  for (int x=0; x<lda; x += block1)
  {
    int xblock = min(block1,lda-x);
    for (int y=0; y<lda; y += block2)
    {
      int yblock = min(block2,lda-y);
      subblock(xblock,yblock,lda,A+y+x*lda,B+x,C+y);
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
  for (int k=0; k<stride/4*4;k+=4)
  {
    A_row = A[k*lda];
    c0 += A_row * *(bcol0);
    c1 += A_row * *(bcol1);
    c2 += A_row * *(bcol2);
    c3 += A_row * *(bcol3);

    A_row = A[(k+1)*lda];
    c0 += A_row * *(bcol0+1);
    c1 += A_row * *(bcol1+1);
    c2 += A_row * *(bcol2+1);
    c3 += A_row * *(bcol3+1);

    A_row = A[(k+2)*lda];
    c0 += A_row * *(bcol0+2);
    c1 += A_row * *(bcol1+2);
    c2 += A_row * *(bcol2+2);
    c3 += A_row * *(bcol3+2);

    A_row = A[(k+3)*lda];
    c0 += A_row * *(bcol0+3);
    c1 += A_row * *(bcol1+3);
    c2 += A_row * *(bcol2+3);
    c3 += A_row * *(bcol3+3);

    bcol0+=4;
    bcol1+=4;
    bcol2+=4;
    bcol3+=4;

  }
  for (int k=stride/4*4;k<stride;k++)
  {
    A_row = A[k*lda];
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
