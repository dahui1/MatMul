const char* dgemm_desc = "New dgemm.";

#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

void add_dot_products(int k, int size, double* a, double* b, double* c)
{
  __m128d c0010, c0111, c0212, c0313, c2030, c2131, c2232, c2333;

  c0010 = _mm_setzero_pd();
  c0111 = _mm_setzero_pd();
  c0212 = _mm_setzero_pd();
  c0313 = _mm_setzero_pd();
  c2030 = _mm_setzero_pd();
  c2131 = _mm_setzero_pd();
  c2232 = _mm_setzero_pd();
  c2333 = _mm_setzero_pd();

  double* bp0, * bp1, * bp2, * bp3;
  bp0 = &b[0];
  bp1 = &b[size];
  bp2 = &b[2 * size];
  bp3 = &b[3 * size];

  __m128d bp0v, bp1v, bp2v, bp3v;

  for (int p = 0; p < k; p += 4)
  {
    __m128d a01_0, a23_0, a01_1, a23_1, a01_2, a23_2, a01_3, a23_3;
    a01_0 = _mm_load_pd((double*) a);
    a23_0 = _mm_load_pd((double*) (a + 2)); 
    a01_1 = _mm_load_pd((double*) (a + 4));
    a23_1 = _mm_load_pd((double*) (a + 6));
    a01_2 = _mm_load_pd((double*) (a + 8));
    a23_2 = _mm_load_pd((double*) (a + 10));
    a01_3 = _mm_load_pd((double*) (a + 12));
    a23_3 = _mm_load_pd((double*) (a + 14));  
    a += 16;

    bp0v = _mm_load1_pd((double*) bp0++);
    bp1v = _mm_load1_pd((double*) bp1++);
    bp2v = _mm_load1_pd((double*) bp2++);
    bp3v = _mm_load1_pd((double*) bp3++);

    c0010 += _mm_mul_pd(a01_0, bp0v);
    c0111 += _mm_mul_pd(a01_0, bp1v);
    c0212 += _mm_mul_pd(a01_0, bp2v);
    c0313 += _mm_mul_pd(a01_0, bp3v);
    c2030 += _mm_mul_pd(a23_0, bp0v);
    c2131 += _mm_mul_pd(a23_0, bp1v);
    c2232 += _mm_mul_pd(a23_0, bp2v);
    c2333 += _mm_mul_pd(a23_0, bp3v);

    bp0v = _mm_loaddup_pd((double*) bp0++);
    bp1v = _mm_loaddup_pd((double*) bp1++);
    bp2v = _mm_loaddup_pd((double*) bp2++);
    bp3v = _mm_loaddup_pd((double*) bp3++);

    c0010 += _mm_mul_pd(a01_1, bp0v);
    c0111 += _mm_mul_pd(a01_1, bp1v);
    c0212 += _mm_mul_pd(a01_1, bp2v);
    c0313 += _mm_mul_pd(a01_1, bp3v);
    c2030 += _mm_mul_pd(a23_1, bp0v);
    c2131 += _mm_mul_pd(a23_1, bp1v);
    c2232 += _mm_mul_pd(a23_1, bp2v);
    c2333 += _mm_mul_pd(a23_1, bp3v);

    bp0v = _mm_loaddup_pd((double*) bp0++);
    bp1v = _mm_loaddup_pd((double*) bp1++);
    bp2v = _mm_loaddup_pd((double*) bp2++);
    bp3v = _mm_loaddup_pd((double*) bp3++);

    c0010 += _mm_mul_pd(a01_2, bp0v);
    c0111 += _mm_mul_pd(a01_2, bp1v);
    c0212 += _mm_mul_pd(a01_2, bp2v);
    c0313 += _mm_mul_pd(a01_2, bp3v);
    c2030 += _mm_mul_pd(a23_2, bp0v);
    c2131 += _mm_mul_pd(a23_2, bp1v);
    c2232 += _mm_mul_pd(a23_2, bp2v);
    c2333 += _mm_mul_pd(a23_2, bp3v);

    bp0v = _mm_loaddup_pd((double*) bp0++);
    bp1v = _mm_loaddup_pd((double*) bp1++);
    bp2v = _mm_loaddup_pd((double*) bp2++);
    bp3v = _mm_loaddup_pd((double*) bp3++);

    c0010 += _mm_mul_pd(a01_3, bp0v);
    c0111 += _mm_mul_pd(a01_3, bp1v);
    c0212 += _mm_mul_pd(a01_3, bp2v);
    c0313 += _mm_mul_pd(a01_3, bp3v);
    c2030 += _mm_mul_pd(a23_3, bp0v);
    c2131 += _mm_mul_pd(a23_3, bp1v);
    c2232 += _mm_mul_pd(a23_3, bp2v);
    c2333 += _mm_mul_pd(a23_3, bp3v);
  }
  
  c[0] += c0010[0], c[size] += c0111[0], c[2 * size] += c0212[0], c[3 * size] += c0313[0];
  c[1] += c0010[1], c[size + 1] += c0111[1], c[2 * size + 1] += c0212[1], c[3 * size + 1] += c0313[1];
  c[2] += c2030[0], c[size + 2] += c2131[0], c[2 * size + 2] += c2232[0], c[3 * size + 2] += c2333[0];
  c[3] += c2030[1], c[size + 3] += c2131[1], c[2 * size + 3] += c2232[1], c[3 * size + 3] += c2333[1];
}

void PackMatrixA( int k, double *a, int lda, double *a_to )
{
  for(int j = 0; j < k; j++)
  {
    double *a_ij_pntr = &a[j * lda];

    *a_to++ = *a_ij_pntr;
    *a_to++ = *(a_ij_pntr + 1);
    *a_to++ = *(a_ij_pntr + 2);
    *a_to++ = *(a_ij_pntr + 3);
  }
}

void square_dgemm (int n, double* A, double* B, double* C)
{
  // Padding the matrices
  int n4 = n;
  if (n4 % 4 != 0) {
    n4 = n - n4 % 4 + 4;
  }

  double* buf = (double*) calloc (3 * n4 * n4, sizeof(double));

  double* newA = buf + 0;
  double* newB = newA + n4 * n4;
  double* newC = newB + n4 * n4;  

  for(int i = 0; i < n; i++)
  {
    memcpy(newA + i * n4, A + i * n, n * sizeof(double));
    memcpy(newB + i * n4, B + i * n, n * sizeof(double));
    memcpy(newC + i * n4, C + i * n, n * sizeof(double));
  }

  // Blocking size
  int kc = 32, mc = 64;
  int xb, yb;

  for (int x = 0; x < n4; x += kc)
  {
    xb = n4 - x < kc ? n4 - x : kc;
    if (xb % 4 != 0)
      xb = xb - xb % 4 + 4;
    for (int y = 0; y < n4; y += mc)
    {
      yb = n4 - y < mc ? n4 - y : mc;
      if (yb % 4 != 0)
        yb = yb - yb % 4 + 4;
      double packedA[yb * xb];
      for (int j = 0; j < n4; j += 4)
        for (int i = 0; i < yb; i += 4) 
        {
          if (j == 0) 
            PackMatrixA(xb, &newA[i + y + x * n4], n4, &packedA[i * xb]);
          add_dot_products(xb, n4, &packedA[i * xb], &newB[x + j * n4], &newC[i + y + j * n4]);
        }

    }
  }

  for (int i = 0; i < n; i++) 
  {
    memcpy(C + i * n, newC + i * n4, n * sizeof(double));
  }
} 