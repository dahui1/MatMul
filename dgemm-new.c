const char* dgemm_desc = "New dgemm.";

#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

void add_dot_products(int k, double* a, double* b, double* c)
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

  double* bp0 = &b[0];
  double* bp1 = &b[k];
  double* bp2 = &b[2 * k];
  double* bp3 = &b[3 * k];

  for (int p = 0; p < k; p += 4)
  {
    __m128d a01_0 = _mm_load_pd((double*) &a[p * k]);
    __m128d a23_0 = _mm_load_pd((double*) &a[p * k + 2]);
    __m128d a01_1 = _mm_load_pd((double*) &a[(p + 1) * k]);
    __m128d a23_1 = _mm_load_pd((double*) &a[(p + 1) * k + 2]);
    __m128d a01_2 = _mm_load_pd((double*) &a[(p + 2)* k]);
    __m128d a23_2 = _mm_load_pd((double*) &a[(p + 2) * k + 2]);
    __m128d a01_3 = _mm_load_pd((double*) &a[(p + 3) * k]);
    __m128d a23_3 = _mm_load_pd((double*) &a[(p + 3) * k + 2]);        

    __m128d bp0v = _mm_loaddup_pd((double*) bp0++);
    __m128d bp1v = _mm_loaddup_pd((double*) bp1++);
    __m128d bp2v = _mm_loaddup_pd((double*) bp2++);
    __m128d bp3v = _mm_loaddup_pd((double*) bp3++);

    c0010 += a01_0 * bp0v;
    c0111 += a01_0 * bp1v;
    c0212 += a01_0 * bp2v;
    c0313 += a01_0 * bp3v;
    c2030 += a23_0 * bp0v;
    c2131 += a23_0 * bp1v;
    c2232 += a23_0 * bp2v;
    c2333 += a23_0 * bp3v;

    bp0v = _mm_loaddup_pd((double*) bp0++);
    bp1v = _mm_loaddup_pd((double*) bp1++);
    bp2v = _mm_loaddup_pd((double*) bp2++);
    bp3v = _mm_loaddup_pd((double*) bp3++);

    c0010 += a01_1 * bp0v;
    c0111 += a01_1 * bp1v;
    c0212 += a01_1 * bp2v;
    c0313 += a01_1 * bp3v;
    c2030 += a23_1 * bp0v;
    c2131 += a23_1 * bp1v;
    c2232 += a23_1 * bp2v;
    c2333 += a23_1 * bp3v;

    bp0v = _mm_loaddup_pd((double*) bp0++);
    bp1v = _mm_loaddup_pd((double*) bp1++);
    bp2v = _mm_loaddup_pd((double*) bp2++);
    bp3v = _mm_loaddup_pd((double*) bp3++);

    c0010 += a01_2 * bp0v;
    c0111 += a01_2 * bp1v;
    c0212 += a01_2 * bp2v;
    c0313 += a01_2 * bp3v;
    c2030 += a23_2 * bp0v;
    c2131 += a23_2 * bp1v;
    c2232 += a23_2 * bp2v;
    c2333 += a23_2 * bp3v;

    bp0v = _mm_loaddup_pd((double*) bp0++);
    bp1v = _mm_loaddup_pd((double*) bp1++);
    bp2v = _mm_loaddup_pd((double*) bp2++);
    bp3v = _mm_loaddup_pd((double*) bp3++);

    c0010 += a01_3 * bp0v;
    c0111 += a01_3 * bp1v;
    c0212 += a01_3 * bp2v;
    c0313 += a01_3 * bp3v;
    c2030 += a23_3 * bp0v;
    c2131 += a23_3 * bp1v;
    c2232 += a23_3 * bp2v;
    c2333 += a23_3 * bp3v;
 }
  
  c[0] += c0010[0], c[k] += c0111[0], c[2 * k] += c0212[0], c[3 * k] += c0313[0];
  c[1] += c0010[1], c[k + 1] += c0111[1], c[2 * k + 1] += c0212[1], c[3 * k + 1] += c0313[1];
  c[2] += c2030[0], c[k + 2] += c2131[0], c[2 * k + 2] += c2232[0], c[3 * k + 2] += c2333[0];
  c[3] += c2030[1], c[k + 3] += c2131[1], c[2 * k + 3] += c2232[1], c[3 * k + 3] += c2333[1];
}
 
void square_dgemm (int n, double* A, double* B, double* C)
{
  int n4 = n;
  if (n4 % 4 != 0) {
    n4 = n - n4 % 4 + 4;
  }

  double* buf = (double*) calloc (3 * n4 * n4, sizeof(double));

  double* newA = buf + 0;
  double* newB = newA + n4 * n4;
  double* newC = newB + n4 * n4;  

  for(int i = 0; i < n; i++){
    memcpy(newA + i * n4, A + i * n, n * sizeof(double));
    memcpy(newB + i * n4, B + i * n, n * sizeof(double));
    memcpy(newC + i * n4, C + i * n, n * sizeof(double));
  }

  for (int j = 0; j < n4; j += 4)
    for (int i = 0; i < n4; i += 4) 
    {
      add_dot_products(n4, &newA[i], &newB[j * n4], &newC[i + j * n4]);
    }

  for (int i = 0; i < n; i++) {
    memcpy(C + i * n, newC + i * n4, n * sizeof(double));
  }
} 