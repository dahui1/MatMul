const char* dgemm_desc = "New dgemm.";

#include <stdlib.h>
#include <string.h>

void add_dot_products14 (int k, double* a, double* b, double* c)
{
  double c0 = 0;
  double c1 = 0;
  double c2 = 0;
  double c3 = 0;
 
  double* bp0 = &b[0];
  double* bp1 = &b[k];
  double* bp2 = &b[2 * k];
  double* bp3 = &b[3 * k];
 
  for (int p = 0; p < k; p += 4)
  {
    double apk = a[p * k];
    c0 += apk * *(bp0);
    c1 += apk * *(bp1);
    c2 += apk * *(bp2);
    c3 += apk * *(bp3);

    apk = a[(p + 1) * k];
    c0 += apk * *(bp0 + 1);
    c1 += apk * *(bp1 + 1);
    c2 += apk * *(bp2 + 1);
    c3 += apk * *(bp3 + 1);

    apk = a[(p + 2) * k];
    c0 += apk * *(bp0 + 2);
    c1 += apk * *(bp1 + 2);
    c2 += apk * *(bp2 + 2);
    c3 += apk * *(bp3 + 2);

    apk = a[(p + 3) * k];
    c0 += apk * *(bp0 + 3);
    c1 += apk * *(bp1 + 3);
    c2 += apk * *(bp2 + 3);
    c3 += apk * *(bp3 + 3); 

    bp0 += 4;
    bp1 += 4;
    bp2 += 4;
    bp3 += 4;
 }
  
  c[0] += c0;
  c[k] += c1;
  c[k * 2] += c2;
  c[k * 3] += c3;
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
    for (int i = 0; i < n4; i++) 
    {
      add_dot_products14(n4, &newA[i], &newB[j * n4], &newC[i + j * n4]);
    }

  for (int i = 0; i < n; i++) {
    memcpy(C + i * n, newC + i * n4, n * sizeof(double));
  }
}

