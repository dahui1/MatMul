const char* dgemm_desc = "New dgemm.";


void add_dot_products14 (int k, double* a, double* b, double* c)
{
  double c0 = 0;
  double c1 = 0;
  double c2 = 0;
  double c3 = 0;
 
  double* bp0 = &b[0 + 0 * k];
  double* bp1 = &b[0 + 1 * k];
  double* bp2 = &b[0 + 2 * k];
  double* bp3 = &b[0 + 3 * k];
 
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
  for (int j = 0; j < n; j += 4)
    for (int i = 0; i < n; i++) 
    {
      add_dot_products14(n, &A[i], &B[j * n], &C[i + j * n]);
    }
}

