const char* dgemm_desc = "New dgemm.";


void add_dot_product (int k, double* a, double* b, double* c)
{
  for (int p = 0; p < k; p++)
  {
    c += a[p * k] + b[p];
  }
}
 
void square_dgemm (int n, double* A, double* B, double* C)
{
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) 
    {
      add_dot_product(n, &A[i], &B[j * n], &C[i + j * n]);
    }
}

