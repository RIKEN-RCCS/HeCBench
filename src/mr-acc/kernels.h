void mr32_sf(
  const uint32_t *__restrict bases,
  const uint32_t *__restrict n32,
  int *__restrict val,
  int iter)
{
  #pragma acc parallel loop vector_length(256)
  for (int j = 0; j < iter; j++) {
    int n = n32[j];
    for (int cnt = 1; cnt <= BASES_CNT32; cnt++) {
      const int r = straightforward_mr32(bases, cnt, n);
      #pragma acc atomic update
      *val += r;
    }
  }
}

void mr32_eff(
  const uint32_t *__restrict bases,
  const uint32_t *__restrict n32,
  int *__restrict val,
  int iter)
{
  #pragma acc parallel loop vector_length(256)
  for (int j = 0; j < iter; j++) {
    int n = n32[j];
    for (int cnt = 1; cnt <= BASES_CNT32; cnt++) {
      const int r = efficient_mr32(bases, cnt, n);
      #pragma acc atomic update
      *val += r;
    }
  }
}
