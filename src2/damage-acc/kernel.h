/* Calculate the damage of each node.
 *
 * nlist - An (n, local_size) array containing the neighbour lists,
 *         a value of -1 corresponds to a broken bond.
 * family - An array of the initial number of neighbours for each node.
 * n_neigh - An array of the number of neighbours (particles bound) for each node.
 * damage - An array of the damage for each node.
 * local_cache - local (local_size) array to store the bond breakages.
 */
void damage_of_node(
  const int n,
  const int *__restrict nlist,
  const int *__restrict family,
        int *__restrict n_neigh,
     double *__restrict damage)
{
  const int num_gangs = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  #pragma acc parallel num_gangs(num_gangs) vector_length(BLOCK_SIZE) \
    present(nlist, family, n_neigh, damage)
  {
    int local_cache[BLOCK_SIZE];

    #pragma acc loop gang
    for (int nid = 0; nid < num_gangs; nid++) {

      //Copy values into local memory
      #pragma acc loop vector
      for (int local_id = 0; local_id < BLOCK_SIZE; local_id++) {
        const int global_id = nid * BLOCK_SIZE + local_id;
        if (global_id < n)
          local_cache[local_id] = nlist[global_id] != -1 ? 1 : 0;
        else
          local_cache[local_id] = 0;
      }

      // Tree reduction
      for (int i = BLOCK_SIZE/2; i > 0; i /= 2) {
        #pragma acc loop vector
        for (int local_id = 0; local_id < BLOCK_SIZE; local_id++) {
          if(local_id < i) {
            local_cache[local_id] += local_cache[local_id + i];
          }
        }
      }

      // Update damage and n_neigh
      int neighbours = local_cache[0];
      n_neigh[nid] = neighbours;
      damage[nid] = 1.0 - (double) neighbours / (double) (family[nid]);
    }
  }
}

void damage_of_node_optimized(
  const int m,
  const int n,
  const int *__restrict nlist,
  const int *__restrict family,
        int *__restrict n_neigh,
     double *__restrict damage)
{
  #pragma acc parallel num_gangs(m) vector_length(BLOCK_SIZE) \
    present(nlist, family, n_neigh, damage)
  #pragma acc loop gang
  for (int nid = 0; nid < m; nid++) {
    int lb = nid * BLOCK_SIZE;
    int ub = (lb + BLOCK_SIZE < n) ? lb + BLOCK_SIZE : n;

    int sum = 0;
    #pragma acc loop vector reduction(+:sum)
    for (int i = lb; i < ub; i++)
      sum += nlist[i] != -1 ? 1 : 0;

    n_neigh[nid] = sum;
    damage[nid] = 1.0 - (double) sum / (double) (family[nid]);
  }
}
