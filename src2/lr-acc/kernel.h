inline void operator+=(float2 &a, const float2 &b)
{
    a.x += b.x;
    a.y += b.y;
}

inline void operator+=(float4 &a, const float4 &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

void linear_regression(
  const int nTeams,
  const float2 *__restrict dataset,
        float4 *__restrict result)
{
  #pragma acc parallel loop gang num_gangs(nTeams) vector_length(TEMP_WORKGROUP_SIZE)
  for( int blk_id = 0; blk_id < nTeams; blk_id++ ) {
    float4 interns[TEMP_WORKGROUP_SIZE];
    size_t loc_size = TEMP_WORKGROUP_SIZE; 

    #pragma acc loop vector
    for( int loc_id = 0; loc_id < loc_size; loc_id++ ) {
      size_t glob_id  = blk_id * loc_size + loc_id;

      /* Initialize local buffer */
      interns[loc_id].x = dataset[glob_id].x;
      interns[loc_id].y = dataset[glob_id].y;
      interns[loc_id].z = (dataset[glob_id].x * dataset[glob_id].y);
      interns[loc_id].w = (dataset[glob_id].x * dataset[glob_id].x);
    }
    // auto barrier
    for (size_t i = (loc_size / 2), old_i = loc_size; i > 0; old_i = i, i /= 2) {
      #pragma acc loop vector
      for( int loc_id = 0; loc_id < loc_size; loc_id++ ) {
        if (loc_id < i) {
          // Only first half of workitems on each workgroup
          interns[loc_id] += interns[loc_id + i];
          if (loc_id == (i - 1) && old_i % 2 != 0) {
            // If there is an odd number of data
            interns[loc_id] += interns[old_i - 1];
          }
        }
      }
      // auto barrier
    }

    #pragma acc loop vector
    for( int loc_id = 0; loc_id < 1; loc_id++ ) {
      result[blk_id] = interns[0];
    }
  }
}

void rsquared(
  const int nTeams,
  const float2 *__restrict dataset,
  const float mean,
  const float2 equation, // [a0,a1]
  float2 *__restrict result)
{
  #pragma acc parallel loop gang num_gangs(nTeams) vector_length(TEMP_WORKGROUP_SIZE)
  for (int blk_id = 0; blk_id < nTeams; blk_id++) {
    float2 dist[TEMP_WORKGROUP_SIZE];
    size_t loc_size = TEMP_WORKGROUP_SIZE; 

    #pragma acc loop vector
    for( int loc_id = 0; loc_id < loc_size; loc_id++ ) {
      size_t glob_id  = blk_id * loc_size + loc_id;

      dist[loc_id].x = powf((dataset[glob_id].y - mean), 2.f);

      float y_estimated = dataset[glob_id].x * equation.y + equation.x;
      dist[loc_id].y = powf((y_estimated - mean), 2.f);
    }
    // auto barrier
    for (size_t i = (loc_size / 2), old_i = loc_size; i > 0; old_i = i, i /= 2) {
      #pragma acc loop vector
      for( int loc_id = 0; loc_id < loc_size; loc_id++ ) {
        if (loc_id < i) {
          // Only first half of workitems on each workgroup
          dist[loc_id] += dist[loc_id + i];
          if (loc_id == (i - 1) && old_i % 2 != 0) {
            // If there is an odd number of data
            dist[loc_id] += dist[old_i - 1];
          }
        }
      }
      // auto barrier
    }

    #pragma acc loop vector
    for( int loc_id = 0; loc_id < 1; loc_id++ ) {
      result[blk_id] = dist[0];
    }
  }
}
