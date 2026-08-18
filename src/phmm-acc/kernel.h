void pair_HMM_forward(
    const int teams,
    const int threads,
    const int cur_i,
    const int cur_j,
    //const double forward_matrix_in[x_dim+1][y_dim+1][batch][states-1],
    const fArray *__restrict forward_matrix_in,
    //const double transitions[x_dim+1][batch][states-1][states],
    const tArray *__restrict transitions,
    //const double emissions[x_dim+1][y_dim+1][batch][states-1],
    const fArray *__restrict emissions,
    // const double likelihood[2][2][batch][states-1],
    const lArray *__restrict likelihood,
    //const double start_transitions[batch][states-1],
    const sArray *__restrict start_transitions,
          //double forward_matrix_out[x_dim+1][y_dim+1][batch][states-1])
          fArray *__restrict forward_matrix_out)
{

  #pragma acc parallel loop gang num_gangs(teams) vector_length(threads)
  for (int batch_id = 0; batch_id < teams; batch_id++)
  {
    double e[batch][states-1];
    double f01[1][batch][2];
    double mul_3d[1][batch][2];
    double mul_4d[4][batch][1][2];
    double t[2][2][batch][2][2];
    double f[2][2][batch][1][2];
    double t01[batch][2][2];

    #pragma acc loop vector
    for (int states_id = 0; states_id < threads; states_id++)
    {
      e[batch_id][states_id] = emissions[cur_i][cur_j][batch_id][states_id];
    
      for (int k = 0; k < 2; k++) {
        for (int l = 0; l < 2; l++) {
          t[0][0][batch_id][k][l] = transitions[cur_i - 1][batch_id][k][l];
          t[0][1][batch_id][k][l] = transitions[cur_i - 1][batch_id][k][l];
          t[1][0][batch_id][k][l] = transitions[cur_i][batch_id][k][l];
          t[1][1][batch_id][k][l] = transitions[cur_i][batch_id][k][l];
        }
      }
      if (cur_i > 0 && cur_j == 0) {
        if (cur_i != 1) {
          f01[0][batch_id][states_id] = 
            forward_matrix_in[cur_i - 1][cur_j][batch_id][states_id];
	}
      }
      else if (cur_i > 0 and cur_j > 0) {
        for (int i = 0; i < 2; i++) {
          f[0][0][batch_id][0][i] = forward_matrix_in[cur_i-1][cur_j-1][batch_id][i];
          f[0][1][batch_id][0][i] = forward_matrix_in[cur_i-1][cur_j][batch_id][i];
          f[1][0][batch_id][0][i] = forward_matrix_in[cur_i][cur_j-1][batch_id][i];
          f[1][1][batch_id][0][i] = forward_matrix_in[cur_i][cur_j][batch_id][i];
        }
      }
    }
    
    #pragma acc loop vector
    for (int states_id = 0; states_id < threads; states_id++)
    {
      if (cur_i > 0 && cur_j == 0) {
        if (cur_i == 1) {
	}
	else {
          for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
              t01[batch_id][j][k] = t[0][1][batch_id][j][k];
            }
          }
	}
      }
    }

    #pragma acc loop vector
    for (int states_id = 0; states_id < threads; states_id++)
    {
      if (cur_i > 0 && cur_j == 0) {
        if (cur_i == 1) {
          forward_matrix_out[1][0][batch_id][states_id] = 
            start_transitions[batch_id][states_id] * e[0][states_id];
        }
        else {
          double s = 0.0;
          for (int k = 0; k < 2; k++)
            s += f01[0][batch_id][k] * t01[batch_id][k][states_id];
          s *= (e[batch_id][states_id] * likelihood[0][1][batch_id][states_id]);
          mul_3d[0][batch_id][states_id] = s;
    
	}
      }
      else if (cur_i > 0 and cur_j > 0) {
        double s0 = 0.0;
        double s1 = 0.0;
        double s2 = 0.0;
        double s3 = 0.0;
    
        for (int k = 0; k < 2; k++) {
          s0 += f[0][0][batch_id][0][k] * t[0][0][batch_id][k][states_id];
          s1 += f[0][1][batch_id][0][k] * t[0][1][batch_id][k][states_id];
          s2 += f[1][0][batch_id][0][k] * t[1][0][batch_id][k][states_id];
          s3 += f[1][1][batch_id][0][k] * t[1][1][batch_id][k][states_id];
        }
        s0 *= likelihood[0][0][batch_id][states_id];
        s1 *= likelihood[0][1][batch_id][states_id];
        s2 *= likelihood[1][0][batch_id][states_id];
        s3 *= likelihood[1][1][batch_id][states_id];
        mul_4d[0][batch_id][0][states_id] = s0;
        mul_4d[1][batch_id][0][states_id] = s1;
        mul_4d[2][batch_id][0][states_id] = s2;
        mul_4d[3][batch_id][0][states_id] = s3;
      }
    }

    #pragma acc loop vector
    for (int states_id = 0; states_id < threads; states_id++)
    {
      if (cur_i > 0 && cur_j == 0) {
        if (cur_i == 1) {
          forward_matrix_out[1][0][batch_id][states_id] = 
            start_transitions[batch_id][states_id] * e[0][states_id];
        }
        else {
          forward_matrix_out[cur_i][0][batch_id][states_id] = mul_3d[0][batch_id][states_id];
        }
      }
      else if (cur_i > 0 and cur_j > 0) {
    
        for (int j = 0; j < 2; j++) {
          double summation = mul_4d[0][batch_id][0][j] + 
                             mul_4d[1][batch_id][0][j] +
                             mul_4d[2][batch_id][0][j] +
                             mul_4d[3][batch_id][0][j];
    
          summation *= e[batch_id][j];
    
          forward_matrix_out[cur_i][cur_j][batch_id][j] = summation;
        }
      }
    }
  }
}

