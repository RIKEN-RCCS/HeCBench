#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <random>
#include <chrono>
#include <openacc.h>

int main(int argc, char* argv[]) {

  if (argc != 2) {
    printf("Usage: ./%s <iterations>\n", argv[0]);
    return 1;
  }

  // specify the number of test cases
  const int iteration = atoi(argv[1]);

  // number of elements to reverse
  const int len = 256;
  const int elem_size = len * sizeof(int);

  // save device result
  int test[len];

  // save expected results after performing reverse operations even/odd times
  int error = 0;
  int gold_odd[len];
  int gold_even[len];

  for (int i = 0; i < len; i++) {
    gold_odd[i] = len-i-1;
    gold_even[i] = i;
  }

  std::default_random_engine generator (123);
  // bound the number of reverse operations
  std::uniform_int_distribution<int> distribution(100, 9999);

  long time = 0;

  #pragma acc data create(test[0:len])
  {
    for (int i = 0; i < iteration; i++) {
      const int count = distribution(generator);

      memcpy(test, gold_even, elem_size);
      #pragma acc update device(test[0:len])

      auto start = std::chrono::steady_clock::now();

      for (int j = 0; j < count; j++) {
        #pragma acc parallel num_gangs(1) vector_length(len) present(test[0:len])
        {
          int s[len];
	  #pragma acc loop vector
	  for (int t = 0; t < len; t++)
          {
            s[t] = test[t];
	  }
	  #pragma acc loop vector
	  for (int t = 0; t < len; t++)
          {
            test[t] = s[len-t-1];
          }
        }
      }

      #pragma acc wait
      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      #pragma acc update self(test[0:len])

      if (count % 2 == 0)
        error = memcmp(test, gold_even, elem_size);
      else
        error = memcmp(test, gold_odd, elem_size);
      
      if (error) break;
    }
  }

  printf("Total kernel execution time: %f (s)\n", time * 1e-9f);
  printf("%s\n", error ? "FAIL" : "PASS");

  return 0;
}
