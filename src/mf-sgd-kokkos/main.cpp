// Matrix Factorization SGD - Kokkos port
// Implements Hogwild-style parallel SGD for matrix factorization.
// Generates synthetic sparse ratings and runs SGD to minimize squared error.
//
// Usage: ./main [num_users num_items num_ratings k num_iters repeat]
// Defaults: 1000 1000 50000 128 5 1

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Parse command-line arguments
        int num_users   = (argc > 1) ? atoi(argv[1]) : 1000;
        int num_items   = (argc > 2) ? atoi(argv[2]) : 1000;
        int num_ratings = (argc > 3) ? atoi(argv[3]) : 50000;
        int k           = (argc > 4) ? atoi(argv[4]) : 128;
        int num_iters   = (argc > 5) ? atoi(argv[5]) : 5;
        int repeat      = (argc > 6) ? atoi(argv[6]) : 1;

        const float lrate  = 0.001f;
        const float lambda = 0.001f;

        printf("MF-SGD: users=%d items=%d ratings=%d k=%d iters=%d repeat=%d\n",
               num_users, num_items, num_ratings, k, num_iters, repeat);

        // Generate synthetic ratings: random (user, item, rating) triples
        Kokkos::View<int*>   d_users("users",    num_ratings);
        Kokkos::View<int*>   d_items("items",    num_ratings);
        Kokkos::View<float*> d_ratings("ratings", num_ratings);

        {
            auto h_users   = Kokkos::create_mirror_view(d_users);
            auto h_items   = Kokkos::create_mirror_view(d_items);
            auto h_ratings = Kokkos::create_mirror_view(d_ratings);

            srand(42);
            for (int i = 0; i < num_ratings; i++) {
                h_users(i)   = rand() % num_users;
                h_items(i)   = rand() % num_items;
                h_ratings(i) = 1.0f + (float)(rand() % 5); // rating in [1,5]
            }
            Kokkos::deep_copy(d_users,   h_users);
            Kokkos::deep_copy(d_items,   h_items);
            Kokkos::deep_copy(d_ratings, h_ratings);
        }

        // Factor matrices P (users x k) and Q (items x k)
        Kokkos::View<float*> d_P("P", (size_t)num_users * k);
        Kokkos::View<float*> d_Q("Q", (size_t)num_items * k);

        // Save initial state for re-run across repeats
        Kokkos::View<float*> d_P0("P0", (size_t)num_users * k);
        Kokkos::View<float*> d_Q0("Q0", (size_t)num_items * k);
        {
            auto h_P = Kokkos::create_mirror_view(d_P0);
            auto h_Q = Kokkos::create_mirror_view(d_Q0);
            srand(123);
            for (int i = 0; i < num_users * k; i++)
                h_P(i) = 0.1f * ((float)rand() / RAND_MAX);
            for (int i = 0; i < num_items * k; i++)
                h_Q(i) = 0.1f * ((float)rand() / RAND_MAX);
            Kokkos::deep_copy(d_P0, h_P);
            Kokkos::deep_copy(d_Q0, h_Q);
        }

        double total_time = 0.0;

        for (int r = 0; r < repeat; r++) {
            // Reset factors to initial state
            Kokkos::deep_copy(d_P, d_P0);
            Kokkos::deep_copy(d_Q, d_Q0);
            Kokkos::fence();

            Kokkos::Timer timer;

            for (int iter = 0; iter < num_iters; iter++) {
                // SGD sweep: process all ratings in parallel (Hogwild-style).
                // Uses atomic_add to accumulate gradient updates safely.
                Kokkos::parallel_for(
                    "sgd_epoch", num_ratings,
                    KOKKOS_LAMBDA(int idx) {
                        const int   u = d_users(idx);
                        const int   v = d_items(idx);
                        const float r_uv = d_ratings(idx);

                        // Compute prediction = dot(P[u], Q[v])
                        float predict = 0.0f;
                        for (int f = 0; f < k; f++)
                            predict += d_P(u * k + f) * d_Q(v * k + f);

                        const float err = r_uv - predict;

                        // Gradient step with L2 regularization (atomic for race safety)
                        for (int f = 0; f < k; f++) {
                            const float pu = d_P(u * k + f);
                            const float qv = d_Q(v * k + f);
                            Kokkos::atomic_add(&d_P(u * k + f),
                                               lrate * (err * qv - lambda * pu));
                            Kokkos::atomic_add(&d_Q(v * k + f),
                                               lrate * (err * pu - lambda * qv));
                        }
                    });
                Kokkos::fence();
            }

            total_time += timer.seconds();
        }

        const double avg_ms = total_time * 1000.0 / repeat;
        printf("Average SGD time (%d iters): %.3f ms\n", num_iters, avg_ms);

        // Compute final RMSE
        float rmse = 0.0f;
        Kokkos::parallel_reduce(
            "rmse", num_ratings,
            KOKKOS_LAMBDA(int idx, float& sum) {
                const int   u    = d_users(idx);
                const int   v    = d_items(idx);
                const float r_uv = d_ratings(idx);
                float pred = 0.0f;
                for (int f = 0; f < k; f++)
                    pred += d_P(u * k + f) * d_Q(v * k + f);
                const float e = r_uv - pred;
                sum += e * e;
            },
            rmse);
        Kokkos::fence();

        rmse = sqrtf(rmse / num_ratings);
        printf("Final RMSE: %.6f\n", rmse);
    }
    Kokkos::finalize();
    return 0;
}
