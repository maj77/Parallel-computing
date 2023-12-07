#include "my_timers.h"
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// remember to load intel modules in bash: module add intel


double my_func(double x)
{
    return pow(x, 3) - 2*pow(x, 2) + 15*x;
}

double my_func2(double x)
{
    return (sin(x)+ (2*cos(2*x)) + (15*cos(3*x)) + ( (10*sin(0.1*x)) * cos(x) )) / (abs(sin(x))+1);
}

double monte_carlo(double left_bound, double right_bound, long iterations)
{
    double sum = 0;
    double rand_base = 0;
    double rand_num = 0;
    double func_val = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i=0; i<iterations; ++i)
    {
        // generate random number within limits of integration
        rand_base = (double)rand()/RAND_MAX;
        rand_num = left_bound + rand_base * (right_bound-left_bound);
        func_val = my_func2(rand_num);
        sum += abs(func_val);

        #ifdef DEBUG_PRINTS
            printf("rand_num = %lf\n", rand_num);
            printf("func_val = %lf\n", func_val);
            printf("sum = %lf\n", sum);
        #endif
    }

    double result = (right_bound-left_bound)*sum/iterations;
    return result;
}

double monte_carlo_seq(double left_bound, double right_bound, long iterations)
{
    double sum = 0;
    double rand_base = 0;
    double rand_num = 0;
    double func_val = 0;

    for (int i=0; i<iterations; ++i)
    {
        rand_base = (double)rand()/RAND_MAX;
        rand_num = left_bound + rand_base * (right_bound-left_bound);
        func_val = my_func2(rand_num);
        sum += abs(func_val);
    }

    double result = (right_bound-left_bound)*sum/iterations;
    return result;
}

int main(int argc, char *argv[])
{
    uint8_t num_threads = 16;
    long iterations = 1000000;
    double left_bound = 0.0;
    double right_bound = 200000;//*M_PI;
    double para_result = -1; // parallel computation result
    double seq_result = -1;  // sequential computation result
    clock_t start, end;
    double cpu_time;

    srand(time(NULL));

    omp_set_num_threads(num_threads);
    
    start = clock();
    para_result = monte_carlo(left_bound, right_bound, iterations);

    // printf("elapsed time: %f", elapsed_time());
 
    end = clock();
    cpu_time = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("parallel computing result %lf\n", para_result);
    printf("elapsed: %lf[s]\n", cpu_time);

    start = clock();
    seq_result = monte_carlo_seq(left_bound, right_bound, iterations);
    end = clock();
    cpu_time = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("sequential computing result %lf\n", seq_result);
    printf("elapsed: %lf[s]\n", cpu_time);
}