/***************************************************
/**
/** interactive mode params:    srun -N 1 --ntasks-per-node=8 -p plgrid -A plgpiask2023-cpu 
/**                                  --reservation=piask_mon -t 1:30:00 --pty /bin/bash
/**
/**************************************************/

#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>


typedef struct range{
    double left_bound;
    double right_bound;
} Range;

double my_func(double x)
{
    return pow(x, 3) - 2*pow(x, 2) + 15*x;
}

double my_func2(double x)
{
    return (sin(x) + (2*cos(2*x)) + (15*cos(3*x)) + ( (10*sin(0.1*x)) * cos(x) )) / (abs(sin(x))+1);
}

double monte_carlo(Range *range, long iterations, int myrank)
{
    double sum = 0;
    double rand_base = 0;
    double rand_num = 0;
    double func_val = 0;
    double left_bound = range->left_bound;
    double right_bound = range->right_bound;

    printf("[PARA] myrank is %d, iterations %d, integration area: <%lf, %lf>\n", myrank, iterations, left_bound, right_bound);
    for (int i=0; i<iterations; ++i){
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

    return (right_bound-left_bound)*sum/iterations;
}

double monte_carlo_seq(double left_bound, double right_bound, long iterations)
{
    double sum = 0;
    double rand_base = 0;
    double rand_num = 0;
    double func_val = 0;
    
    printf("[SEQ] iterations: %d, integration area:  <%lf, %lf>\n", iterations, left_bound, right_bound);

    for (int i=0; i<iterations; ++i){
        rand_base = (double)rand()/RAND_MAX;
        rand_num = left_bound + rand_base * (right_bound-left_bound);
        func_val = my_func2(rand_num);
        sum += abs(func_val);
    }

    return (right_bound-left_bound)*sum/iterations;
}


int main(int argc, char *argv[])
{
    //
    // declare variables and initialize MPI
    //
    long iterations = 16384;    // 2^14
    double left_bound = 0.0;
    double right_bound = 200000;
    double single_result = -1;
    double para_result = -1; // parallel computation result
    double seq_result = -1;  // sequential computation result
    clock_t start, end;
    double cpu_time;
    double cpu_time_buf;
    int myrank;
    int size;
    int root=0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int para_iterations = iterations/size;

    //
    // create MPI type for Range struct
    //
    const int nitems = 2;
    int blocklengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Datatype mpi_range_t;
    MPI_Aint offsets[2] = {offsetof(Range, left_bound), offsetof(Range, right_bound)};

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_range_t);
    MPI_Type_commit(&mpi_range_t);

    Range *ranges = malloc(sizeof(Range)*size);
    Range *ranges_buf = malloc(sizeof(Range));

    //
    // initializations
    //
    ranges[0].left_bound = left_bound;
    ranges[0].right_bound = right_bound/size;
    
    // initialize ranges array in first process
    if(myrank == 0) { 
        for (int i=1; i<size; i++){
            ranges[i].left_bound = ranges[i-1].right_bound;
            ranges[i].right_bound = (i+1)*right_bound/size;
        }
    }

    // send ranges to processes
    MPI_Scatter(ranges, 1, mpi_range_t, 
                ranges_buf, 1, mpi_range_t, 
                root, MPI_COMM_WORLD);

    srand(time(NULL));

    //
    // parallel calculations of integral
    //
    start = clock();
    single_result = monte_carlo(ranges_buf, para_iterations, myrank);
    end = clock();
    cpu_time = ((double)(end-start))/CLOCKS_PER_SEC;
    MPI_Reduce(&single_result, &para_result, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Reduce(&cpu_time, &cpu_time_buf, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
    if (myrank == root) {
        printf("[PARA] parallel computing result %lf\n", para_result); // other ranks will have para_result = -1;
        printf("[PARA] elapsed: %lf[s]\n", cpu_time_buf);
    }

    //
    // sequential calculations of integral
    //
    if(myrank == root){
        start = clock();
        seq_result = monte_carlo_seq(left_bound, right_bound, iterations);
        end = clock();
        cpu_time = ((double)(end-start))/CLOCKS_PER_SEC;
        printf("[SEQ] sequential computing result %lf\n", seq_result);
        printf("[SEQ] elapsed: %lf[s]\n", cpu_time);
    }

    MPI_Finalize();
}