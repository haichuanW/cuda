#include <stdio.h>
#include <math.h>
#include <omp.h>

long int NSTEPS=10000000;

/*
 * This program computes pi as
 * \pi = 4 arctan(1)
 *     = 4 \int _0 ^1 \frac{1} {1 + x^2} dx
 */
int main(int argc, char** argv)
{
    long i;
    double dx = 1.0 / NSTEPS;
    double pi = 0.0;
    // double localsum = 0;
 
    omp_set_num_threads(4);

    int p = omp_get_num_threads();

    double start_time = omp_get_wtime();

#pragma omp parallel shared(dx) private(i) 
{
    int pid = omp_get_thread_num();

    int myfirst = NSTEPS/p*pid;
    int mylast = NSTEPS/p*(pid+1);
    
    double localsum = 0;
 
    for (i = myfirst; i < mylast; i++)
    {
        double x = (i + 0.5) * dx;
        
        localsum += 4.0 / (1.0 + x * x)*dx;

 
    }
    // printf("%lf\n",localsum);
    for(int j=pid;j<p;++j){
        #pragma omp critical 
        pi+=localsum;

    }



}

 

    double run_time = omp_get_wtime() - start_time;
    double ref_pi = 4.0 * atan(1.0);

    printf("pi with %ld steps is %.10f in %.6f seconds (error=%e)\n",
           NSTEPS, pi, run_time, fabs(ref_pi - pi));

    

    return 0;
}
