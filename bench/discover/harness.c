// Times tang_run() of a generated unit; inputs s ~ N(0, 3²), v ~ 4 + N(0,1).
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
void tang_run(void);
void *tang_input(int); int tang_input_len(int);
void *tang_output(int); int tang_output_len(int);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static float nrm(void){double u=(rand()+1.0)/(RAND_MAX+2.0),w=rand()/(RAND_MAX+1.0);return sqrt(-2*log(u))*cos(6.283185307*w);}
int main(void){
  srand(1);
  float*s=tang_input(0),*v=tang_input(1); int n=tang_input_len(0);
  for(int i=0;i<n;i++){s[i]=3*nrm();v[i]=4+nrm();}
  tang_run();
  double best=1e9; for(int r=0;r<20;r++){double t=now();tang_run();t=now()-t;if(t<best)best=t;}
  float*o=tang_output(0); int k=tang_output_len(0); double cs=0; for(int i=0;i<k;i++)cs+=o[i];
  printf("%.3f ms  checksum %.6g  (%.1f GB/s of s,v)\n",best*1e3,cs,2.0*n*4/best/1e9);
}
