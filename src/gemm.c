#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

/* sy_added */
static int lnum = 0;

void ysk_bngemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc,
		float* mean, float* variance, float* scale, float* biase)
{
	int8_t *bwgt, *bbis, *binp, *bout;
	float *inp_m;
    int i,j,k;
	float bn_a, bn_b;
    #pragma omp parallel for
	for(i = 0; i < M; ++i) {
		bn_a = mean[i];
		bn_b = sqrt(variance[i]) + 0.000001f;
		for(j = 0; j < N; ++j) {
			float sum = .0f;
			for(k = 0; k < K; ++k) {
				int8_t qb, qa;
				float bn_w, input;
				if (lnum == 0){
					bn_w = A[i*K+k];
					input = B[k*N+j]*(scale[i]/bn_b);
				} else {
					bn_w  = A[i*K+k]*(scale[i]/bn_b);
					input = B[k*N+j];
				}
				if (bn_w >= 1.984375) bn_w = 1.984375;
				else if (bn_w <= -2) bn_w = -2;
				if (input >= 15.875) input = 15.875;
				else if (input <= -16) input = -16;
				qa = ((int8_t)(bn_w*64));
				qb = ((int8_t)(input*8+0.5));

				// integer MAC
				int32_t qacc = (int32_t)(sum*(1<<16));
				int16_t qmult = qa * qb;
				qacc = (int32_t)qmult*(1<<7) + qacc;
				sum = (float) qacc / (1<<16);
			}
			float bn_biase = (biase[i] - ((bn_a/bn_b)*scale[i]));
			int8_t qbn_biase;
			if (bn_biase >= 7.937500) bn_biase = 7.937500; // 4-4
			else if (bn_biase <= -8) bn_biase = -8;
			qbn_biase = ((int8_t)(bn_biase*16));

			// integer MAC
			int32_t qacc = (int32_t)(sum*(1<<16));
			qacc = (int32_t)qbn_biase*(1<<12) + qacc;  // 12 = 16 - fraction_bit
			sum  = (float)qacc / (1<<16);
			C[i*N+j] = sum;
		}
	}
}

void ysk_gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc, const float* biase)
{
    int i,j,k;
	int8_t *bwgt, *bbis, *binp, *bout;
    #pragma omp parallel for
	for(i = 0; i < M; ++i){
		for(j = 0; j < N; ++j){
			float sum = .0f;
			for(k = 0; k < K; ++k){
				float wgt = A[i*K+k];
				float input = B[k*N+j];
				int8_t qa, qb;

				// will be added to functions.h
				if (wgt >= 1.984375) wgt = 1.984375;
				else if (wgt <= -2) wgt = -2;
				if (input >= 15.875) input = 15.875;
				else if (input <= -16) input = -16;
				qa = ((int8_t)(wgt*64)); // -2^7 ~
				qb = ((int8_t)(input*8+0.5)); // -2^7 ~
				// integer mult
				int32_t qacc = (int32_t)(sum*(1<<16)); //-2^31 ~
				int16_t qmult = qa * qb; // -2^14 ~ --> -2^5 ~
				qacc = (int32_t)qmult*(1<<7) + qacc;
				sum = (float) qacc / (1<<16);
			}
			int8_t qb;
			int32_t qacc;

			float q_bias = biase[i];
			if (q_bias >= 7.937500)
				q_bias = 7.937500;
			else if (q_bias <= -8)
				q_bias = -8;
			qb = (int8_t)(q_bias*16);
			qacc = (int32_t)(sum*(1<<16));
			qacc = (int32_t)qb*(1<<12) + qacc;
			sum  = (float) qacc / (1<<16);
			C[i*N+j] = sum;
		}
    }
}

void sy_bngemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc, 
		float* mean, float* variance, float* scale, float* biase, unsigned int toggle)
{
    sy_bngemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc, mean, variance, scale, biase, toggle);
}


void sy_bngemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc,
		float* mean, float* variance, float* scale, float* biase)
{
	int8_t *bwgt, *bbis, *binp, *bout;
	float *inp_m;
    int i,j,k;
	float bn_a, bn_b;
    #pragma omp parallel for
	for(i = 0; i < M; ++i) {
		bn_a = mean[i];
		bn_b = sqrt(variance[i]) + 0.000001f;
		for(j = 0; j < N; ++j) {
			float sum = .0f;
			for(k = 0; k < K; ++k) {
				int8_t qb, qa;
#if defined(QUANTIZE)
				float bn_w, input;
				if (lnum == 0){
					bn_w = A[i*K+k];
					input = B[k*N+j]*(scale[i]/bn_b);
				} else {
					bn_w  = A[i*K+k]*(scale[i]/bn_b);
					input = B[k*N+j];
				}
				if (bn_w >= 1.984375) bn_w = 1.984375;
				else if (bn_w <= -2) bn_w = -2;
				if (input >= 15.875) input = 15.875;
				else if (input <= -16) input = -16;
				qa = ((int8_t)(bn_w*64));
				qb = ((int8_t)(input*8+0.5));

				// integer MAC
				int32_t qacc = (int32_t)(sum*(1<<16));
				int16_t qmult = qa * qb;
				qacc = (int32_t)qmult*(1<<7) + qacc;
				sum = (float) qacc / (1<<16);
#else
				sum += A[i*K + k]*(scale[i]/bn_b)*B[k*N + j];
#endif
			}
			float bn_biase = (biase[i] - ((bn_a/bn_b)*scale[i]));
			int8_t qbn_biase;
#if defined(QUANTIZE)
			if (bn_biase >= 7.937500) bn_biase = 7.937500; // 4-4
			else if (bn_biase <= -8) bn_biase = -8;
			qbn_biase = ((int8_t)(bn_biase*16));

			// integer MAC
			int32_t qacc = (int32_t)(sum*(1<<16));
			qacc = (int32_t)qbn_biase*(1<<12) + qacc;  // 12 = 16 - fraction_bit
			sum  = (float)qacc / (1<<16);
			C[i*N+j] = sum;
#else
			C[i*N + j] += (sum + bn_biase);
#endif
		}
	}
}

void sy_gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc, const float* biase)
{
    int i,j,k;
	int8_t *bwgt, *bbis, *binp, *bout;
    #pragma omp parallel for
	for(i = 0; i < M; ++i){
		for(j = 0; j < N; ++j){
			float sum = .0f;
			for(k = 0; k < K; ++k){
				float wgt = A[i*K+k];
				float input = B[k*N+j];
				int8_t qa, qb;

#if defined(QUANTIZE)
				// will be added to functions.h
				if (wgt >= 1.984375) wgt = 1.984375;
				else if (wgt <= -2) wgt = -2;
				if (input >= 15.875) input = 15.875;
				else if (input <= -16) input = -16;
				qa = ((int8_t)(wgt*64)); // -2^7 ~
				qb = ((int8_t)(input*8+0.5)); // -2^7 ~
				// integer mult
				int32_t qacc = (int32_t)(sum*(1<<16)); //-2^31 ~
				int16_t qmult = qa * qb; // -2^14 ~ --> -2^5 ~
				qacc = (int32_t)qmult*(1<<7) + qacc;
				sum = (float) qacc / (1<<16);
#else
				sum += wgt*input;
#endif
			}
			int8_t qb;
			int32_t qacc;

#if defined(QUANTIZE)
			float q_bias = biase[i];
			if (q_bias >= 7.937500)
				q_bias = 7.937500;
			else if (q_bias <= -8)
				q_bias = -8;
			qb = (int8_t)(q_bias*16);
			qacc = (int32_t)(sum*(1<<16));
			qacc = (int32_t)qb*(1<<12) + qacc;
			sum  = (float) qacc / (1<<16);
			C[i*N+j] = sum;
#else
			C[i*N+j] += (sum + biase[i]);

#endif
		}
    }
}

void sy_bngemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc,
		float* mean, float* variance, float* scale, float* biase, unsigned int toggle)
{
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB){
		printf("layer %d launched\n", lnum);
		if (toggle == 1)
	        sy_bngemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc, mean, variance, scale, biase);
		else if (toggle == 2)
        	//sy_gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc, biase);
        	ysk_gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc, biase);
		else
			assert(0 && "mm launch failed\n");
	}
    else if(TA && !TB){
		assert(0);
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
	}
    else if(!TA && TB){
		assert(0);
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
	}
    else{
		assert(0);
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
	}
	if (lnum == 15)
		lnum = 0;
	else
		lnum++;
}
/* end added */

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}




void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB){
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
	}
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

#ifdef GPU

#include <math.h>

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,192,729,1600); 
       time_gpu(0,0,384,196,1728); 
       time_gpu(0,0,256,196,3456); 
       time_gpu(0,0,256,196,2304); 
       time_gpu(0,0,128,4096,12544); 
       time_gpu(0,0,128,4096,4096); 
     */
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,576,12544); 
    time_gpu(0,0,256,2304,784); 
    time_gpu(1,1,2304,256,784); 
    time_gpu(0,0,512,4608,196); 
    time_gpu(1,1,4608,512,196); 

    return 0;
}
#endif

