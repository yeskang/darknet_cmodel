#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
#include <stdint.h>
#include <unistd.h>
}

/* syoh added: cuda matmul based on fixed-point quantization */
#define BLOCK_SIZE 16
static int lnum = 0;
__global__ void quant_cuda_bnmm(float *a, float *b, float *c, 
		int m, int n, int k, float *mean, float *variance, float *scale, float* biase, const int lnum)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
	float bn_b = sqrtf(variance[row]) + 0.000001f;
	float bn_a = mean[row];
    if( col < k && row < m ){
        for(int i = 0; i < n; i++){  // n is k in normal matmul
			float bn_w, input;
			if (lnum == 0){
				bn_w = a[row*n+i];
				input = (b[i*k+col]/bn_b)*scale[row];
			} else {
				bn_w = (a[row*n+i]/bn_b)*scale[row];
				input = b[i*k+col];
			}
			// quantization
			if (bn_w >= 1.984375)  // 2-6
				bn_w = 1.984375;
			else if (bn_w <= -2)
				bn_w = -2;
			if (input >= 15.875)   // 5-3
				input = 15.875;
			else if (input <= -16)
				input = -16;
			int8_t qa = ((int8_t)(bn_w*64));
			int8_t qb = ((int8_t)(input*8+0.5));

			// integer MAC
			int32_t qacc = (int32_t)(sum*(1<<16));
			int16_t qmult = qa * qb;
			qacc = (int32_t)qmult*(1<<7) + qacc;
			sum = (float) qacc / (1<<16);
		}
		float bn_biase   = (biase[row] - ((bn_a/bn_b)*scale[row]));
		// quantization
		if (bn_biase >= 7.937500) // 4-4
			bn_biase = 7.937500;
		else if (bn_biase <= -8)
			bn_biase = -8;
		int8_t qbn_biase = ((int8_t)(bn_biase*16));

		// integer MAC
		int32_t qacc = (int32_t)(sum*(1<<16));
		qacc = (int32_t)qbn_biase*(1<<12) + qacc;  // 12 = 16 - fraction_bit
		sum  = (float)qacc / (1<<16);
        c[row * k + col] = sum;
        __syncthreads();
    }
} 


__global__ void quant_cuda_mm(float *a, float *b, float *c, int m, int n, int k, float* biase)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m ){
        for(int i = 0; i < n; i++){  // n is k in normal matmul
			/* original matmul */
            //sum += a[row*n+i] * b[i*k+col];

			/* quantization */
			float aval = a[row*n+i];
			float bval = b[i*k+col];
			if (aval >= 1.984375)
				aval = 1.984375;
			else if (aval <= -2)
				aval = -2;
			if (bval >= 15.875)
				bval = 15.875;
			else if (bval <= -16)
				bval = -16;
			int8_t qa = ((int8_t)(aval*64)); // -2^7 ~
			int8_t qb = ((int8_t)(bval*8+0.5)); // -2^7 ~
			// integer mult
			int32_t qacc = (int32_t)(sum*(1<<16)); //-2^31 ~
			int16_t qmult = qa * qb; // -2^14 ~ --> -2^5 ~
			qacc = (int32_t)qmult*(1<<7) + qacc;
			sum = (float) qacc / (1<<16);
		}
		float bn_biase   = biase[row];
		// quantization
		if (bn_biase >= 7.937500) // 4-4
			bn_biase = 7.937500;
		else if (bn_biase <= -8)
			bn_biase = -8;
		int8_t qbn_biase = ((int8_t)(bn_biase*16));
		
		// integer MAC
		int32_t qacc = (int32_t)(sum*(1<<16));
		qacc = (int32_t)qbn_biase*(1<<12) + qacc;  // 12 = 16 - fraction_bit
		sum  = (float)qacc / (1<<16);

        c[row * k + col] = sum;
        __syncthreads();
    }
} 

__global__ void quant_cuda_sqmm(float *d_a, float *d_b, float *d_result, int n) 
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int idx;
    float tmp = 0;

    for (int sub = 0; sub < gridDim.x; ++sub){
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n)
            tile_a[threadIdx.y][threadIdx.x] = 0;
        else
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n)
            tile_b[threadIdx.y][threadIdx.x] = 0;
        else
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) 
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        __syncthreads();
    }
    if(row < n && col < n)
        d_result[row * n + col] = tmp;
}

void quant_gpu_mm(int m, int k, int n, float* d_a, float* d_b, float* d_c, float* biase){
	unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    if(m == k && k == n){
        quant_cuda_sqmm<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, k);
	}
    else{
        quant_cuda_mm<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, k, n, biase);
	}
    cudaThreadSynchronize();
}

void quant_gpu_bnmm(int m, int k, int n, float* d_a, float* d_b, float* d_c, float* mean, float* variance, float* scale, float* biase){
	unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	if (lnum == 15)
		lnum = 0;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    if(m == k && k == n){
        quant_cuda_sqmm<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, k);
	}
    else{
        quant_cuda_bnmm<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, k, n, mean, variance, scale, biase, lnum++);
	}
    cudaThreadSynchronize();
}
/* end added */

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += fabsf(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += fabsf(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}

// DONE_180725
float* get_qinput(const int m, const int channel, const int height, const int width, float* input, float* inputtest){
	printf("lnum: %d, quantizing input.. dim info ( m: %d, c: %d, h: %d, w: %d )\n", lnum, m, channel, height, width);
	float* gtocinput = (float*)malloc(channel*height*width*sizeof(float));
	int8_t* qinput   = (int8_t*)malloc(channel*height*width*sizeof(int8_t));
	cuda_pull_array(input, gtocinput, channel*height*width);

	// scale, var, mean, biases: size M
	for (int c=0; c<channel; c++){ // channel - height - width
		for(int h=0; h<height; h++){
			for(int w=0; w<width; w++){
				int access = (c*height + h)*width + w;
				float input = gtocinput[access];
				if (input >= 15.875)   // 5-3
					input = 15.875;
				else if (input <= -16)
					input = -16;
				int8_t qb = ((int8_t)(input*8+0.5));
				qinput[access] = qb;
				inputtest[access] = gtocinput[access];
			}
		}
	}
	char buf_bnw[256];
	sprintf(buf_bnw, "./DAC_final_weights_180726/input_raw_%d.data", lnum);
	FILE* bnw_fp = fopen(buf_bnw,    "w");
	fwrite(qinput, sizeof(int8_t),  channel*height*width, bnw_fp);
	free(qinput);
	free(gtocinput);
	fclose(bnw_fp);
	if (lnum == 15)
		sleep(10);
	float* ctoginput = cuda_make_array(inputtest, channel*height*width);
	return ctoginput;
}

void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(net.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        net.input_gpu = l.binary_input_gpu;
    }

#ifdef CUDNN
    float one = 1;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                net.input_gpu,
                l.weightDesc,
                l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);
#else
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;

			float *test = (float*)malloc(l.c*l.w*l.h*sizeof(float));
			float *newinput = get_qinput(l.n, l.c, l.h, l.w, net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w, test); // validated
            //im2col_gpu(net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w,
            //  l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            im2col_gpu(newinput,
                l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);

			/* syoh: toggle to custom cuda matmul (quant added) */
            //gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
			if (l.batch_normalize)
				quant_gpu_bnmm(m, k, n, a, b, c, l.rolling_mean_gpu, l.rolling_variance_gpu, l.scales_gpu, l.biases_gpu); // batchnorm_combined 
			else
				quant_gpu_mm(m, k, n, a, b, c, l.biases_gpu); // orig cuda_mm
		}
	}
#endif
	/*
    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
	*/
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}

__global__ void smooth_kernel(float *x, int n, int w, int h, int c, int size, float rate, float *delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -(size/2.f);
    int h_offset = -(size/2.f);

    int out_index = j + w*(i + h*(k + c*b));
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w*(cur_h + h*(k + b*c));
            int valid = (cur_h >= 0 && cur_h < h &&
                    cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
        }
    }
}

extern "C" void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
    check_error(cudaPeekAtLastError());
}

void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    if(l.smooth){
        smooth_layer(l, 5, l.smooth);
    }
    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);


    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    float *original_input = net.input_gpu;

    if(l.xnor) net.input_gpu = l.binary_input_gpu;
#ifdef CUDNN
    float one = 1;
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            net.input_gpu,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            net.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc,
            l.weight_updates_gpu);

    if(net.delta_gpu){
        if(l.binary || l.xnor) swap_binary(&l);
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.weightDesc,
                l.weights_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                net.delta_gpu);
        if(l.binary || l.xnor) swap_binary(&l);
        if(l.xnor) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
    }

#else
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    int i, j;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta_gpu + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates_gpu + j*l.nweights/l.groups;

            float *im = net.input_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;

            im2col_gpu(im, l.c/l.groups, l.h, l.w,
                    l.size, l.stride, l.pad, b);
            gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if(net.delta_gpu){
                if(l.binary || l.xnor) swap_binary(&l);
                a = l.weights_gpu + j*l.nweights/l.groups;
                b = l.delta_gpu + (i*l.groups + j)*m*k;
                c = net.workspace;

                gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);

                col2im_gpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, 
                    l.pad, net.delta_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w);
                if(l.binary || l.xnor) {
                    swap_binary(&l);
                }
            }
            if(l.xnor) gradient_array_gpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w);
        }
    }
#endif
}

void pull_convolutional_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_convolutional_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void update_convolutional_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);

        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
}


