// HW_3
__kernel void matmul_HW3( 
		const int M,
		const int N,
		const int K,
		const __global float *A, 
		const __global float *B, 
		__global float *C)
{
	int tidx = get_global_id(0); // i
	int tidy = get_global_id(1); // j

	if (tidx < M && tidy < N)
	{
		float Csub = 0.0f;
		for(int k = 0; k < K; k++) // k
			Csub += A[k * M + tidx] * B[tidy * K + k];

		C[tidy * M + tidx] = Csub;
	}
}

