// HW_4
__kernel void matmul_HW4( 
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

		for(int k = 0; k < K; k += 8) // k
		{
			if (k < K)
			{
				Csub += A[(k + 0) * M + tidx] * B[tidy * K + (k + 0)];
				Csub += A[(k + 1) * M + tidx] * B[tidy * K + (k + 1)];
				Csub += A[(k + 2) * M + tidx] * B[tidy * K + (k + 2)];
				Csub += A[(k + 3) * M + tidx] * B[tidy * K + (k + 3)];
				Csub += A[(k + 4) * M + tidx] * B[tidy * K + (k + 4)];
				Csub += A[(k + 5) * M + tidx] * B[tidy * K + (k + 5)];
				Csub += A[(k + 6) * M + tidx] * B[tidy * K + (k + 6)];
				Csub += A[(k + 7) * M + tidx] * B[tidy * K + (k + 7)];

			}
		}

		C[tidx * N + tidy] = Csub;
	}
}

