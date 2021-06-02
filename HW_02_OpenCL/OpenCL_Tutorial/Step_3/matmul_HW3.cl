// HW_3
__kernel void matmul_HW3(const int M, 
						const int N, 
						const int K,
						const __global float* A,
						const __global float* B,
						__global float* C) {
    
    // Thread identifiers
    const int tidx = get_global_id(0); // Row ID of C (0..M)
    const int tidy = get_global_id(1); // Col ID of C (0..N)

    // Compute a single element (loop over K)
    float Csub = 0.0f;
    for (int k=0; k<K; k++) {
        Csub += A[k * M + tidx] * B[tidy * K + k];
    }

    // Store the result
    C[tidy*M + tidx] = Csub;
}
