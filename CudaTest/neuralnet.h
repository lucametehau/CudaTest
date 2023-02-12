#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <random>
#include <cstring>
#include <cassert>
#include <type_traits>
#include <thread>
#include <mutex>
#include <atomic>
#include <immintrin.h>
#include <inttypes.h>
#include <omp.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include "chess.h"

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

using namespace std;

const string LOG_PATH0 = "log0.txt";
const string LOG_PATH1 = "log1.txt";
ofstream log0(LOG_PATH0);
ofstream log1(LOG_PATH1);

const int INPUT_NEURONS = 3072;
const int SIDE_NEURONS = 768;
const int HIDDEN_NEURONS = 2 * SIDE_NEURONS;

const float BETA1 = 0.9;
const float BETA2 = 0.999;
const float SIGMOID_SCALE = 0.00447111749925f;
float LR = 0.01f;

const int NO_ACTIV = 0;
const int SIGMOID = 1;
const int RELU = 2;

string testPos[9] = {
    "3k4/8/8/8/8/8/8/2QK4 w", ///KQvK
    "3k4/8/8/8/8/8/8/2RK4 w", ///KRvK
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w", /// startpos
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b", /// startpos e2e4
    "2r3k1/5pp1/r7/Np5P/1P2pP2/3nP2q/3Q4/R2R2K1 w", /// king safety
    "8/8/4k3/2B2n2/6K1/5p2/8/5q2 w", /// something random
    "r1bqkbnr/pppppppp/2n5/8/3P4/8/PPPKPPPP/RNBQ1BNR b kq - 2 2", /// weird position
    "2k5/8/1P6/2KN3P/8/8/8/8 w - - 1 71", /// again
    "2b1r3/r4ppk/p7/2pNP3/8/q6P/2P2PP1/3RR1K1 w - - 0 2" /// idk
};

mutex M;

namespace tools {
    mt19937_64 gen(time(0));
    uniform_int_distribution <int> bin(0, 1);
    uniform_int_distribution <uint64_t> integer;
};

struct OutputValues {
    float* outputstm;
    float* outputopstm;

    void init() {
        cudaMallocManaged(&outputopstm, SIDE_NEURONS * sizeof(float));
        cudaMallocManaged(&outputstm, SIDE_NEURONS * sizeof(float));
    }

    void free() {
        cudaFree(outputopstm);
        cudaFree(outputstm);
    }
};

struct Gradient {
    float m1, m2;
};

struct OutputValues_CPU {
    float outputstm[SIDE_NEURONS];
    float outputopstm[SIDE_NEURONS];
};

struct ThreadGradients_cpu {
    void reset() {
        memset(inputBiasesGradients, 0, SIDE_NEURONS * sizeof(float));
        memset(inputWeightsGradients, 0, INPUT_NEURONS * SIDE_NEURONS * sizeof(float));
        memset(outputWeightsGradients, 0, HIDDEN_NEURONS * sizeof(float));
        memset(outputBiasGradient, 0, sizeof(float));
    }

    void backprop(int16_t* inputs_stm, int16_t* inputs_opstm, float output_stm[], float output_opstm[], float outputError, float outputWeights[]) {
        float error_stm[SIDE_NEURONS], error_opstm[SIDE_NEURONS];

        for (int i = 0; i < SIDE_NEURONS; i++) {
            error_stm[i] = outputError * output_stm[i];
            outputWeightsGradients[i] += error_stm[i];
            error_stm[i] *= outputWeights[i];
        }

        for (int i = 0; i < SIDE_NEURONS; i++) {
            error_opstm[i] = outputError * output_opstm[i];
            outputWeightsGradients[i + SIDE_NEURONS] += error_opstm[i];
            error_opstm[i] *= outputWeights[i + SIDE_NEURONS];
        }

        for (int i = 0; i < SIDE_NEURONS; i++)
            inputBiasesGradients[i] += error_stm[i] + error_opstm[i];

        outputBiasGradient[0] += outputError;
        for (int i = 0; i < 32; i++) {
            if (inputs_stm[i] == -1)
                break;
            int n0 = inputs_stm[i] * SIDE_NEURONS;
            for (int j = 0; j < SIDE_NEURONS; j++)
                inputWeightsGradients[n0 + j] += error_stm[j];

            n0 = inputs_opstm[i] * SIDE_NEURONS;
            for (int j = 0; j < SIDE_NEURONS; j++)
                inputWeightsGradients[n0 + j] += error_opstm[j];
        }
    }

    float inputBiasesGradients[SIDE_NEURONS];
    float inputWeightsGradients[INPUT_NEURONS * SIDE_NEURONS];
    float outputBiasGradient[1];
    float outputWeightsGradients[HIDDEN_NEURONS];
};

class Network {
public:
    ThreadGradients_cpu thg;
    Network() {
        float k = sqrt(2.0 / INPUT_NEURONS);
        normal_distribution <float> rng(0, k);
        for (int i = 0; i < INPUT_NEURONS; i++) {
            for (int j = 0; j < SIDE_NEURONS; j++) {
                inputWeights[i * SIDE_NEURONS + j] = rng(tools::gen);
            }
        }

        k = sqrt(2.0 / HIDDEN_NEURONS);
        normal_distribution <float> rng2(0, k);

        for (int i = 0; i < HIDDEN_NEURONS; i++)
            outputWeights[i] = rng2(tools::gen);

        memset(inputBiases, 0, sizeof(inputBiases));
        outputBias[0] = 0;

        inputs_stm = (int16_t*)malloc(32 * sizeof(int16_t));
        inputs_opstm = (int16_t*)malloc(32 * sizeof(int16_t));
    }

    float activationFunction(float x, int type) {
        if (type == RELU)
            return max(x, 0.0f);

        return 1.0f / (1.0f + exp(-SIGMOID_SCALE * x));
    }

    float activationFunctionDerivative(float x, int type) {
        if (type == RELU) {
            return (x > 0);
        }

        //float value = activationFunction(x, type);
        return x * (1 - x) * SIGMOID_SCALE;
    }

    float inverseSigmoid(float val) {
        return log(val / (1 - val)) / SIGMOID_SCALE;
    }

    float feedForward(NetInput& input) {
        setInput(inputs_stm, inputs_opstm, input);
        memcpy(outputstm, inputBiases, sizeof(outputstm));
        memcpy(outputopstm, inputBiases, sizeof(outputopstm));

        for (int i = 0; i < 32; i++) {
            //std::cout << input_v.v[0][i] / 128 << " " << (input_v.v[0][i] % 128) / 64 << " " << (input_v.v[0][i] % 64) << '\n';
            if (inputs_stm[i] == -1)
                break;
            int n0 = inputs_stm[i] * SIDE_NEURONS;
            for (int j = 0; j < SIDE_NEURONS; j++)
                outputstm[j] += inputWeights[n0 + j];

            n0 = inputs_opstm[i] * SIDE_NEURONS;
            for (int j = 0; j < SIDE_NEURONS; j++)
                outputopstm[j] += inputWeights[n0 + j];
        }

        float sum = outputBias[0];

        for (int i = 0; i < SIDE_NEURONS; i++) {
            //std::cout << outputstm[i] << " si " << inputBiases[i] << "\n";
            outputstm[i] = max<float>(0.0, outputstm[i]);
            outputopstm[i] = max<float>(0.0, outputopstm[i]);
        }

        for (int i = 0; i < SIDE_NEURONS; i++) {
            sum += outputstm[i] * outputWeights[i];
        }

        for (int i = 0; i < SIDE_NEURONS; i++) {
            sum += outputopstm[i] * outputWeights[i + SIDE_NEURONS];
        }

        //log0 << " " << sum << " " << activationFunction(sum, SIGMOID) << "\n";
        //cout << sum << " xdxdxdxdxdxd\n";

        return activationFunction(sum, SIGMOID);
    }

    float feedForward(int16_t* input_stm, int16_t* input_opstm, float target, int idx) {
        memcpy(outputstm, inputBiases, sizeof(outputstm));
        memcpy(outputopstm, inputBiases, sizeof(outputopstm));

        for (int i = 0; i < 32; i++) {
            //std::cout << input_v.v[0][i] / 128 << " " << (input_v.v[0][i] % 128) / 64 << " " << (input_v.v[0][i] % 64) << '\n';
            if (input_stm[i] == -1)
                break;
            int n0 = input_stm[i] * SIDE_NEURONS;
            for (int j = 0; j < SIDE_NEURONS; j++)
                outputstm[j] += inputWeights[n0 + j];

            n0 = input_opstm[i] * SIDE_NEURONS;
            for (int j = 0; j < SIDE_NEURONS; j++)
                outputopstm[j] += inputWeights[n0 + j];
        }

        float sum = outputBias[0];

        for (int i = 0; i < SIDE_NEURONS; i++) {
            //std::cout << outputstm[i] << " si " << inputBiases[i] << "\n";
            outputstm[i] = max<float>(0.0, outputstm[i]);
            outputopstm[i] = max<float>(0.0, outputopstm[i]);
        }

        for (int i = 0; i < SIDE_NEURONS; i++) {
            sum += outputstm[i] * outputWeights[i];
        }

        for (int i = 0; i < SIDE_NEURONS; i++) {
            sum += outputopstm[i] * outputWeights[i + SIDE_NEURONS];
        }

        sum = activationFunction(sum, SIGMOID);
        float outputError = 2 * (sum - target) * activationFunctionDerivative(sum, SIGMOID);
        //cout << outputError << "\n";
        thg.backprop(input_stm, input_opstm, outputstm, outputopstm, outputError, outputWeights);
        //cout << "done with all\n";
        //log0 << idx << " " << outputError << "\n";
        //cout << sum << " xdxdxdxdxdxd\n";

        return activationFunction(sum, SIGMOID);
    }

    void save(string path) {
        FILE* f = fopen(path.c_str(), "wb");
        int cnt = 3, x;

        x = fwrite(&cnt, sizeof(int), 1, f);
        assert(x == 1);

        int sz = SIDE_NEURONS;

        x = fwrite(inputBiases, sizeof(float), sz, f);
        assert(x == sz);

        x = fwrite(inputBiasesGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        sz = INPUT_NEURONS * SIDE_NEURONS;
        x = fwrite(inputWeights, sizeof(float), sz, f);
        assert(x == sz);

        x = fwrite(inputWeightsGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        sz = 1;

        x = fwrite(outputBias, sizeof(float), sz, f);
        assert(x == sz);

        x = fwrite(outputBiasGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        x = fwrite(outputWeights, sizeof(float), HIDDEN_NEURONS, f);
        assert(x == HIDDEN_NEURONS);

        x = fwrite(outputWeightsGrad, sizeof(Gradient), HIDDEN_NEURONS, f);
        assert(x == HIDDEN_NEURONS);

        fclose(f);
    }

    void load(string path) {
        FILE* f = fopen(path.c_str(), "rb");
        int cnt = 3, x;

        x = fread(&cnt, sizeof(int), 1, f);
        assert(x == 1);

        int sz = SIDE_NEURONS;

        x = fread(inputBiases, sizeof(float), sz, f);
        assert(x == sz);

        x = fread(inputBiasesGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        sz = INPUT_NEURONS * SIDE_NEURONS;
        x = fread(inputWeights, sizeof(float), sz, f);
        assert(x == sz);

        x = fread(inputWeightsGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        sz = 1;

        x = fread(outputBias, sizeof(float), sz, f);
        assert(x == sz);

        x = fread(outputBiasGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        x = fread(outputWeights, sizeof(float), HIDDEN_NEURONS, f);
        assert(x == HIDDEN_NEURONS);

        x = fread(outputWeightsGrad, sizeof(Gradient), HIDDEN_NEURONS, f);
        assert(x == HIDDEN_NEURONS);

        fclose(f);
    }

    int evaluate(string fen) {
        NetInput input = fenToInput(fen);

        float ans = feedForward(input);
        cout << "Fen: " << fen << " ; stm = " << input.stm << " ; eval = " << inverseSigmoid(ans) << "\n";

        return int(inverseSigmoid(ans));
    }

    void evalTestPos() {
        for (int i = 0; i < 9; i++) {
            evaluate(testPos[i]);
        }
    }

    float outputstm[SIDE_NEURONS];
    float outputopstm[SIDE_NEURONS];
    float inputBiases[SIDE_NEURONS], outputBias[1];
    float inputWeights[INPUT_NEURONS * SIDE_NEURONS];
    float outputWeights[HIDDEN_NEURONS];

    Gradient inputWeightsGrad[INPUT_NEURONS * SIDE_NEURONS], outputWeightsGrad[HIDDEN_NEURONS];
    Gradient inputBiasesGrad[SIDE_NEURONS], outputBiasGrad[1];

    int16_t* inputs_stm, * inputs_opstm;
};

__global__ void check(Gradient* g, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        if (g[i].m1 != 0.0 || g[i].m2 != 0.0) {
            printf("%d\n", i);
        }
    }
}

struct GPU_Network {
    float* inputBiases;
    float* outputBias;
    float* inputWeights;
    float* outputWeights;

    Gradient* inputWeightsGrad, * outputWeightsGrad;
    Gradient* inputBiasesGrad;
    Gradient* outputBiasGrad;

    void init() {
        float* empty_vec;
        empty_vec = (float*)malloc(SIDE_NEURONS * sizeof(float));
        for (int i = 0; i < SIDE_NEURONS; i++)
            empty_vec[i] = 0;

        normal_distribution <float> rng1(0, sqrt(2.0 / INPUT_NEURONS)), rng2(0, sqrt(2.0 / HIDDEN_NEURONS));
        float* aux1, * aux2;
        aux1 = (float*)malloc(INPUT_NEURONS * SIDE_NEURONS * sizeof(float));
        aux2 = (float*)malloc(HIDDEN_NEURONS * sizeof(float));
        for (int i = 0; i < INPUT_NEURONS * SIDE_NEURONS; i++)
            aux1[i] = rng1(tools::gen);
        for (int i = 0; i < HIDDEN_NEURONS; i++)
            aux2[i] = rng2(tools::gen);

        cudaDeviceSynchronize();
        cudaMallocManaged(&inputWeights, INPUT_NEURONS * SIDE_NEURONS * sizeof(float)), cudaMemcpy(inputWeights, aux1, INPUT_NEURONS * SIDE_NEURONS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMallocManaged(&outputWeights, HIDDEN_NEURONS * sizeof(float)), cudaMemcpy(outputWeights, aux2, HIDDEN_NEURONS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMallocManaged(&inputBiases, SIDE_NEURONS * sizeof(float)), cudaMemcpy(inputBiases, empty_vec, SIDE_NEURONS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMallocManaged(&outputBias, sizeof(float));

        cudaDeviceSynchronize();

        cudaMallocManaged(&inputWeightsGrad, INPUT_NEURONS * SIDE_NEURONS * sizeof(Gradient));
        cudaMallocManaged(&outputWeightsGrad, HIDDEN_NEURONS * sizeof(Gradient));
        cudaMallocManaged(&inputBiasesGrad, SIDE_NEURONS * sizeof(Gradient));
        cudaMallocManaged(&outputBiasGrad, sizeof(Gradient));

        cudaDeviceSynchronize();
    }

    void save(string path) {
        FILE* f = fopen(path.c_str(), "wb");
        int cnt = 3, x;

        x = fwrite(&cnt, sizeof(int), 1, f);
        assert(x == 1);

        int sz = SIDE_NEURONS;

        x = fwrite(inputBiases, sizeof(float), sz, f);
        assert(x == sz);

        x = fwrite(inputBiasesGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        sz = INPUT_NEURONS * SIDE_NEURONS;
        x = fwrite(inputWeights, sizeof(float), sz, f);
        assert(x == sz);

        x = fwrite(inputWeightsGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        sz = 1;

        x = fwrite(outputBias, sizeof(float), sz, f);
        assert(x == sz);

        x = fwrite(outputBiasGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        x = fwrite(outputWeights, sizeof(float), HIDDEN_NEURONS, f);
        assert(x == HIDDEN_NEURONS);

        x = fwrite(outputWeightsGrad, sizeof(Gradient), HIDDEN_NEURONS, f);
        assert(x == HIDDEN_NEURONS);

        fclose(f);
    }

    void load(string path) {
        FILE* f = fopen(path.c_str(), "rb");
        int cnt = 3, x;

        x = fread(&cnt, sizeof(int), 1, f);
        assert(x == 1);

        int sz = SIDE_NEURONS;

        x = fread(inputBiases, sizeof(float), sz, f);
        assert(x == sz);

        x = fread(inputBiasesGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        sz = INPUT_NEURONS * SIDE_NEURONS;
        x = fread(inputWeights, sizeof(float), sz, f);
        assert(x == sz);

        x = fread(inputWeightsGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        sz = 1;

        x = fread(outputBias, sizeof(float), sz, f);
        assert(x == sz);

        x = fread(outputBiasGrad, sizeof(Gradient), sz, f);
        assert(x == sz);

        x = fread(outputWeights, sizeof(float), HIDDEN_NEURONS, f);
        assert(x == HIDDEN_NEURONS);

        x = fread(outputWeightsGrad, sizeof(Gradient), HIDDEN_NEURONS, f);
        assert(x == HIDDEN_NEURONS);

        fclose(f);
    }
};

struct ThreadGradients {
    void reset() {
        cudaMemset(inputBiasesGradients, 0, SIDE_NEURONS * sizeof(float));
        cudaMemset(inputWeightsGradients, 0, INPUT_NEURONS * SIDE_NEURONS * sizeof(float));
        cudaMemset(outputWeightsGradients, 0, HIDDEN_NEURONS * sizeof(float));
        cudaMemset(outputBiasGradient, 0, sizeof(float));
    }
    void init() {
        cudaMallocManaged(&inputBiasesGradients, SIDE_NEURONS * sizeof(float));
        cudaMallocManaged(&inputWeightsGradients, INPUT_NEURONS * SIDE_NEURONS * sizeof(float));
        cudaMallocManaged(&outputWeightsGradients, HIDDEN_NEURONS * sizeof(float));
        cudaMallocManaged(&outputBiasGradient, sizeof(float));

        reset();
    }

    void free() {
        cudaFree(inputBiasesGradients);
        cudaFree(inputWeightsGradients);
        cudaFree(outputWeightsGradients);
        cudaFree(outputBiasGradient);
    }

    float* inputBiasesGradients;
    float* outputBiasGradient;
    float* inputWeightsGradients;
    float* outputWeightsGradients;
};

#define BUCKET_SIZE 3
const int FF_BATCH_SIZE = SIDE_NEURONS / BUCKET_SIZE;
//const int UPDATE_BATCH_SIZE = 512;

cublasHandle_t handle;

float* getVectorFromCuda(float* V, int N) {
    float* v = (float*)malloc(N * sizeof(float));
    cudaMemcpy(v, V, N * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

__global__ void updateWeights(float* weights, Gradient* grads, float* newGrads, int N, float LR, float BETA1, float BETA2) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N) {
        float grad = newGrads[i];
        grads[i].m1 = grads[i].m1 * BETA1 + grad * (1.0 - BETA1);
        grads[i].m2 = grads[i].m2 * BETA2 + grad * grad * (1.0 - BETA2);

        weights[i] -= LR * grads[i].m1 / (sqrt(grads[i].m2) + 1e-8f);
        newGrads[i] = 0;
    }
}

void cublas_mat_mult(float* A, float* B, float* C, int M, int K, int N, const float add_prev) {
    const float alpha = 1.0;
    const float* alpha_p = &alpha, * beta_p = &add_prev;
    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha_p, B, N, A, K, beta_p, C, N);
    assert(status == 0);
}

__global__ void feedForwardBatch(float* inputBiases, float* inputWeights, float* output, int16_t* inputs_, int l, int r) {
    int idx = blockIdx.x;

    if (idx + l >= r)
        return;

    //const int SIZE = SIDE_NEURONS / FF_BATCH_SIZE;
    int offset = threadIdx.x * BUCKET_SIZE;

    __shared__ float output_shared[SIDE_NEURONS];
    float* output_shared_bucket = output_shared + offset;
    float* output_bucket = output + idx * SIDE_NEURONS + offset;
    float* inputBiases_ = inputBiases + offset;


#pragma unroll
    for (int i = 0; i < BUCKET_SIZE; i++) {
        output_shared_bucket[i] = inputBiases_[i];
    }

    int inputs_offset = idx * 32;
    int16_t* inputs = inputs_ + inputs_offset;

    for (int i = 0; i < 32; i++) {
        if (inputs[i] == -1)
            break;
        float* inputWeights_bucket = inputWeights + inputs[i] * SIDE_NEURONS + offset;
#pragma unroll
        for (int j = 0; j < BUCKET_SIZE; j++)
            output_shared_bucket[j] += inputWeights_bucket[j];
    }

#pragma unroll
    for (int i = 0; i < BUCKET_SIZE; i++) {
        output_bucket[i] = (output_shared_bucket[i] > 0 ? output_shared_bucket[i] : 0);
    }
}

__global__ void addOutputBias(float* sums, float* outputBias, float* outputs, float* outputErrors, float* loss, int l, int r) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (l + idx >= r)
        return;

    constexpr float SIGMOID_SCALE = 0.00447111749925f;
    float output = sums[idx];

    output += outputBias[0];
    output = 1.0f / (1.0f + exp(-SIGMOID_SCALE * output));
    outputErrors[idx] = 2 * (output - outputs[idx]) * output * (1.0 - output) * SIGMOID_SCALE;

    loss[idx] = (output - outputs[idx]) * (output - outputs[idx]);
    sums[idx] = 0;
}

void cudaMemory() {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);

    printf("%lld free out of %lld\n", free, total);
}
// (1 * BATCH_SIZE) x (BATCH_SIZE * SIDE_NEURONS)
// error[i][j] = outputError[i] * output[i][j]

__global__ void backPropBatch(float* inputBiasesGradients, float* inputWeightsGradients, 
    float* outputWeights, float* outputWeightsGradients, float* outputErrors, float* output, int16_t* inputs_, int l, int r) {
    int idx = blockIdx.x;
    //printf("%d %d\n", idx, ty);
    if (idx + l >= r)
        return;

    int offset = threadIdx.x * BUCKET_SIZE;
    float* new_output = output + idx * SIDE_NEURONS + offset;
    float* outputWeightsGradients_ = outputWeightsGradients + offset;
    float* inputBiasesGradients_ = inputBiasesGradients + offset;
    float* outputWeights_ = outputWeights + offset;
    float outputError = outputErrors[idx];
    __shared__ float error_shared[SIDE_NEURONS];
    float* error_bucket = error_shared + offset;
    int inputs_offset = idx * 32;
    int16_t* inputs = inputs_ + inputs_offset;

    for (int j = 0; j < BUCKET_SIZE; j++) {
        error_bucket[j] = outputError * new_output[j];
        if (error_bucket[j] == 0)
            continue;
        atomicAdd(&outputWeightsGradients_[j], error_bucket[j]);
        error_bucket[j] *= outputWeights_[j];
        atomicAdd(&inputBiasesGradients_[j], error_bucket[j]);
        for (int i = 0; i < 32; i++) {
            if (inputs[i] == -1)
                break;
            float* inputWeights_bucket = inputWeightsGradients + inputs[i] * SIDE_NEURONS + offset;
            atomicAdd(&inputWeights_bucket[j], error_bucket[j]);
        }
    }
}

__global__ void updateBiasGrad(float* biasGrad, float* outputErrors, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
        atomicAdd(&biasGrad[0], outputErrors[i]);
}
// pt fiecare idx:
// error = output_error[idx] * output[idx]
// owg[j] += error[j]
// owg[j] = sum(idx = 0...INPUT-1, output_error[idx] * output[idx][j])
// (1 x BATCH) * (BATCH x SIDE) = (1 x SIDE)

// error[i][j] = outputError[i] * output[i][j]
// outputWeightGradients[j] = sum(error[i][j]) = sum(outputError[i] * output[i][j])
// inputBiasGradients[j] = sum(outputError[i] * output[i][j] * outputWeights[j])

const int BATCH_SIZE = 16384 * 4;
const int BATCHES_TO_LOAD = 64;
const int LOAD_SIZE = BATCH_SIZE * BATCHES_TO_LOAD;

GPU_Network* nn;
ThreadGradients* grads;
Network cpu_nn;
float* outputstm, *outputopstm;
int16_t* inputs_gpu_stm, * inputs_gpu_opstm;
float* outputs_gpu;
float* sums, * outputErrors, * loss;
int16_t* inputs_stm, * inputs_opstm;
float* outputs;

void prealloc_on_cuda() {
    cudaMallocManaged(&nn, sizeof(GPU_Network));
    nn->init();

    cudaMallocManaged(&grads, sizeof(ThreadGradients));
    grads->init();
    cudaMallocManaged(&outputstm, BATCH_SIZE * SIDE_NEURONS * sizeof(float));
    cudaMallocManaged(&outputopstm, BATCH_SIZE * SIDE_NEURONS * sizeof(float));


    cudaMallocManaged(&inputs_gpu_stm, LOAD_SIZE * 32 * sizeof(int16_t));
    cudaMallocManaged(&inputs_gpu_opstm, LOAD_SIZE * 32 * sizeof(int16_t));
    cudaMallocManaged(&outputs_gpu, LOAD_SIZE * sizeof(float));


    cudaMallocManaged(&sums, BATCH_SIZE * sizeof(float));
    cudaMallocManaged(&outputErrors, BATCH_SIZE * sizeof(float));

    cudaMallocManaged(&loss, BATCH_SIZE * sizeof(float));

    inputs_stm = (int16_t*)malloc(LOAD_SIZE * 32 * sizeof(int16_t));
    inputs_opstm = (int16_t*)malloc(LOAD_SIZE * 32 * sizeof(int16_t));
    outputs = (float*)malloc(LOAD_SIZE * sizeof(float));

    cudaMemset(sums, 0, BATCH_SIZE * sizeof(float));

    cublasCreate(&handle);

    cudaDeviceSynchronize();
}

void load_data(FILE* bin_file) {
    chessTraining::readNextBatch(bin_file, inputs_stm, inputs_opstm, outputs, LOAD_SIZE);

    //cudaMemset(sums, 0, SIZE * sizeof(float));
    cudaMemcpy(inputs_gpu_stm, inputs_stm, LOAD_SIZE * 32 * sizeof(int16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(inputs_gpu_opstm, inputs_opstm, LOAD_SIZE * 32 * sizeof(int16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(outputs_gpu, outputs, LOAD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
}

void trainOnBatch(GPU_Network*& nn, int l, int r, float& trainingLoss, FILE* bin_file, int ind) {
    int SIZE = r - l;
    int THREADS = min(SIZE, 1024);
    int16_t* inputs_gpu_stm_ = inputs_gpu_stm + ind;
    int16_t* inputs_gpu_opstm_ = inputs_gpu_opstm + ind;
    float* outputs_gpu_ = outputs_gpu + ind / 32;

    cudaDeviceSynchronize();
    feedForwardBatch << <SIZE, FF_BATCH_SIZE >> > (nn->inputBiases, nn->inputWeights, outputstm, inputs_gpu_stm_, l, r);
    cudaDeviceSynchronize();
    cublas_mat_mult(outputstm, nn->outputWeights, sums, SIZE, SIDE_NEURONS, 1, 0.0);
    cudaDeviceSynchronize();

    feedForwardBatch << <SIZE, FF_BATCH_SIZE >> > (nn->inputBiases, nn->inputWeights, outputopstm, inputs_gpu_opstm_, l, r);
    cudaDeviceSynchronize();
    cublas_mat_mult(outputopstm, &nn->outputWeights[SIDE_NEURONS], sums, SIZE, SIDE_NEURONS, 1, 1.0);
    cudaDeviceSynchronize();

    addOutputBias << <(SIZE + THREADS - 1) / THREADS, THREADS >> > (sums, nn->outputBias, outputs_gpu_, outputErrors, loss, l, r);
    cudaDeviceSynchronize();

    backPropBatch << <SIZE, FF_BATCH_SIZE>> > (grads->inputBiasesGradients, grads->inputWeightsGradients,
        nn->outputWeights, grads->outputWeightsGradients, outputErrors, outputstm, inputs_gpu_stm_, l, r);
    cudaDeviceSynchronize();
    backPropBatch << <SIZE, FF_BATCH_SIZE>> > (grads->inputBiasesGradients, grads->inputWeightsGradients,
        &nn->outputWeights[SIDE_NEURONS], &grads->outputWeightsGradients[SIDE_NEURONS], outputErrors, outputopstm, inputs_gpu_opstm_, l, r);
    cudaDeviceSynchronize();

    updateBiasGrad << <(SIZE + THREADS - 1) / THREADS, THREADS >> > (grads->outputBiasGradient, outputErrors, SIZE);
    cudaDeviceSynchronize();

    float* loss_cpu = (float*)malloc(SIZE * sizeof(float));

    cudaMemcpy(loss_cpu, loss, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < r - l; i++)
        trainingLoss += loss_cpu[i];

    //cudaDeviceSynchronize();
    updateWeights << <(INPUT_NEURONS * SIDE_NEURONS + 1023) / 1024, 1024 >> > (nn->inputWeights, nn->inputWeightsGrad, grads->inputWeightsGradients, INPUT_NEURONS * SIDE_NEURONS, LR, BETA1, BETA2);
    cudaDeviceSynchronize();
    updateWeights << <(HIDDEN_NEURONS + 1023) / 1024, 1024 >> > (nn->outputWeights, nn->outputWeightsGrad, grads->outputWeightsGradients, HIDDEN_NEURONS, LR, BETA1, BETA2);
    cudaDeviceSynchronize();
    updateWeights << <1, SIDE_NEURONS >> > (nn->inputBiases, nn->inputBiasesGrad, grads->inputBiasesGradients, SIDE_NEURONS, LR, BETA1, BETA2);
    cudaDeviceSynchronize();
    updateWeights << <1, 1 >> > (nn->outputBias, nn->outputBiasGrad, grads->outputBiasGradient, 1, LR, BETA1, BETA2);
    cudaDeviceSynchronize();

    //free(loss_cpu);
    free(loss_cpu);
}

void calcErrorOnBatch(GPU_Network*& nn, int l, int r, float& totalLoss, FILE* bin_file) {
    int SIZE = r - l;

    double start_time = clock();

    int THREADS = min(SIZE, 1024);

    chessTraining::readNextBatch(bin_file, inputs_stm, inputs_opstm, outputs, SIZE);

    //cudaMemset(sums, 0, SIZE * sizeof(float));
    cudaMemcpy(inputs_gpu_stm, inputs_stm, SIZE * 32 * sizeof(int16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(inputs_gpu_opstm, inputs_opstm, SIZE * 32 * sizeof(int16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(outputs_gpu, outputs, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    feedForwardBatch << <SIZE, FF_BATCH_SIZE >> > (nn->inputBiases, nn->inputWeights, outputstm, inputs_gpu_stm, l, r);
    cudaDeviceSynchronize();
    cublas_mat_mult(outputstm, nn->outputWeights, sums, SIZE, SIDE_NEURONS, 1, 0.0);
    cudaDeviceSynchronize();
    feedForwardBatch << <SIZE, FF_BATCH_SIZE >> > (nn->inputBiases, nn->inputWeights, outputopstm, inputs_gpu_opstm, l, r);
    cudaDeviceSynchronize();
    cublas_mat_mult(outputopstm, &nn->outputWeights[SIDE_NEURONS], sums, SIZE, SIDE_NEURONS, 1, 1.0);
    cudaDeviceSynchronize();
    addOutputBias << <SIZE / THREADS + 1, THREADS >> > (sums, nn->outputBias, outputs_gpu, outputErrors, loss, l, r);
    cudaDeviceSynchronize();

    float* loss_cpu = (float*)malloc(SIZE * sizeof(float));

    cudaMemcpy(loss_cpu, loss, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < r - l; i++)
        totalLoss += loss_cpu[i];

    free(loss_cpu);
}

float calcLoss(GPU_Network*& nn, int l, int r, FILE* bin_file) {
    float totalLoss = 0;
    const int batch_size = BATCH_SIZE;
    for (int i = l; i < r; i += batch_size) {
        //float t1 = clock();
        cout << "Batch " << (i - l) / batch_size + 1 << "/" << (r - l) / batch_size + 1 << "\r";
        //nn->save(savePath);
        //cpu_nn.load(savePath);
        calcErrorOnBatch(nn, i, min(i + batch_size, r), totalLoss, bin_file);
        //trainOnBatch(cpu_nn, dataset, i, min(i + BATCH_SIZE, trainSize));
    }
    return 1.0 * totalLoss / (r - l);
}

void runTraining(int dataSize, int epochs, float split, string loadPath, string dataPath, bool load) {
    int trainSize = dataSize * (1.0 - split);

    string rootPath = "nn_";

    prealloc_on_cuda();

    nn->save(rootPath + "0.nn");
    cpu_nn.load(rootPath + "0.nn");

    if (load) {
        nn->load(loadPath);
        cudaDeviceSynchronize();
        FILE* bin_file = fopen(dataPath.c_str(), "rb");
        float trainingLoss = calcLoss(nn, 0, trainSize, bin_file), validationLoss = calcLoss(nn, trainSize, dataSize, bin_file);
        cout << "Total   Training Loss : " << 1.0 * trainingLoss << "\n";
        cout << "Total Validation Loss : " << 1.0 * validationLoss << "\n";
        fclose(bin_file);
    }

    /// train

    for (int epoch = 1; epoch <= epochs; epoch++) {
        FILE* bin_file = fopen(dataPath.c_str(), "rb");
        cout << "----------------------------------------- Epoch " << epoch << "/" << epochs << " -----------------------------------------\n";

        float tStart = clock();
        float trainingLoss = 0, validationLoss = 0;
        string currentPath = rootPath + to_string(epoch) + ".nn";

        for (int i = 0, batch_id = 0; i < trainSize; i += BATCH_SIZE, batch_id++) {
            //float t1 = clock();
            cout << "Batch " << batch_id + 1 << "/" << trainSize / BATCH_SIZE + 1 << "\r";
            //nn->save(savePath);
            //cpu_nn.load(savePath);
            if (batch_id % BATCHES_TO_LOAD == 0)
                load_data(bin_file);

            trainOnBatch(nn, i, min(i + BATCH_SIZE, trainSize), trainingLoss, bin_file, 32 * BATCH_SIZE * (batch_id % BATCHES_TO_LOAD));
            //nn->save(currentPath);
            //cpu_nn.load(currentPath);
            //trainOnBatch(cpu_nn, dataset, i, min(i + BATCH_SIZE, trainSize));
        }
        float tEnd = clock();
        cout << "\nValidation process:\n";

        validationLoss = calcLoss(nn, trainSize, dataSize, bin_file);
        trainingLoss /= trainSize;

        cout << "\n";

        float elapsed_time = (tEnd - tStart) / CLOCKS_PER_SEC;
        int pos_per_sec = round(1.0 * trainSize / elapsed_time);

        cout << "Time for training   : " << elapsed_time << "s\n";
        cout << "Positions per second: " << pos_per_sec << "\n";

        cout << "Total   Training Loss : " << 1.0 * trainingLoss << "\n";
        cout << "Total Validation Loss : " << 1.0 * validationLoss << "\n";

        cout << "\n";

        nn->save(currentPath);
        cpu_nn.load(currentPath);
        cpu_nn.evalTestPos();

        cout << cpu_nn.outputBias[0] << "\n";


        if (epoch % 30 == 0)
            LR /= 2;

        fclose(bin_file);

        //exit(0);
    }

    cudaMallocManaged(&nn, sizeof(GPU_Network));
    nn->init();

    grads->free();
    cudaFree(outputstm);
    cudaFree(outputopstm);


    cudaFree(inputs_gpu_stm);
    cudaFree(inputs_gpu_opstm);
    cudaFree(outputs_gpu);

    cudaFree(sums);
    cudaFree(outputErrors);

    cudaFree(loss);

    cublasDestroy(handle);

    cudaDeviceSynchronize();
}
