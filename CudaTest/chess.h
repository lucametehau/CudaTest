#include "neuralnet.h"
#include "data.h"
#include <cstring>
#include <thread>
#include <unordered_map>
#include <random>
#include <mutex>

using namespace std;

const int DATASET_SIZE = (int)1e9;

mt19937_64 gen(0xBEEF);
uniform_int_distribution <uint64_t> rng;

namespace chessTraining {

    int ind = 0;
    int badSamples = 0;

    long long getDatasetSize(string path) {
        FILE* bin_file = fopen(path.c_str(), "rb");
        long long nr = 0;
        for (int id = 0; ; id++) {
            NetInput inp;
            float score;
            if (!fread(&inp, sizeof(NetInput), 1, bin_file) || id == DATASET_SIZE)
                break;
            fread(&score, sizeof(float), 1, bin_file);

            if (__builtin_popcountll(inp.occ) == 3) {
                int p1 = (inp.pieces[0] & 15), p2 = ((inp.pieces[0] >> 4) & 15), p3 = ((inp.pieces[0] >> 8) & 15);
                int p = 0;
                if (p1 % 6 == 0)
                    p = (p2 % 6 == 0 ? p3 : p2);
                else
                    p = p1;
                if (p > 6)
                    p -= 6;
                if (p != 1) {
                    assert(p != 2 && p != 3);
                    badSamples += !(score < 0.1 || score > 0.9);
                }
            }

            nr++;

            if (nr % 1000000 == 0)
                cout << badSamples << " bad samples out of " << nr << "\n";
        }

        fclose(bin_file);

        return nr;
    }
    
    void readNextBatch(FILE* bin_file, int16_t*& inputs_stm, int16_t*& inputs_opstm, float*& outputs, int batchSize) {
        double start = clock();

        for (int id = 0; id < batchSize; id++) {
            NetInput inp;
            float score;
            if (!fread(&inp, sizeof(NetInput), 1, bin_file))
                break;

            fread(&score, sizeof(float), 1, bin_file);

            setInput(inputs_stm + 32 * id, inputs_opstm + 32 * id, inp);

            outputs[id] = score;
        }
    }
};
