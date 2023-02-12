#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <ctime>
#include <iostream>
#include <cassert>
#include "neuralnet.h"
//#include "chess.h"

using namespace std;

int main(int argc, char **argv) {
    int dataSize, nrEpochs;
    float split;
    const string path = "C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_v4_shuffled.bin";

    if (argc > 1) {
        if (!strncmp(argv[1], "lr", 1)) {
            if (argc == 2) {
                cout << "Wrong Learning Rate!\n";
                return 0;
            }
            string s = argv[2];
            LR = stof(s);
            cout << "Set Learning Rate to " << LR << "!\n";
        }
    }

    cout << "Welcome!\nTrainer takes input [DATA_SIZE] [NR_EPOCHS] [SPLIT]\n";
    cout << "[DATA_SIZE] - how many positions do you want to train on\n";
    cout << "[NR_EPOCHS] - how many epochs you want to train on\n";
    cout << "[SPLIT]     - split between training and validation data (0 is 100% training data, 1 is 100% validation data)\n";

    cin >> dataSize >> nrEpochs >> split;

    //FILE* bin_file = fopen(path.c_str(), "rb");

    /*if (dataSize > (int)6e8)
        dataSize = min<long long>(dataSize, chessTraining::getDatasetSize(path));*/
    dataSize = min(dataSize, 728092467);
    cout << "Actual datasize : " << dataSize << "\n";
    runTraining(dataSize, nrEpochs, split, "4buckets_768_30.nn", path, true);
    return 0;
}