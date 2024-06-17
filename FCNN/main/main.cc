#include <iostream>
#include <vector>
#include <chrono>
#include "fcnn.h"
#include "tools.h"
#include "load_data.h"

int main()
{
    int iters = 100;
    int batch_size = 100;
    double learning_rate = 0.1;

    int input_size = 784;
    int hidden_size = 100;
    int output_size = 10;

    FCNN::NeuralNetwork nn(input_size, hidden_size, output_size);

    uint32_t train_num = 60000;
    uint32_t test_num = 10000;
    Eigen::MatrixXd x_train;
    Eigen::MatrixXd y_train;
    Eigen::MatrixXd x_test;
    Eigen::MatrixXd y_test;

    x_train = FCNN::LoadImageInfoFromUbyte("../dateset/train-images.idx3-ubyte", train_num);
    x_test = FCNN::LoadImageInfoFromUbyte("../dateset/t10k-images.idx3-ubyte", test_num);
    y_train = FCNN::LoadLabelInfoFromUbyte("../dateset/train-labels.idx1-ubyte", train_num);
    y_test = FCNN::LoadLabelInfoFromUbyte("../dateset/t10k-labels.idx1-ubyte", test_num);
    
    normalization(x_train);
    normalization(x_test);

    Eigen::MatrixXd log_loss(iters, 1);
    std::vector<double> log_acc;
    
    for (int i = 0; i < iters; i++){
        std::cout << "*******************" << std::endl;
        std::cout << "this is " << i << " " << "iter" << std::endl;
        Eigen::MatrixXd x_batch;
        Eigen::MatrixXd y_batch;
        std::tie(x_batch, y_batch) = random_choice(
                x_train,
                y_train,
                batch_size);

//        std::cout << x_batch.rows() << ", " << x_batch.cols() << std::endl;
//        auto st = std::chrono::steady_clock::now();
//        nn.predict(x_batch);
//        auto et = std::chrono::steady_clock::now();
//        auto time = std::chrono::duration<double>(et-st).count();
//        std::cout << time << std::endl;
        
        auto st = std::chrono::steady_clock::now();
        nn.backward(x_batch, y_batch, learning_rate);
        double loss = nn.loss(x_batch, y_batch);
        std::cout << "loss:" << loss << std::endl;
        log_loss(i, 0) = loss;
    
        auto et = std::chrono::steady_clock::now();
        auto time = std::chrono::duration<double>(et-st).count();
        std::cout << "time cost:" << time << std::endl;
        
        if (i % 10 == 0){
            double acc = nn.accuracy(x_test, y_test);
            log_acc.push_back(acc);
            std::cout << "accuracy:" << acc << std::endl;
        }
    }
    return 0;
};
