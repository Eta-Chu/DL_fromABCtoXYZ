#include <iostream>
#include <tuple>
#include <vector>
#include "fcnn.h"
#include "tools.h"
#include "load_data.h"

int main()
{
    int iters;
    int batch_size;
    double learning_rate;

    int input_size = 10;
    int hidden_size = 20;
    int output_size = 4;

    FCNN::NeuralNetwork nn(input_size, hidden_size, output_size);

//    Eigen::MatrixXd train_x= Eigen::MatrixXd::Random(100, 10);
//    Eigen::ArrayXd y = (Eigen::ArrayXd::Random(100) + 1) * 5;
//    Eigen::ArrayXi y1 = y.cast<int>();
//    Eigen::VectorXi train_y = y1.matrix();
    
    Eigen::MatrixXd x_train;
    Eigen::MatrixXd y_train;
    Eigen::MatrixXd x_test;
    Eigen::MatrixXd y_test;

    x_train = FCNN::LoadImageInfoFromUbyte("../dateset/train-images.idx3-ubyte");
    x_test = FCNN::LoadImageInfoFromUbyte("../dateset/t10k-images.idx3-ubyte");
    y_train = FCNN::LoadLabelInfoFromUbyte("../dateset/train-labels.idx1-ubyte");
    y_test = FCNN::LoadLabelInfoFromUbyte("../dateset/t10k-labels.idx1-ubyte");
    
    Eigen::MatrixXd log_loss(iters, 1);
    std::vector<double> log_acc;

    for (int i = 0; i < iters; i++){
        Eigen::MatrixXd x_batch;
        Eigen::MatrixXd y_batch;
        std::tie(x_batch, y_batch) = random_choice(
                x_train,
                y_train,
                batch_size);
        nn.backward(x_batch, y_batch, learning_rate);
        double loss = nn.loss(x_batch, y_batch);
        log_loss(i, 0) = loss;

        if (i % 10 == 0){
            double acc = nn.accuracy(x_test, y_test);
            log_acc.push_back(acc);
            std::cout << acc << std::endl;
        }
    }
    
    return 0;
};
