#include <iostream>
#include "fcnn.h"

using namespace std;

int main()
{
    int input_size = 10;
    int hidden_size = 20;
    int output_size = 4;

    FCNN::NeuralNetwork nn(input_size, hidden_size, output_size);
    std::cout << nn.input_size << std::endl;

    Eigen::MatrixXd dateset = Eigen::MatrixXd::Random(100, 10);
    Eigen::MatrixXi target = Eigen::MatrixXi::Random(100, 4);
};
