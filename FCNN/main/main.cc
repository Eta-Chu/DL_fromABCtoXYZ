#include <iostream>
#include "fcnn.h"
#include "tools.h"

using namespace std;

int main()
{
    int input_size = 10;
    int hidden_size = 20;
    int output_size = 4;

    FCNN::NeuralNetwork nn(input_size, hidden_size, output_size);
    std::cout << nn.input_size << std::endl;

    Eigen::MatrixXd train_x= Eigen::MatrixXd::Random(100, 10);
    Eigen::ArrayXd y = (Eigen::ArrayXd::Random(100) + 1) * 5;
    Eigen::ArrayXi y1 = y.cast<int>();
    Eigen::VectorXi train_y = y1.matrix();
    
    Eigen::MatrixXd target = one_hot(train_y);
    
    cout << target << endl;

    cout << nn.accuracy(train_x, target) << endl;

    return 0;
};
