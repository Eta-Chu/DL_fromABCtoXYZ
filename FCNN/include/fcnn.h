#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <Eigen/Dense>

using namespace std;

namespace FCNN {

class NeuralNetwork{
private:
            

public:
    int input_size;
    int hidden_size;
    int output_size;
    double weight_init_std;
    
    map<string, Eigen::MatrixXd> weight;
    map<string, Eigen::VectorXd> threshold;

    NeuralNetwork();

    NeuralNetwork(int s_input, int s_hidden, int s_output);

    ~NeuralNetwork();

    Eigen::MatrixXd predict(Eigen::MatrixXd& input_x);
    
    double loss(Eigen::MatrixXd& input_x, Eigen::MatrixXd& target);

    double accuracy(Eigen::MatrixXd& input_x, Eigen::MatrixXd& target);
};

}  // namespace FCNN
