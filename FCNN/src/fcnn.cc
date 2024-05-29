#include "fcnn.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <system_error>
#include "tools.h"


using namespace std;

namespace FCNN {

NeuralNetwork::NeuralNetwork()
{

}

NeuralNetwork::NeuralNetwork(int s_input, int s_hidden, int s_output)
{
    input_size = s_input;
    hidden_size = s_hidden;
    output_size = s_output;

    weight.insert({"w1", Eigen::MatrixXd::Random(input_size, hidden_size)});
    weight.insert({"w2", Eigen::MatrixXd::Random(hidden_size, output_size)});
        
    threshold.insert({"b1", Eigen::VectorXd::Zero(hidden_size)});
    threshold.insert({"b2", Eigen::VectorXd::Zero(output_size)});
}

NeuralNetwork::~NeuralNetwork()
{

}

Eigen::MatrixXd NeuralNetwork::predict(Eigen::MatrixXd& input_x)
{
    Eigen::MatrixXd a1 = (input_x * weight["w1"]).rowwise() 
        + threshold["b1"].transpose();
    Eigen::MatrixXd z1 = sigmoid(a1);
    Eigen::MatrixXd a2 = (z1 * weight["w2"]).rowwise()
        + threshold["b2"].transpose();
    Eigen::MatrixXd y = softmax(a2);

    return y;
}

double NeuralNetwork::loss(Eigen::MatrixXd& input_x, Eigen::MatrixXd& target)
{
    Eigen::MatrixXd y = predict(input_x);
    double loss_value = cross_entropy(input_x, target);
    return loss_value;
}

double NeuralNetwork::accuracy(Eigen::MatrixXd& input_x, 
        Eigen::MatrixXd& target)
{
    Eigen::MatrixXd y = predict(input_x);
    Eigen::VectorXi y_max_index = row_argmax(y);
    Eigen::VectorXi t_max_index = row_argmax(target);
    Eigen::Array<bool, Eigen::Dynamic, 1> comp_yt = (y_max_index.array() == t_max_index.array());
    
    double acc = comp_yt.cast<int>().sum() / y.rows();
    return acc;
}

} // namespace FCNN
