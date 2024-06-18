#include "fcnn.h"
#include "tools.h"
#include "log_utils.h"

#include <Eigen/Dense>
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <type_traits>
#include <omp.h>

using namespace std;

namespace FCNN {

NeuralNetwork::NeuralNetwork()
{
    omp_set_dynamic(0); //自动调整
}

NeuralNetwork::NeuralNetwork(int s_input, int s_hidden, int s_output)
{
    input_size = s_input;
    hidden_size = s_hidden;
    output_size = s_output;

    weight.insert({"w1", Eigen::MatrixXd::Random(input_size, hidden_size)});
    weight.insert({"b1", Eigen::MatrixXd::Zero(hidden_size, 1)});
    weight.insert({"w2", Eigen::MatrixXd::Random(hidden_size, output_size)});
    weight.insert({"b2", Eigen::MatrixXd::Zero(output_size, 1)});
}

NeuralNetwork::~NeuralNetwork()
{

}

Eigen::MatrixXd NeuralNetwork::predict(Eigen::MatrixXd& input_x)
{
    Eigen::MatrixXd a1 = (input_x * weight["w1"]).rowwise() 
        + weight["b1"].col(0).transpose();
    Eigen::MatrixXd z1 = sigmoid(a1);
    Eigen::MatrixXd a2 = (z1 * weight["w2"]).rowwise()
        + weight["b2"].col(0).transpose();
    Eigen::MatrixXd y = softmax(a2);

    return y;
}

double NeuralNetwork::loss(Eigen::MatrixXd& input_x, Eigen::MatrixXd& target)
{
    Eigen::MatrixXd y = predict(input_x);
    double loss_value = cross_entropy(y, target);
    return loss_value;
}

double NeuralNetwork::accuracy(Eigen::MatrixXd& input_x, 
        Eigen::MatrixXd& target)
{
    Eigen::MatrixXd y = predict(input_x);
    Eigen::VectorXi y_max_index = row_argmax(y);
    Eigen::VectorXi t_max_index = row_argmax(target);
    Eigen::Array<bool, Eigen::Dynamic, 1> comp_yt = (y_max_index.array() == t_max_index.array());
    double right_num = comp_yt.cast<int>().sum();
    int counts = y.rows();
    double acc = right_num / counts;
    
    return acc;
}

template <typename MatrixType>
MatrixType NeuralNetwork::cal_gradient(
        Eigen::MatrixBase<MatrixType>& param,
        Eigen::MatrixXd x, 
        Eigen::MatrixXd t,
        float h)
{
    int i_row = param.rows();
    int i_col = param.cols();
    MatrixType grad = MatrixType::Zero(i_row, i_col);
    
    if (i_col == 1){
        #pragma omp parallel for
        for (int i = 0; i < i_row; i++){
//            std::cout << i << std::endl;
            double f1, f2;
            double val = param(i);
            param(i) = val + h;
            f1 = this->loss(x, t);

            param(i) = val - h;
            f2 = this->loss(x, t);

            grad(i) = (f1 - f2) / (2 * h);
            param(i) = val;
        }
    } else {
        #pragma omp parallel for
        for (int i = 0; i < i_row; i++){
            #pragma omp parallel for
            for (int j = 0; j < i_col; j++){
//                std::cout << i << "," << j << std::endl;
                double f1, f2;
                double val = param(i, j);
                param(i, j) = val + h;
                f1 = this->loss(x, t);

                param(i, j) = val - h;
                f2 = this->loss(x, t);

                grad(i, j) = (f1 - f2) / (2 * h);
                param(i, j) = val;
            }
        }
    }
    
    return grad;
}

void NeuralNetwork::backward(Eigen::MatrixXd& x_batch, 
        Eigen::MatrixXd& t_batch,
        double learning_rate)
{
    LOGI << "Using " << Eigen::nbThreads() << " threads" << std::endl;
    std::vector<std::string> key_order = {"w1", "b1", "w2", "b2"};
    for (auto key : key_order){
//        auto start_time = std::chrono::steady_clock::now();
//        std::cout << key << std::endl;
        auto grad = this->cal_gradient(this->weight[key], x_batch, t_batch);
        this->weight[key] -= grad * learning_rate;
//        auto end_time = std::chrono::steady_clock::now();
//        auto duration = std::chrono::duration<double>(end_time - start_time).count();
//        std::cout << duration << std::endl;
    }
}

} // namespace FCNN
