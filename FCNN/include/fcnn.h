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

    NeuralNetwork();

    NeuralNetwork(int s_input, int s_hidden, int s_output);

    ~NeuralNetwork();

    Eigen::MatrixXd predict(Eigen::MatrixXd& input_x);
    
    double loss(Eigen::MatrixXd& input_x, Eigen::MatrixXd& target);

    double accuracy(Eigen::MatrixXd& input_x, Eigen::MatrixXd& target);

    template <typename Derived>
    Eigen::MatrixXd cal_gradient(Eigen::MatrixBase<Derived>& param,
        Eigen::MatrixXd x, 
        Eigen::MatrixXd t,
        float h = 1e-4);

    void backward(Eigen::MatrixXd& x_batch, Eigen::MatrixXd& t_batch);
};

}  // namespace FCNN
