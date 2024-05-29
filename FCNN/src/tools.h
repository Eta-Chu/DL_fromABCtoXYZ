#include <Eigen/Dense>

Eigen::MatrixXd sigmoid(Eigen::MatrixXd x);

Eigen::MatrixXd softmax(Eigen::MatrixXd x);

double mse(Eigen::MatrixXd y, Eigen::MatrixXd t);

double cross_entropy(Eigen::MatrixXd y, Eigen::MatrixXd t);

Eigen::VectorXi row_argmax(Eigen::MatrixXd& y);
