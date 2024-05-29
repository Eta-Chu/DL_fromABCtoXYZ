#include <Eigen/Dense>

Eigen::MatrixXd sigmoid(Eigen::MatrixXd x)
{
    Eigen::ArrayXXd y = 1 / (1 + (1 / x.array().exp()));
    return y.matrix();
}

Eigen::MatrixXd softmax(Eigen::MatrixXd x)
{
    Eigen::ArrayXXd x_array = x.array();
    Eigen::ArrayXd c = x_array.rowwise().maxCoeff();
    Eigen::ArrayXXd a = (x_array.colwise() - c).exp();
    Eigen::ArrayXd sum_a = a.rowwise().sum();
    Eigen::ArrayXXd y = a.colwise() / sum_a;

    return y.matrix();
}

double mse(Eigen::MatrixXd y, Eigen::MatrixXd t)
{
    Eigen::ArrayXXd loss = (y.array() - t.array()).square();
    double res = loss.sum() / 2;
    return res;
}

double cross_entropy(Eigen::MatrixXd y, Eigen::MatrixXd t)
{
    double res = - (t.array() * (y.array() + 1e-7).log()).sum();
    res = res / y.rows();
    return res;
}

Eigen::VectorXi row_argmax(Eigen::MatrixXd& y)
{
    Eigen::VectorXi row_max = Eigen::VectorXi::Zero(y.rows());
    Eigen::Index index;

    for (int i = 0; i < y.rows(); i++){
        y.row(i).maxCoeff(&index);
        row_max(i) = index;
    }

    return row_max;
}
