#include <Eigen/Dense>
#include <random>
#include <utility>

Eigen::MatrixXd sigmoid(Eigen::MatrixXd x)
{
    Eigen::MatrixXd y = 1 / (1 + ((-x.array()).exp()));
    return y;
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

Eigen::MatrixXd one_hot(Eigen::VectorXi& y)
{
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(y.rows(), 10);

    for (int i = 0; i < y.rows(); i++){
        res(i, y(i)) = 1;
    }

    return res;
}

void normalization(Eigen::MatrixXd& x)
{
    x = ((x.array() / 255) * 0.99) + 0.01;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> random_choice(
        Eigen::MatrixXd& x,
        Eigen::MatrixXd& y,
        int pieces)
{
    int k = x.cols();
    int j = x.rows();
    Eigen::MatrixXd x_batch(pieces, k);
    Eigen::MatrixXd y_batch(pieces, 10);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, j-1);
    
    for (int i = 0; i < pieces; i++){
        int index = dis(gen);
        x_batch.row(i) = x.row(index);
        y_batch.row(i) = y.row(index);
    }
    
    return std::make_pair(x_batch, y_batch);
}

Eigen::MatrixXd normal_matrix(int m, int n)
{
    Eigen::MatrixXd random_matrix(m, n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            random_matrix(i, j) = dis(gen);
        }
    }

    return random_matrix;
}
