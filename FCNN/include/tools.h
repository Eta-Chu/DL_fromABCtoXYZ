#pragma once

#include <Eigen/Dense>
#include <utility>

Eigen::MatrixXd sigmoid(Eigen::MatrixXd x);

Eigen::MatrixXd softmax(Eigen::MatrixXd x);

double mse(Eigen::MatrixXd y, Eigen::MatrixXd t);

double cross_entropy(Eigen::MatrixXd y, Eigen::MatrixXd t);

Eigen::VectorXi row_argmax(Eigen::MatrixXd& y);

Eigen::MatrixXd one_hot(Eigen::VectorXi& y);

void normalization(Eigen::MatrixXd& x);

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> random_choice(
        Eigen::MatrixXd& x,
        Eigen::MatrixXd& y,
        int pieces);

Eigen::MatrixXd normal_matrix(int m, int n);

