#include <gtest/gtest.h>
#include <Eigen/Dense>

TEST(testCase, test_matrix_add) 
{
    Eigen::MatrixXd mat1(2, 2);
    Eigen::MatrixXd mat2(2, 2);
    
    // 初始化矩阵
    mat1 << 1, 2, 3, 4;
    mat2 << 5, 6, 7, 8;

    // 期望的结果
    Eigen::MatrixXd expected_result(2, 2);
    expected_result << 6, 8, 10, 12;

    // 执行矩阵加法
    Eigen::MatrixXd result = mat1 + mat2;

    // 检查结果是否与预期一致
    ASSERT_EQ(result, expected_result);
}

TEST(testCase, test_matrix_sub) 
{
    Eigen::MatrixXd mat1(2, 2);
    Eigen::MatrixXd mat2(2, 2);
    
    // 初始化矩阵
    mat1 << 6, 7, 8, 9;
    mat2 << 1, 2, 3, 4;

    // 期望的结果
    Eigen::MatrixXd expected_result(2, 2);
    expected_result << 5, 5, 5, 5;

    // 执行矩阵加法
    Eigen::MatrixXd result = mat1 - mat2;

    // 检查结果是否与预期一致
    ASSERT_EQ(result, expected_result);
}
TEST(case1, test_random_matrix_generation)
{
    
}
