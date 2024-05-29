#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include "tools.h"

using namespace std;

int main()
{
    Eigen::MatrixXd m1, m2;
    m1.setRandom(5, 6);
    m2.setRandom(5, 6);

    Eigen::VectorXi y1(5), y2(5);
    y1 << 1, 2, 3, 4, 5;
    y2 << 1, 2, 4 ,5 ,6;
    
    Eigen::Array<bool, 5, 1> res = (y1.array() == y2.array()).array();
    
    cout << res.cast<int>().sum() << endl;
    return 0;
}
