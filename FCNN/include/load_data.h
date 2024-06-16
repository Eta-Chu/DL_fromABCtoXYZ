#pragma once

#include <Eigen/Dense>
#include <string>

namespace FCNN {

Eigen::MatrixXd LoadImageInfoFromUbyte(std::string file_path, uint32_t& img_count);

Eigen::MatrixXd LoadLabelInfoFromUbyte(std::string file_path, uint32_t& label_count);

} // namespace FCNN