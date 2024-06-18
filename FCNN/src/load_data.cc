#include "load_data.h"
#include "log_utils.h"

#include <fstream>
#include <vector>
#include <iostream>

namespace FCNN {

/**
 * @file_path[in]: 数据集文件的路径
 * @img_count[out]: 数据集图片个数，传出参数
 * @return: 二维矩阵，每一行代表一张图片的像素信息
*/
Eigen::MatrixXd LoadImageInfoFromUbyte(std::string file_path, uint32_t& img_count) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file " + file_path);
    }

    // MNIST 数据集文件头部信息
    uint32_t magic_number = 0;
    uint32_t image_count = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&image_count, sizeof(image_count));
    image_count = __builtin_bswap32(image_count);
    file.read((char*)&rows, sizeof(rows));
    rows = __builtin_bswap32(rows);
    file.read((char*)&cols, sizeof(cols));
    cols = __builtin_bswap32(cols);

    LOGI << "magic_number: " << magic_number;
    LOGI << "image_count: " << image_count;
    LOGI << "rows: " << rows;
    LOGI << "cols: " << cols;

    uint32_t image_size = rows * cols;
    Eigen::MatrixXd images(image_count, image_size);

    for (uint32_t i = 0; i < image_count; ++i) {
        std::vector<unsigned char> image(image_size);
        file.read((char*)image.data(), image_size);
        for (uint32_t j = 0; j < image_size; ++j) {
            images(i, j) = image[j];
        }
    }

    file.close();

    img_count = image_count;
    return images;
}

/**
 * @file_path[in]: label文件的路径
 * @label_count[out]: label个数，传出参数
 * @return: 二维矩阵(n*10)，每一行代表一个label信息，对应列数 m 的值为1，则标识是数字 m
*/
Eigen::MatrixXd LoadLabelInfoFromUbyte(std::string file_path, uint32_t& label_count) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    // MNIST 标注集文件头部信息
    uint32_t magic_number = 0;
    uint32_t count = 0;

    
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&count, sizeof(count));
    count = __builtin_bswap32(count);

    LOGI << "magic_number: " << magic_number;
    LOGI << "label_count: " << count;

    // 读取标签数据
    std::vector<uint8_t> labels(count);
    file.read(reinterpret_cast<char*>(labels.data()), count);

    Eigen::MatrixXd labels_matrix = Eigen::MatrixXd::Zero(count, 10);

    for(uint32_t i = 0; i < count; i++) {
        labels_matrix(i, labels[i]) = 1;
    }

    label_count = count;
    return labels_matrix;
}


} // namespace FCNN
