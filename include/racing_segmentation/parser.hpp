#ifndef PARSER_H
#define PARSER_H

// C/C++ Standard Librarys
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

// Thrid Party Librarys
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

// RDK BPU libDNN API
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include "dnn/hb_sys.h"

#include <nlohmann/json.hpp>
#include <omp.h>
#include <fstream>

class RacingSegmentation
{
public:
    int load_config();
    int load_bin_model();
    int detect(uint8_t* ynv12);
    int postprocessing();
    int release_model();

private:
    std::string model_file;
    int class_num;
    std::string dnn_parser;
    std::vector<std::string> cls_names_list;
    int preprocess_type;    // 0: Resize, 1: Letterbox
    float score_threshold;
    float nms_threshold;
    int nms_top_k;
    int reg;                // Regression, default 16
    int mces;               // Mask Coefficients, default 32
    bool is_point;          // Whether to generate and draw contour points
    float font_size;        // Font size for drawing labels, default 1.0
    float font_thickness;   // Font thickness for drawing labels, default 1.0
    float line_size;        // Line width for drawing bounding boxes, default 2.0
    std::vector<cv::Scalar> rdk_colors = {
    cv::Scalar(56, 56, 255), cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255), cv::Scalar(29, 178, 255), cv::Scalar(49, 210, 207), cv::Scalar(10, 249, 72), cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61), cv::Scalar(52, 147, 26), cv::Scalar(187, 212, 0), cv::Scalar(168, 153, 44), cv::Scalar(255, 194, 0), cv::Scalar(147, 69, 52), cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0), cv::Scalar(255, 56, 132), cv::Scalar(133, 0, 82), cv::Scalar(255, 56, 203), cv::Scalar(200, 149, 255), cv::Scalar(199, 55, 255)};
    int order[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int32_t H_4 = input_H / 4;
    int32_t H_8 = input_H / 8;
    int32_t H_16 = input_H / 16;
    int32_t H_32 = input_H / 32;
    int32_t W_4 = input_W / 4;
    int32_t W_8 = input_W / 8;
    int32_t W_16 = input_W / 16;
    int32_t W_32 = input_W / 32;
    hbDNNTensorProperties input_properties;
    hbDNNHandle_t dnn_handle;
    hbPackedDNNHandle_t packed_dnn_handle;
    int32_t input_H, input_W;
    int32_t output_count = 0;


    int rdk_check_success(int value, const std::string &errmsg);
};

#endif // PARSER_H
