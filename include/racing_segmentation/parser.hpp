#ifndef PARSER_H
#define PARSER_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <nlohmann/json.hpp>
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/hb_sys.h"
#include <omp.h>

// DetectionResult 只存储最终在原始图像上的坐标
struct DetectionResult {
    int class_id;
    std::string class_name;
    float score;
    cv::Rect2f box;
    cv::Mat mask; 
};

class RacingSegmentation
{
public:
    RacingSegmentation() = default;
    ~RacingSegmentation();

    int load_config();
    int load_bin_model();
    int detect(uint8_t* ynv12, int original_w, int original_h, std::vector<DetectionResult>& results);
    int release_model();

private:
    std::string model_file;
    int class_num;
    std::string dnn_parser;
    std::vector<std::string> cls_names_list;
    int preprocess_type;
    float score_threshold;
    float nms_threshold;
    int nms_top_k;
    int reg;
    int mces;
    int model_input_w;
    int model_input_h;

    hbPackedDNNHandle_t packed_dnn_handle = nullptr;
    hbDNNHandle_t dnn_handle = nullptr;
    hbDNNTensorProperties input_properties;
    int32_t input_W = 0;
    int32_t input_H = 0;
    int32_t output_count = 0;
    int order[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    int rdk_check_success(int value, const std::string &errmsg);
};

#endif // PARSER_H
