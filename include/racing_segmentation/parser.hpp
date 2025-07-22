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


    int rdk_check_success(int value, const std::string &errmsg);
};

#endif // PARSER_H
