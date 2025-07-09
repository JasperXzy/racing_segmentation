#include "racing_segmentation/parser.hpp"

int RacingSegmentation::rdk_check_success(int value, const std::string &errmsg)
{
    if (value != 0)
    {
        std::cerr << "[ERROR] " << errmsg << ", error code: " << value << std::endl;
        return value;
    }
    return 0;
}

int RacingSegmentation::load_config()
{
    std::cout << "================================================" << std::endl;
    std::cout << "[INFO] Loading Configuration From config/yolo11-seg.json" << std::endl;
    std::ifstream config_file("config/yolo11-seg.json");
    if (!config_file.is_open())
    {
        std::cerr << "[ERROR] Failed to open config file." << std::endl;
        return -1;
    }
 
    nlohmann::json config;
    config_file >> config;
 
    model_file = config["model_file"];
    class_num = config["class_num"];
    dnn_parser = config["dnn_Parser"];
    cls_names_list = config["cls_names_list"].get<std::vector<std::string>>();
    preprocess_type = config["preprocess_type"];
    score_threshold = config["score_threshold"];
    nms_threshold = config["nms_threshold"];
    nms_top_k = config["nms_top_k"];
    reg = config["reg"];
    mces = config["mces"];
    is_point = config["is_point"];
    font_size = config["font_size"];
    font_thickness = config["font_thickness"];
    line_size = config["line_size"];

    config_file.close();
    
    std::cout << "[INFO] Model File: " << model_file << std::endl;
    std::cout << "[INFO] DNN Parser: " << dnn_parser << std::endl;
    std::cout << "[INFO] Class Number: " << class_num << std::endl;
    std::cout << "[INFO] Class Names List: ";
    for (const auto& name : cls_names_list)
    {
        std::cout << name << " ";
    }
    std::cout << std::endl;
    if (preprocess_type == 0)
    {
        std::cout << "[INFO] Preprocess Type: Resize" << std::endl;
    }
    else if (preprocess_type == 1)
    {
        std::cout << "[INFO] Preprocess Type: Letterbox" << std::endl;
    }
    std::cout << "[INFO] Score Threshold: " << score_threshold << std::endl;
    std::cout << "[INFO] NMS Threshold: " << nms_threshold << std::endl;
    std::cout << "[INFO] NMS Top K: " << nms_top_k << std::endl;
    std::cout << "[INFO] Regression: " << reg << std::endl;
    std::cout << "[INFO] Mask Coefficients: " << mces << std::endl;
    std::cout << "[INFO] Is Point: " << (is_point ? "True" : "False") << std::endl;
    std::cout << "[INFO] Font Size: " << font_size << std::endl;
    std::cout << "[INFO] Font Thickness: " << font_thickness << std::endl;
    std::cout << "[INFO] Line Size: " << line_size << std::endl;
    std::cout << "[INFO] Load Configuration Successfully!" << std::endl;
    std::cout << "================================================" << std::endl << std::endl;

    return 0;
}

int RacingSegmentation::load_bin_model()
{
    std::cout << "================================================" << std::endl;
    std::cout << "[INFO] Loading Binary Model From: " << model_file << std::endl;
    std::cout << "[INFO] OpenCV Version: " << CV_VERSION << std::endl;


    // 1. 加载bin模型
    auto begin_time = std::chrono::system_clock::now();
    hbPackedDNNHandle_t packed_dnn_handle;
    const char *model_file_name = MODEL_PATH;
    rdk_check_success(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
        "hbDNNInitializeFromFiles failed");
    std::cout << "\033[31m[INFO] Load D-Robotics Quantize model time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

    // 2. 打印模型信息
    const char **model_name_list;
    int model_count = 0;
    rdk_check_success(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
        "hbDNNGetModelNameList failed");

    // 如果这个bin模型有多个打包，则只使用第一个，一般只有一个
    if (model_count > 1)
    {
        std::cout << "[WARN] This model file have more than 1 model, only use model 0.";
    }
    const char *model_name = model_name_list[0];
    std::cout << "[INFO] Model Name: " << model_name << std::endl;

    // 3. 获得Packed模型的第一个模型的handle
    hbDNNHandle_t dnn_handle;
    rdk_check_success(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "hbDNNGetModelHandle failed");

    // 4. 模型输入检查
    int32_t input_count = 0;
    rdk_check_success(
        hbDNNGetInputCount(&input_count, dnn_handle),
        "hbDNNGetInputCount failed");

    hbDNNTensorProperties input_properties;
    rdk_check_success(
        hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
        "hbDNNGetInputTensorProperties failed");

    // 4.1. D-Robotics YOLO11-Seg *.bin 模型应该为单输入
    if (input_count > 1)
    {
        std::cout << "[ERROR] Your Model have more than 1 input, please check!" << std::endl;
        return -1;
    }

    // 4.2. D-Robotics YOLO11-Seg *.bin 模型输入Tensor类型应为nv12
    if (input_properties.tensorType == HB_DNN_IMG_TYPE_NV12)
    {
        std::cout << "[INFO] Input Tensor Type: HB_DNN_IMG_TYPE_NV12" << std::endl;
    }
    else
    {
        std::cout << "[ERROR] Input Tensor Type is not HB_DNN_IMG_TYPE_NV12, please check!" << std::endl;
        return -1;
    }

    // 4.3. D-Robotics YOLO11-Seg *.bin 模型输入Tensor数据排布应为NCHW
    if (input_properties.tensorLayout == HB_DNN_LAYOUT_NCHW)
    {
        std::cout << "[INFO] Input Tensor Layout: HB_DNN_LAYOUT_NCHW" << std::endl;
    }
    else
    {
        std::cout << "[ERROR] Input Tensor Layout is not HB_DNN_LAYOUT_NCHW, please check!" << std::endl;
        return -1;
    }

    // 4.4. D-Robotics YOLO11-Seg *.bin 模型输入Tensor数据的valid shape应为(1,3,H,W)
    int32_t input_H, input_W;
    if (input_properties.validShape.numDimensions == 4)
    {
        input_H = input_properties.validShape.dimensionSize[2];
        input_W = input_properties.validShape.dimensionSize[3];
        std::cout << "[INFO] Input Tensor Valid Shape: (" << input_properties.validShape.dimensionSize[0];
        std::cout << ", " << input_properties.validShape.dimensionSize[1];
        std::cout << ", " << input_H;
        std::cout << ", " << input_W << ")" << std::endl;
    }
    else
    {
        std::cout << "[ERROR] Input Tensor Valid Shape.numDimensions is not 4 such as (1,3,640,640), please check!" << std::endl;
        return -1;
    }

    // 5. 模型输出检查
    int32_t output_count = 0;
    rdk_check_success(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "hbDNNGetOutputCount failed");

    // 5.1. D-Robotics YOLO11-Seg *.bin 模型应该有10个输出
    if (output_count == 10)
    {
        for (int i = 0; i < 10; i++)
        {
            hbDNNTensorProperties output_properties;
            rdk_check_success(
                hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
                "hbDNNGetOutputTensorProperties failed");
            std::cout << "[INFO] Output[" << i << "] ";
            std::cout << "Valid Shape: (" << output_properties.validShape.dimensionSize[0];
            std::cout << ", " << output_properties.validShape.dimensionSize[1];
            std::cout << ", " << output_properties.validShape.dimensionSize[2];
            std::cout << ", " << output_properties.validShape.dimensionSize[3] << "), ";
            if (output_properties.quantiType == SHIFT)
                std::cout << "QuantiType: SHIFT" << std::endl;
            if (output_properties.quantiType == SCALE)
                std::cout << "QuantiType: SCALE" << std::endl;
            if (output_properties.quantiType == NONE)
                std::cout << "QuantiType: NONE" << std::endl;
        }
    }
    else
    {
        std::cout << "[ERROR] Your Model's outputs num is not 10, please check!" << std::endl;
        return -1;
    }

    // 6. 调整输出头顺序的映射
    int order[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int32_t H_4 = input_H / 4;
    int32_t H_8 = input_H / 8;
    int32_t H_16 = input_H / 16;
    int32_t H_32 = input_H / 32;
    int32_t W_4 = input_W / 4;
    int32_t W_8 = input_W / 8;
    int32_t W_16 = input_W / 16;
    int32_t W_32 = input_W / 32;
    int32_t order_we_want[10][3] = {
        {H_8, W_8, class_num},      // output[order[0]]: (1, H // 8,  W // 8,  class_num)
        {H_8, W_8, 4 * reg},        // output[order[1]]: (1, H // 8,  W // 8,  64)
        {H_8, W_8, mces},           // output[order[2]]: (1, H // 8,  W // 8,  mces)
        {H_16, W_16, class_num},    // output[order[3]]: (1, H // 16, W // 16, class_num)
        {H_16, W_16, 4 * reg},      // output[order[4]]: (1, H // 16, W // 16, 64)
        {H_16, W_16, mces},         // output[order[5]]: (1, H // 16, W // 16, mces)
        {H_32, W_32, class_num},    // output[order[6]]: (1, H // 32, W // 32, class_num)
        {H_32, W_32, 4 * reg},      // output[order[7]]: (1, H // 32, W // 32, 64)
        {H_32, W_32, mces},         // output[order[8]]: (1, H // 32, W // 32, mces)
        {H_4, W_4, mces}            // output[order[9]]: (1, H // 4, W // 4, mces)
    };
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            hbDNNTensorProperties output_properties;
            rdk_check_success(
                hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, j),
                "hbDNNGetOutputTensorProperties failed");
            int32_t h = output_properties.validShape.dimensionSize[1];
            int32_t w = output_properties.validShape.dimensionSize[2];
            int32_t c = output_properties.validShape.dimensionSize[3];
            if (h == order_we_want[i][0] && w == order_we_want[i][1] && c == order_we_want[i][2])
            {
                order[i] = j;
                break;
            }
        }
    }

    // 7. 打印并检查调整后的输出头顺序的映射
    if (order[0] + order[1] + order[2] + order[3] + order[4] + order[5] + order[6] + order[7] + order[8] + order[9] == 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)
    {
        std::cout << "[INFO] Outputs order check SUCCESS, continue." << std::endl;
        std::cout << "Order = {";
        for (int i = 0; i < 10; i++)
        {
            std::cout << order[i] << ", ";
        }
        std::cout << "}" << std::endl;
    }
    else
    {
        std::cout << "[INFO] Outputs order check FAILED, use default" << std::endl;
        for (int i = 0; i < 10; i++)
            order[i] = i;
    }

    return 0;
}
