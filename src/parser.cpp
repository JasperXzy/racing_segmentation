#include "racing_segmentation/parser.hpp"

int RacingObstacleDetection::rdk_check_success(int value, const std::string &errmsg)
{
    if (value != 0)
    {
        std::cerr << "[ERROR] " << errmsg << ", error code: " << value << std::endl;
        return value;
    }
    return 0;
}

int RacingObstacleDetection::load_config()
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
