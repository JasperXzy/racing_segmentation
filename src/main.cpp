#include "racing_segmentation/parser.hpp"
#include "racing_segmentation/image_utils.hpp"
#include <iostream>
#include <iomanip> 

cv::Rect2f scale_box_to_original(const cv::Rect2f& box_on_model, 
                                 int model_w, int model_h, 
                                 int original_w, int original_h) {
    // 1. 计算缩放比例和填充量 (这部分逻辑与 image_to_nv12 中的逻辑完全对应)
    double scale = std::min(static_cast<double>(model_w) / original_w, static_cast<double>(model_h) / original_h);
    int scaled_w = static_cast<int>(original_w * scale);
    int scaled_h = static_cast<int>(original_h * scale);
    int pad_x = (model_w - scaled_w) / 2; // 左边填充
    int pad_y = (model_h - scaled_h) / 2; // 上边填充

    // 2. 将坐标从模型空间转换回原始图像空间
    float original_x = (box_on_model.x - pad_x) / scale;
    float original_y = (box_on_model.y - pad_y) / scale;
    float original_width = box_on_model.width / scale;
    float original_height = box_on_model.height / scale;

    return cv::Rect2f(original_x, original_y, original_width, original_height);
}

int main()
{
    // --- 1. 初始化 ---
    RacingSegmentation racing_segmentation;
    if (racing_segmentation.load_config() != 0) {
        std::cerr << "Failed to load config." << std::endl;
        return -1;
    }
    if (racing_segmentation.load_bin_model() != 0) {
        std::cerr << "Failed to load model." << std::endl;
        return -1;
    }

    // --- 2. 图像预处理 ---
    const std::string image_path = "/userdata/racing_ws/src/racing_segmentation/demo/test.jpg";
    
    // 我们需要先加载原始图片，以获取其尺寸用于坐标转换
    cv::Mat original_image = cv::imread(image_path);
    if (original_image.empty()) {
        std::cerr << "Error: Failed to read image from " << image_path << std::endl;
        return -1;
    }
    const int original_img_w = original_image.cols;
    const int original_img_h = original_image.rows;

    // 模型输入尺寸 (硬编码为640x640)
    const int model_input_w = 640;
    const int model_input_h = 640;
    
    // 调用你提供的 image_to_nv12 函数
    cv::Mat nv12_mat;
    try {
        nv12_mat = image_to_nv12(image_path, model_input_w, model_input_h);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    uint8_t* nv12_data_ptr = nv12_mat.data;

    // --- 3. 执行检测 ---
    std::cout << "Starting detection on " << image_path << "..." << std::endl;
    std::vector<DetectionResult> detection_results;
    if (racing_segmentation.detect(nv12_data_ptr, detection_results) != 0) {
        std::cerr << "Detection process failed." << std::endl;
        return -1;
    }
    
    // --- 4. 打印结果，并进行坐标转换 ---
    std::cout << "\n================ Detection Results ================" << std::endl;
    std::cout << "Found " << detection_results.size() << " objects." << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    if (!detection_results.empty())
    {
        std::cout << std::fixed << std::setprecision(4);

        for (size_t i = 0; i < detection_results.size(); ++i) {
            const auto& res = detection_results[i];

            // 将检测框坐标转换回原始图像坐标系
            cv::Rect2f original_box = scale_box_to_original(res.box, model_input_w, model_input_h, original_img_w, original_img_h);
            
            std::cout << "Object " << i + 1 << ":" << std::endl;
            std::cout << "  - Class:      " << res.class_name << " (ID: " << res.class_id << ")" << std::endl;
            std::cout << "  - Confidence: " << res.score << std::endl;
            
            // 打印在模型输入图上的坐标 (640x640)
            std::cout << "  - BBox on Model [x, y, w, h]: [" 
                      << res.box.x << ", " << res.box.y << ", " 
                      << res.box.width << ", " << res.box.height << "]" << std::endl;

            // 打印在原始图像上的坐标
            std::cout << "  - BBox on Original [x, y, w, h]: ["
                      << original_box.x << ", " << original_box.y << ", "
                      << original_box.width << ", " << original_box.height << "]" << std::endl;

            // 打印掩码信息（尺寸）
            std::cout << "  - Mask Size [w, h]:  [" 
                      << res.mask.cols << ", " << res.mask.rows << "]" << std::endl;
            
            std::cout << "-------------------------------------------------" << std::endl;
        }
    } else {
        std::cout << "No objects were detected." << std::endl;
    }
    std::cout << "=================================================" << std::endl;

    return 0;
}