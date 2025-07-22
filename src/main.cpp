#include "racing_segmentation/parser.hpp"
#include "racing_segmentation/image_utils.hpp"
#include <iostream>
#include <iomanip> 


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
    
    cv::Mat original_image = cv::imread(image_path);
    if (original_image.empty()) {
        std::cerr << "Error: Failed to read image from " << image_path << std::endl;
        return -1;
    }
    const int original_img_w = original_image.cols;
    const int original_img_h = original_image.rows;

    const int model_input_w = 640;
    const int model_input_h = 640;
    
    cv::Mat nv12_mat = image_to_nv12(image_path, model_input_w, model_input_h);
    uint8_t* nv12_data_ptr = nv12_mat.data;

    // --- 3. 执行检测 ---
    std::cout << "Starting detection on " << image_path << "..." << std::endl;
    std::vector<DetectionResult> detection_results;

    // 传入原始图像的宽高
    if (racing_segmentation.detect(nv12_data_ptr, original_img_w, original_img_h, detection_results) != 0) {
        std::cerr << "Detection process failed." << std::endl;
        return -1;
    }
    
    // --- 4. 打印结果 ---
    std::cout << "\n================ Detection Results on Original Image ================" << std::endl;
    std::cout << "Found " << detection_results.size() << " objects." << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    if (!detection_results.empty())
    {
        std::cout << std::fixed << std::setprecision(4);

        for (size_t i = 0; i < detection_results.size(); ++i) {
            const auto& res = detection_results[i];
            
            std::cout << "Object " << i + 1 << ":" << std::endl;
            std::cout << "  - Class:      " << res.class_name << " (ID: " << res.class_id << ")" << std::endl;
            std::cout << "  - Confidence: " << res.score << std::endl;
            
            // **直接打印 box，它已经是原始图像上的坐标**
            std::cout << "  - BBox [x, y, w, h]: [" 
                      << res.box.x << ", " << res.box.y << ", " 
                      << res.box.width << ", " << res.box.height << "]" << std::endl;

            std::cout << "  - Mask Size [w, h]:  [" 
                      << res.mask.cols << ", " << res.mask.rows << "]" << std::endl;
            
            std::cout << "-------------------------------------------------" << std::endl;
        }
    } else {
        std::cout << "No objects were detected." << std::endl;
    }
    std::cout << "================================================================" << std::endl;

    return 0;
}
