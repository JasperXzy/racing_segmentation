#include "racing_segmentation/parser.hpp"
#include "racing_segmentation/image_utils.hpp"
#include "rclcpp/rclcpp.hpp"
#include "ai_msgs/msg/perception_targets.hpp"
#include "sensor_msgs/msg/region_of_interest.hpp"
#include "geometry_msgs/msg/point32.hpp"
#include <chrono>
#include <memory>
#include <iostream>
#include <iomanip> 


int main(int argc, char * argv[])
{
    // --- 1. 初始化 ROS 2 ---
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("racing_segmentation_node");

    RCLCPP_INFO(node->get_logger(), "ROS 2 Racing Segmentation Node has been started.");

    // --- 2. 初始化推理模型 ---
    RacingSegmentation racing_segmentation;
    if (racing_segmentation.load_config() != 0) {
        RCLCPP_ERROR(node->get_logger(), "Failed to load config.");
        rclcpp::shutdown();
        return -1;
    }
    if (racing_segmentation.load_bin_model() != 0) {
        RCLCPP_ERROR(node->get_logger(), "Failed to load model.");
        rclcpp::shutdown();
        return -1;
    }
    RCLCPP_INFO(node->get_logger(), "Inference model loaded successfully.");

    // --- 3. 创建发布者 ---
    // 话题名 /racing_segmentation, 消息类型 PerceptionTargets, 队列大小 10
    auto publisher = node->create_publisher<ai_msgs::msg::PerceptionTargets>("/racing_segmentation", 10);
    
    // --- 4. 准备固定的测试图片 ---
    const std::string image_path = "/userdata/racing_ws/src/racing_segmentation/demo/test.jpg";
    const int model_input_w = 640;
    const int model_input_h = 640;
    
    cv::Mat original_image = cv::imread(image_path);
    if (original_image.empty()) {
        RCLCPP_ERROR(node->get_logger(), "Failed to read test image from %s", image_path.c_str());
        return -1;
    }
    const int original_img_w = original_image.cols;
    const int original_img_h = original_image.rows;

    cv::Mat nv12_mat = image_to_nv12(image_path, model_input_w, model_input_h);
    uint8_t* nv12_data_ptr = nv12_mat.data;
    
    // --- 5. 设置循环发布，频率为 1 Hz (每秒1次) ---
    rclcpp::WallRate loop_rate(1.0);

    while (rclcpp::ok()) {
        RCLCPP_INFO(node->get_logger(), "Running detection...");
        
        // --- 5.1 执行检测 ---
        std::vector<DetectionResult> detection_results;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (racing_segmentation.detect(nv12_data_ptr, original_img_w, original_img_h, detection_results) != 0) {
            RCLCPP_WARN(node->get_logger(), "Detection returned with an error.");
        }
        
        // --- 5.2 准备和填充 ROS 消息 ---
        auto perception_msg = std::make_unique<ai_msgs::msg::PerceptionTargets>();
        
        // 填充消息头
        perception_msg->header.stamp = node->get_clock()->now();
        perception_msg->header.frame_id = "camera_output_frame";
        
        // 转换检测结果
        racing_segmentation.convert_to_ros_msg(detection_results, *perception_msg);

        // --- 5.3 发布消息 ---
        publisher->publish(std::move(perception_msg));
        RCLCPP_INFO(node->get_logger(), "Published %zu perception targets.", detection_results.size());

        // --- 5.4 等待下一个循环 ---
        rclcpp::spin_some(node); // 处理ROS事件，例如订阅回调（虽然我们这里没有）
        loop_rate.sleep();     // 等待直到1秒钟结束
    }

    // --- 6. 关闭ROS 2 ---
    rclcpp::shutdown();
    RCLCPP_INFO(node->get_logger(), "Node shutting down.");
    return 0;
}
