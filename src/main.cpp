#include "racing_segmentation/parser.hpp"
#include "racing_segmentation/image_utils.hpp"

#include "rclcpp/rclcpp.hpp"
#include "hbm_img_msgs/msg/hbm_msg1080_p.hpp"
#include "ai_msgs/msg/perception_targets.hpp"

#include <chrono>
#include <memory>
#include <vector>

class SegmentationNode : public rclcpp::Node
{
public:
    SegmentationNode(RacingSegmentation& detector) 
        : Node("segmentation_node"), 
          detector_(detector),
          frame_count_(0)
    {
        RCLCPP_INFO(this->get_logger(), "Initializing SegmentationNode...");

        publisher_ = this->create_publisher<ai_msgs::msg::PerceptionTargets>("/racing_segmentation", 10);

        subscription_ = this->create_subscription<hbm_img_msgs::msg::HbmMsg1080P>(
            "/hbmem_img", 
            rclcpp::SensorDataQoS(),
            std::bind(&SegmentationNode::image_callback, this, std::placeholders::_1));
            
        last_time_ = std::chrono::steady_clock::now();
        RCLCPP_INFO(this->get_logger(), "Node initialized. Subscribing to /hbmem_img, publishing to /racing_segmentation.");
    }

private:
    void image_callback(const hbm_img_msgs::msg::HbmMsg1080P::ConstSharedPtr msg)
    {
        profiler_.start("Total_Callback"); 
        frame_count_++;

        const int src_w = 640; 
        const int src_h = 480;
        const int dst_w = 640;
        const int dst_h = 640;
 
        std::vector<uint8_t> output_nv12(dst_w * dst_h * 3 / 2);
 
        int x_shift, y_shift;
        float scale;

        profiler_.start("A_Preprocess_Letterbox");
        letterbox_nv12(msg->data.data(), src_w, src_h, output_nv12.data(), dst_w, dst_h, x_shift, y_shift, scale, scale);
        profiler_.stop("A_Preprocess_Letterbox");
        profiler_.count("A_Preprocess_Letterbox");

        std::vector<DetectionResult> detection_results;
        profiler_.start("B_Detect_Total"); 
        detector_.detect(output_nv12.data(), detection_results, x_shift, y_shift, scale, profiler_);
        profiler_.stop("B_Detect_Total");
        profiler_.count("B_Detect_Total");
 
        profiler_.start("C_Publish_Results");
        publish_results(detection_results);
        profiler_.stop("C_Publish_Results");
        profiler_.count("C_Publish_Results");

        profiler_.stop("Total_Callback");
        profiler_.count("Total_Callback");

        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_time_).count();
        
        if (duration >= 1) {
            RCLCPP_INFO(this->get_logger(), "Processed %ld frames in %ld seconds (%.2f FPS)",
                  frame_count_, duration, static_cast<float>(frame_count_) / duration); 
            
            profiler_.report(this->get_logger());
            
            profiler_.reset(); 
            frame_count_ = 0;
            last_time_ = now;
        }
    }

    void publish_results(const std::vector<DetectionResult>& results)
    {
        auto targets_msg = std::make_unique<ai_msgs::msg::PerceptionTargets>();

        targets_msg->header.stamp = this->get_clock()->now();
        targets_msg->header.frame_id = "camera_optical_frame";
        
        profiler_.start("C.1_ConvertToROS_Msg");
        detector_.convert_to_ros_msg(results, *targets_msg);
        profiler_.stop("C.1_ConvertToROS_Msg");
        profiler_.count("C.1_ConvertToROS_Msg");
        
        profiler_.start("C.2_ROS_Publish");
        publisher_->publish(std::move(targets_msg));
        profiler_.stop("C.2_ROS_Publish");
        profiler_.count("C.2_ROS_Publish");
    }
 
    RacingSegmentation& detector_;
    rclcpp::Subscription<hbm_img_msgs::msg::HbmMsg1080P>::SharedPtr subscription_;
    rclcpp::Publisher<ai_msgs::msg::PerceptionTargets>::SharedPtr publisher_;
    size_t frame_count_; 
    std::chrono::steady_clock::time_point last_time_;
    std::vector<uint8_t> output_nv12_;
    SimpleProfiler profiler_;
};

int main(int argc, char * argv[]) 
{
    RacingSegmentation detector;
    if (detector.load_config() != 0) {
        std::cerr << "[ERROR] Failed to load detector config." << std::endl;
        return -1;
    }
    if (detector.load_bin_model() != 0) {
        std::cerr << "[ERROR] Failed to load detector model." << std::endl;
        return -1;
    }
    if (detector.init_inference_buffers() != 0) {
        std::cerr << "[ERROR] Failed to initialize inference buffers." << std::endl;
        return -1;
    }
    std::cout << "[INFO] Racing Segmentation Detector initialized successfully." << std::endl;
    std::cout << "==========================================================" << std::endl;

    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<SegmentationNode>(detector);
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}
