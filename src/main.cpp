#include "racing_segmentation/parser.hpp"
#include "racing_segmentation/image_utils.hpp"

int main()
{
    const std::string image_path = "/userdata/racing_ws/src/racing_segmentation/demo/test.jpg"; 
    const int input_W = 640;
    const int input_H = 640;

    RacingSegmentation racing_segmentation;
    racing_segmentation.load_config();
    racing_segmentation.load_bin_model();
    
    cv::Mat nv12_mat = image_to_nv12(image_path, input_W, input_H);
    uint8_t* nv12_data_ptr = nv12_mat.data;
    racing_segmentation.detect(nv12_data_ptr);

    return 0;
}
