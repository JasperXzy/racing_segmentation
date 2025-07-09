#include "racing_segmentation/parser.hpp"
#include "racing_segmentation/image_utils.hpp"

int main()
{
    RacingSegmentation racing_segmentation;
    racing_segmentation.load_config();
    racing_segmentation.load_bin_model();

    return 0;
}
