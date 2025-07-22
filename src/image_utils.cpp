#include "racing_segmentation/image_utils.hpp"

void letterbox_nv12(const uint8_t* nv12, int src_w, int src_h,
                    uint8_t* out_nv12, int dst_w, int dst_h,
                    int& x_shift, int& y_shift, float& x_scale, float& y_scale) {
    // 1. 拆分Y/UV
    const uint8_t* y_in = nv12;
    const uint8_t* uv_in = nv12 + src_w * src_h;
 
    // 2. 构建Y和UV的Mat，分别处理
    cv::Mat y_plane_in(src_h, src_w, CV_8UC1, (void*)y_in);
    cv::Mat uv_plane_in(src_h/2, src_w/2, CV_8UC2, (void*)uv_in);
 
    // 3. 计算缩放 & padding
    x_scale = std::min(1.0 * dst_h / src_h, 1.0 * dst_w / src_w);
    y_scale = x_scale;
 
    int new_w = int(src_w * x_scale + 0.5);
    int new_h = int(src_h * y_scale + 0.5);
 
    x_shift = (dst_w - new_w) / 2;
    int x_other = dst_w - new_w - x_shift;
    y_shift = (dst_h - new_h) / 2;
    int y_other = dst_h - new_h - y_shift;
 
    // 4. 分别resize、letterbox Y和UV平面
 
    // Y
    cv::Mat y_plane_resize, y_plane_out;
    cv::resize(y_plane_in, y_plane_resize, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(y_plane_resize, y_plane_out, y_shift, y_other, x_shift, x_other, cv::BORDER_CONSTANT, cv::Scalar(127));
 
    // UV（UV分辨率减半）
    cv::Mat uv_resize, uv_out;
    int uv_new_w = new_w / 2;
    int uv_new_h = new_h / 2;
    int uv_x_shift = x_shift / 2;
    int uv_x_other = (dst_w/2) - uv_new_w - uv_x_shift;
    int uv_y_shift = y_shift / 2;
    int uv_y_other = (dst_h/2) - uv_new_h - uv_y_shift;
 
    cv::resize(uv_plane_in, uv_resize, cv::Size(uv_new_w, uv_new_h), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(uv_resize, uv_out, uv_y_shift, uv_y_other, uv_x_shift, uv_x_other, cv::BORDER_CONSTANT, cv::Scalar(127, 127));
 
    // 5. 合并为 NV12
    // Y
    memcpy(out_nv12, y_plane_out.data, dst_w * dst_h);
    // UV
    memcpy(out_nv12 + dst_w * dst_h, uv_out.data, (dst_w/2) * (dst_h/2) * 2);
}

cv::Mat image_to_nv12(const std::string& image_path, int target_width, int target_height)
{
    // 1. 加载图片
    cv::Mat bgr_img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr_img.empty())
    {
        throw std::runtime_error("无法加载图片: " + image_path);
    }

    // 2. 保持宽高比进行缩放和填充
    cv::Mat processed_img;
    const int img_h = bgr_img.rows;
    const int img_w = bgr_img.cols;

    // 计算缩放比例
    double scale = std::min(static_cast<double>(target_width) / img_w, static_cast<double>(target_height) / img_h);
    
    int scaled_w = static_cast<int>(img_w * scale);
    int scaled_h = static_cast<int>(img_h * scale);

    cv::Mat resized_img;
    cv::resize(bgr_img, resized_img, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);

    // 计算需要填充的边界大小
    int top_pad = (target_height - scaled_h) / 2;
    int bottom_pad = target_height - scaled_h - top_pad;
    int left_pad = (target_width - scaled_w) / 2;
    int right_pad = target_width - scaled_w - left_pad;

    // 填充图像以达到目标尺寸
    cv::copyMakeBorder(resized_img, processed_img, top_pad, bottom_pad, left_pad, right_pad, 
                       cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));

    // 3. 将BGR图像转换为YUV I420格式
    cv::Mat yuv_i420_mat;
    cv::cvtColor(processed_img, yuv_i420_mat, cv::COLOR_BGR2YUV_I420);

    // 4. 从I420手动转换为NV12
    
    // 创建用于存放NV12数据
    // 高度为 1.5 * H, 宽度为 W, 单通道
    cv::Mat nv12_mat(target_height * 3 / 2, target_width, CV_8UC1);

    // 获取数据指针
    const uint8_t* y_i420 = yuv_i420_mat.data;
    const int y_size = target_width * target_height;
    
    const int uv_plane_size = target_width * target_height / 4;
    const uint8_t* u_i420 = y_i420 + y_size;
    const uint8_t* v_i420 = u_i420 + uv_plane_size;

    uint8_t* nv12_data = nv12_mat.data;

    // 4.1 复制Y平面
    memcpy(nv12_data, y_i420, y_size);

    // 4.2 交错存储U和V平面
    uint8_t* uv_nv12 = nv12_data + y_size;
    for (int i = 0; i < uv_plane_size; ++i)
    {
        uv_nv12[2 * i]     = u_i420[i]; // U
        uv_nv12[2 * i + 1] = v_i420[i]; // V
    }

    return nv12_mat;
}
