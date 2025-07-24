#include "racing_segmentation/parser.hpp"

RacingSegmentation::~RacingSegmentation() 
{
    release_inference_buffers();
    if (packed_dnn_handle) {
        release_model();
    }
}

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
    std::cout << "[INFO] Loading Configuration From config/yolo11_seg.json" << std::endl;
    std::ifstream config_file("config/yolo11_seg.json");
    if (!config_file.is_open())
    {
        std::cerr << "[ERROR] Failed to open config file." << std::endl;
        return -1;
    }
 
    nlohmann::json config;
    try {
        config_file >> config;
        model_file = config.at("model_file");
        class_num = config.at("class_num");
        dnn_parser = config.at("dnn_Parser");
        cls_names_list = config.at("cls_names_list").get<std::vector<std::string>>();
        preprocess_type = config.at("preprocess_type");
        score_threshold = config.at("score_threshold");
        nms_threshold = config.at("nms_threshold");
        nms_top_k = config.at("nms_top_k");
        reg = config.at("reg");
        mces = config.at("mces");
        model_input_w = config.at("model_input_w");
        model_input_h = config.at("model_input_h");

    } catch (nlohmann::json::exception& e) {
        std::cerr << "[ERROR] JSON parse error: " << e.what() << std::endl;
        return -1;
    }

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
    std::cout << "[INFO] Model Input Width: " << model_input_w << std::endl;
    std::cout << "[INFO] Model Input Height: " << model_input_h << std::endl;
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
    const char *model_file_name = model_file.c_str();
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

    // 如果这个bin模型有多个打包，则只使用第一个
    if (model_count > 1)
    {
        std::cout << "[WARN] This model file have more than 1 model, only use model 0.";
    }
    const char *model_name = model_name_list[0];
    std::cout << "[INFO] Model Name: " << model_name << std::endl;

    // 3. 获得Packed模型的第一个模型的handle
    rdk_check_success(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "hbDNNGetModelHandle failed");

    // 4. 模型输入检查
    int32_t input_count = 0;
    rdk_check_success(
        hbDNNGetInputCount(&input_count, dnn_handle),
        "hbDNNGetInputCount failed");

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
        std::cout << "[INFO] Outputs Order Check SUCCESS" << std::endl;
        std::cout << "[INFO] Order = {";
        for (int i = 0; i < 10; i++)
        {
            std::cout << order[i] << ", ";
        }
        std::cout << "}" << std::endl;
    }
    else
    {
        std::cout << "[INFO] Outputs Order Check FAILED, use default" << std::endl;
        for (int i = 0; i < 10; i++)
            order[i] = i;
    }

    std::cout << "[INFO] Load Binary Model Successfully!" << std::endl;
    std::cout << "================================================" << std::endl << std::endl;

    return 0;
}

int RacingSegmentation::detect(uint8_t* ynv12, int original_w, int original_h, std::vector<DetectionResult>& results)
{
    results.clear();

    // 1. 准备输入输出和推理
    memcpy(input_tensor.sysMem[0].virAddr, ynv12, int(3 * input_H * input_W / 2));
    hbSysFlushMem(&input_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    
    // 2. 执行推理
    hbDNNTaskHandle_t task_handle = nullptr;
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    hbDNNInfer(&task_handle, &output, &input_tensor, dnn_handle, &infer_ctrl_param);
    hbDNNWaitTaskDone(task_handle, 0);

    // 后处理
    // 在函数内部计算维度
    const int32_t H_4 = input_H / 4, W_4 = input_W / 4;
    const int32_t H_8 = input_H / 8, W_8 = input_W / 8;
    const int32_t H_16 = input_H / 16, W_16 = input_W / 16;
    const int32_t H_32 = input_H / 32, W_32 = input_W / 32;

    float CONF_THRES_RAW = -log(1 / score_threshold - 1);
    std::vector<std::vector<cv::Rect2d>> bboxes(class_num);
    std::vector<std::vector<float>> scores(class_num);
    std::vector<std::vector<std::vector<float>>> maskes(class_num);

    // 3.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    if (output[order[9]].properties.quantiType != SCALE)
    {
        std::cout << "[Error] output[order[9]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }
    // 3.2 对缓存的BPU内存进行刷新
    hbSysFlushMem(&(output[order[9]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    // 3.3 将BPU推理完的内存地址转换为对应类型的指针
    auto *proto_data = reinterpret_cast<int16_t *>(output[order[9]].sysMem[0].virAddr);
    float proto_scale_data = output[order[9]].properties.scale.scaleData[0];
    // 3.4 反量化
    std::vector<float> proto(H_4 * W_4 * mces);

    for (int h = 0; h < H_4; h++)
    {
        for (int w = 0; w < W_4; w++)
        {
            for (int c = 0; c < mces; c++)
            {
                // 索引计算需要修正，根据tensor的存储布局 (h, w, c)
                int index = (h * W_4 * mces) + (w * mces) + c;
                proto[index] = static_cast<float>(proto_data[index]) * proto_scale_data;
            }
        }
    }
    
    // 4. 小目标特征图
    // output[order[0]]: (1, H // 8,  W // 8,  class_num)
    // output[order[1]]: (1, H // 8,  W // 8,  4 * reg)
    // output[order[2]]: (1, H // 8,  W // 8,  mces)

    // 4.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    if (output[order[0]].properties.quantiType != NONE)
    {
        std::cout << "[Error] output[order[0]] QuantiType is not NONE, please check!" << std::endl;
        return -1;
    }
    if (output[order[1]].properties.quantiType != SCALE)
    {
        std::cout << "[Error] output[order[1]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }
    if (output[order[2]].properties.quantiType != SCALE)
    {
        std::cout << "[Error] output[order[2]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }

    // 4.2 对缓存的BPU内存进行刷新
    hbSysFlushMem(&(output[order[0]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[1]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[2]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    // 4.3 将BPU推理完的内存地址转换为对应类型的指针
    auto *s_cls_raw = reinterpret_cast<float *>(output[order[0]].sysMem[0].virAddr);
    auto *s_bbox_raw = reinterpret_cast<int32_t *>(output[order[1]].sysMem[0].virAddr);
    auto *s_bbox_scale = reinterpret_cast<float *>(output[order[1]].properties.scale.scaleData);
    auto *s_mces_raw = reinterpret_cast<int32_t *>(output[order[2]].sysMem[0].virAddr);
    auto *s_mces_scale = reinterpret_cast<float *>(output[order[2]].properties.scale.scaleData);
    for (int h = 0; h < H_8; h++)
    {
        for (int w = 0; w < W_8; w++)
        {
            // 4.4 取对应H和W位置的C通道, 记为数组的形式
            // cls对应class_num个分数RAW值, 也就是Sigmoid计算之前的值，这里利用函数单调性先筛选, 再计算
            // bbox对应4个坐标乘以reg的RAW值, 也就是DFL计算之前的值, 仅仅分数合格了, 才会进行这部分的计算
            float *cur_s_cls_raw = s_cls_raw;
            int32_t *cur_s_bbox_raw = s_bbox_raw;
            int32_t *cur_s_mces_raw = s_mces_raw;

            // 4.5 找到分数的最大值索引, 如果最大值小于阈值，则舍去
            int cls_id = 0;
            for (int i = 1; i < class_num; i++)
            {
                if (cur_s_cls_raw[i] > cur_s_cls_raw[cls_id])
                {
                    cls_id = i;
                }
            }

            // 4.6 不合格则直接跳过, 避免无用的反量化, DFL和dist2bbox计算
            if (cur_s_cls_raw[cls_id] < CONF_THRES_RAW)
            {
                s_cls_raw += class_num;
                s_bbox_raw += reg * 4;
                s_mces_raw += mces;
                continue;
            }

            // 4.7 计算这个目标的分数
            float score = 1 / (1 + std::exp(-cur_s_cls_raw[cls_id]));

            // 4.8 对bbox_raw信息进行反量化, DFL计算
            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++)
            {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < reg; j++)
                {
                    int index_id = reg * i + j;
                    dfl = std::exp(float(cur_s_bbox_raw[index_id]) * s_bbox_scale[index_id]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            // 4.9 剔除不合格的框
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
            {
                s_cls_raw += class_num;
                s_bbox_raw += reg * 4;
                s_mces_raw += mces;
                continue;
            }

            // 4.10 dist 2 bbox (ltrb 2 xyxy)
            float x1 = (w + 0.5 - ltrb[0]) * 8.0;
            float y1 = (h + 0.5 - ltrb[1]) * 8.0;
            float x2 = (w + 0.5 + ltrb[2]) * 8.0;
            float y2 = (h + 0.5 + ltrb[3]) * 8.0;

            // 4.11 对应类别加入到对应的std::vector中
            bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            scores[cls_id].push_back(score);

            // 提取掩码系数并反量化
            std::vector<float> mask_coeffs(mces);
            for (int i = 0; i < mces; i++)
            {
                mask_coeffs[i] = float(cur_s_mces_raw[i]) * s_mces_scale[i];
            }
            maskes[cls_id].push_back(mask_coeffs);

            s_cls_raw += class_num;
            s_bbox_raw += reg * 4;
            s_mces_raw += mces;
        }
    }

    // 5. 中目标特征图
    // output[order[3]]: (1, H // 16,  W // 16,  class_num)
    // output[order[4]]: (1, H // 16,  W // 16,  4 * reg)
    // output[order[5]]: (1, H // 16,  W // 16,  mces)

    // 5.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    if (output[order[3]].properties.quantiType != NONE)
    {
        std::cout << "[Error] output[order[3]] QuantiType is not NONE, please check!" << std::endl;
        return -1;
    }
    if (output[order[4]].properties.quantiType != SCALE)
    {
        std::cout << "[Error] output[order[4]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }
    if (output[order[5]].properties.quantiType != SCALE)
    {
        std::cout << "[Error] output[order[5]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }

    // 5.2 对缓存的BPU内存进行刷新
    hbSysFlushMem(&(output[order[3]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[4]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[5]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    // 5.3 将BPU推理完的内存地址转换为对应类型的指针
    auto *m_cls_raw = reinterpret_cast<float *>(output[order[3]].sysMem[0].virAddr);
    auto *m_bbox_raw = reinterpret_cast<int32_t *>(output[order[4]].sysMem[0].virAddr);
    auto *m_bbox_scale = reinterpret_cast<float *>(output[order[4]].properties.scale.scaleData);
    auto *m_mces_raw = reinterpret_cast<int32_t *>(output[order[5]].sysMem[0].virAddr);
    auto *m_mces_scale = reinterpret_cast<float *>(output[order[5]].properties.scale.scaleData);

    for (int h = 0; h < H_16; h++)
    {
        for (int w = 0; w < W_16; w++)
        {
            // 5.4 取对应H和W位置的C通道, 记为数组的形式
            // cls对应class_num个分数RAW值, 也就是Sigmoid计算之前的值，这里利用函数单调性先筛选, 再计算
            // bbox对应4个坐标乘以reg的RAW值, 也就是DFL计算之前的值, 仅仅分数合格了, 才会进行这部分的计算
            float *cur_m_cls_raw = m_cls_raw;
            int32_t *cur_m_bbox_raw = m_bbox_raw;
            int32_t *cur_m_mces_raw = m_mces_raw;

            // 5.5 找到分数的最大值索引, 如果最大值小于阈值，则舍去
            int cls_id = 0;
            for (int i = 1; i < class_num; i++)
            {
                if (cur_m_cls_raw[i] > cur_m_cls_raw[cls_id])
                {
                    cls_id = i;
                }
            }

            // 5.6 不合格则直接跳过, 避免无用的反量化, DFL和dist2bbox计算
            if (cur_m_cls_raw[cls_id] < CONF_THRES_RAW)
            {
                m_cls_raw += class_num;
                m_bbox_raw += reg * 4;
                m_mces_raw += mces;
                continue;
            }

            // 5.7 计算这个目标的分数
            float score = 1 / (1 + std::exp(-cur_m_cls_raw[cls_id]));

            // 5.8 对bbox_raw信息进行反量化, DFL计算
            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++)
            {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < reg; j++)
                {
                    int index_id = reg * i + j;
                    dfl = std::exp(float(cur_m_bbox_raw[index_id]) * m_bbox_scale[index_id]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            // 5.9 剔除不合格的框
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
            {
                m_cls_raw += class_num;
                m_bbox_raw += reg * 4;
                m_mces_raw += mces;
                continue;
            }

            // 5.10 dist 2 bbox (ltrb 2 xyxy)
            float x1 = (w + 0.5 - ltrb[0]) * 16.0;
            float y1 = (h + 0.5 - ltrb[1]) * 16.0;
            float x2 = (w + 0.5 + ltrb[2]) * 16.0;
            float y2 = (h + 0.5 + ltrb[3]) * 16.0;

            // 5.11 对应类别加入到对应的std::vector中
            bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            scores[cls_id].push_back(score);

            // 提取掩码系数并反量化
            std::vector<float> mask_coeffs(mces);
            for (int i = 0; i < mces; i++)
            {
                mask_coeffs[i] = float(cur_m_mces_raw[i]) * m_mces_scale[i];
            }
            maskes[cls_id].push_back(mask_coeffs);

            m_cls_raw += class_num;
            m_bbox_raw += reg * 4;
            m_mces_raw += mces;
        }
    }

    // 6. 大目标特征图
    // output[order[6]]: (1, H // 32,  W // 32,  class_num)
    // output[order[7]]: (1, H // 32,  W // 32,  4 * reg)
    // output[order[8]]: (1, H // 32,  W // 16,  mces)

    // 6.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    if (output[order[6]].properties.quantiType != NONE)
    {
        std::cout << "[Error] output[order[6]] QuantiType is not NONE, please check!" << std::endl;
        return -1;
    }
    if (output[order[7]].properties.quantiType != SCALE)
    {
        std::cout << "[Error] output[order[7]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }
    if (output[order[8]].properties.quantiType != SCALE)
    {
        std::cout << "[Error] output[order[8]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }

    // 6.2 对缓存的BPU内存进行刷新
    hbSysFlushMem(&(output[order[6]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[7]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[8]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    // 6.3 将BPU推理完的内存地址转换为对应类型的指针
    auto *l_cls_raw = reinterpret_cast<float *>(output[order[6]].sysMem[0].virAddr);
    auto *l_bbox_raw = reinterpret_cast<int32_t *>(output[order[7]].sysMem[0].virAddr);
    auto *l_bbox_scale = reinterpret_cast<float *>(output[order[7]].properties.scale.scaleData);
    auto *l_mces_raw = reinterpret_cast<int32_t *>(output[order[8]].sysMem[0].virAddr);
    auto *l_mces_scale = reinterpret_cast<float *>(output[order[8]].properties.scale.scaleData);

    for (int h = 0; h < H_32; h++)
    {
        for (int w = 0; w < W_32; w++)
        {
            // 6.4 取对应H和W位置的C通道, 记为数组的形式
            // cls对应class_num个分数RAW值, 也就是Sigmoid计算之前的值，这里利用函数单调性先筛选, 再计算
            // bbox对应4个坐标乘以reg的RAW值, 也就是DFL计算之前的值, 仅仅分数合格了, 才会进行这部分的计算
            float *cur_l_cls_raw = l_cls_raw;
            int32_t *cur_l_bbox_raw = l_bbox_raw;
            int32_t *cur_l_mces_raw = l_mces_raw;

            // 6.5 找到分数的最大值索引, 如果最大值小于阈值，则舍去
            int cls_id = 0;
            for (int i = 1; i < class_num; i++)
            {
                if (cur_l_cls_raw[i] > cur_l_cls_raw[cls_id])
                {
                    cls_id = i;
                }
            }

            // 6.6 不合格则直接跳过, 避免无用的反量化, DFL和dist2bbox计算
            if (cur_l_cls_raw[cls_id] < CONF_THRES_RAW)
            {
                l_cls_raw += class_num;
                l_bbox_raw += reg * 4;
                l_mces_raw += mces;
                continue;
            }

            // 6.7 计算这个目标的分数
            float score = 1 / (1 + std::exp(-cur_l_cls_raw[cls_id]));

            // 6.8 对bbox_raw信息进行反量化, DFL计算
            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++)
            {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < reg; j++)
                {
                    int index_id = reg * i + j;
                    dfl = std::exp(float(cur_l_bbox_raw[index_id]) * l_bbox_scale[index_id]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            // 6.9 剔除不合格的框
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
            {
                l_cls_raw += class_num;
                l_bbox_raw += reg * 4;
                l_mces_raw += mces;
                continue;
            }

            // 6.10 dist 2 bbox (ltrb 2 xyxy)
            float x1 = (w + 0.5 - ltrb[0]) * 32.0;
            float y1 = (h + 0.5 - ltrb[1]) * 32.0;
            float x2 = (w + 0.5 + ltrb[2]) * 32.0;
            float y2 = (h + 0.5 + ltrb[3]) * 32.0;

            // 6.11 对应类别加入到对应的std::vector中
            bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            scores[cls_id].push_back(score);

            // 提取掩码系数并反量化
            std::vector<float> mask_coeffs(mces);
            for (int i = 0; i < mces; i++)
            {
                mask_coeffs[i] = float(cur_l_mces_raw[i]) * l_mces_scale[i];
            }
            maskes[cls_id].push_back(mask_coeffs);

            l_cls_raw += class_num;
            l_bbox_raw += reg * 4;
            l_mces_raw += mces;
        }
    }

    // 7. 使用OpenCV的NMS进行过滤
    std::vector<std::vector<cv::Rect2d>> nms_bboxes(class_num);
    std::vector<std::vector<float>> nms_scores(class_num);
    std::vector<std::vector<std::vector<float>>> nms_maskes(class_num);

    for (int cls_id = 0; cls_id < class_num; cls_id++)
    {
        if (bboxes[cls_id].empty()) continue;
        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes[cls_id], scores[cls_id], score_threshold, nms_threshold, indices, 1.0f, nms_top_k);
        for (const auto idx : indices) {
            nms_bboxes[cls_id].push_back(bboxes[cls_id][idx]);
            nms_scores[cls_id].push_back(scores[cls_id][idx]);
            nms_maskes[cls_id].push_back(maskes[cls_id][idx]);
        }
    }

    // 8. 将NMS后的坐标转换为原始图像上的坐标
    // 8.1 计算缩放比例和填充量
    // input_W 和 input_H 是模型输入尺寸
    double scale = std::min(static_cast<double>(input_W) / original_w, static_cast<double>(input_H) / original_h);
    int scaled_w = static_cast<int>(original_w * scale);
    int scaled_h = static_cast<int>(original_h * scale);
    int pad_x = (input_W - scaled_w) / 2; // 左边填充
    int pad_y = (input_H - scaled_h) / 2; // 上边填充

    for (int cls_id = 0; cls_id < class_num; cls_id++)
    {
        for (size_t i = 0; i < nms_bboxes[cls_id].size(); i++)
        {
            // 8.2 将 NMS 后的模型坐标框进行逆变换
            const cv::Rect2d& box_on_model = nms_bboxes[cls_id][i];
            
            float original_box_x = (box_on_model.x - pad_x) / scale;
            float original_box_y = (box_on_model.y - pad_y) / scale;
            float original_box_width = box_on_model.width / scale;
            float original_box_height = box_on_model.height / scale;

            cv::Rect2f final_box(original_box_x, original_box_y, original_box_width, original_box_height);
            
            // 裁剪 box 到原始图像边界内
            float x1_model = std::max(0.0, box_on_model.x);
            float y1_model = std::max(0.0, box_on_model.y);
            float x2_model = std::min(static_cast<double>(input_W), x1_model + box_on_model.width);
            float y2_model = std::min(static_cast<double>(input_H), y1_model + box_on_model.height);
            int mask_w = static_cast<int>(x2_model - x1_model);
            int mask_h = static_cast<int>(y2_model - y1_model);

            if (mask_h <= 0 || mask_w <= 0) continue;

            // 计算掩码
            std::vector<float>& mask_coeffs = nms_maskes[cls_id][i];
            cv::Mat mask_mat(mask_h, mask_w, CV_32F);

            #pragma omp parallel for
            for (int r = 0; r < mask_h; ++r) {
                for (int c = 0; c < mask_w; ++c) {
                    float sum = 0.0f;
                    int proto_y = static_cast<int>((r + y1_model) / 4);
                    int proto_x = static_cast<int>((c + x1_model) / 4);
                    if (proto_y < H_4 && proto_x < W_4) {
                        for (int k = 0; k < mces; ++k) {
                           sum += mask_coeffs[k] * proto[(proto_y * W_4 + proto_x) * mces + k];
                        }
                    }
                    mask_mat.at<float>(r, c) = 1.0f / (1.0f + std::exp(-sum));
                }
            }
            
            cv::Mat binary_mask;
            cv::threshold(mask_mat, binary_mask, 0.5, 255, cv::THRESH_BINARY);
            binary_mask.convertTo(binary_mask, CV_8U);

            // 8.3 填充结果结构体，直接使用转换后的坐标
            DetectionResult res;
            res.class_id = cls_id;
            res.class_name = cls_names_list[cls_id];
            res.score = nms_scores[cls_id][i];
            res.box = final_box;
            res.mask = binary_mask;
            
            results.push_back(res);
        }
    }

    // 9. 释放资源
    hbDNNReleaseTask(task_handle);

    return 0;
}

void RacingSegmentation::convert_to_ros_msg(const std::vector<DetectionResult>& results, ai_msgs::msg::PerceptionTargets& msg) const
{
    msg.targets.clear();
    for (const auto& det_res : results) {
        ai_msgs::msg::Target target;
        target.type = det_res.class_name;

        // 填充边界框信息
        ai_msgs::msg::Roi roi;
        roi.rect.x_offset = static_cast<uint32_t>(det_res.box.x);
        roi.rect.y_offset = static_cast<uint32_t>(det_res.box.y);
        roi.rect.width = static_cast<uint32_t>(det_res.box.width);
        roi.rect.height = static_cast<uint32_t>(det_res.box.height);
        roi.confidence = det_res.score;
        target.rois.push_back(roi);

        // 填充分割掩码轮廓信息
        if (!det_res.mask.empty()) {
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(det_res.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (const auto& contour : contours) {
                ai_msgs::msg::Point point_set;
                point_set.type = "segmentation_contour";
                
                for (const auto& cv_point : contour) {
                    geometry_msgs::msg::Point32 ros_point;
                    ros_point.x = static_cast<float>(cv_point.x);
                    ros_point.y = static_cast<float>(cv_point.y);
                    ros_point.z = 0.0f;
                    point_set.point.push_back(ros_point);
                }
                target.points.push_back(point_set);
            }
        }

        msg.targets.push_back(target);
    }
}

int RacingSegmentation::release_model()
{
    std::cout << "[INFO] Releasing model..." << std::endl;
    if (packed_dnn_handle) {
         hbDNNRelease(packed_dnn_handle);
         packed_dnn_handle = nullptr;
         dnn_handle = nullptr;
    }
    std::cout << "[INFO] Model released." << std::endl;
    return 0;
}

int RacingSegmentation::init_inference_buffers() {
    input_tensor.properties = input_properties;
    rdk_check_success(hbSysAllocCachedMem(&input_tensor.sysMem[0], int(3 * input_H * input_W / 2)), 
                      "hbSysAllocCachedMem for input tensor failed");

    output = new hbDNNTensor[output_count];
    for (int i = 0; i < output_count; i++) {
        hbDNNTensorProperties &output_properties = output[i].properties;
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
        int out_aligned_size = output_properties.alignedByteSize;
        hbSysMem &mem = output[i].sysMem[0];
        rdk_check_success(hbSysAllocCachedMem(&mem, out_aligned_size), 
                          "hbSysAllocCachedMem for output tensor failed");
    }
    return 0;
}

int RacingSegmentation::release_inference_buffers() {
    if (input_tensor.sysMem[0].virAddr) {
        hbSysFreeMem(&(input_tensor.sysMem[0]));
        input_tensor.sysMem[0].virAddr = nullptr;
    }
    if (output) {
        for (int i = 0; i < output_count; i++) {
            if (output[i].sysMem[0].virAddr) {
                hbSysFreeMem(&(output[i].sysMem[0]));
            }
        }
        delete[] output;
        output = nullptr;
    }
    return 0;
}
