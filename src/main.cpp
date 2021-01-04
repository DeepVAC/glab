/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "glab.h"
#include "gemfield.h"
#include "syszux_ocr_pse.h"

using namespace deepvac;

cv::Mat crop_rect(cv::Mat img_, cv::RotatedRect rect) {
    cv::Mat img = img_.clone();
    cv::Point center;
    cv::Size size;
    center.x = (int)rect.center.x;
    center.y = (int)rect.center.y;
    size.width = (int)rect.size.width;
    size.height = (int)rect.size.height;
    float angle = rect.angle;

    if (rect.size.width < rect.size.height) {
        angle = angle + 90;
        auto tmp = size.width;
        size.width = size.height;
        size.height = tmp;
    }
    int height =  img.rows;
    int width = img.cols;
    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1);
    cv::Mat img_rot, img_crop;
    cv::warpAffine(img, img_rot, M, cv::Size(width, height), cv::INTER_CUBIC);
    cv::getRectSubPix(img_rot, size, center, img_crop);
    return img_crop;
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        GEMFIELD_E("usage: deepvac <device> <det_model_path> <img_path>");
        return -1;
    }
    std::string device = argv[1];
    std::string det_model_path = argv[2];
    std::string img_path = argv[3];
    
    int long_size = 1280;
    int crop_gap = 10;
    int text_min_area = 300;
    float text_mean_score = 0.90;

    SyszuxOcrPse ocr_detect;
    ocr_detect.setDevice(device);
    ocr_detect.setModel(det_model_path);

    ocr_detect.set(long_size, crop_gap, text_min_area, text_mean_score);

    auto mat_opt = gemfield_org::img2CvMat(img_path);
    if(!mat_opt){
        throw std::runtime_error("illegal image detected");
        return 1;
    }
	
    auto mat_out = mat_opt.value();
    auto detect_out_opt = ocr_detect.process(mat_out);
    if(!detect_out_opt){
        throw std::runtime_error("no text detected");
    }
    std::vector<cv::RotatedRect> detect_out = detect_out_opt.value();
    std::cout << "before glab rect: " << detect_out.size() << std::endl;

    DeepvacOcrFrame ocr_frame(mat_out, detect_out);
    auto result_opt = ocr_frame();
    if (!result_opt) {
        throw std::runtime_error("illegal image detected");
        return 1;
    }
    std::vector<AggressiveBox> result = result_opt.value();
    std::cout << "after glab rect: " << result.size() << std::endl;
    for (int i; i<result.size(); i++) {
        auto box = result[i].getRect();
        cv::Mat img_crop = crop_rect(mat_out, box);
        std::string save_name = "ocr2_"+std::to_string(i)+".jpg";
        cv::imwrite("vis/"+save_name, img_crop);
    }
    return 0;
}

