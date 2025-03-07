#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

void convolve1d(const cv::Mat* input, cv::Mat* output, const double* kernel, int kernel_size);
void convolve_through_time(const std::vector<cv::Mat>* input_stack, cv::Mat* output, const double* kernel, int kernel_size);
void convolve_through_time2(const std::vector<cv::Mat>* input_stack, std::vector<cv::Mat>* output_stack, const double* kernel, int kernel_size);
void sobel_y2(const cv::Mat* input, cv::Mat* output);
void sobel_x2(const cv::Mat* input, cv::Mat* output);

#endif
