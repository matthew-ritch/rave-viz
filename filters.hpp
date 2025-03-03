#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>

void convolve1d(const cv::Mat* input, cv::Mat* output, const double* kernel, int kernel_size);
void sobel_y2(const cv::Mat* input, cv::Mat* output);
void sobel_x2(const cv::Mat* input, cv::Mat* output);

// GPU-accelerated convolution
void convolve1d_gpu(cv::cuda::GpuMat* src, cv::cuda::GpuMat* dst, double* kernel, int kernel_size);

#endif
