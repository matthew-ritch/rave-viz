#include "filters.hpp"
#include <string.h>
#include <iostream>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>

void convolve1d(const cv::Mat* input, cv::Mat* output, const double* kernel, int kernel_size) {
    // Check for null pointers
    if (input == nullptr || output == nullptr || kernel == nullptr) {
        std::cout << "DEBUG: ERROR - Null pointer passed to convolve1d!" << std::endl;
        return;
    }
    
    // Check if input is empty
    if (input->empty()) {
        std::cout << "DEBUG: ERROR - Empty input matrix in convolve1d!" << std::endl;
        return;
    }
    
    // Check matrix types and sizes
    std::cout << "DEBUG: convolve1d - Input size: " << input->size() << ", channels: " << input->channels() << std::endl;
    std::cout << "DEBUG: convolve1d - Output size: " << output->size() << ", channels: " << output->channels() << std::endl;
    
    // Initialize output if necessary
    if (output->empty() || output->size() != input->size() || output->type() != input->type()) {
        std::cout << "DEBUG: Initializing output matrix to match input" << std::endl;
        *output = cv::Mat::zeros(input->size(), input->type());
    }
    
    int rows = input->rows;
    int cols = input->cols;
    int channels = input->channels();
    
    try {
        for (int ch = 0; ch < channels; ch++) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < kernel_size; k++) {
                        int idx = (i + k - kernel_size/2 + rows) % rows;
                        
                        // Bounds check
                        if (idx < 0 || idx >= rows) {
                            std::cout << "DEBUG: ERROR - Index out of bounds in convolve1d: " << idx << std::endl;
                            continue;
                        }
                        
                        sum += kernel[k] * input->at<cv::Vec3d>(idx, j)[ch];
                    }
                    output->at<cv::Vec3d>(i, j)[ch] = sum;
                }
            }
        }
    } catch (const cv::Exception& e) {
        std::cout << "DEBUG: OpenCV exception in convolve1d: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "DEBUG: Standard exception in convolve1d: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "DEBUG: Unknown exception in convolve1d!" << std::endl;
    }
}

void convolve1d_gpu(cv::cuda::GpuMat* src, cv::cuda::GpuMat* dst, double* kernel, int kernel_size) {
    try {
        // Create a CPU kernel Matrix
        cv::Mat cpuKernel(1, kernel_size, CV_64FC1);
        for (int i = 0; i < kernel_size; i++) {
            cpuKernel.at<double>(0, i) = kernel[i];
        }
        
        // Convert to floating point format compatible with CUDA filters
        cv::Mat kernelFloat;
        cpuKernel.convertTo(kernelFloat, CV_32FC1);
        
        // Create temporary src/dst if needed to match filter requirements
        cv::cuda::GpuMat srcFloat, dstFloat;
        if (src->type() != CV_32FC3) {
            src->convertTo(srcFloat, CV_32FC3);
        } else {
            srcFloat = *src;
        }
        
        // Create a separable filter for each channel
        cv::Ptr<cv::cuda::Filter> rowFilter = cv::cuda::createSeparableLinearFilter(
            CV_32FC3,
            CV_32FC3,
            kernelFloat,
            cv::Mat::ones(1, 1, CV_32FC1) // Identity for other dimension
        );
        
        // Apply filter
        dstFloat.create(src->size(), CV_32FC3);
        rowFilter->apply(srcFloat, dstFloat);
        
        // Convert back to original format if needed
        if (dst->type() != CV_32FC3) {
            dstFloat.convertTo(*dst, dst->type());
        } else {
            dstFloat.copyTo(*dst);
        }
    }
    catch (const cv::Exception& e) {
        std::cerr << "CUDA filter error: " << e.what() << std::endl;
        
        // Fallback to CPU
        cv::Mat cpu_src, cpu_dst;
        src->download(cpu_src);
        
        // Initialize dst if needed
        if (cpu_dst.size() != cpu_src.size() || cpu_dst.type() != cpu_src.type()) {
            cpu_dst = cv::Mat::zeros(cpu_src.size(), cpu_src.type());
        }
        
        convolve1d(&cpu_src, &cpu_dst, kernel, kernel_size);
        dst->upload(cpu_dst);
    }
}

void sobel_y2(const cv::Mat* input, cv::Mat* output) {
    cv::Sobel(*input, *output, CV_64F, 0, 2, 1);
}

void sobel_x2(const cv::Mat* input, cv::Mat* output) {
    cv::Sobel(*input, *output, CV_64F, 2, 0, 1);
}
