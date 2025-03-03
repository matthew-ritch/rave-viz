#include "filters.hpp"
#include <string.h>
#include <iostream>

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

void sobel_y2(const cv::Mat* input, cv::Mat* output) {
    cv::Sobel(*input, *output, CV_64F, 0, 2, 1);
}

void sobel_x2(const cv::Mat* input, cv::Mat* output) {
    cv::Sobel(*input, *output, CV_64F, 2, 0, 1);
}
