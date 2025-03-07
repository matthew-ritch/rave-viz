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

void convolve_through_time(const std::vector<cv::Mat>* input_stack, cv::Mat* output, const double* kernel, int kernel_size) {
    // Check for null pointers
    if (input_stack == nullptr || output == nullptr || kernel == nullptr) {
        std::cout << "DEBUG: ERROR - Null pointer passed to convolve1d!" << std::endl;
        return;
    }
    
    // Check if input is empty
    if (input_stack->empty()) {
        std::cout << "DEBUG: ERROR - Empty input stack in convolve1d!" << std::endl;
        return;
    }
    
    // Get dimensions from the first image in the stack
    const cv::Mat& first_frame = input_stack->front();
    int rows = first_frame.rows;
    int cols = first_frame.cols;
    int channels = first_frame.channels();
    int stack_size = input_stack->size();
    
    std::cout << "DEBUG: convolve1d - Stack size: " << stack_size << std::endl;
    std::cout << "DEBUG: convolve1d - Frame size: " << first_frame.size() << ", channels: " << channels << std::endl;
    
    // Initialize output if necessary
    if (output->empty() || output->size() != first_frame.size() || output->type() != first_frame.type()) {
        std::cout << "DEBUG: Initializing output matrix to match input" << std::endl;
        *output = cv::Mat::zeros(first_frame.size(), first_frame.type());
    }
    
    try {
        // For each position in the 2D image
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // For each color channel
                for (int ch = 0; ch < channels; ch++) {
                    double sum = 0.0;
                    
                    // Apply kernel through the time dimension (stack)
                    for (int k = 0; k < kernel_size; k++) {
                        // Calculate frame index with proper wrapping
                        int frame_idx = (stack_size + (k - kernel_size/2)) % stack_size;
                        
                        // Bounds check (shouldn't be needed with proper modulo, but kept for safety)
                        if (frame_idx < 0 || frame_idx >= stack_size) {
                            std::cout << "DEBUG: ERROR - Frame index out of bounds: " << frame_idx << std::endl;
                            continue;
                        }
                        
                        // Access the frame and apply kernel weight
                        const cv::Mat& frame = (*input_stack)[frame_idx];
                        sum += kernel[k] * frame.at<cv::Vec3d>(i, j)[ch];
                    }
                    
                    // Store the result
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

void convolve_through_time2(const std::vector<cv::Mat>* input_stack, std::vector<cv::Mat>* output_stack, const double* kernel, int kernel_size){
    // Check for null pointers
    if (input_stack == nullptr || output_stack == nullptr || kernel == nullptr) {
        std::cout << "DEBUG: ERROR - Null pointer passed to convolve1d!" << std::endl;
        return;
    }
    
    // Check if input is empty
    if (input_stack->empty()) {
        std::cout << "DEBUG: ERROR - Empty input stack in convolve1d!" << std::endl;
        return;
    }
    
    // Get dimensions from the first image in the stack
    const cv::Mat& first_frame = input_stack->front();
    int rows = first_frame.rows;
    int cols = first_frame.cols;
    int channels = first_frame.channels();
    int stack_size = input_stack->size();
    
    std::cout << "DEBUG: convolve1d - Stack size: " << stack_size << std::endl;
    std::cout << "DEBUG: convolve1d - Frame size: " << first_frame.size() << ", channels: " << channels << std::endl;
    
    // Initialize output if necessary
    if (output_stack->empty() || output_stack->front().size() != first_frame.size() || output_stack->front().type() != first_frame.type()) {
        std::cout << "DEBUG: Initializing output matrix to match input" << std::endl;
        output_stack->clear();
        output_stack->resize(stack_size);
        for (int t = 0; t < stack_size; t++) {
            (*output_stack)[t] = cv::Mat::zeros(first_frame.size(), first_frame.type());
        }
    }
    
    try {
        // For each position in the 2D image
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // For each color channel
                for (int ch = 0; ch < channels; ch++) {
                    double sum = 0.0;
                    // For each layer in the stack
                    for (int t = 0; t < stack_size; t++) {
                        // Apply kernel through the time dimension (stack)
                        for (int k = 0; k < kernel_size; k++) {
                            // Calculate frame index with proper wrapping
                            int frame_idx = (stack_size + (k - kernel_size/2)) % stack_size;
                            
                            // Bounds check (shouldn't be needed with proper modulo, but kept for safety)
                            if (frame_idx < 0 || frame_idx >= stack_size) {
                                std::cout << "DEBUG: ERROR - Frame index out of bounds: " << frame_idx << std::endl;
                                continue;
                            }
                            
                            // Access the frame and apply kernel weight
                            const cv::Mat& frame = (*input_stack)[frame_idx];
                            sum += kernel[k] * frame.at<cv::Vec3d>(i, j)[ch];
                        }
                        
                        // Store the result
                        (*output_stack)[t].at<cv::Vec3d>(i, j)[ch] = sum;
                    }
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