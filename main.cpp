#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp> // Add this for CUDA arithmetic operations
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include "filters.hpp"

#define MAX_FRAMES 10
#define MIN_FRAMES 5
#define N_REPS 3
#define USE_GPU true // Flag to enable/disable GPU processing

typedef struct {
    cv::Mat* frames;
    cv::Mat* u_t;
    int n_frames;
} VideoBuffer;

void free_video_buffer(VideoBuffer* vb) {
    if (vb == nullptr) {
        return;
    }
    
    if (vb->u_t != nullptr) {
        delete[] vb->u_t;
        vb->u_t = nullptr;
    }
    
    if (vb->frames != nullptr) {
        delete[] vb->frames;
        vb->frames = nullptr;
    }
    
    vb->n_frames = 0;
}

void init_video_buffer(VideoBuffer* vb, cv::VideoCapture* cap, int n) {
    if (vb == nullptr || cap == nullptr) {
        std::cerr << "Error - null pointer passed to init_video_buffer" << std::endl;
        return;
    }
    
    // Free any previously allocated memory
    free_video_buffer(vb);
    
    vb->n_frames = n;
    
    try {
        vb->frames = new cv::Mat[n];
        vb->u_t = new cv::Mat[n];
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        free_video_buffer(vb);
        return;
    }
    
    cv::Mat frame;
    for (int i = 0; i < n; i++) {
        if (!cap->read(frame)) {
            continue;
        }
        
        frame.convertTo(vb->frames[i], CV_64FC3, 1.0/255.0);
        cv::flip(vb->frames[i], vb->frames[i], 1);
        vb->u_t[i] = cv::Mat::zeros(frame.size(), CV_64FC3);
    }
}

// Function to process a single frame - can be executed in parallel
void process_frame(cv::Mat* frame, cv::Mat* u_t, double* filt) {
    cv::Mat delta = cv::Mat::zeros(frame->size(), CV_64FC3);
    
    for (int k = 0; k < N_REPS; k++) {
        convolve1d(frame, &delta, filt, 3);
        *u_t += delta;
        *frame += *u_t;
    }
    
    // Clip values
    cv::max(*frame, 0.0, *frame);
    cv::min(*frame, 1.0, *frame);
}

// GPU version of process_frame using CUDA
void process_frame_gpu(cv::Mat* frame, cv::Mat* u_t, double* filt) {
    try {
        // Upload data to GPU
        cv::cuda::GpuMat d_frame, d_u_t, d_delta;
        d_frame.upload(*frame);
        d_u_t.upload(*u_t);
        d_delta = cv::cuda::GpuMat(frame->size(), CV_64FC3, cv::Scalar(0, 0, 0));
        
        for (int k = 0; k < N_REPS; k++) {
            // Use optimized GPU convolution from filters.hpp
            convolve1d_gpu(&d_frame, &d_delta, filt, 3);
            
            // Update u_t and frame directly on GPU using cv::cuda::add
            cv::cuda::addWeighted(d_u_t, 1.0, d_delta, 1.0, 0.0, d_u_t);
            cv::cuda::addWeighted(d_frame, 1.0, d_u_t, 1.0, 0.0, d_frame);
        }
        
        // Clip values on GPU
        // Instead of threshold, use min/max operations
        cv::cuda::GpuMat zeros(d_frame.size(), d_frame.type(), cv::Scalar(0, 0, 0));
        cv::cuda::GpuMat ones(d_frame.size(), d_frame.type(), cv::Scalar(1, 1, 1));
        
        // Max operation (replaces threshold TOZERO)
        cv::cuda::max(d_frame, zeros, d_frame);
        
        // Min operation (replaces threshold TRUNC)
        cv::cuda::min(d_frame, ones, d_frame);
        
        // Download results back to CPU
        d_frame.download(*frame);
        d_u_t.download(*u_t);
    }
    catch (const cv::Exception& e) {
        std::cerr << "CUDA error: " << e.what() << std::endl;
        // Fall back to CPU processing
        process_frame(frame, u_t, filt);
    }
}

bool check_gpu_support() {
    int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
    if (deviceCount == 0) {
        std::cout << "No CUDA support available" << std::endl;
        return false;
    }
    
    std::cout << "CUDA enabled devices detected: " << deviceCount << std::endl;
    cv::cuda::printCudaDeviceInfo(0); // Print info for the first device
    return true;
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        return -1;
    }
    
    // Set camera parameters
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 960);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 600);
    
    // Check for GPU support
    bool use_gpu = USE_GPU && check_gpu_support();
    
    // Initialize random seed
    srand(time(NULL));
    
    // Initialize video buffer
    VideoBuffer vb = {nullptr, nullptr, 0};
    init_video_buffer(&vb, &cap, 5);
    
    cv::namedWindow("stream", cv::WINDOW_NORMAL);
    
    int i = 0, j = 0;
    double filt[] = {1.0, -1.0};
    
    std::vector<std::thread> threads;
    
    while (true) {
        if (j == 0) {
            int n = MIN_FRAMES + rand() % (MAX_FRAMES - MIN_FRAMES + 1);
            init_video_buffer(&vb, &cap, n);
            i = 0;
        }
        
        // Display current frame without resizing
        cv::Mat display;
        if (!vb.frames[i].empty()) {
            vb.frames[i].convertTo(display, CV_8UC3, 255.0);
            cv::imshow("stream", display);
        }
        
        // Capture and update current frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            break;
        }
        
        frame.convertTo(vb.frames[i], CV_64FC3, 1.5/255.0);
        cv::flip(vb.frames[i], vb.frames[i], 1);
        
        // Apply effects using either CPU or GPU
        if (use_gpu) {
            process_frame_gpu(&vb.frames[i], &vb.u_t[i], filt);
        } else {
            // Create worker thread for processing
            threads.push_back(std::thread(process_frame, &vb.frames[i], &vb.u_t[i], filt));
            
            // Limit max number of threads to avoid oversubscription
            if (threads.size() >= std::thread::hardware_concurrency()) {
                for (auto& t : threads) {
                    if (t.joinable()) {
                        t.join();
                    }
                }
                threads.clear();
            }
        }
        
        i = (i + 1) % vb.n_frames;
        j = (j + 1) % (vb.n_frames * 5);
        
        char key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
        if (key == 'a') {
            j = 0;
        }
    }
    
    // Join any remaining threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    // Clean up resources
    free_video_buffer(&vb);
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
