#include <opencv2/opencv.hpp>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <functional>
#include "filters.hpp"

#define MAX_FRAMES 7
#define MIN_FRAMES 5
#define N_REPS 3
#define N_CYCLES 7
#define SPEED .15

#pragma GCC optimize("O3")
#pragma GCC target("avx2")

typedef struct {
    std::vector<cv::Mat> frames;
    std::vector<cv::Mat> u_t;
    int n_frames;
} VideoBuffer;

// Helper function to parallelize work using std::thread
template<typename Iterator, typename Function>
void parallel_for(Iterator begin, Iterator end, Function fn, size_t num_threads) {
    size_t range = std::distance(begin, end);
    if (range <= 0)
        return;
    
    if (num_threads <= 0)
        num_threads = std::thread::hardware_concurrency();
    
    size_t chunk_size = (range + num_threads - 1) / num_threads;
    
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads && i * chunk_size < range; ++i) {
        Iterator chunk_begin = std::next(begin, i * chunk_size);
        Iterator chunk_end = (i + 1) * chunk_size < range ? 
                             std::next(begin, (i + 1) * chunk_size) : 
                             end;
        
        threads.push_back(std::thread([=]() {
            for (Iterator it = chunk_begin; it != chunk_end; ++it) {
                fn(*it);
            }
        }));
    }
    
    for (auto& thread : threads) {
        if (thread.joinable())
            thread.join();
    }
}

void free_video_buffer(VideoBuffer* vb) {
    if (vb == nullptr) {
        return;
    }
    
    vb->u_t.clear();
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
    vb->frames.resize(n);
    vb->u_t.resize(n);
    
    // Pre-allocate for better performance
    cv::Mat frame;
    cap->read(frame);
    
    // Use C++ threads for parallel initialization
    std::vector<int> indices(n);
    for (int i = 0; i < n; i++) indices[i] = i;
    
    parallel_for(indices.begin(), indices.end(), [&](int i) {
        vb->frames[i] = cv::Mat(frame.size(), CV_64FC3);
        vb->u_t[i] = cv::Mat::zeros(frame.size(), CV_64FC3);
    }, std::thread::hardware_concurrency());
    
    // Now read all frames with pre-allocated memory
    for (int i = 0; i < n; i++) {
        if (!cap->read(frame)) {
            std::cerr << "Failed to read initial frame " << i << std::endl;
            continue;
        }
        
        frame.convertTo(vb->frames[i], CV_64FC3, 1.0/255.0);
        cv::flip(vb->frames[i], vb->frames[i], 1);
    }
}

// Function to process frames through time
void process_frames(VideoBuffer* vb, double* filt, int kernel_size) {
    if (vb == nullptr || filt == nullptr || vb->frames.empty()) {
        std::cerr << "Invalid parameters in process_frames" << std::endl;
        return;
    }
    
    // Use static allocation to avoid repeated memory allocation
    static std::vector<cv::Mat> deltas;
    if (deltas.size() != vb->n_frames) {
        deltas.resize(vb->n_frames);
    }
    for (int i = 0; i < vb->n_frames; i++) {
        deltas[i] = cv::Mat::zeros(vb->frames[0].size(), CV_64FC3);
    }
    
    // Create a vector of indices to process
    std::vector<int> indices(vb->n_frames);
    for (int i = 0; i < vb->n_frames; i++) indices[i] = i;

    for (int k = 0; k < N_REPS; k++) {
        convolve_through_time2(&vb->frames, &deltas, filt, kernel_size);

    }

    for (int i = 0; i < vb->n_frames; i++) {
        cv::Mat& delta = deltas[i];

        // Update velocity and position (in-place operations)
        cv::scaleAdd(delta, SPEED, vb->u_t[i], vb->u_t[i]);  // u_t += SPEED * delta
        vb->frames[i] += vb->u_t[i];
        
        // Clip values using efficient OpenCV functions
        cv::threshold(vb->frames[i], vb->frames[i], 0.0, 0.0, cv::THRESH_TOZERO);
        cv::threshold(vb->frames[i], vb->frames[i], 1.0, 1.0, cv::THRESH_TRUNC);
    }
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        return -1;
    }
    
    // Set camera parameters for better performance
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 960);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 600);
    cap.set(cv::CAP_PROP_FPS, 30); // Set FPS explicitly
    cap.set(cv::CAP_PROP_BUFFERSIZE, MAX_FRAMES); // Optimize buffer size
    
    // Initialize random seed
    srand(time(NULL));
    
    // Initialize video buffer
    VideoBuffer vb = {std::vector<cv::Mat>(), std::vector<cv::Mat>(), 0};
    init_video_buffer(&vb, &cap, MIN_FRAMES);
    
    cv::namedWindow("stream", cv::WINDOW_NORMAL);
    
    // Pre-allocate display matrix to avoid repeated allocation
    cv::Mat display;
    cv::Mat frame;
    
    int i = 0, j = 0;
    double filt[] = {1.0, -1.0};
    const int kernel_size = 2; // Size of the filter kernel
    
    // Print number of threads being used
    std::cout << "Using " << std::thread::hardware_concurrency() << " threads for processing" << std::endl;
    
    while (true) {
        // If we're starting a new cycle, process all frames
        if (j == 0) {
            // Generate new buffer size
            int n = MIN_FRAMES + rand() % (MAX_FRAMES - MIN_FRAMES + 1);
            init_video_buffer(&vb, &cap, n);
            i = 0;
        }
        
        // Display current frame without resizing
        if (i >= 0 && i < vb.n_frames && !vb.frames[i].empty()) {
            vb.frames[i].convertTo(display, CV_8UC3, 255.0);
            cv::imshow("stream", display);
        }
        
        // Capture and update current frame
        if (!cap.read(frame)) {
            std::cerr << "Failed to read frame" << std::endl;
            break;
        }
        
        if (i >= 0 && i < vb.n_frames) {
            // Use more efficient conversion with pre-allocated memory
            frame.convertTo(vb.frames[i], CV_64FC3, 1.5/255.0);
            cv::flip(vb.frames[i], vb.frames[i], 1);
            
            // Process all frames with temporal convolution
            process_frames(&vb, filt, kernel_size);
        }
        
        i = (i + 1) % vb.n_frames;
        j = (j + 1) % (vb.n_frames * N_CYCLES);
        
        char key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
        if (key == 'a') {
            // Reset j to trigger buffer reinitialization
            j = 0;
        }
    }
    
    // Clean up resources
    free_video_buffer(&vb);
    cap.release();
    cv::destroyAllWindows();
    return 0;
}