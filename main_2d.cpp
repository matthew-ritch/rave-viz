#include <opencv2/opencv.hpp>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include "filters.hpp"

#define MAX_FRAMES 10
#define MIN_FRAMES 10
#define N_REPS 3
#define N_CYCLES 30
#define SPEED .15


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
            std::cerr << "Failed to read initial frame " << i << std::endl;
            continue;
        }
        
        frame.convertTo(vb->frames[i], CV_64FC3, 1.0/255.0);
        cv::flip(vb->frames[i], vb->frames[i], 1);
        vb->u_t[i] = cv::Mat::zeros(frame.size(), CV_64FC3);
    }
}

// Function to process a single frame - can be executed in parallel
void process_frame(cv::Mat* frame, cv::Mat* u_t, double* filt) {
    if (frame == nullptr || u_t == nullptr || filt == nullptr) {
        std::cerr << "Null pointer in process_frame" << std::endl;
        return;
    }
    
    if (frame->empty()) {
        std::cerr << "Empty frame in process_frame" << std::endl;
        return;
    }
    
    cv::Mat delta = cv::Mat::zeros(frame->size(), CV_64FC3);
    
    for (int k = 0; k < N_REPS; k++) {
        convolve1d(frame, &delta, filt, 3);
        *u_t += SPEED*delta;
        *frame += *u_t;
    }
    
    // Clip values
    cv::max(*frame, 0.0, *frame);
    cv::min(*frame, 1.0, *frame);
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
    
    // Initialize random seed
    srand(time(NULL));
    
    // Initialize video buffer
    VideoBuffer vb = {nullptr, nullptr, 0};
    init_video_buffer(&vb, &cap, MIN_FRAMES);
    
    cv::namedWindow("stream", cv::WINDOW_NORMAL);
    
    int i = 0, j = 0;
    double filt[] = {1.0, -1.0};
    
    std::vector<std::thread> threads;
    
    while (true) {
        // If we're starting a new cycle, join all threads first
        if (j == 0) {
            // Join any existing threads before reinitializing buffer
            for (auto& t : threads) {
                if (t.joinable()) {
                    t.join();
                }
            }
            threads.clear();
            
            // Generate new buffer size
            int n = MIN_FRAMES + rand() % (MAX_FRAMES - MIN_FRAMES + 1);
            init_video_buffer(&vb, &cap, n);
            i = 0;
        }
        
        // Display current frame without resizing
        if (i >= 0 && i < vb.n_frames && !vb.frames[i].empty()) {
            cv::Mat display;
            vb.frames[i].convertTo(display, CV_8UC3, 255.0);
            cv::imshow("stream", display);
        }
        
        // Capture and update current frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "Failed to read frame" << std::endl;
            break;
        }
        
        if (i >= 0 && i < vb.n_frames) {
            frame.convertTo(vb.frames[i], CV_64FC3, 1.5/255.0);
            cv::flip(vb.frames[i], vb.frames[i], 1);
            
            // Create worker thread for processing
            threads.push_back(std::thread(process_frame, &vb.frames[i], &vb.u_t[i], filt));
            
            // Limit number of threads and join completed threads periodically
            const unsigned int max_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
            if (threads.size() >= max_threads) {
                for (auto& t : threads) {
                    if (t.joinable()) {
                        t.join();
                    }
                }
                threads.clear();
            }
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
