# rave-viz

Visuals for walls at parties

## To Use Locally

1. Install cmake
    ```bash
    brew install cmake
    ```

2. Install OpenCV
    ```bash
    brew install opencv
    ```

3. Update your CPLUS_INCLUDE_PATH to include that install
    ```bash
    export CPLUS_INCLUDE_PATH="/opt/homebrew/opt/opencv/include/opencv4:$CPLUS_INCLUDE_PATH"
    ```

4. Clone the repository
    ```bash
    git clone https://github.com/matthew-ritch/rave-viz
    ```

5. Build and run
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ./rav
    ```

## To use in browser
Coming soon to WASM near you
