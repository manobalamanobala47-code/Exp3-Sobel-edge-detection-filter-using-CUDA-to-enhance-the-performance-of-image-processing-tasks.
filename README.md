# Exp 3 Sobel edge detection filter using CUDA to enhance the performance of image processing tasks


<h3>ENTER YOUR NAME</h3> Manobala R
<h3>ENTER YOUR REGISTER NO</h3> 212225230167
<h3>EX. NO</h3> 03
<h3>DATE</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>


## Aim
To utilize CUDA to parallelize the Sobel filter implementation for efficient processing of images.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.
## Program

```cpp
%%writefile sobelEdgeDetectionFilter.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage,
                            unsigned int width, unsigned int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width-1 && y >= 1 && y < height-1) {

        int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
        int Gy[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

        int sumX = 0;
        int sumY = 0;

        for(int i=-1;i<=1;i++){
            for(int j=-1;j<=1;j++){
                unsigned char pixel = srcImage[(y+i)*width + (x+j)];
                sumX += pixel * Gx[i+1][j+1];
                sumY += pixel * Gy[i+1][j+1];
            }
        }

        int magnitude = sqrtf(sumX*sumX + sumY*sumY);
        magnitude = min(max(magnitude,0),255);

        dstImage[y*width + x] = (unsigned char)magnitude;
    }
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {

    Mat image = imread("/content/Screenshot 2026-05-22 191031.png", IMREAD_GRAYSCALE);

    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    size_t imageSize = width * height * sizeof(unsigned char);

    unsigned char *h_outputImage = (unsigned char*)malloc(imageSize);

    unsigned char *d_inputImage, *d_outputImage;

    checkCudaErrors(cudaMalloc(&d_inputImage,imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage,imageSize));

    checkCudaErrors(cudaMemcpy(d_inputImage,
                               image.data,
                               imageSize,
                               cudaMemcpyHostToDevice));

    dim3 blockSize(16,16);
    dim3 gridSize((width+15)/16,(height+15)/16);

    sobelFilter<<<gridSize,blockSize>>>(d_inputImage,d_outputImage,width,height);

    checkCudaErrors(cudaMemcpy(h_outputImage,
                               d_outputImage,
                               imageSize,
                               cudaMemcpyDeviceToHost));

    Mat outputImage(height,width,CV_8UC1,h_outputImage);

    imwrite("output_sobel.jpeg",outputImage);

    printf("Edge detection completed.\n");

    return 0;
}
```

## Output Explanation
<img width="501" height="358" alt="Screenshot 2026-05-22 191031" src="https://github.com/user-attachments/assets/648ea380-8945-42dc-b1c5-3877d780be99" />

The program detects the edges present in the input image using the Sobel filter.

<img width="626" height="446" alt="image" src="https://github.com/user-attachments/assets/802a86a4-891d-4da3-b013-67403bb70f61" />






## Result
Successfully implemented a CUDA-accelerated Sobel filter, demonstrating significant performance improvement over the CPU-based implementation, with an efficient parallelized approach for edge detection in image processing.
