#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>

extern "C++" bool cuda_raw2gray(int width, int height, unsigned char *img,
                                unsigned char *res);

void cpu_grayscale(const cv::Mat &img, cv::Mat &res) {
  int width = int(img.cols / 2), height = int(img.rows / 2);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      float pixel = 0.7152f * 0.5 *
                        (img.data[2 * i * (2 * width) + 2 * j] +
                         img.data[(2 * i + 1) * (2 * width) + 2 * j + 1]) +
                    0.2126f * img.data[2 * i * (2 * width) + 2 * j + 1] +
                    0.0722f * img.data[(2 * i + 1) * (2 * width) + 2 * j];
      res.data[i * width + j] = (unsigned char)pixel;
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "usage: raw2gray filename" << std::endl;
    return -1;
  }

  cv::Mat img = cv::imread(argv[1], 0);
  int width = int(img.cols / 2), height = int(img.rows / 2);
  cv::Mat res(height, width, 0);

  cuda_raw2gray(width, height, img.data, res.data);

  //   cpu_grayscale(img, res);

  cv::imwrite("res.png", res);
  return 0;
}