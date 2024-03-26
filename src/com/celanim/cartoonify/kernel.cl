// Guassian Blur kernel
__kernel void gaussianBlur(__global int *pixels, __global int *newPixels,
                           const int width, const int height) {
__kernel void gaussianBlur(
    __global const int *oldPixels,  // Input image
    __global int *newPixels,        // Output image
    __const int width,              // Image width
    __const int height              // Image height
) {
    // Get the work-item's unique ID
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Ensure the pixel is within image bounds
    if (x >= width || y >= height) {
        return;
    }

    // Gaussian filter definition (use your actual filter values)
    const int filterSize = 5;
    const int filterHalf = filterSize / 2;
    const float filter[25] = {
        2, 4, 5, 4, 2, // sum=17
        4, 9, 12, 9, 4, // sum=38
        5, 12, 15, 12, 5, // sum=49
        4, 9, 12, 9, 4, // sum=38
        2, 4, 5, 4, 2  // sum=17
    };
    const float filterSum = 159.0;  // Replace with the actual sum of filter

    // Convolution calculation
    float red = 0.0f;
    float green = 0.0f;
    float blue = 0.0f;

    for (int filterY = -filterHalf; filterY <= filterHalf; filterY++) {
        for (int filterX = -filterHalf; filterX <= filterHalf; filterX++) {
            int sampleX = x + filterX;
            int sampleY = y + filterY;

            // Handle image boundaries (replace with your boundary handling method)
            sampleX = clamp(sampleX, 0, width - 1);
            sampleY = clamp(sampleY, 0, height - 1);

            int pixelIndex = sampleY * width + sampleX;
            int rgb = oldPixels[pixelIndex];

            int filterIndex = (filterY + filterHalf) * filterSize + (filterX + filterHalf);
            float filterVal = filter[filterIndex];

            red += (float) ((rgb >> 16) & 0xFF) * filterVal;
            green += (float) ((rgb >> 8) & 0xFF) * filterVal;
            blue += (float) (rgb & 0xFF) * filterVal;
        }
    }

    // Normalize by the filter sum
    red /= filterSum;
    green /= filterSum;
    blue /= filterSum;

    // Clamp the color values to the valid range
    red = clamp(red, 0.0f, 255.0f);
    green = clamp(green, 0.0f, 255.0f);
    blue = clamp(blue, 0.0f, 255.0f);

    // Construct the new pixel value
    int newPixel = ((int)red << 16) | ((int)green << 8) | (int)blue;

    // Store the result
    int index = y * width + x;
    newPixels[index] = newPixel;
}

__kernel void sobelEdgeDetect(__global int *pixels, __global int *newPixels,
                              const int width, const int height, const int edgeThreshold) {

}


__kernel void reduceColours(__global int *oldPixels, __global int *newPixels,
		                    const int width, const int height, const int numColours) {

}

__kernel void mergeMask(__global int *maskPixels, __global int *photoPixels, __global int *newPixels,
		                const int maskColour, const int width) {

}

