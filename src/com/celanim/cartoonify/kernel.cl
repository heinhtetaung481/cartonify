int customClamp(float value) {
    int result = (int) (value + 0.5); // round to nearest integer
    if (result <= 0) {
        return 0;
    } else if (result > (1 << 8) - 1) {
        return 255;
    } else {
        return result;
    }
}

int wrap(int pos, int size) {
    if (pos < 0) {
        pos = -1 - pos;
    } else if (pos >= size) {
        pos = (size - 1) - (pos - size);
    }
    return pos;
}

int convolution(__global const int *oldPixels, int x, int y, int width, int height, const int *filter, int filterSize, int colour) {
    int filterHalf = filterSize / 2;
    int sum = 0;
    for (int filterY = 0; filterY < filterSize; filterY++) {
        int centerY = wrap(y + filterY - filterHalf, height);
        for (int filterX = 0; filterX < filterSize; filterX++) {
            int centerX = wrap(x + filterX - filterHalf, width);
            int rgb = oldPixels[centerY * width + centerX];
            int filterVal = filter[filterY * filterSize + filterX];
            sum += ((rgb >> (colour * 8)) & 0xFF) * filterVal;
        }
    }
    return sum;
}
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


__kernel void sobelEdgeDetect(__global int *oldPixels, __global int *newPixels, const int width, const int height) {
    // Get the global thread IDs
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Sobel filter for edge detection
    const int Gx[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    const int Gy[9] = {
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1
    };

    int gradientX = 0;
    int gradientY = 0;

    // Apply the Sobel filter
    for (int filterY = -1; filterY <= 1; filterY++) {
        for (int filterX = -1; filterX <= 1; filterX++) {
            int sampleX = x + filterX;
            int sampleY = y + filterY;

            // Handle image boundaries
            sampleX = clamp(sampleX, 0, width - 1);
            sampleY = clamp(sampleY, 0, height - 1);

            int pixelIndex = sampleY * width + sampleX;
            int rgb = oldPixels[pixelIndex];

            int filterIndex = (filterY + 1) * 3 + (filterX + 1);

            // Extract the RGB values
            int r = (rgb >> 16) & 0xFF;
            int g = (rgb >> 8) & 0xFF;
            int b = rgb & 0xFF;

            // Apply the filter to each color channel
            gradientX += r * Gx[filterIndex];
            gradientY += r * Gy[filterIndex];

            gradientX += g * Gx[filterIndex];
            gradientY += g * Gy[filterIndex];

            gradientX += b * Gx[filterIndex];
            gradientY += b * Gy[filterIndex];
        }
    }

    // Gradient magnitude approximation
    int magnitude = abs(gradientX) + abs(gradientY);

    // Thresholding
    int newPixel = (magnitude >= 256) ? 0x00000000 : 0xFFFFFFFF;  // White or black

    // Store the result
    int index = y * width + x;
    newPixels[index] = newPixel;
}


// Reduce Colours kernel
__kernel void reduceColours(__global int *oldPixels, __global int *newPixels,
                            const int width, const int height) {
    // Get the global thread IDs
    int x = get_global_id(0);
    int y = get_global_id(1);

    int numColours = 3;  // Number of colours to quantize to
    int COLOUR_MASK = 255;  // Colour mask

    // Get the pixel color
    int idx = y * width + x;
    int color = oldPixels[idx];

    // Extract the RGB values
    int r = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = color & 0xFF;

    // Quantize each color channel
    r = round((float)r / (COLOUR_MASK + 1.0f) * numColours - 0.49999f) * COLOUR_MASK / (numColours - 1);
    g = round((float)g / (COLOUR_MASK + 1.0f) * numColours - 0.49999f) * COLOUR_MASK / (numColours - 1);
    b = round((float)b / (COLOUR_MASK + 1.0f) * numColours - 0.49999f) * COLOUR_MASK / (numColours - 1);

    // Write the new pixel to the output image
    int newColor = (r << 16) | (g << 8) | b;
    newPixels[idx] = newColor;
}

// Merge Mask kernel
__kernel void mergeMask(__global int *maskPixels, __global int *photoPixels, __global int *newPixels,
                        const int maskColour, const int width, const int height) {
    // Get the global thread IDs
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Get the pixel color from the mask and the photo
    int idx = y * width + x;
    int maskColor = maskPixels[idx];
    int photoColor = photoPixels[idx];

    // If the mask color matches the specified mask color, use the photo color
    // Otherwise, use the mask color
    int newColor = ((maskColor & 0xFFFFFF) == (maskColour & 0xFFFFFF)) ? photoColor : maskColor;

    // Write the new pixel to the output image
    newPixels[idx] = newColor;
}
