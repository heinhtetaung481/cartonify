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

int convolution(__global const int *oldPixels, int x, int y, int width, int height, const __constant int *filter, int filterSize, int colour) {
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

__constant int filter[25] = {
    2, 4, 5, 4, 2,
    4, 9, 12, 9, 4,
    5, 12, 15, 12, 5,
    4, 9, 12, 9, 4,
    2, 4, 5, 4, 2
};

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

    const float filterSum = 159.0;
    int red = customClamp(convolution(oldPixels, x, y, width, height, filter, filterSize, 2) / filterSum);
    int green = customClamp(convolution(oldPixels, x, y, width, height, filter, filterSize, 1) / filterSum);
    int blue = customClamp(convolution(oldPixels, x, y, width, height, filter, filterSize, 0) / filterSum);
    newPixels[y * width + x] = (red << 16) + (green << 8) + blue;
}

__constant int Gx[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};
__constant int Gy[9] = {
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1
};
__kernel void sobelEdgeDetect(__global int *oldPixels, __global int *newPixels, const int width, const int height, int threshold) {
    // Get the global thread IDs
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Apply the Sobel filter
    int redVertical = convolution(oldPixels, x, y, width, height, Gx, 3, 2);
    int greenVertical = convolution(oldPixels, x, y, width, height, Gx, 3, 1);
    int blueVertical = convolution(oldPixels, x, y, width, height, Gx, 3, 0);
    int redHorizontal = convolution(oldPixels, x, y, width, height, Gy, 3, 2);
    int greenHorizontal = convolution(oldPixels, x, y, width, height, Gy, 3, 1);
    int blueHorizontal = convolution(oldPixels, x, y, width, height, Gy, 3, 0);
    int verticalGradient = abs(redVertical) + abs(greenVertical) + abs(blueVertical);
    int horizontalGradient = abs(redHorizontal) + abs(greenHorizontal) + abs(blueHorizontal);
    // we could take use sqrt(vertGrad^2 + horizGrad^2), but simple addition catches most edges.
    int totalGradient = verticalGradient + horizontalGradient;
    if (totalGradient >= threshold) {
        newPixels[y * width + x] = 0x000000; // we colour the edges black
    } else {
        newPixels[y * width + x] = 0xFFFFFF;
    }
}


// Reduce Colours kernel
__kernel void reduceColours(__global int *oldPixels, __global int *newPixels,
                            const int width, const int height, int numColours) {
    // Get the global thread IDs
    int x = get_global_id(0);
    int y = get_global_id(1);

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
