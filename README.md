# Cartonify Project
## Background
An animated movie production company called GelAnim wants to turn millions of photos (eg. from short video clips) into cartoon-style photos, so that they can incorporate them into the movies they are making. For example: 

 	
GelAnim already have a Java program that does this, but it takes about FIVE seconds per 8 MPixel photo, which is far too slow. So they are asking you to see if you can port parts of the program to run on a GPU, so that it runs faster.


Their program uses two main techniques to turn an input photo into a cartoon-like photo:


* [Edge Detection](http://en.wikipedia.org/wiki/Edge_detection): They use the first couple of stages of the [Canny Edge Detector](http://en.wikipedia.org/wiki/Canny_edge_detector). This first blurs the image using a 5x5 Gaussian filter to smooth out the effects of any noisy pixels, and then uses a [Sobel filter](http://en.wikipedia.org/wiki/Sobel_operator) to detect edges in the photo. This gives a new image where all the edges are black and the other pixels are white;

* [Colour Quantization](http://en.wikipedia.org/wiki/Color_quantization): They use a very simple kind of colour quantization, which just rounds the colour values of each channel (red, green blue) down to just a few values. This gives a new image that uses only a few colours, and looks more like it is hand-painted;

* [Image Masking](http://en.wikipedia.org/wiki/Mask_(computing)): Finally, they use 'masking' to put the black edges on top of the quantized-colour photo, so that the final image has the edges outlined in black.

Note: this 'cartoonify' process is quite similar to the Cel Shading that is popular in some computer games, such as The Legend of Zelda: The Wind Waker.

The following sequence of images shows the output of each of the stages in GelAnim's program: 

<table style="width:100%">
  <tbody>
     <tr><th> Step </th><th>Result Image</th></tr>
<tr><td >1. The original image, showing just one small 200x200 area of the image. </td><td><img src="https://elearn.waikato.ac.nz/pluginfile.php/3285561/mod_resource/content/1/eg_bumblebee.jpg" width="450"/></td></tr>
<tr><td >2. After applying the 5x5 Gaussian Blur filter. This takes each colour channel of each pixel and combines it with the values of the adjacent pixels, by multiplying them by the following matrix (so the pixel's own value is multiplied by 15, etc).
               
                [2,  4,  5,  4,  2]
                [4,  9, 12,  9,  4]
                [5, 12, 15, 12,  5]
                [4,  9, 12,  9,  4]
                [2,  4,  5,  4,  2]
                
</td><td><img src="https://elearn.waikato.ac.nz/pluginfile.php/3285562/mod_resource/content/1/eg_bumblebee_blurred.jpg"  /></td></tr>
<tr><td>
     3. After applying the horizontal and vertical Sobel edge filters to the blurred image:   
   
          vertical = [-1,  0, +1]    horizontal = [+1, +2, +1]
                     [-2,  0, +2]                 [ 0,  0,  0]
                     [-1,  0, +1]                 [-1, -2, -1]
  
                 
</td><td><img src="https://elearn.waikato.ac.nz/pluginfile.php/3285557/mod_resource/content/1/eg_bumblebee_edges.jpg" /></td></tr>
<tr><td >4. After applying the simple colour quantization algorithm to each pixel of the original image. This example uses just THREE values per channel, 
so all the red channel values are rounded to the nearest of 0, 127 or 255, and similarly for the green and blue channels.</td><td><img src="https://elearn.waikato.ac.nz/pluginfile.php/3285556/mod_page/content/1/eg_bumblebee_colours.jpg" /></td></tr>
<tr><td >5. The final 'cartoon' image is formed by drawing the black edges on top of the colour-reduced image. </td><td> <img src="https://elearn.waikato.ac.nz/pluginfile.php/3285558/mod_resource/content/1/eg_bumblebee_edited.jpg" /></td></tr>
    </tbody>
</table>

### GPU version of Cartoonify
Their GelAnim Cartoonify program has a command line interface with the following parameters. You should retain this usage, so that your program is backwards compatible with their existing workflow.

```
  Arguments:[-d] [-e EdgeThreshold] [-c NumColours] photo1.jpg photo2.jpg ...
    -d means turn on debugging, which saves intermediate photos.
    -e EdgeThreshold values can range from 0 (everything is an edge) up to about 1000 or more.
    -c NumColours is the number of discrete values within each colour channel (2..256).
```

A new "-g" flag before the existing "-d" flag is added for GPU. Eg.

```
  Arguments: [-g] [-d] [-e EdgeThreshold] [-c NumColours] photo1.jpg photo2.jpg ...
    -g use the GPU, to speed up photo processing.
    -d means turn on debugging, which saves intermediate photos.
    ...
```

If this "-g" flag is specified, then some or all of the photo processing steps will be done on the GPU, for faster performance. (Note: When the "-g" flag is specified, you can disable the debugging features if necessary, if they would slow down the speed of the program.) If the "-g" flag is omitted, then all the steps will continue to be done on the CPU using the existing code. So this original non-gpu program will be your benchmark, so that you compare the speed of your OpenCL programs against it to see how much speedup you have achieved.

## Important notes

*  GPU and CPU implementation produce the same image outputs.
*  processPhotoOpenCL is the entry point of GPU implementation. It should contain the code (or methods) that initialize the environment and load the resources required by GPU version.

