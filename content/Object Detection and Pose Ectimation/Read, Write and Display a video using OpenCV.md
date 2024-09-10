# first up, what is a video?
A video is a sequence of fast moving images. 
How fast are the pictures moving? The measure of how fast the images are transitioning is given by a metric called **frames per second(FPS)**
When someone says that the video has an FPS of 40, it means that 40 images are being displayed every second. Alternatively, after every 25 milliseconds, a new frame is displayed. The other important attributes are the width and height of the frame.
# Reading a Video
In OpenCV, a video can be read either by :
1. using the feed from a camera connected to a computer 
2. reading a video file.
The first step towards reading a video file is to create a **VideoCapture** object. Its argument can be either the device index or the name of the video file to be read.

```python

cap = `cv2.VideoCapture(``'chaplin.mp4'``)`

```
After the VideoCapture object is created, we can capture the video frame by frame.
## Displaying a video

- we can display the video frame by frame. 
- A frame of a video is simply an image
- we display each frame the same way we display images, i.e., we use the function **imshow()**.
- we use the **waitKey()** after imshow() function to pause each frame in the video.
- we need to pass a number greater than ‘0’ to the waitKey() function. This number is equal to the time in milliseconds we want each frame to be displayed.
- While reading the frames from a webcam, using waitKey(1) is appropriate because the display frame rate will be limited by the frame rate of the webcam even if we specify a delay of 1 ms in waitKey.
- While reading frames from a video that you are processing, it may still be appropriate to set the time delay to 1 ms so that the thread is freed up to do the processing we want to do.
- In rare cases, when the playback needs to be at a certain framerate, we may want the delay to be higher than 1 ms.

## Writing a video

After we are done with capturing and processing the video frame by frame, the next step we would want to do is to save the video.

 We need to create a **VideoWriter** object. We should specify:
 1. the output file name with its format (eg: output.avi). 
 2. the **FourCC** code 
 3. the number of frames per second (FPS). 
 4. the frame size.
`out=cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(``'M'``,``'J'``,``'P'``,``'G'``),` `10``, (frame_width,frame_height))`

[FourCC](http://en.wikipedia.org/wiki/FourCC) is a 4-byte code used to specify the video codec. The list of available codes can be found at [fourcc.org](http://www.fourcc.org/codecs.php).


---
https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/#:~:text=Read%2C%20Write%20and%20Display%20a%20video%20using%20OpenCV,by%20frame.%20...%203%20Writing%20a%20video%20