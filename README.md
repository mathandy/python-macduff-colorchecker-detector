# Python-Macduff
This is a Python port of [Macduff](https://github.com/ryanfb/macduff), a tool for finding the Macbeth ColorChecker chart in an image.  

The translation to python was done by Andrew Port and was partially supported by [Rare](https://rare.org) as part of work done for the [Fish Forever](http://www.fishforever.org/) project.

## Some Changes from the Original Algorithm
1. A parameter `MIN_RELATIVE_SQUARE_SIZE` has been added as a work-around for an issue where the algorithm would choke on images where the ColorChecker was smaller than a certain hard-coded size relative to the image dimensions.

2. The original code by ryanfb made use of an undocumented OpenCV function
named `CV_IS_SEQ_HOLE`.  The related check was removed in this Python
version of the algorithm.  This ommission has not been noted to cause any effect.

## Usage
  
    $ python macduff.py test.jpg result.png > result.csv

## DESCRIPTION

![Macduff result](https://ryanfb.s3.amazonaws.com/images/macduff.png)

Macduff will try its best to find your ColorChecker. If you specify an output
image, it will be written with the "found" ColorChecker overlaid on the input
image with circles on each patch (the outer circle is the "reference" value,
the inner circle is the average value from the actual image). Macduff outputs
various useless debug info on stderr, and useful information in CSV-style
on stdout. The first 24 lines will be the ColorChecker patch locations and
average values:

    x,y,r,g,b

The last two lines contain the patch square size (i.e. you can feed another
program this and the location and safely use a square of `size` with the top
left corner at `x-size/2,y-size/2` for each patch) and error against the
reference chart. The patches are output in row order from the typical
ColorChecker orientation ("dark skin" top left, "black" bottom right):

![ColorChecker layout](https://ryanfb.s3.amazonaws.com/images/CC_Avg20_orig_layout.png)

See also: [Automatic ColorChecker Detection, a Survey](http://ryanfb.github.io/etc/2015/07/08/automatic_colorchecker_detection.html)

## LICENSE
The original Macduff code is protected under the 3-clause BSD and includes some code taken from OpenCV.  See LICENSE.TXT.
