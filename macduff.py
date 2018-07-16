#!/usr/bin/env python
"""Python-Macduff: "the Macbeth ColorChecker finder", ported to Python.

Original C++ code: github.com/ryanfb/macduff/

Usage:
    # if pixel-width of color patches is unknown,
    $ python macduff.py examples/test.jpg result.png > result.csv

    # if pixel-width of color patches is known to be, e.g. 65,
    $ python macduff.py examples/test.jpg result.png 65 > result.csv
"""
from __future__ import print_function, division
import cv2
import numpy as np
from math import sqrt
from sys import stderr, argv
from copy import copy

# Each color square must takes up more than this percentage of the image
MIN_RELATIVE_SQUARE_SIZE = 0.0001

DEBUG = False

MACBETH_WIDTH = 6
MACBETH_HEIGHT = 4
MACBETH_SQUARES = MACBETH_WIDTH * MACBETH_HEIGHT

MAX_CONTOUR_APPROX = 50  # default was 7
MAX_RGB_DISTANCE = 444


# BabelColor averages in sRGB:
#   http://www.babelcolor.com/main_level/ColorChecker.htm
# (converted to BGR order for comparison)
colorchecker_srgb = np.array([
        [
            (67, 81, 115),
            (129, 149, 196),
            (157, 123, 93),
            (65, 108, 90),
            (176, 129, 130),
            (171, 191, 99)
        ],
        [
            (45, 123, 220),
            (168, 92, 72),
            (98, 84, 195),
            (105, 59, 91),
            (62, 189, 160),
            (41, 161, 229)
        ],
        [
            (147, 62, 43),
            (72, 149, 71),
            (56, 48, 176),
            (22, 200, 238),
            (150, 84, 188),
            (166, 136, 0)
        ],
        [
            (240, 245, 245),
            (201, 201, 200),
            (161, 161, 160),
            (121, 121, 120),
            (85, 84, 83),
            (50, 50, 50)
        ]
    ], dtype='uint8')


def dist2(a, b):
    """Euclidean distance between two 2-vectors."""
    x, y = (b - a).ravel()
    return sqrt(x*x + y*y)


def lab_distance(p_1, p_2):
    """Converts to Lab color space then takes Euclidean distance.

    Note: this is vectorized below.  (wrapped with `np.vectorize`)"""
    convert = np.array([p_1, p_2], dtype='uint8').reshape(2, 1, 3)
    convert = cv2.cvtColor(convert, cv2.COLOR_BGR2Lab)
    l, a, b = convert[0, 0] - convert[1, 0]
    return sqrt(l*l + a*a + b*b)


# a few classes to simplify the translation from c++

class Box2D:
    """
    Note: The Python equivalent of `RotatedRect` and `Box2D` objects 
    are tuples, `((center_x, center_y), (w, h), rotation)`.
    Example:
    >>> cv2.boxPoints(((0, 0), (2, 1), 0))
    array([[-1. ,  0.5],
           [-1. , -0.5],
           [ 1. , -0.5],
           [ 1. ,  0.5]], dtype=float32)
    >>> cv2.boxPoints(((0, 0), (2, 1), 90))
    array([[-0.5, -1. ],
           [ 0.5, -1. ],
           [ 0.5,  1. ],
           [-0.5,  1. ]], dtype=float32)
    """
    def __init__(self, center=None, size=None, angle=0, rrect=None):
        if rrect is not None:
            center, size, angle = rrect

        # self.center = Point2D(*center)
        # self.size = Size(*size)
        self.center = center
        self.size = size
        self.angle = angle  # in degrees

    def rrect(self):
        return self.center, self.size, self.angle


class Rect:
    def __init__(self, x=0, y=0, width=0, height=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


def rect_average(rect, image):
    """Returns mean color in intersection of `image` and `rectangle`."""

    # crop out intersection of `rect` and `image`
    x, y, w, h = rect.x, rect.y, rect.width, rect.height
    intersection = image[int(max(y, 0)): int(min(y + h, image.shape[0])),
                         int(max(x, 0)): int(min(x + w, image.shape[1]))]
    return intersection.mean(axis=(0, 1))


def contour_average(contour, image):
    """Assuming `contour` is a polygon, returns the mean color inside it.

    Note: This function is inefficiently implemented!!! 
    Maybe using drawing/fill functions would improve speed.
    """

    # find up-right bounding box
    xbb, ybb, wbb, hbb = cv2.boundingRect(contour)

    # now found which points in bounding box are inside contour and sum
    def is_inside_contour(pt):
        return cv2.pointPolygonTest(contour, pt, False) > 0

    from itertools import product as catesian_product
    from operator import add
    from functools import reduce
    bb = catesian_product(range(max(xbb, 0), min(xbb + wbb,  image.shape[1])),
                          range(max(ybb, 0), min(ybb + hbb,  image.shape[0])))
    pts_inside_of_contour = [xy for xy in bb if is_inside_contour(xy)]

    # pts_inside_of_contour = list(filter(is_inside_contour, bb))
    color_sum = reduce(add, (image[y, x] for x, y in pts_inside_of_contour))
    return color_sum / len(pts_inside_of_contour)


def rotate_box(box_corners):
    """NumPy equivalent of `[arr[i-1] for i in range(len(arr)]`"""
    return np.roll(box_corners, 1, 0)


def check_colorchecker(colorchecker):
    """Find deviation of `colorchecker` values from expected values."""
    h, w = MACBETH_HEIGHT, MACBETH_WIDTH
    diff = (colorchecker[:h, :w] - colorchecker_srgb[:h, :w]).ravel(order='K')
    return sqrt(np.dot(diff, diff))


def draw_colorchecker(colors, centers, image, radius):
    for observed_color, expected_color, pt in zip(colors.reshape(-1, 3),
                                                  colorchecker_srgb.reshape(-1, 3),
                                                  centers.reshape(-1, 2)):
        x, y = pt
        cv2.circle(image, (x, y), radius//2,    expected_color.tolist(), -1)
        cv2.circle(image, (x, y), radius//4, observed_color.tolist(), -1)
    return image


class ColorChecker:
    def __init__(self, error, values, points, size):
        self.error = error
        self.values = values
        self.points = points
        self.size = size


def find_colorchecker(boxes, image, original_image, debug_image=None):

    rotated_box = False
    
    points = np.array([[box.center[0], box.center[1]] for box in boxes])
    passport_box = cv2.minAreaRect(points.astype('float32'))

    if debug_image is not None:
        image_copy = copy(image)
        for box in boxes:
            pts = cv2.boxPoints(box.rrect()).astype(np.int32)
            cv2.polylines(image_copy, [pts], True, (255, 0, 0))

        cv2.polylines(image_copy, [cv2.boxPoints(passport_box).astype(np.int32)],
                      True, (0, 0, 255))
        cv2.imwrite('debug_passport_box.png', image_copy)

    (x, y), (w, h), a = passport_box
    print("Box:\n"
          "\tCenter: %f,%f\n"
          "\tSize: %f,%f\n"
          "\tAngle: %f\n" % (x, y, w, h, a),
          file=stderr)

    if a < 0.0:
        rotated_box = True

    box_corners = cv2.boxPoints(passport_box)
    
    d10 = dist2(box_corners[1], box_corners[0])
    d12 = dist2(box_corners[1], box_corners[2])
    if d10 < d12:
        print("Box is upright, rotating\n", file=stderr)
        box_corners = rotate_box(box_corners)
        h_spacing = d12/(MACBETH_WIDTH - 1)
        v_spacing = d10/(MACBETH_HEIGHT - 1)
    else:
        h_spacing = d10/(MACBETH_WIDTH - 1)
        v_spacing = d12/(MACBETH_HEIGHT - 1)

    dx, dy = box_corners[1] - box_corners[0]
    h_slope = dy/dx
    h_mag = sqrt(1 + h_slope * h_slope)
    dx, dy = box_corners[3] - box_corners[0]
    v_slope = dy/dx
    v_mag = sqrt(1 + v_slope*v_slope)
    h_orientation = -1 if box_corners[0][0] < box_corners[1][0] else 1
    v_orientation = -1 if box_corners[0][1] < box_corners[3][1] else 1
        
    print("Spacing is %f %f\n" % (h_spacing, v_spacing), file=stderr)
    print("Slope is %f %f\n" % (h_slope, v_slope), file=stderr)

    average_size = int(sum(min(box.size) for box in boxes) / len(boxes))
    print("Average contained rect size is %d\n" % average_size, file=stderr)

    checker_dims = (MACBETH_HEIGHT, MACBETH_WIDTH)
    this_colorchecker = np.empty(checker_dims + (3,), dtype='float32')
    this_colorchecker_points = np.empty(checker_dims + (2,), dtype='float32')
    
    # calculate the averages for our oriented colorchecker
    for x in range(MACBETH_WIDTH):
        for y in range(MACBETH_HEIGHT):
            row_start = [None, None]
            tmp = v_spacing * y / v_mag
            if not rotated_box:
                row_start[0] = box_corners[0][0] + tmp
                row_start[1] = box_corners[0][1] + tmp * v_slope
            else:
                row_start[0] = box_corners[0][0] - tmp
                row_start[1] = box_corners[0][1] - tmp * v_slope

            rect = Rect(0, 0, average_size, average_size)
            rect.x = row_start[0] - x * h_spacing * h_orientation / h_mag
            rect.y = row_start[1] - x * h_spacing * v_orientation * h_slope / h_mag
            
            this_colorchecker_points[y, x] = (rect.x, rect.y)

            rect.x -= average_size//2
            rect.y -= average_size//2

            this_colorchecker[y, x] = rect_average(rect, original_image)

    orient_1_error = check_colorchecker(this_colorchecker)
    this_colorchecker = this_colorchecker[::-1, ::-1]
    orient_2_error = check_colorchecker(this_colorchecker)
    
    print("Orientation 1: %f\n" % orient_1_error, file=stderr)
    print("Orientation 2: %f\n" % orient_2_error, file=stderr)

    if orient_1_error < orient_2_error:
        this_colorchecker = this_colorchecker[::-1, ::-1]
    else:
        this_colorchecker_points = this_colorchecker_points[::-1, ::-1]

    return ColorChecker(error=min(orient_1_error, orient_2_error),
                        values=this_colorchecker,
                        points=this_colorchecker_points,
                        size=average_size)


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


# https://github.com/opencv/opencv/blob/master/samples/python/squares.py
def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, _hierarchy = \
                cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if (len(cnt) == 4 and cv2.contourArea(cnt) > 1000
                        and cv2.isContourConvex(cnt)):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = max([angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i + 2) % 4])
                                   for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


def maybe_patch(square, patch_size, rtol=.25):
    cw = abs(np.linalg.norm(square[0] - square[1]) -  patch_size) < rtol*patch_size
    ch = abs(np.linalg.norm(square[0] - square[-1]) - patch_size) < rtol*patch_size
    return cw and ch


# stolen from icvGenerateQuads
def find_quad(src_contour, min_size, debug_image=None):

    for max_error in range(2, MAX_CONTOUR_APPROX + 1):
        dst_contour = cv2.approxPolyDP(src_contour, max_error, closed=True)
        if len(dst_contour) == 4:
            break

        # we call this again on its own output, because sometimes
        # cvApproxPoly() does not simplify as much as it should.
        dst_contour = cv2.approxPolyDP(dst_contour, max_error, closed=True)
        if len(dst_contour) == 4:
            break

    # reject non-quadrangles
    is_acceptable_quad = False
    is_quad = False
    if len(dst_contour) == 4 and cv2.isContourConvex(dst_contour):
        is_quad = True
        perimeter = cv2.arcLength(dst_contour, closed=True)
        area = cv2.contourArea(dst_contour, oriented=False)

        d1 = dist2(dst_contour[0], dst_contour[2])
        d2 = dist2(dst_contour[1], dst_contour[3])
        d3 = dist2(dst_contour[0], dst_contour[1])
        d4 = dist2(dst_contour[1], dst_contour[2])

        # philipg.  Only accept those quadrangles which are more square
        # than rectangular and which are big enough
        cond = (d3/1.1 < d4 < d3*1.1 and
                d3*d4/1.5 < area and
                min_size < area and
                d1 >= 0.15*perimeter and
                d2 >= 0.15*perimeter)

        if not cv2.CALIB_CB_FILTER_QUADS or area > min_size and cond:
            is_acceptable_quad = True
            # return dst_contour
    if debug_image is not None:
        cv2.drawContours(debug_image, [src_contour], -1, (255, 0, 0), thickness=1)
        if is_acceptable_quad:
            cv2.drawContours(debug_image, [dst_contour], -1, (0, 255, 0), thickness=1)
        elif is_quad:
            cv2.drawContours(debug_image, [dst_contour], -1, (0, 0, 255), thickness=1)
        return debug_image

    if is_acceptable_quad:
        return dst_contour
    return None


def find_macbeth(img, patch_size=None):
    macbeth_img = img
    if isinstance(img, str):
        macbeth_img = cv2.imread(img)
    macbeth_original = copy(macbeth_img)
    macbeth_split = cv2.split(macbeth_img)
    
    block_size = int(min(macbeth_img.shape[:2]) * 0.02) | 1
    # block_size = int(min(macbeth_img.shape[:2]) * .009) | 1
    print("Using %d as block size\n" % block_size, file=stderr)

    # threshold each channel and OR results together
    macbeth_split_thresh = []
    for channel in macbeth_split:
        res = cv2.adaptiveThreshold(channel,
                                    255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV,
                                    block_size,
                                    C=6)
        # _, res = cv2.threshold(channel, 50, 255, cv2.THRESH_BINARY_INV)
        macbeth_split_thresh.append(res)
    adaptive = np.bitwise_or(*macbeth_split_thresh)
    if DEBUG:
        cv2.imwrite('debug_threshold.png', np.vstack(macbeth_split_thresh+[adaptive]))

    element_size = int(2 + block_size/10)
    print("Using %d as element size\n" % element_size, file=stderr)

    # do an opening on the threshold image
    shape, ksize = cv2.MORPH_RECT, (element_size, element_size)
    element = cv2.getStructuringElement(shape, ksize)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, element)
    if DEBUG:
        cv2.imwrite('debug_adaptive-open.png', adaptive)

    # find contours in the threshold image
    adaptive, contours, _ = cv2.findContours(image=adaptive,
                                             mode=cv2.RETR_LIST,
                                             method=cv2.CHAIN_APPROX_SIMPLE)

    if DEBUG:
        show_contours = cv2.cvtColor(copy(adaptive), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(show_contours, contours, -1, (0, 255, 0))
        cv2.imwrite('debug_all_contours.png', show_contours)

    min_size = np.product(macbeth_img.shape[:2]) * MIN_RELATIVE_SQUARE_SIZE

    def is_seq_hole(c):
        return cv2.contourArea(c, oriented=True) > 0

    def is_big_enough(contour):
        _, (w, h), _ = cv2.minAreaRect(contour)
        return w * h >= min_size

    # filter out contours that are too small or clockwise
    contours = [c for c in contours if is_big_enough(c) and is_seq_hole(c)]

    if DEBUG:
        show_contours = cv2.cvtColor(copy(adaptive), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(show_contours, contours, -1, (0, 255, 0))
        cv2.imwrite('debug_big_contours.png', show_contours)

    if contours:
        ### debug
        if DEBUG:
            debug_img = cv2.cvtColor(copy(adaptive), cv2.COLOR_GRAY2BGR)
            for c in contours:
                debug_img = find_quad(c, min_size, debug_image=debug_img)
            cv2.imwrite("debug_quads.png", debug_img)
        ### end of debug
        if patch_size is None:
            initial_quads = [find_quad(c, min_size) for c in contours]
        else:
            initial_quads = [s for s in find_squares(macbeth_original)
                             if maybe_patch(s, patch_size)]
        initial_quads = [q for q in initial_quads if q is not None]
        initial_boxes = [Box2D(rrect=cv2.minAreaRect(q)) for q in initial_quads]

        if DEBUG:
            show_quads = cv2.cvtColor(copy(adaptive), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(show_quads, initial_quads, -1, (0, 255, 0))
            cv2.imwrite('debug_quads2.png', show_quads)

        print("%d initial quads found", len(initial_quads), file=stderr)
        if len(initial_quads) > MACBETH_SQUARES:
            print(" (probably a Passport)\n", stderr)

            # set up the points sequence for cvKMeans2, using the box centers
            points = np.array([box.center for box in initial_boxes], dtype='float32')

            # partition into two clusters: passport and colorchecker
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        10, 1.0)
            compactness, clusters, centers = cv2.kmeans(data=points,
                                                        K=2,
                                                        bestLabels=None,
                                                        criteria=criteria,
                                                        attempts=100,
                                                        flags=cv2.KMEANS_RANDOM_CENTERS)

            partitioned_quads = [[], []]
            partitioned_boxes = [[], []]
            for i, cluster in enumerate(clusters.ravel()):
                partitioned_quads[cluster].append(initial_quads[i])
                partitioned_boxes[cluster].append(initial_boxes[i])

            # check each of the two partitioned sets for the best colorchecker
            if DEBUG:
                partitioned_checkers = \
                    [find_colorchecker(partitioned_boxes[i],
                                       macbeth_img,
                                       macbeth_original,
                                       'debug_find_colorchecker{}.jpg'.format(i))
                     for i in [0, 1]]
            else:
                partitioned_checkers = [find_colorchecker(partitioned_boxes[i],
                                                          macbeth_img,
                                                          macbeth_original)
                                        for i in [0, 1]]

            # use the colorchecker with the lowest error
            if partitioned_checkers[0].error < partitioned_checkers[1].error:
                found_colorchecker = partitioned_checkers[0]
            else:
                found_colorchecker = partitioned_checkers[1]

        else:  # just one colorchecker to test
            debug_img = None
            if DEBUG:
                debug_img = "debug_find_colorchecker.jpg"
            print("\n", file=stderr)
            found_colorchecker = \
                find_colorchecker(initial_boxes, macbeth_img, macbeth_original,
                                  debug_image=debug_img)

        # render the found colorchecker
        draw_colorchecker(found_colorchecker.values,
                          found_colorchecker.points,
                          macbeth_img,
                          found_colorchecker.size)

        # # write results
        # def write_results(filename, colorchecker):
        #     with open(filename, 'w+') as f:
        #         colors = colorchecker.values.reshape(1, 3)
        #         for k, (b, g, r) in enumerate(colors):
        #             f.write('{},{},{},{}\n'.format(k, r, g, b))
        # write_results('results.csv, found_colorchecker')

        # print out the colorchecker info
        for color, pt in zip(found_colorchecker.values.reshape(-1, 3),
                             found_colorchecker.points.reshape(-1, 2)):
            b, g, r = color
            x, y = pt
            print("%.0f,%.0f,%.0f,%.0f,%.0f\n" % (x, y, r, g, b))
        print("%0.f\n%f\n" % (found_colorchecker.size, found_colorchecker.error))

    return macbeth_img, found_colorchecker


if __name__ == '__main__':
    if len(argv) == 3:
        out, _ = find_macbeth(argv[1])
        cv2.imwrite(argv[2], out)
    elif len(argv) == 4:
        out, _ = find_macbeth(argv[1], patch_size=float(argv[3]))
        cv2.imwrite(argv[2], out)
    else:
        print("Usage: %s <input_image> <output_image> <(optional) patch_size>\n" % argv[0],
              file=stderr)
