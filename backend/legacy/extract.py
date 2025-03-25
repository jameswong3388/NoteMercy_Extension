import numpy as np
import cv2
import math
# from matplotlib import pyplot as plt

# Global constants for thresholding and measurements
ANCHOR_POINT = 6000
MIDZONE_THRESHOLD = 15000
MIN_HANDWRITING_HEIGHT_PIXEL = 20

# Global feature variables
BASELINE_ANGLE = 0.0
TOP_MARGIN = 0.0
LETTER_SIZE = 0.0
LINE_SPACING = 0.0
WORD_SPACING = 0.0
PEN_PRESSURE = 0.0
SLANT_ANGLE = 0.0


def bilateralFilter(image, d):
    """
    Apply bilateral filtering to reduce noise while preserving edges.

    Parameters:
        image: Input image.
        d: Diameter of each pixel neighborhood.

    Returns:
        Filtered image.
    """
    return cv2.bilateralFilter(image, d, 50, 50)


def medianFilter(image, d):
    """
    Apply median filtering to the image.

    Parameters:
        image: Input image.
        d: Kernel size.

    Returns:
        Filtered image.
    """
    return cv2.medianBlur(image, d)


def threshold(image, t):
    """
    Convert the image to grayscale and apply inverted binary thresholding.

    Parameters:
        image: Input image.
        t: Threshold value.

    Returns:
        Thresholded image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
    return thresh


def dilate(image, kernelSize):
    """
    Dilate objects in the image using a specified kernel.

    Parameters:
        image: Input image.
        kernelSize: Tuple specifying the size of the kernel.

    Returns:
        Dilated image.
    """
    kernel = np.ones(kernelSize, np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def erode(image, kernelSize):
    """
    Erode objects in the image using a specified kernel.

    Parameters:
        image: Input image.
        kernelSize: Tuple specifying the size of the kernel.

    Returns:
        Eroded image.
    """
    kernel = np.ones(kernelSize, np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def straighten(image):
    """
    Straighten handwritten lines by detecting contours and rotating them horizontally.
    Updates the global BASELINE_ANGLE based on the mean rotation angle.

    Parameters:
        image: Input image.

    Returns:
        Straightened image.
    """
    global BASELINE_ANGLE

    angle_sum = 0.0
    contour_count = 0

    # Apply bilateral filtering and thresholding to highlight text lines
    filtered = bilateralFilter(image, 3)
    thresh = threshold(filtered, 120)
    dilated = dilate(thresh, (5, 100))

    contours, hier = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        # Ignore contours that are not likely to be text lines
        if h > w or h < MIN_HANDWRITING_HEIGHT_PIXEL:
            continue

        roi = image[y:y + h, x:x + w]
        # Skip small contours to avoid inaccurate angle estimation
        if w < image.shape[1] / 2:
            image[y:y + h, x:x + w] = 255
            continue

        rect = cv2.minAreaRect(ctr)
        angle = rect[2]
        if angle < -45.0:
            angle += 90.0
        elif angle > 45.0:
            angle -= 90.0

        rot = cv2.getRotationMatrix2D(((x + w) / 2, (y + h) / 2), angle, 1)
        extract = cv2.warpAffine(roi, rot, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        image[y:y + h, x:x + w] = extract

        angle_sum += angle
        contour_count += 1

    mean_angle = angle_sum / contour_count if contour_count else angle_sum
    BASELINE_ANGLE = mean_angle
    return image


def horizontalProjection(img):
    """
    Calculate the horizontal projection of the image pixel rows.

    Parameters:
        img: Input image.

    Returns:
        List of sums of pixel values for each row.
    """
    h, w = img.shape[:2]
    return [np.sum(img[j:j + 1, 0:w]) for j in range(h)]


def verticalProjection(img):
    """
    Calculate the vertical projection of the image pixel columns.

    Parameters:
        img: Input image.

    Returns:
        List of sums of pixel values for each column.
    """
    h, w = img.shape[:2]
    return [np.sum(img[0:h, j:j + 1]) for j in range(w)]


def extractLines(img):
    """
    Extract lines of handwritten text using horizontal projection.
    Updates the global features LETTER_SIZE, LINE_SPACING, and TOP_MARGIN.

    Parameters:
        img: Input image.

    Returns:
        A list of [start, end] indices representing the vertical boundaries of each line.
    """
    global LETTER_SIZE, LINE_SPACING, TOP_MARGIN

    filtered = bilateralFilter(img, 5)
    thresh = threshold(filtered, 160)
    hpList = horizontalProjection(thresh)

    # Determine the top margin by counting blank rows at the top
    topMarginCount = 0
    for value in hpList:
        if value <= 255:
            topMarginCount += 1
        else:
            break

    # Detect lines by tracking transitions in the horizontal projection
    lineTop = 0
    lineBottom = 0
    spaceTop = 0
    spaceBottom = 0
    indexCount = 0
    setLineTop = True
    setSpaceTop = True
    includeNextSpace = True
    space_zero = []
    lines = []

    for i, value in enumerate(hpList):
        if value == 0:
            if setSpaceTop:
                spaceTop = indexCount
                setSpaceTop = False
            indexCount += 1
            spaceBottom = indexCount
            if i < len(hpList) - 1 and hpList[i + 1] == 0:
                continue
            if includeNextSpace:
                space_zero.append(spaceBottom - spaceTop)
            else:
                previous = space_zero.pop() if space_zero else 0
                space_zero.append(previous + spaceBottom - lineTop)
            setSpaceTop = True

        if value > 0:
            if setLineTop:
                lineTop = indexCount
                setLineTop = False
            indexCount += 1
            lineBottom = indexCount
            if i < len(hpList) - 1 and hpList[i + 1] > 0:
                continue
            if lineBottom - lineTop < 20:
                includeNextSpace = False
                setLineTop = True
                continue
            includeNextSpace = True
            lines.append([lineTop, lineBottom])
            setLineTop = True

    # Split contours that may contain multiple lines into individual lines
    fineLines = []
    for line in lines:
        anchor = line[0]
        anchorPoints = []
        upHill = True
        downHill = False
        segment = hpList[line[0]:line[1]]

        for j, value in enumerate(segment):
            if upHill:
                if value < ANCHOR_POINT:
                    anchor += 1
                    continue
                anchorPoints.append(anchor)
                upHill = False
                downHill = True
            elif downHill:
                if value > ANCHOR_POINT:
                    anchor += 1
                    continue
                anchorPoints.append(anchor)
                downHill = False
                upHill = True

        if len(anchorPoints) < 2:
            continue

        lineTop = line[0]
        for x in range(1, len(anchorPoints) - 1, 2):
            lineMid = (anchorPoints[x] + anchorPoints[x + 1]) // 2
            if lineMid - lineTop < 20:
                continue
            fineLines.append([lineTop, lineMid])
            lineTop = lineMid
        if line[1] - lineTop >= 20:
            fineLines.append([lineTop, line[1]])

    # Calculate features: average line spacing and letter size
    space_nonzero_row_count = 0
    midzone_row_count = 0
    lines_having_midzone_count = 0
    for line in fineLines:
        segment = hpList[int(line[0]):int(line[1])]
        flag = False
        for value in segment:
            if value < MIDZONE_THRESHOLD:
                space_nonzero_row_count += 1
            else:
                midzone_row_count += 1
                flag = True
        if flag:
            lines_having_midzone_count += 1

    if lines_having_midzone_count == 0:
        lines_having_midzone_count = 1

    total_space_row_count = space_nonzero_row_count + np.sum(space_zero[1:-1])
    average_line_spacing = float(total_space_row_count) / lines_having_midzone_count
    average_letter_size = float(midzone_row_count) / lines_having_midzone_count
    LETTER_SIZE = average_letter_size if average_letter_size else 1
    LINE_SPACING = average_line_spacing / LETTER_SIZE
    TOP_MARGIN = float(topMarginCount) / LETTER_SIZE

    return fineLines


def extractWords(image, lines):
    """
    Extract words from each line using vertical projection.
    Updates the global feature WORD_SPACING.

    Parameters:
        image: Input image.
        lines: List of line boundaries from extractLines.

    Returns:
        A list of word coordinates [y1, y2, x1, x2] for each detected word.
    """
    global LETTER_SIZE, WORD_SPACING

    filtered = bilateralFilter(image, 5)
    thresh = threshold(filtered, 180)
    width = thresh.shape[1]
    space_zero = []
    words = []

    for line in lines:
        extract_img = thresh[int(line[0]):int(line[1]), 0:width]
        vp = verticalProjection(extract_img)

        wordStart = 0
        wordEnd = 0
        spaceStart = 0
        spaceEnd = 0
        indexCount = 0
        setWordStart = True
        setSpaceStart = True
        includeNextSpace = True
        spaces = []

        for j, value in enumerate(vp):
            if value == 0:
                if setSpaceStart:
                    spaceStart = indexCount
                    setSpaceStart = False
                indexCount += 1
                spaceEnd = indexCount
                if j < len(vp) - 1 and vp[j + 1] == 0:
                    continue
                if (spaceEnd - spaceStart) > int(LETTER_SIZE / 2):
                    spaces.append(spaceEnd - spaceStart)
                setSpaceStart = True
            if value > 0:
                if setWordStart:
                    wordStart = indexCount
                    setWordStart = False
                indexCount += 1
                wordEnd = indexCount
                if j < len(vp) - 1 and vp[j + 1] > 0:
                    continue
                # Count non-zero pixels to filter out noise components (like punctuation)
                count = 0
                for k in range(int(round(line[1] - line[0]))):
                    row = thresh[line[0] + k:line[0] + k + 1, wordStart:wordEnd]
                    if np.sum(row):
                        count += 1
                if count > int(LETTER_SIZE / 2):
                    words.append([line[0], line[1], wordStart, wordEnd])
                setWordStart = True

        space_zero.extend(spaces[1:-1])

    space_columns = np.sum(space_zero)
    space_count = len(space_zero) if len(space_zero) > 0 else 1
    average_word_spacing = float(space_columns) / space_count
    relative_word_spacing = average_word_spacing / LETTER_SIZE if LETTER_SIZE else average_word_spacing
    WORD_SPACING = relative_word_spacing

    return words


def extractSlant(img, words):
    """
    Determine the average slant angle of the handwriting using vertical strokes.
    Applies affine transformation on each word to measure the effect of shifting columns.
    Updates the global SLANT_ANGLE.

    Parameters:
        img: Input image.
        words: List of word coordinates.
    """
    global SLANT_ANGLE

    theta = [-0.785398, -0.523599, -0.261799, -0.0872665,
             0.01, 0.0872665, 0.261799, 0.523599, 0.785398]
    s_function = [0.0] * 9
    count_ = [0] * 9

    filtered = bilateralFilter(img, 5)
    thresh = threshold(filtered, 180)

    for i, angle in enumerate(theta):
        s_temp = 0.0
        count = 0
        for word in words:
            original = thresh[int(word[0]):int(word[1]), int(word[2]):int(word[3])]
            height = word[1] - word[0]
            width = word[3] - word[2]
            shift = (math.tan(angle) * height) / 2
            pad_length = abs(int(shift))
            blank_image = np.zeros((int(height), int(width) + pad_length * 2, 3), np.uint8)
            new_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
            new_image[:, pad_length:width + pad_length] = original

            h, w = new_image.shape[:2]
            x1 = w / 2
            y1 = 0
            x2 = w / 4
            y2 = h
            x3 = 3 * w / 4
            y3 = h

            pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3]])
            pts2 = np.float32([[x1 + shift, y1], [x2 - shift, y2], [x3 - shift, y3]])
            M = cv2.getAffineTransform(pts1, pts2)
            deslanted = cv2.warpAffine(new_image, M, (w, h))
            vp = verticalProjection(deslanted)

            for k, col_sum in enumerate(vp):
                if col_sum == 0:
                    continue
                num_fgpixel = col_sum / 255
                if num_fgpixel < int(height / 3):
                    continue
                column = deslanted[0:height, k:k + 1].flatten()
                for l, pixel in enumerate(column):
                    if pixel != 0:
                        break
                for m, pixel in enumerate(column[::-1]):
                    if pixel != 0:
                        break
                delta_y = height - (l + m)
                h_sq = (float(num_fgpixel) / delta_y) ** 2
                h_wted = (h_sq * num_fgpixel) / height
                s_temp += h_wted
                count += 1

        s_function[i] = s_temp
        count_[i] = count

    max_value = max(s_function)
    max_index = s_function.index(max_value)

    if max_index == 0:
        angle = 45
    elif max_index == 1:
        angle = 30
    elif max_index == 2:
        angle = 15
    elif max_index == 3:
        angle = 5
    elif max_index == 5:
        angle = -5
    elif max_index == 6:
        angle = -15
    elif max_index == 7:
        angle = -30
    elif max_index == 8:
        angle = -45
    elif max_index == 4:
        p = s_function[4] / s_function[3] if s_function[3] != 0 else s_function[4]
        q = s_function[4] / s_function[5] if s_function[5] != 0 else s_function[4]
        if ((p <= 1.2 and q <= 1.2) or (p > 1.4 and q > 1.4)) or (
                (p <= 1.2 and q - p > 0.4) or (q <= 1.2 and p - q > 0.4)):
            angle = 0
        else:
            angle = 180

    SLANT_ANGLE = angle
    return


def barometer(image):
    """
    Calculate the average pen pressure of the handwriting by inverting and thresholding the image.
    Updates the global PEN_PRESSURE.

    Parameters:
        image: Input image.

    Returns:
        None.
    """
    global PEN_PRESSURE

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    inverted = np.zeros_like(gray)
    for x in range(h):
        for y in range(w):
            inverted[x][y] = 255 - gray[x][y]

    filtered = bilateralFilter(inverted, 3)
    ret, thresh = cv2.threshold(filtered, 100, 255, cv2.THRESH_TOZERO)

    total_intensity = 0
    pixel_count = 0
    for x in range(h):
        for y in range(w):
            if thresh[x][y] > 0:
                total_intensity += thresh[x][y]
                pixel_count += 1

    average_intensity = float(total_intensity) / pixel_count
    PEN_PRESSURE = average_intensity
    return


def start(file_name):
    """
    Main function to process the handwriting image and extract features.

    Parameters:
        file_name: Name of the image file (located in the 'images/' directory).

    Returns:
        List of extracted features: [BASELINE_ANGLE, TOP_MARGIN, LETTER_SIZE,
        LINE_SPACING, WORD_SPACING, PEN_PRESSURE, SLANT_ANGLE].
    """
    global BASELINE_ANGLE, TOP_MARGIN, LETTER_SIZE, LINE_SPACING, WORD_SPACING, PEN_PRESSURE, SLANT_ANGLE

    image = cv2.imread('images/' + file_name)
    barometer(image)
    straightened = straighten(image)
    lineIndices = extractLines(straightened)
    wordCoordinates = extractWords(straightened, lineIndices)
    extractSlant(straightened, wordCoordinates)

    BASELINE_ANGLE = round(BASELINE_ANGLE, 2)
    TOP_MARGIN = round(TOP_MARGIN, 2)
    LETTER_SIZE = round(LETTER_SIZE, 2)
    LINE_SPACING = round(LINE_SPACING, 2)
    WORD_SPACING = round(WORD_SPACING, 2)
    PEN_PRESSURE = round(PEN_PRESSURE, 2)
    SLANT_ANGLE = round(SLANT_ANGLE, 2)

    return [BASELINE_ANGLE, TOP_MARGIN, LETTER_SIZE, LINE_SPACING, WORD_SPACING, PEN_PRESSURE, SLANT_ANGLE]
