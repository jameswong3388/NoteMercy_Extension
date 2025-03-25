function results = compute_vertical_stroke_proportion(image_path, debug)
    % Check for debug argument
    if nargin < 2
        debug = false;
    end

    % Read the image
    img = imread(image_path);
    if isempty(img)
        error('Image file not found: %s', image_path);
    end

    % Convert to grayscale if needed
    if size(img, 3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end

    % Apply thresholding using Otsu's method and invert binary image
    level = graythresh(gray);
    bw = imbinarize(gray, level);
    binary = ~bw;  % Invert: letters become white on black background

    % Find connected components (8-connected)
    cc = bwconncomp(binary, 8);
    stats = regionprops(cc, 'BoundingBox');

    % Initialize boxes and collect component heights
    boxes = [];
    heights = [];

    % Loop through each connected component
    for i = 1:numel(stats)
        bbox = stats(i).BoundingBox;  % [x, y, width, height]
        % Filter out small noise components (width and height > 5 pixels)
        if bbox(3) > 5 && bbox(4) > 5
            boxes = [boxes; bbox];
            heights = [heights; bbox(4)];
        end
    end

    results = struct();
    if ~isempty(heights)
        sortedHeights = sort(heights);
        median_height = median(sortedHeights);
        max_height = max(sortedHeights);
        ascender_ratio = max_height / median_height;
        results.median_height = median_height;
        results.max_height = max_height;
        results.ascender_ratio = ascender_ratio;
        fprintf('Ascender-to-xHeight ratio: %.2f\n', ascender_ratio);
    else
        warning('No connected components found with the given size threshold.');
    end

    % Debug visualization if requested
    if debug && ~isempty(boxes)
        % Prepare a copy of the image for drawing bounding boxes.
        % If the image is grayscale, convert it to RGB.
        if size(img, 3) == 1
            img_with_boxes = repmat(img, [1, 1, 3]);
        else
            img_with_boxes = img;
        end

        % Create a figure with three subplots
        figure('Name', 'Vertical Stroke Proportion Analysis', 'NumberTitle', 'off');

        % Subplot 1: Original Image
        subplot(1, 3, 1);
        imshow(img);
        title('Original Image');

        % Subplot 2: Binary Image
        subplot(1, 3, 2);
        imshow(binary);
        title('Binary Image');

        % Subplot 3: Bounding Boxes over Original Image
        subplot(1, 3, 3);
        imshow(img_with_boxes);
        hold on;
        for i = 1:size(boxes, 1)
            bbox = boxes(i, :);
            rectangle('Position', bbox, 'EdgeColor', 'g', 'LineWidth', 2);
        end
        title(sprintf('Bounding Boxes (Ascender Ratio: %.2f)', ascender_ratio));
        hold off;
    end
end

% Example usage:
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/3.png';
results = compute_vertical_stroke_proportion(image_path, true);
disp(results);
