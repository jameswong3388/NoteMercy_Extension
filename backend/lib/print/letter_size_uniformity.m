function results = compute_letter_size_uniformity(image_path, debug)
    % Set default for debug if not provided
    if nargin < 2
        debug = false;
    end

    % Read the image
    img = imread(image_path);
    if isempty(img)
        disp(['Error: Could not read image at ', image_path]);
        results = struct();
        return;
    end

    % Convert to grayscale if image is RGB
    if size(img, 3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end

    % Perform analysis on original grayscale for intensity measurements
    % We'll use this to analyze pen pressure through pixel intensity

    % Apply binary thresholding and invert for component analysis
    threshold = 127 / 255;
    binary = imbinarize(gray, threshold);
    binary = ~binary;  % Inversion to match cv2.THRESH_BINARY_INV

    % Find connected components (letters/characters)
    cc = bwconncomp(binary);
    stats = regionprops(cc, 'BoundingBox', 'Area', 'Centroid', 'PixelIdxList');
    
    % Filter out small noise components
    min_area = 20;  % Minimum area threshold
    valid_indices = [];
    for i = 1:length(stats)
        if stats(i).Area > min_area
            valid_indices = [valid_indices, i];
        end
    end
    stats = stats(valid_indices);
    
    % Extract heights and widths of bounding boxes
    heights = zeros(1, length(stats));
    widths = zeros(1, length(stats));
    aspect_ratios = zeros(1, length(stats));
    
    % For pen pressure analysis
    mean_intensities = zeros(1, length(stats));
    intensity_variations = zeros(1, length(stats));
    stroke_widths = zeros(1, length(stats));
    
    for i = 1:length(stats)
        % Get bounding box dimensions
        bbox = stats(i).BoundingBox;
        widths(i) = bbox(3);
        heights(i) = bbox(4);
        aspect_ratios(i) = widths(i) / heights(i);
        
        % Get pixel indices for this component
        pixelIdx = stats(i).PixelIdxList;
        
        % Extract intensity values for this letter
        letter_intensity = 255 - double(gray(pixelIdx));  % Invert values so higher = darker
        
        % Calculate mean intensity as a measure of pen pressure
        mean_intensities(i) = mean(letter_intensity);
        
        % Calculate variation in intensity within the letter
        intensity_variations(i) = std(letter_intensity);
        
        % Estimate stroke width using distance transform
        % Create a mask for just this letter
        letter_mask = false(size(binary));
        letter_mask(pixelIdx) = true;
        
        % Distance transform to find the distance to the nearest background pixel
        dist_transform = bwdist(~letter_mask);
        
        % The maximum distance gives half the maximum stroke width
        max_dist = max(dist_transform(pixelIdx));
        stroke_widths(i) = 2 * max_dist;  % Double to get full width
    end
    
    % Check if any letters were found
    if isempty(heights)
        results = struct('height_uniformity', 0, ...
                        'width_uniformity', 0, ...
                        'aspect_ratio_uniformity', 0, ...
                        'pen_pressure_uniformity', 0, ...
                        'stroke_width_uniformity', 0, ...
                        'avg_pen_pressure', 0, ...
                        'avg_stroke_width', 0, ...
                        'letter_count', 0);
        return;
    end
    
    % Calculate uniformity as 1 - coefficient of variation
    % Lower variation means higher uniformity
    height_cv = std(heights) / mean(heights);
    width_cv = std(widths) / mean(widths);
    aspect_ratio_cv = std(aspect_ratios) / mean(aspect_ratios);
    
    % Pen pressure related metrics
    pressure_cv = std(mean_intensities) / mean(mean_intensities);
    stroke_width_cv = std(stroke_widths) / mean(stroke_widths);
    
    height_uniformity = max(0, 1 - height_cv);
    width_uniformity = max(0, 1 - width_cv);
    aspect_ratio_uniformity = max(0, 1 - aspect_ratio_cv);
    pen_pressure_uniformity = max(0, 1 - pressure_cv);
    stroke_width_uniformity = max(0, 1 - stroke_width_cv);
    
    % Pack results into a structure
    results = struct(...
        'height_uniformity', height_uniformity, ...
        'width_uniformity', width_uniformity, ...
        'aspect_ratio_uniformity', aspect_ratio_uniformity, ...
        'pen_pressure_uniformity', pen_pressure_uniformity, ...
        'stroke_width_uniformity', stroke_width_uniformity, ...
        'avg_pen_pressure', mean(mean_intensities) / 255, ...
        'avg_stroke_width', mean(stroke_widths), ...
        'letter_count', length(stats) ...
    );
    
    % Debug visualization if requested
    if debug
        figure('Name', 'Letter Size and Pen Pressure Analysis', 'NumberTitle', 'off');
        
        % Original Image
        subplot(2,3,1);
        imshow(img);
        title('Original Image');
        
        % Binary Image with Bounding Boxes
        subplot(2,3,2);
        imshow(binary);
        hold on;
        for i = 1:length(stats)
            bbox = stats(i).BoundingBox;
            rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 1);
            % Add letter index for reference
            text(stats(i).Centroid(1), stats(i).Centroid(2), num2str(i), ...
                'Color', 'g', 'FontSize', 8, 'FontWeight', 'bold');
        end
        hold off;
        title('Letter Detection');
        
        % Height Distribution
        subplot(2,3,3);
        histogram(heights, min(20, ceil(length(heights)/2)));
        title(sprintf('Height Distribution\nUniformity: %.3f', height_uniformity));
        
        % Width Distribution
        subplot(2,3,4);
        histogram(widths, min(20, ceil(length(widths)/2)));
        title(sprintf('Width Distribution\nUniformity: %.3f', width_uniformity));
        
        % Pen Pressure Distribution
        subplot(2,3,5);
        histogram(mean_intensities, min(20, ceil(length(mean_intensities)/2)));
        title(sprintf('Pen Pressure Distribution\nUniformity: %.3f', pen_pressure_uniformity));
        
        % Stroke Width Distribution
        subplot(2,3,6);
        histogram(stroke_widths, min(20, ceil(length(stroke_widths)/2)));
        title(sprintf('Stroke Width Distribution\nUniformity: %.3f', stroke_width_uniformity));
        
        % Print debug information in the command window
        fprintf('Letter count: %d\n', length(stats));
        fprintf('Height uniformity: %.3f\n', height_uniformity);
        fprintf('Width uniformity: %.3f\n', width_uniformity);
        fprintf('Aspect ratio uniformity: %.3f\n', aspect_ratio_uniformity);
        fprintf('Pen pressure uniformity: %.3f\n', pen_pressure_uniformity);
        fprintf('Stroke width uniformity: %.3f\n', stroke_width_uniformity);
        fprintf('Average pen pressure (0-1): %.3f\n', results.avg_pen_pressure);
        fprintf('Average stroke width: %.3f\n', results.avg_stroke_width);
    end
end

% === Main Script ===
% Replace with your actual image file path.
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png';
uniformity_results = compute_letter_size_uniformity(image_path, true);
disp(uniformity_results);
