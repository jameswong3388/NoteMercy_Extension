function results = calculate_uppercase_ratio(image_path, debug)
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

    % Apply binary thresholding and invert
    threshold = 127 / 255;
    binary = imbinarize(gray, threshold);
    binary = ~binary;  % Inversion to match typical document analysis

    % Find connected components (potential characters)
    cc = bwconncomp(binary);
    stats = regionprops(cc, 'BoundingBox', 'Area', 'Extent');
    
    % Filter out noise (very small components)
    areas = [stats.Area];
    median_area = median(areas);
    valid_idx = areas > (median_area * 0.1);  % Keep components larger than 10% of median area
    
    stats = stats(valid_idx);
    if isempty(stats)
        results = struct('uppercase_ratio', 0, 'character_count', 0);
        return;
    end
    
    % Extract bounding box heights
    heights = zeros(1, length(stats));
    for i = 1:length(stats)
        heights(i) = stats(i).BoundingBox(4);
    end
    
    % Normalize heights by the median height
    median_height = median(heights);
    normalized_heights = heights / median_height;
    
    % Calculate total image height (for reference)
    img_height = size(binary, 1);
    
    % Determine uppercase/lowercase classification threshold
    % Characters with height > 0.8 of median height are considered uppercase-like
    uppercase_threshold = 0.8;
    uppercase_count = sum(normalized_heights >= uppercase_threshold);
    total_chars = length(normalized_heights);
    
    % Calculate the uppercase ratio
    uppercase_ratio = uppercase_count / total_chars;
    
    % Apply additional criteria for block lettering
    extents = [stats.Extent];  % Extent is the ratio of component pixels to bounding box pixels
    median_extent = median(extents);
    
    % Pack results into a structure
    results = struct(...
        'uppercase_ratio', uppercase_ratio, ...
        'character_count', total_chars, ...
        'median_height_ratio', median_height / img_height, ...
        'median_extent', median_extent ...
    );
    
    % Debug visualization if requested
    if debug
        figure('Name', 'Uppercase Ratio Analysis', 'NumberTitle', 'off');
        
        % Original Image
        subplot(2,2,1);
        imshow(img);
        title('Original Image');
        
        % Binary Image
        subplot(2,2,2);
        imshow(binary);
        title('Binary Image');
        
        % Visualize character bounding boxes
        subplot(2,2,3);
        imshow(img);
        hold on;
        for i = 1:length(stats)
            bbox = stats(i).BoundingBox;
            % Color code: red for uppercase, blue for lowercase
            if normalized_heights(i) >= uppercase_threshold
                rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
            else
                rectangle('Position', bbox, 'EdgeColor', 'b', 'LineWidth', 2);
            end
        end
        hold off;
        title('Character Classification (Red=Uppercase, Blue=Lowercase)');
        
        % Histogram of normalized heights
        subplot(2,2,4);
        histogram(normalized_heights, 10);
        hold on;
        xline(uppercase_threshold, 'r--', 'LineWidth', 2);
        hold off;
        title(sprintf('Height Distribution\nUppercase Ratio: %.2f', uppercase_ratio));
        
        % Print debug information
        fprintf('Uppercase ratio: %.2f\n', uppercase_ratio);
        fprintf('Character count: %d\n', total_chars);
        fprintf('Median height ratio: %.3f\n', results.median_height_ratio);
        fprintf('Median extent: %.3f\n', results.median_extent);
    end
end

% === Example usage ===
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/3.png';
results = calculate_uppercase_ratio(image_path, true);
disp(results);
