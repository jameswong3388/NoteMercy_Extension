function results = analyze_block_letter_characteristics(image_path, debug)
    % Analyzes an image to determine if it contains block lettering by measuring
    % angular characteristics and corner features
    
    % Set default for debug if not provided
    if nargin < 2
        debug = false;
    end

    % Read and validate input image
    img = imread(image_path);
    if isempty(img)
        disp(['Error: Could not read image at ', image_path]);
        results = struct();
        return;
    end

    % Convert to grayscale if needed
    if size(img, 3) == 3
        gray_img = rgb2gray(img);
    else
        gray_img = img;
    end

    % Extract letter shapes
    binary_letters = preprocess_image(gray_img);
    
    % Find letter contours
    [contours, ~] = bwboundaries(binary_letters, 'noholes');
    
    % Initialize storage for analysis
    corner_angles = [];
    simplified_contours = {};

    % Analyze each letter contour
    for i = 1:length(contours)
        current_contour = contours{i};
        
        % Filter out noise (very small shapes)
        if calculate_contour_area(current_contour) < 50
            continue;
        end
        
        % Simplify contour and analyze angles
        [simplified_shape, angles] = analyze_contour_geometry(current_contour);
        simplified_contours{end+1} = simplified_shape;
        corner_angles = [corner_angles, angles];
    end

    % Calculate block letter metrics
    results = compute_block_metrics(corner_angles, contours);

    % Visualize results if debug mode is enabled
    if debug
        visualize_analysis(img, binary_letters, simplified_contours, corner_angles, results);
    end
end

function binary = preprocess_image(gray_img)
    % Converts grayscale image to binary, optimized for letter extraction
    threshold = 127 / 255;
    binary = imbinarize(gray_img, threshold);
    binary = ~binary;  % Invert to make letters white
end

function area = calculate_contour_area(contour)
    % Calculates the area of a contour
    area = polyarea(contour(:,2), contour(:,1));
end

function [simplified_poly, angles] = analyze_contour_geometry(boundary)
    % Analyzes the geometric properties of a contour
    
    % Convert to x,y coordinates
    xy_points = [boundary(:,2), boundary(:,1)];
    
    % Calculate simplification parameter
    perimeter = sum(sqrt(diff(xy_points(:,1)).^2 + diff(xy_points(:,2)).^2));
    epsilon = 0.01 * perimeter;
    
    % Normalize epsilon based on bounding box
    bbox_diagonal = sqrt((max(xy_points(:,1)) - min(xy_points(:,1)))^2 + ...
                        (max(xy_points(:,2)) - min(xy_points(:,2)))^2);
    norm_epsilon = min(epsilon / bbox_diagonal, 1);
    
    % Simplify contour and measure angles
    simplified_poly = reducepoly(xy_points, norm_epsilon);
    angles = measure_corner_angles(simplified_poly);
end

function angles = measure_corner_angles(polygon)
    % Measures angles between consecutive segments in a polygon
    angles = [];
    num_points = size(polygon, 1);
    
    for i = 1:num_points
        % Get three consecutive vertices (with wraparound)
        p1 = polygon(i, :);
        p2 = polygon(mod(i, num_points) + 1, :);
        p3 = polygon(mod(i + 1, num_points) + 1, :);
        
        % Calculate angle deviation from 90 degrees
        v1 = p2 - p1;
        v2 = p3 - p2;
        angle = atan2d(abs(det([v1; v2])), dot(v1, v2));
        deviation = min(abs(angle - [0, 90, 180, 270]));
        angles = [angles, deviation];
    end
end

function metrics = compute_block_metrics(angles, contours)
    % Computes metrics indicating presence of block lettering
    if isempty(angles)
        metrics = struct('avg_deviation', 0, ...
                        'median_deviation', 0, ...
                        'max_deviation', 0, ...
                        'corner_count', 0, ...
                        'shape_count', length(contours));
        return;
    end
    
    metrics = struct(...
        'avg_deviation', mean(angles), ...
        'median_deviation', median(angles), ...
        'max_deviation', max(angles), ...
        'corner_count', length(angles), ...
        'shape_count', length(contours) ...
    );
end

function visualize_analysis(original, binary, contours, angles, metrics)
    % Creates visualization plots for debugging
    figure('Name', 'Block Letter Analysis', 'NumberTitle', 'off');
    
    subplot(2,2,1);
    imshow(original); title('Original Image');
    
    subplot(2,2,2);
    imshow(binary); title('Extracted Letters');
    
    subplot(2,2,3);
    imshow(original);
    hold on;
    for k = 1:length(contours)
        poly = contours{k};
        closed_poly = [poly; poly(1,:)];
        plot(closed_poly(:,1), closed_poly(:,2), 'r', 'LineWidth', 2);
    end
    hold off;
    title('Corner Detection');
    
    subplot(2,2,4);
    histogram(angles, 20);
    hold on;
    xline(metrics.avg_deviation, 'r--', 'LineWidth', 2);
    hold off;
    title(sprintf('Angle Distribution\nMean Deviation: %.2f°', metrics.avg_deviation));
    
    % Print analysis results
    fprintf('\nBlock Letter Analysis Results:\n');
    fprintf('Average angle deviation: %.2f°\n', metrics.avg_deviation);
    fprintf('Median angle deviation: %.2f°\n', metrics.median_deviation);
    fprintf('Maximum angle deviation: %.2f°\n', metrics.max_deviation);
    fprintf('Total corners detected: %d\n', metrics.corner_count);
    fprintf('Number of letter shapes: %d\n', metrics.shape_count);
end

% === Example Usage ===
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/2.png';
pressure_results = analyze_block_letter_characteristics(image_path, true);
disp(pressure_results);