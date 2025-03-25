function results = compute_angular_deviation(image_path, debug)
    % Set default for debug if not provided
    if nargin < 2
        debug = false;
    end

    % Read and preprocess the image
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

    % Binarize and invert the image
    threshold = 127 / 255;
    binary = imbinarize(gray, threshold);
    binary = ~binary;

    % Find contours
    [B, ~] = bwboundaries(binary, 'noholes');

    all_angles = [];
    all_polys = {};

    % Process each contour
    for k = 1:length(B)
        boundary = B{k};
        
        % Skip small contours (noise)
        area_val = polyarea(boundary(:,2), boundary(:,1));
        if area_val < 50
            continue;
        end
        
        % Calculate perimeter for epsilon
        d = sqrt(diff(boundary(:,2)).^2 + diff(boundary(:,1)).^2);
        perimeter = sum(d);
        epsilon = 0.01 * perimeter;
        
        % Convert to [x,y] coordinates
        xyBoundary = [boundary(:,2), boundary(:,1)];
        
        % Normalize epsilon
        bbox_diag = sqrt((max(xyBoundary(:,1)) - min(xyBoundary(:,1)))^2 + ...
                        (max(xyBoundary(:,2)) - min(xyBoundary(:,2)))^2);
        normalized_epsilon = min(epsilon / bbox_diag, 1);
        
        % Simplify the contour
        poly = reducepoly(xyBoundary, normalized_epsilon);
        all_polys{end+1} = poly;
        
        % Calculate angles between consecutive segments
        nPoints = size(poly, 1);
        for i = 1:nPoints
            % Get three consecutive points (with wrap-around)
            p1 = poly(i, :);
            if i < nPoints
                p2 = poly(i+1, :);
            else
                p2 = poly(1, :);
            end
            if i < nPoints-1
                p3 = poly(i+2, :);
            elseif i == nPoints-1
                p3 = poly(1, :);
            else
                p3 = poly(2, :);
            end
            
            % Calculate vectors
            v1 = p2 - p1;
            v2 = p3 - p2;
            
            % Calculate angle between vectors
            angle = atan2d(abs(det([v1; v2])), dot(v1, v2));
            
            % Calculate deviation from nearest 90-degree multiple
            deviation = min(abs(angle - [0, 90, 180, 270]));
            all_angles = [all_angles, deviation];
        end
    end

    % Compute metrics
    if isempty(all_angles)
        results = struct('mean_deviation', 0, ...
                        'median_deviation', 0, ...
                        'max_deviation', 0, ...
                        'angle_count', 0, ...
                        'total_contours', length(B));
        return;
    end

    % Pack results
    results = struct(...
        'mean_deviation', mean(all_angles), ...
        'median_deviation', median(all_angles), ...
        'max_deviation', max(all_angles), ...
        'angle_count', length(all_angles), ...
        'total_contours', length(B) ...
    );

    % Debug visualization
    if debug
        figure('Name', 'Angular Deviation Analysis', 'NumberTitle', 'off');
        
        % Original Image
        subplot(2,2,1);
        imshow(img);
        title('Original Image');
        
        % Binary Image
        subplot(2,2,2);
        imshow(binary);
        title('Binary Image');
        
        % Contours and angles
        subplot(2,2,3);
        imshow(img);
        hold on;
        for k = 1:length(all_polys)
            poly = all_polys{k};
            poly_closed = [poly; poly(1,:)];
            plot(poly_closed(:,1), poly_closed(:,2), 'r', 'LineWidth', 2);
        end
        hold off;
        title('Simplified Polygons');
        
        % Histogram of angle deviations
        subplot(2,2,4);
        histogram(all_angles, 20);
        hold on;
        xline(results.mean_deviation, 'r--', 'LineWidth', 2);
        hold off;
        title(sprintf('Angle Deviation Distribution\nMean: %.2f°', results.mean_deviation));
        
        % Print debug information
        fprintf('Mean angle deviation: %.2f°\n', results.mean_deviation);
        fprintf('Median angle deviation: %.2f°\n', results.median_deviation);
        fprintf('Maximum angle deviation: %.2f°\n', results.max_deviation);
        fprintf('Number of angles analyzed: %d\n', results.angle_count);
        fprintf('Number of contours: %d\n', results.total_contours);
    end
end

% === Example Usage ===
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png';
pressure_results = compute_angular_deviation(image_path, true);
disp(pressure_results);