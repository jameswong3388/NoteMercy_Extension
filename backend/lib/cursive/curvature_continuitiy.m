function results = compute_stroke_curvature_continuity(image_path, debug)
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

    % Apply binary thresholding (threshold = 127) and invert
    threshold = 127 / 255;
    binary = imbinarize(gray, threshold);
    binary = ~binary;  % Inversion to match cv2.THRESH_BINARY_INV

    % Find external contours using bwboundaries with 'noholes'
    [B, ~] = bwboundaries(binary, 'noholes');

    segment_lengths = [];
    all_polys = {};  % To store the simplified polygons

    % Process each detected boundary
    for k = 1:length(B)
        boundary = B{k};  % Each boundary is an N-by-2 matrix [row, col]
        
        % Skip small contours (noise) using polygon area
        area_val = polyarea(boundary(:,2), boundary(:,1));
        if area_val < 50
            continue;
        end
        
        % Calculate perimeter (sum of distances between consecutive points)
        d = sqrt(diff(boundary(:,2)).^2 + diff(boundary(:,1)).^2);
        perimeter = sum(d);
        
        % Set tolerance (epsilon = 1% of the perimeter)
        epsilon = 0.01 * perimeter;
        
        % Convert to [x,y] coordinates for reducepoly (x = col, y = row)
        xyBoundary = [boundary(:,2), boundary(:,1)];
        
        % Normalize tolerance relative to bounding box diagonal
        x_min = min(xyBoundary(:,1));
        x_max = max(xyBoundary(:,1));
        y_min = min(xyBoundary(:,2));
        y_max = max(xyBoundary(:,2));
        diag_length = sqrt((x_max - x_min)^2 + (y_max - y_min)^2);
        normalized_epsilon = min(epsilon / diag_length, 1);
        
        % Simplify the contour (approximate polygon)
        poly = reducepoly(xyBoundary, normalized_epsilon);
        all_polys{end+1} = poly;
        
        % Calculate lengths of each segment in the approximated polygon
        nPoints = size(poly, 1);
        for i = 1:nPoints
            % Get current point
            x1 = poly(i, 1); 
            y1 = poly(i, 2);
            % Wrap-around: next point (or first point if at end)
            if i < nPoints
                x2 = poly(i+1, 1); 
                y2 = poly(i+1, 2);
            else
                x2 = poly(1, 1); 
                y2 = poly(1, 2);
            end
            seg_len = sqrt((x2 - x1)^2 + (y2 - y1)^2);
            segment_lengths = [segment_lengths, seg_len];
        end
    end

    % Check if any segments were found
    if isempty(segment_lengths)
        results = struct('avg_normalized_segment_length', 0, ...
                         'median_normalized_segment_length', 0, ...
                         'segment_count', 0, ...
                         'total_contours', length(B));
        return;
    end

    % Normalize segment lengths by the image height
    H = size(binary, 1);
    normalized_segment_lengths = segment_lengths / H;

    % Compute average and median normalized segment lengths
    avg_normalized_segment_length = mean(normalized_segment_lengths);
    median_normalized_segment_length = median(normalized_segment_lengths);

    % Pack results into a structure
    results = struct(...
        'avg_normalized_segment_length', avg_normalized_segment_length, ...
        'median_normalized_segment_length', median_normalized_segment_length, ...
        'segment_count', length(segment_lengths), ...
        'total_contours', length(B) ...
    );

    % Debug visualization if requested
    if debug
        figure('Name', 'Stroke Curvature Continuity Analysis', 'NumberTitle', 'off');
        
        % Original Image
        subplot(2,2,1);
        imshow(img);
        title('Original Image');
        
        % Binary Image
        subplot(2,2,2);
        imshow(binary);
        title('Binary Image');
        
        % Overlay contours and simplified polygons on the original image
        subplot(2,2,3);
        imshow(img);
        hold on;
        % Draw original contours in green
        for k = 1:length(B)
            boundary = B{k};
            plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 2);
        end
        % Draw simplified polygons in red
        for k = 1:length(all_polys)
            poly = all_polys{k};  % poly is in [x,y]
            % Close the polygon by appending the first point
            poly_closed = [poly; poly(1,:)];
            plot(poly_closed(:,1), poly_closed(:,2), 'r', 'LineWidth', 2);
        end
        hold off;
        title('Contours (green) and Simplified Polygons (red)');
        
        % Histogram of normalized segment lengths
        subplot(2,2,4);
        histogram(normalized_segment_lengths, 20);
        hold on;
        xline(avg_normalized_segment_length, 'r--', 'LineWidth', 2);
        hold off;
        title(sprintf('Segment Length Distribution\nAvg: %.3f', avg_normalized_segment_length));
        
        % Print debug information in the command window
        fprintf('Average normalized segment length: %.3f\n', avg_normalized_segment_length);
        fprintf('Median normalized segment length: %.3f\n', median_normalized_segment_length);
        fprintf('Number of segments: %d\n', length(segment_lengths));
        fprintf('Number of contours: %d\n', length(B));
    end
end

% === Main Script ===
% Replace with your actual image file path.
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/5.png';
curviness_results = compute_stroke_curvature_continuity(image_path, true);
disp(curviness_results);
