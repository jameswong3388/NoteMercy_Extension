function results = compute_vertical_alignment_consistency(image_path, debug)
    % Function to measure vertical alignment consistency in handwriting
    % This feature helps identify print-style writing which typically shows more consistent alignment
    % Input:
    %   image_path: Path to the image file
    %   debug: Boolean flag to show visualization (default: false)
    % Output:
    %   results: Structure containing vertical alignment metrics
    
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

    % Apply binary thresholding and invert (ink is white, background is black)
    threshold = 127 / 255;
    binary = imbinarize(gray, threshold);
    binary = ~binary;  % Inversion to match cv2.THRESH_BINARY_INV
    
    % Find connected components (letters)
    cc = bwconncomp(binary, 8);
    stats = regionprops(cc, 'BoundingBox', 'Centroid', 'Area');
    
    % Filter out noise (very small components)
    validIdx = [];
    boxes = [];
    centroids = [];
    
    for i = 1:length(stats)
        bbox = stats(i).BoundingBox;
        area = stats(i).Area;
        
        % Filter small components (noise) - adjust threshold as needed
        if area > 50 && bbox(3) > 2 && bbox(4) > 5
            validIdx = [validIdx, i];
            boxes = [boxes; bbox];
            centroids = [centroids; stats(i).Centroid];
        end
    end
    
    % Check if we found any valid components
    if isempty(validIdx)
        results = struct('baseline_deviation', 0, ...
                         'xheight_deviation', 0, ...
                         'overall_alignment_score', 0, ...
                         'component_count', 0);
        return;
    end
    
    % Extract y-coordinates of bottom and top edges of all boxes
    % Box format is [x, y, width, height] where (x,y) is the top-left corner
    bottoms = boxes(:,2) + boxes(:,4);  % y + height = bottom edge y-coordinate
    tops = boxes(:,2);                  % y = top edge y-coordinate
    
    % Determine components on the same line using clustering of bottom y-coordinates
    % We use DBSCAN clustering to group components by their baseline
    bottom_clusters = dbscanCluster(bottoms, 5);  % 5 pixels tolerance
    
    % Process each line separately
    line_metrics = [];
    for lineId = 1:max(bottom_clusters)
        % Get components that belong to this line
        lineIndices = find(bottom_clusters == lineId);
        
        if length(lineIndices) < 3
            % Skip lines with too few components for meaningful statistics
            continue;
        end
        
        line_bottoms = bottoms(lineIndices);
        line_tops = tops(lineIndices);
        line_heights = boxes(lineIndices, 4);
        
        % Calculate median bottom and top positions (baseline and approximate x-height line)
        baseline = median(line_bottoms);
        xheight_line = median(line_tops + line_heights*0.6);  % Approximate x-height at 60% of letter height
        
        % Calculate deviations from baseline and x-height
        baseline_deviations = abs(line_bottoms - baseline);
        xheight_deviations = abs(line_tops + line_heights*0.6 - xheight_line);
        
        % Normalize by median letter height to achieve scale invariance
        median_height = median(line_heights);
        if median_height > 0
            norm_baseline_deviations = baseline_deviations / median_height;
            norm_xheight_deviations = xheight_deviations / median_height;
        else
            norm_baseline_deviations = baseline_deviations;
            norm_xheight_deviations = xheight_deviations;
        end
        
        % Store metrics for this line
        line_metrics = [line_metrics; ...
                        mean(norm_baseline_deviations), ...
                        mean(norm_xheight_deviations), ...
                        length(lineIndices)];
    end
    
    % Calculate overall metrics across all lines
    if ~isempty(line_metrics)
        % Weight metrics by number of components in each line
        weights = line_metrics(:,3) / sum(line_metrics(:,3));
        weighted_baseline_dev = sum(line_metrics(:,1) .* weights);
        weighted_xheight_dev = sum(line_metrics(:,2) .* weights);
        
        % Overall alignment score (higher is better)
        % Convert deviations to a 0-1 score where 1 means perfect alignment
        alignment_score = max(0, 1 - (weighted_baseline_dev + weighted_xheight_dev));
    else
        weighted_baseline_dev = 0;
        weighted_xheight_dev = 0;
        alignment_score = 0;
    end
    
    % Pack results into structure
    results = struct(...
        'baseline_deviation', weighted_baseline_dev, ...
        'xheight_deviation', weighted_xheight_dev, ...
        'overall_alignment_score', alignment_score, ...
        'component_count', length(validIdx) ...
    );
    
    % Debug visualization if requested
    if debug
        figure('Name', 'Vertical Alignment Consistency Analysis', 'NumberTitle', 'off');
        
        % Original Image
        subplot(2,2,1);
        imshow(img);
        title('Original Image');
        
        % Binary Image
        subplot(2,2,2);
        imshow(binary);
        title('Binary Image');
        
        % Image with bounding boxes and baselines
        subplot(2,2,3);
        imshow(img);
        hold on;
        
        % Different color for each line
        colors = {'r', 'g', 'b', 'c', 'm', 'y'};
        for lineId = 1:max(bottom_clusters)
            lineIndices = find(bottom_clusters == lineId);
            if length(lineIndices) < 3
                continue;
            end
            
            % Get color for this line
            color_idx = mod(lineId-1, length(colors)) + 1;
            line_color = colors{color_idx};
            
            % Draw bounding boxes for this line
            for i = 1:length(lineIndices)
                idx = lineIndices(i);
                bbox = boxes(idx,:);
                rectangle('Position', bbox, 'EdgeColor', line_color, 'LineWidth', 1);
            end
            
            % Draw baseline
            line_bottoms = bottoms(lineIndices);
            baseline = median(line_bottoms);
            plot([1, size(img,2)], [baseline, baseline], [line_color, '--'], 'LineWidth', 1.5);
            
            % Draw x-height line
            line_heights = boxes(lineIndices, 4);
            line_tops = tops(lineIndices);
            xheight_line = median(line_tops + line_heights*0.6);
            plot([1, size(img,2)], [xheight_line, xheight_line], [line_color, ':'], 'LineWidth', 1.5);
        end
        hold off;
        title('Character Bounding Boxes & Reference Lines');
        
        % Display deviation metrics
        subplot(2,2,4);
        bar([weighted_baseline_dev, weighted_xheight_dev, alignment_score]);
        set(gca, 'XTickLabel', {'Baseline Dev', 'X-height Dev', 'Alignment Score'});
        ylim([0, 1]);
        title('Alignment Metrics');
        
        % Print debug information in the command window
        fprintf('Baseline deviation (normalized): %.3f\n', weighted_baseline_dev);
        fprintf('X-height deviation (normalized): %.3f\n', weighted_xheight_dev); 
        fprintf('Overall alignment score: %.3f\n', alignment_score);
        fprintf('Number of components analyzed: %d\n', length(validIdx));
    end
end

function clusters = dbscanCluster(values, epsilon)
    % A simple implementation of DBSCAN clustering for 1D data
    % Used to cluster y-coordinates of character bottoms to identify text lines
    
    n = length(values);
    clusters = zeros(n, 1);
    current_cluster = 0;
    
    % Sort values to make neighbor finding easier
    [sorted_values, sorting_indices] = sort(values);
    
    for i = 1:n
        % Skip points already assigned to clusters
        if clusters(sorting_indices(i)) ~= 0
            continue;
        end
        
        % Create a new cluster
        current_cluster = current_cluster + 1;
        clusters(sorting_indices(i)) = current_cluster;
        
        % Find all points in epsilon neighborhood
        queue = [];
        for j = 1:n
            if abs(sorted_values(j) - sorted_values(i)) <= epsilon
                queue = [queue j];
            end
        end
        
        % Process queue
        while ~isempty(queue)
            current = queue(1);
            queue(1) = [];
            
            % If not yet assigned, assign to current cluster
            if clusters(sorting_indices(current)) == 0
                clusters(sorting_indices(current)) = current_cluster;
                
                % Find all points in epsilon neighborhood of current
                for j = 1:n
                    if abs(sorted_values(j) - sorted_values(current)) <= epsilon && clusters(sorting_indices(j)) == 0
                        queue = [queue j];
                    end
                end
            end
        end
    end
end

% === Test Section (Comment out for production) ===
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png';
alignment_results = compute_vertical_alignment_consistency(image_path, true);
disp(alignment_results);
