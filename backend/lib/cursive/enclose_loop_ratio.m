 function results = compute_enclosed_loop_ratio(image_path, debug)
% COMPUTE_ENCLOSED_LOOP_RATIO Analyzes an image for enclosed loop metrics
%
%   results = compute_enclosed_loop_ratio(image_path, debug)
%
%   This function reads a grayscale image, preprocesses it (Gaussian blur,
%   adaptive thresholding, morphological closing), segments text lines and
%   words, and then computes a “loopiness” metric by counting inner (hole)
%   and outer contours. Optionally, debug mode produces a four-panel
%   visualization.
%
%   Example:
%       image_path = 'atest/5.png';
%       results = compute_enclosed_loop_ratio(image_path, true);

    %% 1. Read Image
    img = imread(image_path);
    if size(img,3) == 3
        img = rgb2gray(img);
    end
    if isempty(img)
        fprintf('Could not read image at: %s\n', image_path);
        results = [];
        return;
    end
    original = img;  % Save original for later visualization

    %% 2. Preprocessing
    % Apply Gaussian blur (kernel size ~5x5, sigma ~1)
    img_blur = imgaussfilt(img, 1);

    % Adaptive thresholding: use adaptthresh then binarize.
    % The threshold is computed locally (with a neighborhood of 11) using a gaussian statistic.
    T = adaptthresh(img_blur, 0.5, 'NeighborhoodSize', 11, 'Statistic', 'gaussian');
    binary = imbinarize(img_blur, T);
    binary = ~binary; % Invert so that text becomes white

    % Morphological closing to clean up the binary image
    se = strel('square', 2);
    binary = imclose(binary, se);

    %% 3. Improved Word Segmentation via Text Line Detection
    % Use horizontal projection profile to detect lines
    horiz_proj = sum(binary, 2);
    line_threshold = max(horiz_proj) * 0.1;
    lines = [];
    in_line = false;
    for i = 1:length(horiz_proj)
        if ~in_line && horiz_proj(i) > line_threshold
            in_line = true;
            line_start = i;
        elseif in_line && horiz_proj(i) <= line_threshold
            in_line = false;
            if (i - line_start) > 10  % Minimum line height
                lines = [lines; line_start, i-1];
            end
        end
    end
    % If still in line at the bottom, add final line
    if in_line
        lines = [lines; line_start, length(horiz_proj)];
    end

    % Initialize cell array to store word info (each entry: struct with word image and bounding box)
    words = {};
    for l = 1:size(lines,1)
        ls = lines(l,1);
        le = lines(l,2);
        line_img = binary(ls:le, :);
        
        % Use connected components to get potential word components
        CC = bwconncomp(line_img, 8);
        stats = regionprops(CC, 'BoundingBox');
        if isempty(stats)
            continue;
        end
        
        % Extract bounding boxes and sort them by their x-coordinate (left edge)
        boxes = reshape([stats.BoundingBox], 4, [])';
        [~, sortIdx] = sort(boxes(:,1));
        boxes = boxes(sortIdx,:);
        compIndices = sortIdx;
        
        % Group nearby components as a single word based on horizontal gap
        current_group = compIndices(1);
        groups = {};
        for i = 2:length(compIndices)
            prev_box = boxes(i-1, :); % [x, y, width, height]
            curr_box = boxes(i, :);
            gap = curr_box(1) - (prev_box(1) + prev_box(3));
            if gap < 0.8 * prev_box(3)
                current_group = [current_group, compIndices(i)];
            else
                groups{end+1} = current_group; %#ok<AGROW>
                current_group = compIndices(i);
            end
        end
        groups{end+1} = current_group;
        
        % For each group, determine the union bounding box and extract the word image
        for g = 1:length(groups)
            group_indices = groups{g};
            % Select boxes corresponding to the group
            group_boxes = boxes(ismember(compIndices, group_indices), :);
            x_min = min(group_boxes(:,1));
            y_min = min(group_boxes(:,2));
            x_max = max(group_boxes(:,1) + group_boxes(:,3));
            y_max = max(group_boxes(:,2) + group_boxes(:,4));
            
            % Adjust vertical coordinates to the original image (line offset)
            y_min_global = y_min + ls - 1;
            y_max_global = y_max + ls - 1;
            
            % Convert bounding box to integer indices (with boundary checking)
            x_min_i = max(floor(x_min), 1);
            y_min_i = max(floor(y_min_global), 1);
            x_max_i = min(ceil(x_max), size(binary,2));
            y_max_i = min(ceil(y_max_global), size(binary,1));
            
            word_img = binary(y_min_i:y_max_i, x_min_i:x_max_i);
            % Filter out small regions (noise)
            if (size(word_img,1) > 10) && (size(word_img,2) > 10)
                wordStruct.word_img = word_img;
                % Save bounding box as [x, y, width, height]
                wordStruct.bbox = [x_min_i, y_min_i, x_max_i - x_min_i + 1, y_max_i - y_min_i + 1];
                words{end+1} = wordStruct; %#ok<AGROW>
            end
        end
    end

    if isempty(words)
        fprintf('No word-like regions found in the image.\n');
        results = [];
        return;
    end

    %% 4. Compute Enclosed Loop Metrics
    total_outer_count = 0;
    total_inner_count = 0;
    word_loopiness = zeros(length(words), 1);
    
    % For each word, use connected component EulerNumber to determine hole count.
    % For a single connected component, EulerNumber = (# objects - # holes),
    % so holes = 1 - EulerNumber.
    for i = 1:length(words)
        w = words{i}.word_img;
        CC_word = bwconncomp(w, 8);
        props = regionprops(CC_word, 'EulerNumber');
        outer_count = CC_word.NumObjects;
        inner_count = 0;
        for j = 1:length(props)
            inner_count = inner_count + (1 - props(j).EulerNumber);
        end
        if outer_count > 0
            word_loopiness(i) = inner_count / outer_count;
        else
            word_loopiness(i) = 0;
        end
        total_outer_count = total_outer_count + outer_count;
        total_inner_count = total_inner_count + inner_count;
    end

    if total_outer_count > 0
        global_loopiness = total_inner_count / total_outer_count;
    else
        global_loopiness = 0;
    end
    avg_word_loopiness = mean(word_loopiness);
    std_loopiness = 0;
    if numel(word_loopiness) > 1
        std_loopiness = std(word_loopiness);
    end

    % Store results in a structure
    results.global_loopiness = global_loopiness;
    results.avg_word_loopiness = avg_word_loopiness;
    results.std_loopiness = std_loopiness;
    results.inner_contour_count = total_inner_count;
    results.outer_contour_count = total_outer_count;
    results.word_count = length(words);

    fprintf('Enclosed Loop Ratio Analysis Results:\n');
    fprintf('  global_loopiness: %.3f\n', results.global_loopiness);
    fprintf('  avg_word_loopiness: %.3f\n', results.avg_word_loopiness);
    fprintf('  std_loopiness: %.3f\n', results.std_loopiness);
    fprintf('  inner_contour_count: %d\n', results.inner_contour_count);
    fprintf('  outer_contour_count: %d\n', results.outer_contour_count);
    fprintf('  word_count: %d\n', results.word_count);

    %% 5. Debug Visualization (if enabled)
    if debug
        % Create a color version of the original image for visualization
        vis_img = repmat(original, [1, 1, 3]);
        % Draw bounding boxes around detected words (using insertShape requires Computer Vision Toolbox)
        for i = 1:length(words)
            bbox = words{i}.bbox;  % Format: [x, y, width, height]
            vis_img = insertShape(vis_img, 'Rectangle', bbox, 'Color', 'green', 'LineWidth', 2);
        end

        % Visualize a sample word (middle word)
        sample_idx = ceil(length(words) / 2);
        sample_word = words{sample_idx}.word_img;
        % Get boundaries (including holes)
        [B, ~] = bwboundaries(sample_word, 'holes');
        % Create an RGB image for the sample word (scale logical to uint8)
        sample_vis = repmat(uint8(sample_word)*255, [1, 1, 3]);
        for k = 1:length(B)
            boundary = B{k};
            % Compute signed area using the shoelace formula to determine contour orientation.
            x = boundary(:,2);
            y = boundary(:,1);
            signed_area = 0.5 * sum( x .* circshift(y, -1) - y .* circshift(x, -1) );
            if signed_area > 0
                % Outer contour: draw in green
                sample_vis = insertShape(sample_vis, 'Polygon', boundaryToPolygon(boundary), 'Color', 'green', 'LineWidth', 2);
            else
                % Inner contour (hole): draw in red
                sample_vis = insertShape(sample_vis, 'Polygon', boundaryToPolygon(boundary), 'Color', 'red', 'LineWidth', 2);
            end
        end

        % Create a 2x2 figure for visualizations
        figure;
        subplot(2,2,1);
        imshow(original);
        title('Original Image');

        subplot(2,2,2);
        imshow(binary);
        title('Binarized Image');

        subplot(2,2,3);
        imshow(vis_img);
        title('Detected Words');

        subplot(2,2,4);
        imshow(sample_vis);
        title(sprintf('Sample Word (Loopiness: %.2f)', word_loopiness(sample_idx)));

        % Save the figure to file
        saveas(gcf, [image_path, '_loopiness_analysis.png']);
        close(gcf);
    end
end

%% Helper function to convert boundary points into polygon vector format
function polygon = boundaryToPolygon(boundary)
    % Convert an Mx2 boundary array into a 1x(2*M) vector [x1 y1 x2 y2 ...]
    polygon = reshape(boundary(:, [2 1])', 1, []);
end

%% Main Script Execution (if run as a script)
% Uncomment the following lines to run the analysis directly.
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/5.png';  % Replace with your image file path
results = compute_enclosed_loop_ratio(image_path, true);
