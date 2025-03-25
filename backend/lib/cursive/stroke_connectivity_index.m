function results = compute_cursive_connectivity_index(image_path, debug)
    % Set default for debug if not provided
    if nargin < 2
        debug = false;
    end

    %% Read and Preprocess the Image
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

    % Create a binary image using adaptive thresholding.
    % Invert so that text (ink) is white.
    binary = imbinarize(gray, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.4);
    binary = ~binary;

    % Remove small objects (noise) and perform morphological closing
    binary = bwareaopen(binary, 20);  % remove objects smaller than 20 pixels
    se = strel('disk', 1);
    binary_closed = imclose(binary, se);

    %% Skeletonization and Graph Analysis
    % Skeletonize the binary image to obtain a one-pixel-wide representation.
    skeleton = bwmorph(binary_closed, 'skel', Inf);
    
    % Find branchpoints and endpoints in the skeleton
    branchpoints = bwmorph(skeleton, 'branchpoints');
    endpoints = bwmorph(skeleton, 'endpoints');
    num_branchpoints = sum(branchpoints(:));
    num_endpoints = sum(endpoints(:));
    
    % Compute a skeleton connectivity ratio (if there are any endpoints/branchpoints)
    if (num_branchpoints + num_endpoints) > 0
        skeleton_connectivity = num_branchpoints / (num_branchpoints + num_endpoints);
    else
        skeleton_connectivity = 0;
    end

    %% Connected Component Analysis for Gap-Based Metrics
    [labeled, num_components] = bwlabel(binary_closed);
    stats = regionprops(labeled, 'BoundingBox', 'Area');
    
    % Filter out very small components
    min_area = 20;
    valid_components = find([stats.Area] >= min_area);
    num_valid_components = length(valid_components);
    
    if num_valid_components == 0
        results = struct(...
            'connectivity_index', 0, ...
            'skeleton_connectivity', skeleton_connectivity, ...
            'gap_index', 0, ...
            'avg_gap_ratio', 0, ...
            'ink_density', 0, ...
            'component_density', 0, ...
            'num_components', 0, ...
            'num_branchpoints', num_branchpoints, ...
            'num_endpoints', num_endpoints ...
        );
        return;
    end

    % Extract bounding boxes and sort them left-to-right (assumes horizontal text)
    bboxes = zeros(num_valid_components, 4);
    for i = 1:num_valid_components
        bboxes(i, :) = stats(valid_components(i)).BoundingBox;
    end
    [~, sorted_indices] = sort(bboxes(:,1));
    sorted_components = valid_components(sorted_indices);

    % Compute gap distances and average gap (normalized by average character width)
    if num_valid_components > 1
        component_gaps = zeros(1, num_valid_components-1);
        widths = zeros(1, num_valid_components);
        for i = 1:num_valid_components
            bb = stats(sorted_components(i)).BoundingBox;
            widths(i) = bb(3);
        end
        avg_char_width = mean(widths);
        for i = 1:num_valid_components-1
            bb_current = stats(sorted_components(i)).BoundingBox;
            bb_next = stats(sorted_components(i+1)).BoundingBox;
            curr_right = bb_current(1) + bb_current(3);
            next_left = bb_next(1);
            gap = max(0, next_left - curr_right);
            component_gaps(i) = gap;
        end
        normalized_gaps = component_gaps / avg_char_width;
        avg_gap = mean(normalized_gaps);
        % For gap-based connectivity, a lower gap implies a more connected structure.
        gap_index = 1 - avg_gap;
        gap_index = max(0, min(1, gap_index));
    else
        gap_index = 1; % if only one component, assume full connectivity
        avg_gap = 0;
    end

    %% Combine Metrics to Compute Overall Connectivity Index
    % Here we take a simple average of the skeleton and gap-based connectivity.
    connectivity_index = 0.5 * skeleton_connectivity + 0.5 * gap_index;
    
    % Calculate additional metrics
    [img_height, img_width] = size(binary_closed);
    ink_density = sum(binary_closed(:)) / (img_width * img_height);
    component_density = num_valid_components / (img_width * img_height);
    
    % Pack results into a structure
    results = struct(...
        'connectivity_index', connectivity_index, ...
        'skeleton_connectivity', skeleton_connectivity, ...
        'gap_index', gap_index, ...
        'avg_gap_ratio', avg_gap, ...
        'ink_density', ink_density, ...
        'component_density', component_density, ...
        'num_components', num_valid_components, ...
        'num_branchpoints', num_branchpoints, ...
        'num_endpoints', num_endpoints ...
    );

    %% Debug Visualization
    if debug
        figure('Name', 'Cursive Connectivity Analysis', 'NumberTitle', 'off');
        
        subplot(2,3,1);
        imshow(img);
        title('Original Image');
        
        subplot(2,3,2);
        imshow(binary_closed);
        title('Binary (Closed) Image');
        
        subplot(2,3,3);
        imshow(skeleton);
        title('Skeletonized Image');
        
        subplot(2,3,4);
        rgb_label = label2rgb(labeled, 'jet', 'k', 'shuffle');
        imshow(rgb_label);
        title(sprintf('Connected Components (%d)', num_valid_components));
        
        subplot(2,3,5);
        imshow(skeleton);
        hold on;
        % Overlay branchpoints (red) and endpoints (green)
        [bp_y, bp_x] = find(branchpoints);
        plot(bp_x, bp_y, 'ro', 'MarkerSize', 5);
        [ep_y, ep_x] = find(endpoints);
        plot(ep_x, ep_y, 'go', 'MarkerSize', 5);
        hold off;
        title('Skeleton with Keypoints');
        
        subplot(2,3,6);
        axis off;
        text(0.1, 0.9, sprintf('Connectivity Index: %.3f', connectivity_index), 'FontSize', 10);
        text(0.1, 0.8, sprintf('Skeleton Connectivity: %.3f', skeleton_connectivity), 'FontSize', 10);
        text(0.1, 0.7, sprintf('Gap Index: %.3f', gap_index), 'FontSize', 10);
        text(0.1, 0.6, sprintf('Avg Gap Ratio: %.3f', avg_gap), 'FontSize', 10);
        text(0.1, 0.5, sprintf('Ink Density: %.3f', ink_density), 'FontSize', 10);
        text(0.1, 0.4, sprintf('Component Density: %.3f', component_density), 'FontSize', 10);
        text(0.1, 0.3, sprintf('Components: %d', num_valid_components), 'FontSize', 10);
        text(0.1, 0.2, sprintf('Branchpoints: %d, Endpoints: %d', num_branchpoints, num_endpoints), 'FontSize', 10);
    end
end

%% === Main Script for Testing ===
% Replace the image_path below with the path to your cursive handwriting image.
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png';
connectivity_results = compute_cursive_connectivity_index(image_path, true);
disp(connectivity_results);
