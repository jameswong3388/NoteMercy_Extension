function results = compute_stroke_continuity(image_path, debug)
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

    % Apply binary thresholding and invert
    threshold = 127 / 255;
    binary = imbinarize(gray, threshold);
    binary = ~binary;  % Invert to match cv2.THRESH_BINARY_INV

    % Get connected components
    CC = bwconncomp(binary);
    num_components = CC.NumObjects;

    % Skeletonize the image
    skel = bwmorph(binary, 'thin', Inf);

    % Find endpoints and branch points
    endpts = bwmorph(skel, 'endpoints');
    branchpts = bwmorph(skel, 'branchpoints');
    
    % Count endpoints and branches
    num_endpoints = sum(endpts(:));
    num_branches = sum(branchpts(:));

    % Calculate average components per word
    % (assuming input is word-level, this will typically be 1)
    components_per_word = num_components;

    % Pack results into a structure
    results = struct(...
        'num_components', num_components, ...
        'num_endpoints', num_endpoints, ...
        'num_branches', num_branches, ...
        'components_per_word', components_per_word ...
    );

    % Debug visualization if requested
    if debug
        figure('Name', 'Stroke Continuity Analysis', 'NumberTitle', 'off');
        
        % Original Image
        subplot(2,2,1);
        imshow(img);
        title('Original Image');
        
        % Binary Image
        subplot(2,2,2);
        imshow(binary);
        title('Binary Image');
        
        % Skeleton with endpoints and branch points
        subplot(2,2,3);
        imshow(skel);
        hold on;
        
        % Plot endpoints in green
        [ey, ex] = find(endpts);
        plot(ex, ey, 'go', 'MarkerSize', 10, 'LineWidth', 2);
        
        % Plot branch points in red
        [by, bx] = find(branchpts);
        plot(bx, by, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
        
        hold off;
        title('Skeleton with Endpoints (green) and Branch Points (red)');
        
        % Connected Components
        subplot(2,2,4);
        labeled = labelmatrix(CC);
        rgb_label = label2rgb(labeled, 'jet', 'k', 'shuffle');
        imshow(rgb_label);
        title(sprintf('Connected Components (%d)', num_components));
        
        % Print debug information
        fprintf('Number of connected components: %d\n', num_components);
        fprintf('Number of endpoints: %d\n', num_endpoints);
        fprintf('Number of branch points: %d\n', num_branches);
    end
end

% === Example Usage ===
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png';
continuity_results = compute_stroke_continuity(image_path, true);
disp(continuity_results);
