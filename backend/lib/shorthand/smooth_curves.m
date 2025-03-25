function results = compute_stroke_smoothness(image_path, debug)
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

    % Binarize and get skeleton
    binary = imbinarize(gray);
    binary = ~binary;  % Invert to match expected format
    skel = bwskel(binary);  % Get the skeleton of the writing

    % Find skeleton points
    [y, x] = find(skel);
    
    if length(x) < 3  % Need at least 3 points for meaningful analysis
        results = struct('avg_curvature_change', 0, ...
                        'curvature_variance', 0, ...
                        'direction_changes', 0, ...
                        'smoothness_score', 0);
        return;
    end

    % Sort points to follow the stroke path
    ordered_points = sort_skeleton_points([x y]);
    x = ordered_points(:,1);
    y = ordered_points(:,2);

    % Fit a spline through skeleton points
    pp = cscvn([x'; y']);
    t = linspace(0, 1, 100);  % Parameter for evaluating spline
    spl = fnval(pp, t);
    
    % Calculate direction changes and curvature
    dx = gradient(spl(1,:));
    dy = gradient(spl(2,:));
    theta = atan2d(dy, dx);
    dTheta = diff(unwrap(deg2rad(theta)));
    
    % Calculate metrics
    avg_curvature_change = mean(abs(dTheta));
    curvature_variance = var(dTheta);
    direction_changes = sum(abs(diff(sign(dTheta))) > 0);
    
    % Normalize by stroke length
    stroke_length = sum(sqrt(diff(spl(1,:)).^2 + diff(spl(2,:)).^2));
    normalized_direction_changes = direction_changes / stroke_length;
    
    % Calculate overall smoothness score (lower is smoother)
    % Add small epsilon to prevent multiplication by zero
    epsilon = 1e-6;
    smoothness_score = (max(avg_curvature_change, epsilon) * ...
                       max(normalized_direction_changes, epsilon) * ...
                       max(curvature_variance, epsilon)) ^ (1/3);
    
    % Scale the smoothness score to a more interpretable range [0-100]
    % Higher score means less smooth (more complex) writing
    smoothness_score = min(100, smoothness_score * 10);

    % Pack results
    results = struct(...
        'avg_curvature_change', avg_curvature_change, ...
        'curvature_variance', curvature_variance, ...
        'direction_changes', normalized_direction_changes, ...
        'smoothness_score', smoothness_score ...
    );

    % Debug visualization if requested
    if debug
        figure('Name', 'Stroke Smoothness Analysis', 'NumberTitle', 'off');
        
        % Original Image
        subplot(2,2,1);
        imshow(img);
        title('Original Image');
        
        % Skeleton
        subplot(2,2,2);
        imshow(skel);
        title('Skeleton');
        
        % Spline fit
        subplot(2,2,3);
        plot(spl(1,:), spl(2,:), 'b-', 'LineWidth', 2);
        axis equal;
        title('Fitted Spline');
        
        % Curvature plot
        subplot(2,2,4);
        plot(dTheta);
        title('Curvature Changes');
        
        % Print debug information
        fprintf('Average curvature change: %.3f\n', avg_curvature_change);
        fprintf('Curvature variance: %.3f\n', curvature_variance);
        fprintf('Normalized direction changes: %.3f\n', normalized_direction_changes);
        fprintf('Smoothness score: %.3f\n', smoothness_score);
    end
end

function ordered_points = sort_skeleton_points(points)
    % Helper function to sort skeleton points along the path
    % Start from leftmost point and connect to nearest neighbors
    ordered_points = zeros(size(points));
    remaining_points = points;
    
    % Start with leftmost point
    [~, idx] = min(remaining_points(:,1));
    ordered_points(1,:) = remaining_points(idx,:);
    remaining_points(idx,:) = [];
    
    % Connect remaining points by nearest neighbor
    for i = 2:size(points,1)
        current = ordered_points(i-1,:);
        distances = pdist2(current, remaining_points);
        [~, idx] = min(distances);
        ordered_points(i,:) = remaining_points(idx,:);
        remaining_points(idx,:) = [];
    end
end

% === Example Usage ===
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/2.png';
smoothness_results = compute_stroke_smoothness(image_path, true);
disp(smoothness_results);