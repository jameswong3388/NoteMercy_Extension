function results = compute_artistic_consistency(image_path, debug)
    % Set default for debug if not provided
    if nargin < 2
        debug = false;
    end

    %% Read and Pre-process Image
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

    % Use adaptive thresholding (Otsu or local method) for robustness
    T = adaptthresh(gray, 0.5);
    binary = imbinarize(gray, T);
    binary = imcomplement(binary);  % Make text white on black background

    % Apply morphological filtering to reduce noise and close gaps
    se = strel('disk', 1);
    binary = imopen(binary, se);
    binary = imclose(binary, se);

    %% Stroke Width Measurement (Pressure Consistency)
    % Compute skeleton and distance transform
    skel = bwmorph(binary, 'thin', Inf);
    D = bwdist(~binary);
    
    % Extract stroke widths at skeleton points
    widths = D(skel);
    widths = widths(widths > 0); % Remove zeros if any
    
    if isempty(widths)
        results = struct('pressure_consistency', 0, ...
                        'transition_smoothness', 0, ...
                        'serif_consistency', 0, ...
                        'overall_artistic_score', 0);
        return;
    end
    
    % Use coefficient of variation as a measure (lower is more consistent)
    width_std = std(widths);
    mean_width = mean(widths);
    pressure_consistency = 1 - min(width_std / mean_width, 1);

    %% Transition Smoothness via Curvature Analysis
    % Instead of only measuring junction angles, we trace each stroke component
    CC = bwconncomp(skel);
    curvature_variances = [];
    for i = 1:CC.NumObjects
        % Create mask for the current stroke component
        comp_mask = false(size(skel));
        comp_mask(CC.PixelIdxList{i}) = true;
        
        % Identify endpoints in this component
        comp_endpoints = bwmorph(comp_mask, 'endpoints');
        [ey, ex] = find(comp_endpoints);
        
        % If endpoints exist, trace the boundary starting from one endpoint.
        if ~isempty(ey)
            start_point = [ey(1), ex(1)];
            boundary = bwtraceboundary(comp_mask, start_point, 'N');
            if isempty(boundary) || size(boundary,1) < 5
                % Fallback: use unsorted component pixels if trace fails
                [y_unsorted, x_unsorted] = find(comp_mask);
                boundary = [y_unsorted, x_unsorted];
            end
        else
            % If no endpoints, fallback to unsorted component pixels
            [y_unsorted, x_unsorted] = find(comp_mask);
            boundary = [y_unsorted, x_unsorted];
        end
        
        % Compute directional angles along the traced boundary
        boundary = double(boundary);
        % Calculate differences between consecutive points
        diff_y = diff(boundary(:,1));
        diff_x = diff(boundary(:,2));
        angles = atan2(diff_y, diff_x);  % angles in radians
        
        % Calculate differences between successive angles (curvature)
        if length(angles) > 1
            angle_diffs = abs(diff(angles));
            % Wrap differences so that maximum difference is pi
            angle_diffs(angle_diffs > pi) = 2*pi - angle_diffs(angle_diffs > pi);
            curvature_var = std(angle_diffs);
            curvature_variances = [curvature_variances, curvature_var];
        end
    end
    
    % Average curvature variance normalized to [0,1] (pi is the maximum possible difference)
    if ~isempty(curvature_variances)
        avg_curvature_var = mean(curvature_variances);
        transition_smoothness = 1 - min(avg_curvature_var / pi, 1);
    else
        transition_smoothness = 0.5; % Default value if no valid curvature data
    end

    %% Serif Detection using Local Edge Density
    % Use endpoints from the skeleton as candidate serif locations
    endpoints_mask = bwmorph(skel, 'endpoints');
    [ep_y, ep_x] = find(endpoints_mask);
    serif_scores = [];
    
    for i = 1:length(ep_y)
        % Define a local window around the endpoint
        window_size = 7;
        y1 = max(1, ep_y(i) - window_size);
        y2 = min(size(binary, 1), ep_y(i) + window_size);
        x1 = max(1, ep_x(i) - window_size);
        x2 = min(size(binary, 2), ep_x(i) + window_size);
        
        local_window = binary(y1:y2, x1:x2);
        
        % Use Canny edge detection to highlight serif-like features
        edges_local = edge(local_window, 'Canny');
        % Compute the edge density in the local window as a surrogate measure
        density = sum(edges_local(:)) / numel(edges_local);
        serif_scores = [serif_scores, density];
    end
    
    if ~isempty(serif_scores) && mean(serif_scores) > 0
        serif_consistency = 1 - min(std(serif_scores) / mean(serif_scores), 1);
    else
        serif_consistency = 0.5;
    end

    %% Combine Metrics to Compute Overall Artistic Consistency Score
    % Weights can be tuned based on validation against expert ratings.
    overall_artistic_score = 0.4 * pressure_consistency + ...
                             0.4 * transition_smoothness + ...
                             0.2 * serif_consistency;
    
    % Pack results into a structure
    results = struct(...
        'pressure_consistency', pressure_consistency, ...
        'transition_smoothness', transition_smoothness, ...
        'serif_consistency', serif_consistency, ...
        'overall_artistic_score', overall_artistic_score ...
    );
    
    %% Debug Visualization (if requested)
    if debug
        figure('Name', 'Artistic Consistency Analysis v2', 'NumberTitle', 'off');
        
        % Original Image
        subplot(2,3,1);
        imshow(img);
        title('Original Image');
        
        % Binary Image after Adaptive Thresholding and Cleaning
        subplot(2,3,2);
        imshow(binary);
        title('Binary Image');
        
        % Skeleton with Endpoints
        subplot(2,3,3);
        imshow(binary);
        hold on;
        [skel_y, skel_x] = find(skel);
        scatter(skel_x, skel_y, 1, 'g', 'filled');
        [ep_y2, ep_x2] = find(endpoints_mask);
        scatter(ep_x2, ep_y2, 20, 'b', 'filled');
        hold off;
        title('Skeleton and Endpoints');
        
        % Heat Map of Stroke Width (Distance Transform Values)
        subplot(2,3,4);
        heatmap = zeros(size(binary));
        [sy, sx] = find(skel);
        for i = 1:length(sy)
            heatmap(sy(i), sx(i)) = D(sy(i), sx(i));
        end
        imagesc(heatmap);
        colormap(jet);
        colorbar;
        title('Stroke Width Map');
        
        % Histogram of Stroke Widths
        subplot(2,3,5);
        histogram(widths, 20);
        hold on;
        xline(mean_width, 'r--', 'LineWidth', 2);
        hold off;
        title(sprintf('Stroke Width Distribution\nConsistency: %.2f', pressure_consistency));
        
        % Feature Scores Display
        subplot(2,3,6);
        axis off;
        text(0.1, 0.9, sprintf('Pressure: %.3f', pressure_consistency), 'FontSize', 10);
        text(0.1, 0.75, sprintf('Transition: %.3f', transition_smoothness), 'FontSize', 10);
        text(0.1, 0.6, sprintf('Serif: %.3f', serif_consistency), 'FontSize', 10);
        text(0.1, 0.45, sprintf('Overall: %.3f', overall_artistic_score), 'FontSize', 12, 'FontWeight', 'bold');
        title('Feature Scores');
        
        % Print debug information to command window
        fprintf('Pressure Consistency: %.3f\n', pressure_consistency);
        fprintf('Transition Smoothness: %.3f\n', transition_smoothness);
        fprintf('Serif Consistency: %.3f\n', serif_consistency);
        fprintf('Overall Artistic Score: %.3f\n', overall_artistic_score);
    end
end

% === Example Usage ===
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png';
artistic_results = compute_artistic_consistency(image_path, true);
disp(artistic_results);
