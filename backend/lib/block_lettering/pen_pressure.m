function results = compute_pen_pressure_consistency(image_path, debug)
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
    binary = ~binary;  % Invert to match convention

    % Calculate distance transform
    dist_transform = bwdist(~binary);

    % Get stroke thickness measurements (non-zero values represent thickness)
    thickness_values = dist_transform(binary);

    % Skip processing if no strokes detected
    if isempty(thickness_values)
        results = struct('mean_thickness', 0, ...
                        'thickness_std', 0, ...
                        'thickness_variance', 0, ...
                        'coefficient_of_variation', 0);
        return;
    end

    % Calculate statistical measures
    mean_thickness = mean(thickness_values(:));
    thickness_std = std(thickness_values(:));
    thickness_variance = var(thickness_values(:));
    coefficient_of_variation = thickness_std / mean_thickness;

    % Pack results into a structure
    results = struct(...
        'mean_thickness', mean_thickness, ...
        'thickness_std', thickness_std, ...
        'thickness_variance', thickness_variance, ...
        'coefficient_of_variation', coefficient_of_variation ...
    );

    % Debug visualization if requested
    if debug
        figure('Name', 'Pen Pressure Analysis', 'NumberTitle', 'off');

        % Original Image
        subplot(2,2,1);
        imshow(img);
        title('Original Image');

        % Binary Image
        subplot(2,2,2);
        imshow(binary);
        title('Binary Image');

        % Distance Transform Visualization
        subplot(2,2,3);
        imshow(dist_transform, []);
        colormap(gca, jet);
        colorbar;
        title('Distance Transform (Stroke Thickness)');

        % Histogram of thickness values
        subplot(2,2,4);
        histogram(thickness_values, 30);
        hold on;
        xline(mean_thickness, 'r--', 'LineWidth', 2);
        hold off;
        title(sprintf('Thickness Distribution\nMean: %.2f, CoV: %.2f', ...
              mean_thickness, coefficient_of_variation));

        % Print debug information
        fprintf('Mean stroke thickness: %.3f\n', mean_thickness);
        fprintf('Thickness standard deviation: %.3f\n', thickness_std);
        fprintf('Coefficient of variation: %.3f\n', coefficient_of_variation);
    end
end

% === Example Usage ===
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/1.png';
pressure_results = compute_pen_pressure_consistency(image_path, true);
disp(pressure_results);