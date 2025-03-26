function results = analyze_pen_pressure(image_path, debug)
% ANALYZE_PEN_PRESSURE Analyze pen pressure and ligature frequency in shorthand images.
%
%   results = analyze_pen_pressure(image_path, debug) processes the image at the 
%   specified image_path and computes metrics based on pen pressure and ligature frequency.
%
%   Parameters:
%       image_path - string path to the image file.
%       debug - (optional) boolean flag to enable debug visualization (default: false).
%
%   Returns:
%       results - a structure containing:
%           avg_pressure          - Global average normalized pen pressure.
%           pressure_variation    - Global average pressure variation (std dev).
%           pressure_consistency  - Global average pressure consistency (std/mean).
%           ligature_count        - Number of detected ligatures.
%           ligature_frequency    - Ratio of ligatures to valid components.
%           component_count       - Number of valid components analyzed.
%

    if nargin < 2
        debug = false;
    end

    %% Read and Preprocess the Image
    try
        img = imread(image_path);
    catch ME
        fprintf('Error: Could not read image at %s\n', image_path);
        results = struct();
        return;
    end

    % Convert to grayscale if the image is in RGB
    if size(img, 3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end

    % Use Otsu's method to determine an optimal threshold and binarize
    threshold = graythresh(gray);
    binary = imbinarize(gray, threshold);
    binary = ~binary;  % Invert the binary image for analysis

    %% Connected Component Analysis
    CC = bwconncomp(binary);

    % Initialize arrays for storing valid components and their metrics
    pressure_metrics = [];
    validPixelIdxList = {};  % To store pixel indices for valid components

    minPixels = 50;  % Minimum pixel count to filter out noise

    for i = 1:CC.NumObjects
        pixelList = CC.PixelIdxList{i};
        if numel(pixelList) < minPixels
            continue;
        end

        % Create a mask for the current component
        component_mask = false(size(binary));
        component_mask(pixelList) = true;

        % Extract grayscale intensity values for this component and convert to double
        component_intensities = double(gray(component_mask));

        % Invert intensities so that darker strokes (lower intensity) correspond to higher pressure
        inverted_intensities = 255 - component_intensities;

        % Calculate pressure metrics (normalized to [0,1])
        avg_pressure = mean(inverted_intensities) / 255;
        std_pressure = std(inverted_intensities) / 255;
        max_pressure = max(inverted_intensities) / 255;

        % Obtain shape properties for further analysis (e.g., for detecting ligatures)
        props = regionprops(component_mask, 'Area', 'BoundingBox', 'Solidity', 'Eccentricity');
        if isempty(props)
            continue;
        end
        prop = props(1);

        % Pressure consistency: lower variation relative to the mean indicates more consistent pressure
        pressure_consistency = std_pressure / (avg_pressure + eps);

        % Compute aspect ratio from the bounding box
        bbox = prop.BoundingBox;
        aspect_ratio = bbox(3) / max(bbox(4), 1);

        % Store metrics for the current component
        comp_metrics = struct(...
            'avg_pressure', avg_pressure, ...
            'pressure_variation', std_pressure, ...
            'max_pressure', max_pressure, ...
            'pressure_consistency', pressure_consistency, ...
            'area', prop.Area, ...
            'aspect_ratio', aspect_ratio, ...
            'solidity', prop.Solidity, ...
            'eccentricity', prop.Eccentricity);
        pressure_metrics = [pressure_metrics, comp_metrics]; %#ok<AGROW>
        validPixelIdxList{end+1} = pixelList; %#ok<AGROW>
    end

    % Return default results if no valid components were detected
    if isempty(pressure_metrics)
        results = struct('avg_pressure', 0, ...
                         'pressure_variation', 0, ...
                         'pressure_consistency', 0, ...
                         'ligature_count', 0, ...
                         'ligature_frequency', 0, ...
                         'component_count', 0);
        return;
    end

    %% Aggregate Pressure Metrics
    avg_pressures = [pressure_metrics.avg_pressure];
    pressure_variations = [pressure_metrics.pressure_variation];
    pressure_consistencies = [pressure_metrics.pressure_consistency];

    global_avg_pressure = mean(avg_pressures);
    global_pressure_variation = mean(pressure_variations);
    global_pressure_consistency = mean(pressure_consistencies);

    %% Ligature Detection
    % Criteria: components with high eccentricity and moderate solidity may represent ligatures.
    ligature_count = 0;
    for j = 1:length(pressure_metrics)
        comp = pressure_metrics(j);
        if comp.eccentricity > 0.85 && comp.solidity > 0.4 && comp.solidity < 0.75
            ligature_count = ligature_count + 1;
        end
    end
    ligature_frequency = ligature_count / length(pressure_metrics);

    %% Pack the Results
    results = struct(...
        'avg_pressure', global_avg_pressure, ...
        'pressure_variation', global_pressure_variation, ...
        'pressure_consistency', global_pressure_consistency, ...
        'ligature_count', ligature_count, ...
        'ligature_frequency', ligature_frequency, ...
        'component_count', length(pressure_metrics));

    %% Debug Visualization (if requested)
    if debug
        figure('Name', 'Pen Pressure Analysis', 'NumberTitle', 'off');

        % Original Image
        subplot(2,2,1);
        imshow(img);
        title('Original Image');

        % Binary Image with Labeled Components
        subplot(2,2,2);
        labeled = labelmatrix(CC);
        colored_labels = label2rgb(labeled, 'jet', 'k', 'shuffle');
        imshow(colored_labels);
        title('Connected Components');

        % Pressure Heatmap Visualization
        subplot(2,2,3);
        pressure_map = zeros(size(gray));
        for j = 1:length(validPixelIdxList)
            pressure_val = pressure_metrics(j).avg_pressure * 255;
            pressure_map(validPixelIdxList{j}) = pressure_val;
        end
        imshow(pressure_map, []);
        colormap jet;
        colorbar;
        title('Pressure Distribution');

        % Histogram of Average Pressure Values
        subplot(2,2,4);
        histogram(avg_pressures, 20);
        hold on;
        xline(global_avg_pressure, 'r--', 'LineWidth', 2);
        hold off;
        title(sprintf('Pressure Distribution\nGlobal Avg: %.3f', global_avg_pressure));

        % Output debug information to the command window
        fprintf('Average pressure: %.3f\n', global_avg_pressure);
        fprintf('Pressure variation: %.3f\n', global_pressure_variation);
        fprintf('Pressure consistency: %.3f\n', global_pressure_consistency);
        fprintf('Number of components: %d\n', length(pressure_metrics));
        fprintf('Ligature count: %d\n', ligature_count);
        fprintf('Ligature frequency: %.3f\n', ligature_frequency);
    end
end

% === Test code (uncomment to run) ===
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/6.png';
pressure_results = analyze_pen_pressure(image_path, true);
disp(pressure_results);
