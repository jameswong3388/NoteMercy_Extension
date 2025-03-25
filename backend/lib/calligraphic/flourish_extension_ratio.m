function results = compute_flourish_extension_ratio(image_path, debug)
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

    % Binarize the image
    binary = imbinarize(gray);
    binary = ~binary;  % Invert if needed

    % Get connected components and basic properties
    CC = bwconncomp(binary);
    stats = regionprops(CC, 'BoundingBox', 'Area', 'Perimeter', 'Extent', 'EulerNumber');
    
    if isempty(stats)
        results = struct('flourish_ratio', 0, ...
                        'width_ratio', 0, ...
                        'height_ratio', 0, ...
                        'core_area', 0, ...
                        'complexity_ratio', 0, ...
                        'vertical_proportion', 0);
        return;
    end

    % Find the main letter body (largest connected component)
    [~, mainIdx] = max([stats.Area]);
    mainBox = stats(mainIdx).BoundingBox;
    
    % Calculate core letter body metrics
    coreWidth = mainBox(3);
    coreHeight = mainBox(4);
    coreArea = stats(mainIdx).Area;
    
    % Calculate total extent including flourishes
    totalBox = regionprops(binary, 'BoundingBox');
    totalWidth = totalBox(1).BoundingBox(3);
    totalHeight = totalBox(1).BoundingBox(4);
    
    % Calculate ratios
    widthRatio = totalWidth / coreWidth;
    heightRatio = totalHeight / coreHeight;
    flourishRatio = (widthRatio + heightRatio) / 2;

    % Calculate contour complexity using perimeter-to-area ratio
    totalPerimeter = sum([stats.Perimeter]);
    totalArea = sum([stats.Area]);
    complexityRatio = totalPerimeter^2 / totalArea;

    % Calculate vertical distribution metrics
    verticalProj = sum(binary, 2);  % Horizontal projection
    cumDist = cumsum(verticalProj) / sum(verticalProj);
    
    % Find 5th and 95th percentiles for vertical extent
    lowQuant = find(cumDist > 0.05, 1);  % 5th percentile
    highQuant = find(cumDist < 0.95, 1, 'last');  % 95th percentile
    verticalProportion = (size(binary,1) - highQuant - lowQuant) / size(binary,1);

    % Pack results with new metrics
    results = struct(...
        'flourish_ratio', flourishRatio, ...
        'width_ratio', widthRatio, ...
        'height_ratio', heightRatio, ...
        'core_area', coreArea, ...
        'complexity_ratio', complexityRatio, ...
        'vertical_proportion', verticalProportion ...
    );

    % Enhanced debug visualization
    if debug
        figure('Name', 'Calligraphic Analysis', 'NumberTitle', 'off');
        
        % Original and Binary Images
        subplot(2,3,1);
        imshow(img);
        title('Original Image');
        
        subplot(2,3,2);
        imshow(binary);
        title('Binary Image');
        
        % Bounding Boxes
        subplot(2,3,3);
        imshow(binary);
        hold on;
        rectangle('Position', mainBox, 'EdgeColor', 'r', 'LineWidth', 2);
        rectangle('Position', totalBox(1).BoundingBox, 'EdgeColor', 'b', 'LineWidth', 2);
        hold off;
        title('Bounding Boxes');
        
        % Vertical Distribution
        subplot(2,3,4);
        plot(verticalProj, 1:length(verticalProj));
        title('Vertical Distribution');
        xlabel('Pixel Count');
        ylabel('Vertical Position');
        
        % Metrics Display
        subplot(2,3,[5,6]);
        axis off;
        text(0.1, 0.9, sprintf('Flourish Ratio: %.2f', results.flourish_ratio));
        text(0.1, 0.8, sprintf('Width Ratio: %.2f', results.width_ratio));
        text(0.1, 0.7, sprintf('Height Ratio: %.2f', results.height_ratio));
        text(0.1, 0.6, sprintf('Core Area: %d pixels', results.core_area));
        text(0.1, 0.5, sprintf('Complexity Ratio: %.2f', results.complexity_ratio));
        text(0.1, 0.4, sprintf('Vertical Proportion: %.2f', results.vertical_proportion));
        title('Metrics');
        
        % Print debug information
        fprintf('\nCalligraphic Analysis Results:\n');
        fprintf('Flourish extension ratio: %.2f\n', results.flourish_ratio);
        fprintf('Complexity ratio: %.2f\n', results.complexity_ratio);
        fprintf('Vertical proportion: %.2f\n', results.vertical_proportion);
    end
end

% === Example Usage ===
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/4.png';
flourish_results = compute_flourish_extension_ratio(image_path, true);
disp(flourish_results);