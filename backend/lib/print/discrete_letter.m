function results = compute_discrete_letter_components(image_path, debug)
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

    % Apply binary thresholding and invert
    binary = imbinarize(gray);
    binary = ~binary;  % Invert to match typical text representation

    % Get connected components
    cc = bwconncomp(binary);
    
    % Get region properties
    stats = regionprops(cc, 'BoundingBox', 'Area');
    
    % Filter out noise and dots/accents based on area
    valid_components = [];
    areas = [stats.Area];
    median_area = median(areas);
    
    for i = 1:length(stats)
        % Consider components with area > 25% of median area as valid letters
        % This helps filter out dots and noise while keeping actual letters
        if stats(i).Area > 0.25 * median_area
            valid_components = [valid_components, i];
        end
    end
    
    % Calculate metrics
    num_components = length(valid_components);
    avg_component_area = mean([stats(valid_components).Area]);
    
    % Pack results into a structure
    results = struct(...
        'num_letter_components', num_components, ...
        'avg_component_area', avg_component_area, ...
        'total_components', cc.NumObjects ...
    );

    % Debug visualization if requested
    if debug
        figure('Name', 'Discrete Letter Component Analysis', 'NumberTitle', 'off');
        
        % Original Image
        subplot(2,2,1);
        imshow(img);
        title('Original Image');
        
        % Binary Image
        subplot(2,2,2);
        imshow(binary);
        title('Binary Image');
        
        % Labeled components
        subplot(2,2,3);
        labeled = labelmatrix(cc);
        rgb_label = label2rgb(labeled, 'jet', 'k', 'shuffle');
        imshow(rgb_label);
        title('Connected Components');
        
        % Valid components highlighted
        subplot(2,2,4);
        imshow(img);
        hold on;
        for i = valid_components
            bb = stats(i).BoundingBox;
            rectangle('Position', bb, 'EdgeColor', 'g', 'LineWidth', 2);
        end
        hold off;
        title(sprintf('Valid Letters Found: %d', num_components));
        
        % Print debug information
        fprintf('Number of letter components: %d\n', num_components);
        fprintf('Total components detected: %d\n', cc.NumObjects);
        fprintf('Average component area: %.2f pixels\n', avg_component_area);
    end
end

% === Example Usage ===
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/4.png';
results = compute_discrete_letter_components(image_path, true);
disp(results);
