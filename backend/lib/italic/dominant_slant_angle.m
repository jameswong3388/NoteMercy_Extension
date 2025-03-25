function results = compute_slant_angle(image_path, debug)
    % compute_slant_angle - Calculate the dominant stroke (slant) angle from an image.
    %
    % Syntax: results = compute_slant_angle(image_path, debug)
    %
    % Inputs:
    %    image_path - Path to the image file.
    %    debug      - (Optional) Boolean flag to show visualization (default: false).
    %
    % Outputs:
    %    results - A structure with fields:
    %              avg_slant       - Average slant angle (in degrees).
    %              vertical_slant  - Angle from vertical (for intuitive interpretation).
    %              slant_std       - Standard deviation of the slant angles.
    %              num_components  - Number of components analyzed.
    
    if nargin < 2
        debug = false;
    end

    % Read image
    img = imread(image_path);
    if isempty(img)
        error('Could not load image: %s', image_path);
    end

    % Convert to grayscale if needed
    if size(img,3) > 1
        gray = rgb2gray(img);
    else
        gray = img;
    end

    % Thresholding using Otsu's method and invert binary image
    level = graythresh(gray);
    binary = ~imbinarize(gray, level);

    % Label connected components and get properties
    cc = bwconncomp(binary);
    stats = regionprops(cc, 'Area', 'Centroid', 'PixelIdxList');

    % Filter out very small regions (area threshold = 50)
    min_area = 50;
    slant_angles = [];
    for k = 1:length(stats)
        if stats(k).Area <= min_area
            continue;
        end
        
        % Get pixel coordinates for the component
        [rows, cols] = ind2sub(size(binary), stats(k).PixelIdxList);
        % Centroid is given as [x, y] (i.e. [col, row])
        cx = stats(k).Centroid(1);
        cy = stats(k).Centroid(2);
        
        % Compute central moments: mu11 and mu02
        mu11 = sum((cols - cx) .* (rows - cy));
        mu02 = sum((rows - cy).^2);
        
        % Skip if mu02 is nearly zero to avoid division errors
        if abs(mu02) < 1e-2
            continue;
        end
        
        % Compute skew and corresponding angle in degrees
        skew = mu11 / mu02;
        angle_rad = atan(skew);
        angle_deg = angle_rad * (180 / pi);
        
        slant_angles(end+1) = angle_deg; %#ok<AGROW>
    end

    % Calculate statistics if any components were found
    results = struct('avg_slant', [], 'vertical_slant', [], 'slant_std', [], 'num_components', []);
    if ~isempty(slant_angles)
        avg_slant = mean(slant_angles);
        slant_std = std(slant_angles);
        
        % Convert to angle from vertical (for more intuitive interpretation)
        if avg_slant <= 90
            vertical_slant = 90 - avg_slant;
        else
            vertical_slant = avg_slant - 90;
        end
        
        results.avg_slant = avg_slant;
        results.vertical_slant = vertical_slant;
        results.slant_std = slant_std;
        results.num_components = length(slant_angles);
    end

    % Debug visualization if enabled
    if debug && ~isempty(slant_angles)
        % Display original image with contours drawn over the binary regions
        figure;
        imshow(img);
        hold on;
        boundaries = bwboundaries(binary);
        for k = 1:length(boundaries)
            b = boundaries{k};
            plot(b(:,2), b(:,1), 'g', 'LineWidth', 2);
        end
        title('Contours');
        hold off;
        
        % Plot histogram of slant angles
        figure;
        histogram(slant_angles, 20, 'FaceColor', 'blue', 'FaceAlpha', 0.7);
        hold on;
        xline(avg_slant, 'r--', sprintf('Avg: %.1f°, StdDev: %.1f°', avg_slant, slant_std));
        title('Slant Angle Distribution');
        xlabel('Angle (degrees)');
        ylabel('Frequency');
        legend('Slant Angles');
        hold off;
        
        % Print statistics to the command window
        fprintf('Average slant: %.1f°\n', avg_slant);
        fprintf('Angle from vertical: %.1f°\n', vertical_slant);
        fprintf('Slant consistency (std): %.1f°\n', slant_std);
        fprintf('Components analyzed: %d\n', length(slant_angles));
    end
end


image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/5.png';
results = compute_slant_angle(image_path, true);
disp(results);
