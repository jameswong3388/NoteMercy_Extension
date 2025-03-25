function results = compute_stroke_width_variation(image_path, debug)
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
    threshold = 127 / 255;
    bw = imbinarize(gray, threshold);
    bw = ~bw;  % Invert to make text white on black background

    % Compute distance transform and skeleton
    D = bwdist(~bw);                     % Distance transform
    skel = bwmorph(bw, 'thin', Inf);     % Skeleton of the strokes
    strokeRadii = D(skel);               % Radius at each skeleton point

    % Remove zero values (background points)
    strokeRadii = strokeRadii(strokeRadii > 0);

    % If no valid stroke points found
    if isempty(strokeRadii)
        results = struct('mean_width', 0, ...
                        'width_std', 0, ...
                        'width_ratio', 0, ...
                        'variation_coefficient', 0);
        return;
    end

    % Calculate stroke width metrics
    meanWidth = 2 * mean(strokeRadii);    % Average stroke width
    widthStd = 2 * std(strokeRadii);      % Standard deviation
    widthRatio = max(strokeRadii)/min(strokeRadii);  % Thick-thin ratio
    
    % Coefficient of variation (normalized std)
    variationCoeff = widthStd / meanWidth;

    % Pack results into a structure
    results = struct(...
        'mean_width', meanWidth, ...
        'width_std', widthStd, ...
        'width_ratio', widthRatio, ...
        'variation_coefficient', variationCoeff ...
    );

    % Debug visualization if requested
    if debug
        figure('Name', 'Stroke Width Analysis', 'NumberTitle', 'off');
        
        % Original Image
        subplot(2,2,1);
        imshow(img);
        title('Original Image');
        
        % Binary Image
        subplot(2,2,2);
        imshow(bw);
        title('Binary Image');
        
        % Skeleton overlay
        subplot(2,2,3);
        imshow(bw);
        hold on;
        skelPoints = find(skel);
        [y, x] = ind2sub(size(skel), skelPoints);
        scatter(x, y, 1, 'r', 'filled');
        hold off;
        title('Skeleton Points');
        
        % Histogram of stroke widths
        subplot(2,2,4);
        histogram(2*strokeRadii, 20);
        hold on;
        xline(meanWidth, 'r--', 'LineWidth', 2);
        hold off;
        title(sprintf('Stroke Width Distribution\nMean: %.2f, Ratio: %.2f', ...
              meanWidth, widthRatio));
        
        % Print debug information
        fprintf('Mean stroke width: %.3f\n', meanWidth);
        fprintf('Stroke width std: %.3f\n', widthStd);
        fprintf('Thick-thin ratio: %.3f\n', widthRatio);
        fprintf('Variation coefficient: %.3f\n', variationCoeff);
    end
end

% === Example Usage ===
image_path = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/2.png';
width_results = compute_stroke_width_variation(image_path, true);
disp(width_results);