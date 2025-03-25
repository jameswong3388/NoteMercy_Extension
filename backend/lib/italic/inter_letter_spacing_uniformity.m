function results = computeInterLetterSpacingUniformity(imagePath, debug)
    if nargin < 2
        debug = false;
    end

    % Read the image
    img = imread(imagePath);
    if isempty(img)
        results = struct('error', 'Image not found');
        return;
    end

    % Convert to grayscale if necessary
    if size(img,3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end

    % Binarize the image using Otsu's method and invert the result
    level = graythresh(gray);
    binary = imbinarize(gray, level);
    binary = ~binary; % Invert so letters become white

    % Find connected components and extract bounding boxes
    cc = bwconncomp(binary);
    stats = regionprops(cc, 'BoundingBox');
    if isempty(stats)
        results = struct('error', 'No letters found');
        return;
    end

    % Create a matrix of bounding boxes: each row is [x, y, width, height]
    boxes = zeros(length(stats), 4);
    for i = 1:length(stats)
        boxes(i,:) = stats(i).BoundingBox;
    end

    % Sort boxes by the x coordinate (leftmost)
    boxes = sortrows(boxes, 1);

    % Calculate gaps between adjacent letters
    gaps = [];
    for i = 1:size(boxes,1)-1
        x_current_right = boxes(i,1) + boxes(i,3);
        x_next = boxes(i+1,1);
        gap = x_next - x_current_right;
        if gap > 0
            gaps = [gaps, gap];
        end
    end

    results = struct();
    if ~isempty(gaps)
        % Compute gap statistics
        avg_gap = mean(gaps);
        gap_std = std(gaps);
        letter_widths = boxes(:,3);
        median_width = median(letter_widths);

        % Normalize by median letter width for scale invariance
        if median_width > 0
            norm_avg_gap = avg_gap / median_width;
            norm_gap_std = gap_std / median_width;
        else
            norm_avg_gap = avg_gap;
            norm_gap_std = gap_std;
        end

        % Store results
        results.raw_avg_gap = avg_gap;
        results.raw_gap_std = gap_std;
        results.median_letter_width = median_width;
        results.normalized_avg_gap = norm_avg_gap;
        results.normalized_gap_std = norm_gap_std;
        results.gap_count = length(gaps);

        % Qualitative assessment based on normalized gaps
        if norm_avg_gap >= 0.1 && norm_avg_gap <= 0.5 && norm_gap_std < 0.2
            results.assessment = 'Likely italic handwriting (moderate, consistent spacing)';
        elseif norm_avg_gap < 0.1
            results.assessment = 'Likely cursive handwriting (minimal spacing)';
        elseif norm_avg_gap > 0.5
            results.assessment = 'Likely printed handwriting (large spacing)';
        else
            results.assessment = 'Indeterminate style';
        end

        % Debug visualization if enabled
        if debug
            figure;

            % Subplot 1: Original Image
            subplot(2,2,1);
            imshow(img);
            title('Original Image');

            % Subplot 2: Binary Image with bounding boxes
            subplot(2,2,2);
            imshow(binary);
            hold on;
            for i = 1:size(boxes,1)
                rectangle('Position', boxes(i,:), 'EdgeColor', 'g', 'LineWidth', 2);
            end
            hold off;
            title('Letter Detection');

            % Subplot 3: Histogram of raw gaps
            subplot(2,2,3);
            histogram(gaps, 20);
            hold on;
            yl = ylim;
            plot([avg_gap avg_gap], yl, 'r--', 'LineWidth', 2);
            hold off;
            title(sprintf('Gap Distribution (Mean=%.2f)', avg_gap));

            % Subplot 4: Histogram of normalized gaps
            if median_width > 0
                norm_gaps = gaps / median_width;
                subplot(2,2,4);
                histogram(norm_gaps, 20);
                hold on;
                yl = ylim;
                plot([norm_avg_gap norm_avg_gap], yl, 'r--', 'LineWidth', 2);
                hold off;
                title(sprintf('Normalized Gaps (Mean=%.2f)', norm_avg_gap));
            end
        end
    else
        results.error = 'No gaps found';
    end
end


% Example usage
imagePath = '/Users/jameswong/PycharmProjects/NoteMercy_Extension/backend/atest/5.png';  % Update with your image file path
results = computeInterLetterSpacingUniformity(imagePath, true);
disp(results);
