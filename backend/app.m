function handwritingAnalysisApp
    % Import the analyze_block_letter_characteristics function
    current_dir = fileparts(mfilename('fullpath'));
    addpath(fullfile(current_dir, 'lib', 'block_lettering'));

    % --- Create main figure ---
    f = figure('Name','Handwriting Analysis',...
               'NumberTitle','off',...
               'Position',[100 100 1000 600]);

    % --- Left panel for controls and text boxes ---
    leftPanel = uipanel('Parent',f,...
                        'Units','normalized',...
                        'Position',[0 0 0.2 1],...
                        'Title','Controls');

    % --- Button: Upload Image ---
    uploadBtn = uicontrol('Parent',leftPanel,...
                          'Style','pushbutton',...
                          'String','Upload Image',...
                          'Units','normalized',...
                          'Position',[0.1 0.9 0.8 0.05],...
                          'Callback',@uploadImageCallback);

    % --- Button: Preprocess Image ---
    preprocessBtn = uicontrol('Parent',leftPanel,...
                              'Style','pushbutton',...
                              'String','Preprocess',...
                              'Units','normalized',...
                              'Position',[0.1 0.83 0.8 0.05],...
                              'Callback',@preprocessImageCallback);

    % --- Button: Extract Features ---
    extractBtn = uicontrol('Parent',leftPanel,...
                           'Style','pushbutton',...
                           'String','Extract Features',...
                           'Units','normalized',...
                           'Position',[0.1 0.76 0.8 0.05],...
                           'Callback',@extractFeaturesCallback);

    % --- Label & Text Area for Analysis Results ---
    resultsText = uicontrol('Parent',leftPanel,...
                          'Style','text',...
                          'String','Feature Extraction Results:',...
                          'Units','normalized',...
                          'Position',[0.1 0.41 0.8 0.05],...
                          'FontWeight','bold',...
                          'HorizontalAlignment','left');
    resultsBox = uicontrol('Parent',leftPanel,...
                         'Style','edit',...
                         'Max',10,'Min',0,... % Allow multiple lines
                         'String','',...
                         'Units','normalized',...
                         'Position',[0.1 0.21 0.8 0.2],...
                         'HorizontalAlignment','left');

    % --- Label & Edit box: Handwriting Style ---
    styleText = uicontrol('Parent',leftPanel,...
                          'Style','text',...
                          'String','Handwriting Style:',...
                          'Units','normalized',...
                          'Position',[0.1 0.12 0.8 0.05],...
                          'FontWeight','bold',...
                          'HorizontalAlignment','left');
    styleBox = uicontrol('Parent',leftPanel,...
                         'Style','edit',...
                         'Max',2,'Min',0,...
                         'String','',...
                         'Units','normalized',...
                         'Position',[0.1 0.02 0.8 0.1],...
                         'HorizontalAlignment','left');

    % --- Left panel controls - after feature extraction button ---
    % Add task list panel
    taskListText = uicontrol('Parent',leftPanel,...
                            'Style','text',...
                            'String','Feature Extraction Tasks:',...
                            'Units','normalized',...
                            'Position',[0.1 0.69 0.8 0.05],...
                            'FontWeight','bold',...
                            'HorizontalAlignment','left');
                            
    % Create list of tasks with checkboxes
    taskList = uicontrol('Parent',leftPanel,...
                        'Style','listbox',...
                        'String',{},...  % Replace with your actual task names
                        'Units','normalized',...
                        'Position',[0.1 0.5 0.8 0.18],...
                        'Max',2,'Min',0,...
                        'Value',[]); % Allow multiple selection

    % --- Right panel (tab group) for displaying images and analysis ---
    rightPanel = uipanel('Parent',f,...
                         'Units','normalized',...
                         'Position',[0.2 0 0.8 1]);

    % --- Create tab group ---
    tabGroup = uitabgroup('Parent',rightPanel,...
                          'Units','normalized',...
                          'Position',[0 0 1 1]);

    % --- Tab 1: Original Image ---
    tab1 = uitab('Parent',tabGroup,'Title','Original Image');
    ax1 = axes('Parent',tab1,...
               'Units','normalized',...
               'Position',[0.1 0.1 0.8 0.8]);

    % --- Tab 2: Processed Image ---
    tab2 = uitab('Parent',tabGroup,'Title','Processed Image');
    ax2 = axes('Parent',tab2,...
               'Units','normalized',...
               'Position',[0.1 0.1 0.8 0.8]);

    % --- Tab 3: Features Extracted ---
    tab3 = uitab('Parent', tabGroup, 'Title', 'Features Extracted');
    ax3 = axes('Parent', tab3, ...
            'Units','normalized', ...
            'Position',[0.1 0.1 0.8 0.8]);
               
    % --- Tab 4: Analysis ---
    tab4 = uitab('Parent',tabGroup,'Title','Analysis');
    ax4 = axes('Parent',tab4,...
               'Units','normalized',...
               'Position',[0.1 0.1 0.8 0.8]);

    % --- Store handles in a struct for easy sharing in callbacks ---
    handles.ax1 = ax1;
    handles.ax2 = ax2;
    handles.ax3 = ax3;
    handles.ax4 = ax4;
    handles.tab3 = tab3;
    handles.styleBox = styleBox;
    handles.taskList = taskList;
    handles.resultsBox = resultsBox;

    % Initialize and store in figure
    guidata(f,handles);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                     CALLBACK FUNCTIONS                     %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function uploadImageCallback(hObject,~)
        [file, path] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp','Image Files'});
        if isequal(file,0)
            return; % User canceled
        end
        img = imread(fullfile(path, file));

        % Retrieve the handles
        handles = guidata(hObject);

        % Display on the "Original Image" tab
        axes(handles.ax1);
        imshow(img,[]);
        title('Original Image');

        % Store the image in handles
        handles.imgOriginal = img;
        guidata(hObject, handles);
    end

    function preprocessImageCallback(hObject,~)
        handles = guidata(hObject);

        % Ensure an image has been uploaded
        if ~isfield(handles, 'imgOriginal')
            errordlg('Please upload an image first.', 'No Image Found');
            return;
        end

        % Example preprocessing: Convert to grayscale, then binarize
        grayImg = rgb2gray(handles.imgOriginal);
        bwImg   = imbinarize(grayImg);

        % Display on the "Processed Image" tab
        axes(handles.ax2);
        imshow(bwImg);
        title('Processed Image');

        % Store the preprocessed image
        handles.imgPreprocessed = bwImg;
        guidata(hObject, handles);
    end

    function extractFeaturesCallback(hObject,~)
        handles = guidata(hObject);

        % Ensure a preprocessed image is available
        if ~isfield(handles, 'imgPreprocessed')
            errordlg('Please preprocess the image first.', 'No Preprocessed Image');
            return;
        end

        % Initialize task list and results storage if not exists
        if ~isfield(handles, 'taskResults')
            handles.taskResults = struct();
        end

        % Clear previous task list
        set(handles.taskList, 'String', {});
        completedTasks = {};
        
        % Task 1: Block Letter Analysis
        completedTasks{end+1} = 'Block Letter Analysis';
        set(handles.taskList, 'String', completedTasks);
        drawnow;
        
        % Execute block letter analysis with debug visualization
        results = analyze_block_letter_characteristics(handles.imgPreprocessed, true);
        handles.taskResults.blockLetterAnalysis = results;
        
        % Display results in the Features Extracted tab
        axes(handles.ax3);
        cla;
        
        % Create subplots for visualization
        subplot(2,2,1);
        imshow(handles.imgPreprocessed);
        title('Original Image');
        
        subplot(2,2,2);
        imshow(handles.imgPreprocessed);
        hold on;
        for k = 1:length(results.simplified_contours)
            poly = results.simplified_contours{k};
            closed_poly = [poly; poly(1,:)];
            plot(closed_poly(:,1), closed_poly(:,2), 'r', 'LineWidth', 2);
        end
        hold off;
        title('Corner Detection');
        
        subplot(2,2,3);
        histogram(results.corner_angles, 20);
        hold on;
        xline(results.avg_deviation, 'r--', 'LineWidth', 2);
        hold off;
        title(sprintf('Angle Distribution\nMean Deviation: %.2f°', results.avg_deviation));
        
        subplot(2,2,4);
        bar([results.avg_deviation, results.median_deviation, results.max_deviation]);
        set(gca, 'XTickLabel', {'Avg Dev', 'Med Dev', 'Max Dev'});
        title('Deviation Statistics');
        
        % Add listener for task selection
        set(handles.taskList, 'Callback', @taskSelectionCallback);
        
        % Store the updated handles
        guidata(hObject, handles);
    end

    function taskSelectionCallback(hObject, ~)
        % Get selected task
        handles = guidata(hObject);
        selectedIdx = get(hObject, 'Value');
        tasks = get(hObject, 'String');
        selectedTask = tasks{selectedIdx};

        % Display results based on selected task
        switch selectedTask
            case 'Block Letter Analysis'
                if isfield(handles.taskResults, 'blockLetterAnalysis')
                    results = handles.taskResults.blockLetterAnalysis;
                    
                    % Use the stored tab handle directly
                    tab3 = handles.tab3;
                    
                    % Clear all existing axes in the tab
                    existingAxes = findall(tab3, 'Type', 'axes');
                    delete(existingAxes);
                    
                    % Create new subplots
                    subplot(2,2,1, 'Parent', tab3);
                    imshow(handles.imgPreprocessed);
                    title('Original Image');
                    
                    subplot(2,2,2, 'Parent', tab3);
                    imshow(handles.imgPreprocessed);
                    hold on;
                    for k = 1:length(results.simplified_contours)
                        poly = results.simplified_contours{k};
                        closed_poly = [poly; poly(1,:)];
                        plot(closed_poly(:,1), closed_poly(:,2), 'r', 'LineWidth', 2);
                    end
                    hold off;
                    title('Corner Detection');
                    
                    subplot(2,2,3, 'Parent', tab3);
                    histogram(results.corner_angles, 20);
                    hold on;
                    xline(results.avg_deviation, 'r--', 'LineWidth', 2);
                    hold off;
                    title(sprintf('Angle Distribution\nMean Deviation: %.2f°', results.avg_deviation));
                    
                    subplot(2,2,4, 'Parent', tab3);
                    bar([results.avg_deviation, results.median_deviation, results.max_deviation]);
                    set(gca, 'XTickLabel', {'Avg Dev', 'Med Dev', 'Max Dev'});
                    title('Deviation Statistics');
                    
                    % Update the results box with text
                    resultsStr = sprintf('Block Letter Analysis Results:\n');
                    resultsStr = [resultsStr sprintf('Average angle deviation: %.2f°\n', results.avg_deviation)];
                    resultsStr = [resultsStr sprintf('Median angle deviation: %.2f°\n', results.median_deviation)];
                    resultsStr = [resultsStr sprintf('Maximum angle deviation: %.2f°\n', results.max_deviation)];
                    resultsStr = [resultsStr sprintf('Total corners detected: %d\n', results.corner_count)];
                    resultsStr = [resultsStr sprintf('Number of letter shapes: %d\n', results.shape_count)];
                    set(handles.resultsBox, 'String', resultsStr);
                end
        end
    end

    function displayFeatureGraphs(axesHandle, results)
        % General API for displaying graphs from any feature extraction function
        % parameters:
        %   axesHandle: Handle to the axes where graphs should be displayed
        %   results: Structure containing results from feature extraction
        
        % Get field names from the results structure
        fieldNames = fieldnames(results);
        
        % Count fields that might contain plot data (non-scalar numeric values)
        plotCount = 0;
        plotFields = {};
        
        for i = 1:length(fieldNames)
            field = fieldNames{i};
            value = results.(field);
            
            % Check if the field contains plottable data (non-scalar numeric)
            if isnumeric(value) && (numel(value) > 1 || isfield(results, [field '_x']))
                plotCount = plotCount + 1;
                plotFields{plotCount} = field;
            end
        end
        
        % Special case: if no plottable fields found, try to use metrics
        if plotCount == 0
            % Create bar chart from scalar values as fallback
            scalarFields = {};
            scalarValues = [];
            
            for i = 1:length(fieldNames)
                field = fieldNames{i};
                value = results.(field);
                
                if isnumeric(value) && isscalar(value)
                    scalarFields{end+1} = field;
                    scalarValues(end+1) = value;
                end
            end
            
            if ~isempty(scalarValues)
                bar(axesHandle, scalarValues);
                set(axesHandle, 'XTickLabel', scalarFields);
                title(axesHandle, 'Feature Analysis Metrics');
                return;
            end
        end
        
        % If we have plottable data, create subplots
        if plotCount > 0
            rows = ceil(sqrt(plotCount));
            cols = ceil(plotCount/rows);
            
            % Current axes is the parent for all subplots
            parentFigure = get(axesHandle, 'Parent');
            
            % Clear existing subplots if any
            axHandles = findall(parentFigure, 'Type', 'axes');
            for ax = axHandles(:)'
                if ax ~= axesHandle
                    delete(ax);
                end
            end
            
            % Hide the main axes
            set(axesHandle, 'Visible', 'off');
            
            for i = 1:plotCount
                field = plotFields{i};
                
                % Create subplot
                subAx = subplot(rows, cols, i, 'Parent', parentFigure);
                
                % Check if there's an accompanying X values field
                if isfield(results, [field '_x'])
                    xValues = results.([field '_x']);
                    yValues = results.(field);
                    plot(subAx, xValues, yValues);
                elseif isvector(results.(field))
                    % It's a vector, just plot it
                    plot(subAx, results.(field));
                elseif size(results.(field), 1) == 2 || size(results.(field), 2) == 2
                    % It might be x,y coordinates
                    if size(results.(field), 2) == 2
                        plot(subAx, results.(field)(:,1), results.(field)(:,2));
                    else
                        plot(subAx, results.(field)(1,:), results.(field)(2,:));
                    end
                else
                    % For other 2D data, use imagesc
                    imagesc(subAx, results.(field));
                    colorbar('peer', subAx);
                end
                
                % Add title (convert field name to title format)
                title(subAx, strrep(field, '_', ' '));
            end
        end
    end
    
    function str = formatStructToString(s)
        % Convert a struct to a formatted string for display
        % parameters:
        %   s: Structure to convert
        % returns:
        %   str: Formatted string
        
        str = '';
        fields = fieldnames(s);
        
        for i = 1:length(fields)
            field = fields{i};
            value = s.(field);
            
            % Format based on data type
            if isnumeric(value)
                if isscalar(value)
                    % For scalar, show directly
                    str = [str, field, ': ', num2str(value), newline];
                elseif numel(value) <= 5
                    % For small arrays, show all values
                    str = [str, field, ': [', num2str(value), ']', newline];
                else
                    % For larger arrays, just show size
                    sizes = size(value);
                    sizeStr = num2str(sizes(1));
                    for j = 2:length(sizes)
                        sizeStr = [sizeStr, 'x', num2str(sizes(j))];
                    end
                    str = [str, field, ': [', sizeStr, ' array]', newline];
                end
            elseif ischar(value)
                str = [str, field, ': ', value, newline];
            elseif islogical(value)
                if value
                    str = [str, field, ': true', newline];
                else
                    str = [str, field, ': false', newline];
                end
            elseif isstruct(value)
                % For nested structs, add indentation
                nestedStr = formatStructToString(value);
                lines = strsplit(nestedStr, newline);
                for j = 1:length(lines)
                    if ~isempty(lines{j})
                        str = [str, field, '.', lines{j}, newline];
                    end
                end
            end
        end
    end
end
