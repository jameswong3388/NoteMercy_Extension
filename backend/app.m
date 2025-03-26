function handwritingAnalysisApp
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

    % --- Label & Edit box: Handwriting Style ---
    styleText = uicontrol('Parent',leftPanel,...
                          'Style','text',...
                          'String','Handwriting Style:',...
                          'Units','normalized',...
                          'Position',[0.1 0.67 0.8 0.05],...
                          'FontWeight','bold',...
                          'HorizontalAlignment','left');
    styleBox = uicontrol('Parent',leftPanel,...
                         'Style','edit',...
                         'Max',2,'Min',0,...
                         'String','',...
                         'Units','normalized',...
                         'Position',[0.1 0.57 0.8 0.1],...
                         'HorizontalAlignment','left');

    % --- Label & Edit box: Features ---
    featureText = uicontrol('Parent',leftPanel,...
                            'Style','text',...
                            'String','Features:',...
                            'Units','normalized',...
                            'Position',[0.1 0.48 0.8 0.05],...
                            'FontWeight','bold',...
                            'HorizontalAlignment','left');
    featureBox = uicontrol('Parent',leftPanel,...
                           'Style','edit',...
                           'Max',2,'Min',0,...
                           'String','',...
                           'Units','normalized',...
                           'Position',[0.1 0.38 0.8 0.1],...
                           'HorizontalAlignment','left');

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
    tab3 = uitab('Parent',tabGroup,'Title','Features Extracted');
    ax3 = axes('Parent',tab3,...
               'Units','normalized',...
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
    handles.styleBox = styleBox;
    handles.featureBox = featureBox;

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

        % Example feature extraction:
        % Count white vs. black pixels in the binarized image
        bwImg = handles.imgPreprocessed;
        whitePixels = sum(bwImg(:));
        blackPixels = numel(bwImg) - whitePixels;

        % Display the feature info in the Features box
        featureStr = sprintf('White pixels: %d\nBlack pixels: %d', ...
                             whitePixels, blackPixels);
        set(handles.featureBox, 'String', featureStr);

        % Dummy classification for "Handwriting Style" (just as a demo)
        if whitePixels > blackPixels
            styleStr = 'Dominant White Region';
        else
            styleStr = 'Dominant Black Region';
        end
        set(handles.styleBox, 'String', styleStr);

        % Plot feature extraction in the "Features Extracted" tab
        axes(handles.ax3);
        bar([whitePixels, blackPixels]);
        set(gca, 'XTickLabel', {'White','Black'});
        title('Feature Extraction');
        
        % Plot analysis in the "Analysis" tab
        axes(handles.ax4);
        pie([whitePixels, blackPixels]);
        legend({'White Pixels', 'Black Pixels'});
        title('Handwriting Style Analysis');

        guidata(hObject, handles);
    end
end
