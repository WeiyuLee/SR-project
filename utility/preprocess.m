close all; clear all;

%% Choose the target dataset
preprocess_target = 'train';
%preprocess_target = 'test';

%% Set the data path
Train_data_dir = './Train/';
Test_data_dir = './Test/Set5/';
%Test_data_dir = './Test/Set14/';

%% DON'T MODIFY ANYTHING BELOWS

% Get the file list in the target folder
Train_data_list = dir(Train_data_dir);
Test_data_list = dir(Test_data_dir);

% Assign the dir & list
if (strcmp(preprocess_target, 'train'))
    data_dir = Train_data_dir;
    data_list = Train_data_list;
elseif (strcmp(preprocess_target, 'test'))
    data_dir = Test_data_dir;    
    data_list = Test_data_list;
end

% Set parameters
up_scale = 3;
file_num = length(data_list);

% Make dir to save the image
save_dir = [data_dir 'preprocessed_scale_' num2str(up_scale) '/'];
if (exist(save_dir, 'dir'))
    fprintf('ERROR: The preprocessed directory is already exist!\n');
    return;
else
    mkdir(save_dir);
end

for i = 3:file_num
    im_gnd = imread([data_dir data_list(i).name]);
    
    % Only process Y channel
    if size(im_gnd, 3) > 1
        im_gnd = rgb2ycbcr(im_gnd);
        im_gnd = im_gnd(:, :, 1);
    end
    
    im_gnd = modcrop(im_gnd, up_scale);
    
    % Normalization
    im_gnd = single(im_gnd) / 255;
    
    % Bicubic interpolation
    im_bicubic = imresize(im_gnd, 1/up_scale, 'bicubic');
    im_bicubic = imresize(im_bicubic, up_scale, 'bicubic');
    
    % Save images
    [pathstr,name,ext] = fileparts([data_dir data_list(i).name]);
    save_path = [save_dir name];
    imwrite(uint8(im_bicubic*255), [save_path '_bicubic_scale_' num2str(up_scale) '.bmp']);
    imwrite(uint8(im_gnd*255), [save_path '_org.bmp']);
end
