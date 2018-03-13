% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);

cells_a_template = feature_params.template_size / feature_params.hog_cell_size;
features_neg = zeros(num_samples, cells_a_template^2 * 31);

samples_per_image = ceil(num_samples / num_images);
generated = 1;

while generated < num_samples   
    %Pick a random image
    image_index =  floor(rand(1) * num_images + 1);
    %make sure we get at least one sample from each image
    if generated < num_images
        image_index = generated;
    end
    
    image_file = image_files(image_index);
    
    path = strcat(non_face_scn_path, '/', image_file.name);
    image = rgb2gray(imread(path));
    
    image_size = size(image);
    height = image_size(1);
    width = image_size(2);
    
    min_dimension = min(width, height);
    
    if (min_dimension < feature_params.template_size)
        continue
    end
    
    hog = vl_hog(single(image)/255.0, feature_params.hog_cell_size);
    
    hog_size = size(hog);
    hog_height = hog_size(1);
    hog_width = hog_size(2);
    
    %Get some random x/y points for the hog
    rand_hog_permutation_y = randperm(hog_height-cells_a_template+1);
    rand_hog_permutation_x = randperm(hog_width-cells_a_template+1);
    min_dimension = min(size(rand_hog_permutation_x, 2), size(rand_hog_permutation_y, 2));
    
    for j = 1:samples_per_image
        if (j > min_dimension || generated >= num_samples)
            break
        end
        
        y = rand_hog_permutation_y(j);
        x = rand_hog_permutation_x(j);
        
        feature = hog(y:y + cells_a_template - 1, x:x + cells_a_template - 1,:);
        feature = reshape(feature, 1, []);
        
        features_neg(generated, :) = feature(1, :);
        %keep track of how many images we've generated
        generated = generated + 1;        
    end
end

% placeholder to be deleted
% features_neg = rand(100, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);

end