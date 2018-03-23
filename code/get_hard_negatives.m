function features_hard_neg = get_hard_negatives(hard_negative_path, feature_params, w, b)

image_files = dir( fullfile( hard_negative_path, '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files);

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);

cells_a_template = feature_params.template_size / feature_params.hog_cell_size;
features_hard_neg = zeros(num_images, cells_a_template^2 * 31);

scale_multiplier = 1;
min_confidence_threshold = 0.3;
window_size = feature_params.template_size/feature_params.hog_cell_size;

for i = 1:num_images
    image_file = image_files(i);
    path = strcat(hard_negative_path, '/', image_file.name);
    img = imread(path);
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    img = single(img)/255;
    img_min_dimension = min(size(img, 1), size(img, 2));
    
    %Run the sliding window detector
    cur_bboxes = zeros(0, 4);
    cur_confidences = zeros(0, 1); 
    scale = 1;
    not_found = true;
    
    if (scale_multiplier > 0 && scale_multiplier < 1)
        while (floor(scale * img_min_dimension/feature_params.hog_cell_size) > window_size) && not_found
            scaled_image = imresize(img, scale);
            scaled_hog = vl_hog(scaled_image, feature_params.hog_cell_size);
            width = size(scaled_hog, 2);
            height = size(scaled_hog, 1);

            for x = 1:1:width - window_size + 1
                for y = 1:1:height - window_size + 1
                    feature = scaled_hog(y:y+ window_size - 1, x:x + window_size - 1, :);
                    score = reshape(feature, 1, [])*w + b;

                    if score > min_confidence_threshold
                        res_feature = reshape(feature, 1, []);
                        not_found = false;
                        features_hard_neg(i,:) = res_feature(1,:);
                        bbox = [x, y, x + window_size, y + window_size] * feature_params.hog_cell_size / scale;
                        cur_bboxes = [cur_bboxes; bbox];
                        cur_confidences = [cur_confidences; score];
                    end
                end
            end
            scale = scale * scale_multiplier;  
        end

        %non_max_supr_bbox can actually get somewhat slow with thousands of
        %initial detections. You could pre-filter the detections by confidence,
        %e.g. a detection with confidence -1.1 will probably never be
        %meaningful. You probably _don't_ want to threshold at 0.0, though. You
        %can get higher recall with a lower threshold. You don't need to modify
        %anything in non_max_supr_bbox, but you can.
        [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

        cur_confidences = cur_confidences(is_maximum,:);
        cur_bboxes      = cur_bboxes(     is_maximum,:);

        bboxes      = [bboxes;      cur_bboxes];
        confidences = [confidences; cur_confidences];
    else
        hog = vl_hog(img, feature_params.hog_cell_size);
        width = size(hog, 2);
        height = size(hog, 1);
        for x = 1:1:width - window_size + 1
            for y = 1:1:height - window_size + 1
                feature = hog(y:y+ window_size - 1, x:x + window_size - 1, :);
                score = reshape(feature, 1, [])*w + b;
                if (score > min_confidence_threshold) && not_found
                    res_feature = reshape(feature, 1, []);
                    not_found = false;
                    features_hard_neg(i,:) = res_feature(1,:);
                end
            end
        end
    end
end
end