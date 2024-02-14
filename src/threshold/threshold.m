function threshold(videoPath, startPoint, endPoint)
    
    % change with the network name
    networkName = "DeepFTSG_1";

    maskFolder = strcat('../output/CD2014/' + networkName + '/');
    orgImgFolder='../datasets/test/CD2014/DATASET/';
    thresholdFolder = strcat('../output_th/CD2014/' + networkName + '/');
    
    maskPath = strcat(maskFolder, videoPath);
    orgImgPath = strcat(orgImgFolder, videoPath, "input/");

    thresholdPath = strcat(thresholdFolder, videoPath);
    
    % create threshold folder
    if (0==isdir(thresholdPath))
        mkdir(thresholdPath);
    end

    videoPath
    
    for i = startPoint : endPoint
         % filename formatting
         fileName = num2str(i, '%.6d');
         
         % read image
         mask = imread(fullfile(maskPath, ['bin', fileName, '.png']));
         
         % read orginal frame
         orgImg = imread(fullfile(orgImgPath, ['in', fileName, '.jpg']));
         
         % get size of original image
         iSize = size(orgImg);
     
         % resize mask to original image size
         maskResize = imresize(mask, [iSize(1), iSize(2)], 'bilinear');
         
         % threshold to binary
         maskBinary = im2bw(maskResize, 0.4);

         fullfile(thresholdPath, ['bin', fileName, '.png'])
         imwrite(maskBinary, fullfile(thresholdPath, ['bin', fileName, '.png']));
    end
end
    
