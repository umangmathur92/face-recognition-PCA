% Get training data
imgFiles = dir('D:/PAMI/Matlab/FaceRecognition_Data/ALL/*.TIF');
numFiles = size(imgFiles);
for i = 1:numFiles
    img = imread(strcat('D:/PAMI/Matlab/FaceRecognition_Data/ALL/', imgFiles(i).name));
    X(:, i) = img(:);
end

% Calculating average matrix
avg = mean(X,2);
h1 = figure;
colormap('gray');
subplot(1, 1, 1);
imagesc(reshape(avg, 32, 32))
title('Average Face from training data')

% Removing average from training matrix X
X = double(X);
M = [];
for i = 1:numFiles
    M(:, i) = X(:, i) - avg;
end

% Covariance
covariantMatrix = M*M';
disp(covariantMatrix)

% Eigen values
[V, D] = eig(covariantMatrix);
evalues = diag(D);
[~,indices] = sort(evalues,'descend');

% Selecting top K eigenvalues
K = 16;
W = zeros(32*32, K);
for i=1:K
    W(:, i) = V(:, indices(i));
end

% Testing the training set
K = 16;
testImgFaces = dir('D:/PAMI/Matlab/FaceRecognition_Data/FA/*.TIF');
numTestImages = length(testImgFaces);
DB = zeros(K, numTestImages);
for i=1:numTestImages
    img = imread(strcat('D:/PAMI/Matlab/FaceRecognition_Data/FA/', testImgFaces(i).name));
    DB(:, i) = W' * (reshape(double(img), [], 1) - avg);
end

% Take input image from user
imgName = input('Enter filename of your test image inside FB folder: ','s');
imgPath = strcat('D:/PAMI/Matlab/FaceRecognition_Data/FB/', imgName);
testImg = reshape(double(imread(imgPath)), [], 1);
y = W' * (testImg - avg);
distFromTest = zeros(1, numTestImages);
for i=1:numTestImages
    distFromTest(i) = norm(y - DB(:, i));
end
[sortedDistValues, sortedDistIndices] = sort(distFromTest, 'ascend');

% Display Results
h2 = figure;
subplot(2, 2, 1);
imshow(imgPath)
title('Input image')
subplot(2, 2, 2);
imshow(strcat('D:/PAMI/Matlab/FaceRecognition_Data/FA/', testImgFaces(sortedDistIndices(1)).name))
title(strcat('Best Match, Distance from input Image : ', int2str(sortedDistValues(1))))
subplot(2, 2, 3);
imshow(strcat('D:/PAMI/Matlab/FaceRecognition_Data/FA/', testImgFaces(sortedDistIndices(2)).name))
title(strcat('2nd Best Match, Distance from input Image : ', int2str(sortedDistValues(2))))
subplot(2, 2, 4);
imshow(strcat('D:/PAMI/Matlab/FaceRecognition_Data/FA/', testImgFaces(sortedDistIndices(3)).name))
title(strcat('3rd Best Match, Distance from input Image : ', int2str(sortedDistValues(3))))