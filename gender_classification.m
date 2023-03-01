%
%%
clc;
clear;
close all;
%%
%train data 
train_men = dir(fullfile(['training\men\','*.jpg']));
train_images_men=[];
for i = 1:length(train_men)
    file_name = strcat('training\men\',train_men(i).name);
    image = imread(file_name);
    re=reshape(image,[1 1296]);
    train_images_men=[train_images_men;re];  
end

train_women = dir(fullfile(['training\women\','*.jpg']));
train_images_women=[];
for i = 1:length(train_women)
    file_name = strcat('training\men\',train_women(i).name);
    image = imread(file_name);
    re=reshape(image,[1 1296]);
    train_images_women=[train_images_women;re];  
end

train_images=[train_images_men;train_images_women];
train_images=double(train_images);
%%
%test data
test_men = dir(fullfile(['testing\men\','*.jpg']));
test_images_men=[];
for i = 1:length(test_men)
    file_name = strcat('testing\men\',test_men(i).name);
    image = imread(file_name);
    re=reshape(image,[1 1296]);
    test_images_men=[test_images_men;re];  
end

test_women = dir(fullfile(['testing\women\','*.jpg']));
test_images_women=[];
for i = 1:length(test_women)
    file_name = strcat('testing\men\',test_women(i).name);
    image = imread(file_name);
    re=reshape(image,[1 1296]);
    test_images_women=[test_images_women;re];  
end

test_images=[test_images_men;test_images_women];
test_images=double(test_images);
%%
% Subtract the mean 'face' before performing PCA
h = 36; w = 36;
faces = (train_images)';
numFaces = size(faces,2);
meanFace = mean(faces, 2);
faces = faces - repmat(meanFace, 1, numFaces);
%%
% Perform Singular Value Decomposition
[u,d,v] = svd(faces, 0);
%%
% Pull out eigen values and vectors
eigVals = diag(d);
eigVecs = u;
%%
% Plot the mean sample and the first three principal components
figure;
imagesc((reshape(meanFace, h, w))); colormap(gray); title('Mean Face');

figure;
subplot(1, 3, 1); imagesc(reshape(u(:, 1), h, w)); colormap(gray); title('First Eigenface');
subplot(1, 3, 2); imagesc(reshape(u(:, 2), h, w)); colormap(gray); title('Second Eigenface');
subplot(1, 3, 3); imagesc(reshape(u(:, 3), h, w)); colormap(gray); title('Third Eigenface');
%%
% The cumulative energy content for the m'th eigenvector is the sum of the energy contentacross eigenvalues 1:m
for i = 1:size(faces,1)
    energy(i) = sum(eigVals(1:i));
end
propEnergy = energy./energy(end);
%%
% Determine the number of principal components required to model 90% of data variance
percentMark = min(find(propEnergy > 0.9));

% Pick those principal components
eigenVecs = u(:, 1:percentMark);
%%
% Do something with them; for example, project each of the man and woman faces onto the corresponding eigenfaces
menFaces = faces(:, [1:2500]); womenFaces = faces(:, [2501:5000]);
menWeights = eigenVecs' * menFaces;
womenWeights = eigenVecs' * womenFaces;

test_faces = (test_images)'
test_menFaces = test_faces(:, [1:200]); test_womenFaces = test_faces(:, [201:400]);
%%
for i = 1:length(menFaces)
    test_men=menWeights(:,i);
    test_repeat_men=repmat(test_men,1,(length(menFaces)-1));
    men_weights_no_test=[menWeights(:,1:i-1) menWeights(:,i+1:end)];
    distance_men = test_repeat_men - men_weights_no_test;
    distance_men_val(i)=sum(vecnorm(distance_men))/(length(menFaces)-1);
end

for i = 1:length(menFaces)
    test_men=menWeights(:,i);
    test_repeat_women=repmat(test_men,1,(length(womenFaces)));
    distance_women=test_repeat_women-womenWeights;
    distance_women_val(i)=sum(vecnorm(distance_women))/(length(womenFaces));
end

for i = 1:length(menFaces)
    decision(i)=distance_men_val(i)>= distance_men_val(i);
end
sum(decision)/length(menFaces)
