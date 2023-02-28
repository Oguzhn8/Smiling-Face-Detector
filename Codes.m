% FURKAN KARLITAĞ / 040170210
% OGUZHAN KARAARSLAN / 040170081
%
%%
clc ;
clear ;
close all ;
%%load data
data_set = load ('Yale_32x32.mat') ;
numFaces = length(data_set.gnd);

% % Labeling the faces 
neutral = []; 
smile = [];

for i=3:11:numFaces+2
    smile(end + 1) = i;     %%adding smile face
end
for i=1:numFaces+2
    if i == 22 || i == 66 || i == 74
        smile(end + 1) = i;  %%adding smile faces
    end
end
        
        
for i=1:1:numFaces
    if rem(i,11) ~= 3
        if i == 22 || i == 66 || i == 74    %%smile faces
            continue
        else
            neutral(end+1) = i              %%adding neutral
        end
     end
end           


%%
% 'Mean Face'
h = 32; w = 32;
faces = (data_set.fea)';
meanFace = mean(faces, 2);
faces = faces - repmat(meanFace, 1, 165);
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
for i = 1:numFaces
    energy(i) = sum(eigVals(1:i));
end
propEnergy = energy./energy(end);
%%
% Determine the number of principal components required to model 90% of data variance
percentMark = min(find(propEnergy > 0.9));
%%
% Pick those principal components
eigenVecs = u(:, 1:percentMark);
%%
% Do something with them; for example, project each of the neutral and smiling faces onto the corresponding eigenfaces
neutralFaces = faces(:, neutral); 
smileFaces = faces(:, smile);
neutralWeights = eigenVecs' * neutralFaces;
smileWeights = eigenVecs' * smileFaces;
%%
% Use the coefficients of these projections to classify each smiling face
for i = 1:length(smile)
test_smile=smileWeights(:,i);
test_repeat_smile=repmat(test_smile,1,(length(smile)-1));
smile_weights_no_test=[smileWeights(:,1:i-1) smileWeights(:,i+1:end)];
distance_smile=test_repeat_smile-smile_weights_no_test;
distance_smile_val(i)=sum(vecnorm(distance_smile))/(length(smile)-1);
end

for i = 1:length(smile)
test_smile=smileWeights(:,i);
test_repeat_neutral=repmat(test_smile,1,(length(neutral)));
distance_neutral=test_repeat_neutral-neutralWeights;
distance_neutral_val(i)=sum(vecnorm(distance_neutral))/(length(neutral));
end

for i = 1:length(smile)
decision(i)=distance_neutral_val(i)>= distance_smile_val(i);
end
sum(decision)/length(smile)
