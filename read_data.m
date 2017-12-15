function [X,Xtest,r,c] = read_data(people,poses)
%Function to read in face data
%Can the select the number of people, and poses for the training and test
%set

%Initialize data matrices
X = [];%Traing set
Xtest = [];%Test set


%Load data for the training set
for i = 1:people
    %Populate the training set
    for j = 1:poses
        img = double(imread(strcat('./faces/s',int2str(i),'/',int2str(j),'.pgm')));
        X = [X img(:)];%Append unrolled image
    end
    %Begin populating the test set with remaining poses for the person
    for k = poses+1:10
        img = double(imread(strcat('./faces/s',int2str(i),'/',int2str(k),'.pgm')));
        Xtest = [Xtest img(:)];%Append unrolled image   
    end
end
[r,c] = size(img); % get number of rows and columns in image
end