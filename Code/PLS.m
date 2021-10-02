dataset = readtable('compactiv.dat');     % Read the .dat formate as a table
data = table2array(dataset);              % Changing data table to array

data = zscore(data);                      % Standardisation of data

% 0utliers identify and deleting
idx = find(data(:,22)<-4.0);
data(idx,:) = [];

X = data(:, 1:21);                        % Computer systems activity
y = data(:, 22);                          % Usr data


% PLS modeling
[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(X, y, 21, 'cv', 10);

plot(1:21, cumsum(100*PCTVAR(2,:)), '-bo');
xlabel('Number of PLS component');
ylabel('Percent Variance Explained in y');



% PLS modeling using 9 components
[XL1,yl1,XS1,YS1,beta1,PCTVAR1,MSE1,stats1] = plsregress(X, y, 9, 'cv', 10);

plot(1:9, cumsum(100*PCTVAR1(2,:)), '-bo');
xlabel('Number of PLS component');
ylabel('Percent Variance Explained in y');

% plotting MSE
figure;
plot(0:9, MSE1(2,:));

% fitting regression
yfit1 = [ones(size(X,1),1) X]*beta1;

TSS1 = sum((y-mean(y)).^2);
RSS1 = sum((y-yfit1).^2);
Rsquared1 = 1 - RSS1/TSS1;


% Comparison of regression made PCR and PLS

% Compute PCR
[PCALoadings, PCAScores, PCAVar] = pca(X, 'Economy', false);
betaPCR = regress(y, PCAScores(:, 1:9));

% Transform Beta PCs -> Beta Variables
betaPCR = PCALoadings(:,1:9)*betaPCR;
betaPCR = [mean(y) - mean(X)*betaPCR; betaPCR];

% Ploting the Explained Variance
figure;
a = plot(1:9, 100*cumsum(PCAVar(1:9))/sum(PCAVar(1:9)), 'b');
hold on
b = plot(1:9, 100*cumsum(PCTVAR1(1,:))/sum(PCTVAR1(1,:)), 'r');
c = plot(1:9, 100*cumsum(PCTVAR1(2,:))/sum(PCTVAR1(2,:)), 'k');
xlabel('Number of Components')
ylabel('Explained Variance')
legend([a, b, c], {'PCR: Explained Variance in X', 'PLS: Explained Variance in X', 'PLS: Explained Variance in y'});


% Calculation of MSE for PCR
[n p] = size(X);
PCRmsep = sum(crossval(@pcrsse, X, y, 'KFold', 9), 1)/n;
% Plotting MSE
figure;
title("Mean Squared Error");
hold on
a = plot(0:9, PCRmsep, 'b-o');
b = plot(0:9, MSE1(1,:), 'r-o');
c = plot(0:9, MSE1(2,:), 'k-o');
xlabel("Number of Components");
ylabel("Estimated Mean Squared Prediction Error");
legend([a, b, c], {'PCR: MSE in X', 'PLS: MSE in X', 'PLS: MSE in y'})












