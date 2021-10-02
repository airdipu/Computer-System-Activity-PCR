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

% plotting MSE
figure;
plot(0:21, MSE(2,:));






% PLS modeling
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
Rsquared1 = 1 - RSS/TSS;













