dataset = readtable('compactiv.dat');     % Read the .dat formate as a table
data = zscore(table2array(dataset));      % Changing data table to array & standarization
plot(data);                               % Checking data center
TF = ismissing(data);                     % checking the missing data
tot = sum(TF);
tot;                                      % If missing the sum >0
x = data(:, 1:21);                        % Computer systems activity
y = data(:, 22);                          % Usr data
[n, p] = size(x);
yc = y - mean(y); 

% apply PCA
[PCALoadings, PCAScores, EigenVals, PCAVar] = pca(x, 'Economy', false);

figure;
plot(1:10, 100*cumsum(PCAVar(1:10))/sum(PCAVar(1:10)));
xlabel('Number of Principal Component')
ylabel('Explained Variance in x')

% Fitting regresson
betaPCR = regress(yc, PCAScores(:,1:9));
betaPCR = PCALoadings(:,1:9)*betaPCR;
betaPCR = [mean(y) - mean(x)*betaPCR; betaPCR];

yfitPCR = [ones(n, 1) x]*betaPCR;

figure;
plot(yc, yfitPCR, 'bo');
xlabel('Observed Response');
ylabel('Fitted Response');

