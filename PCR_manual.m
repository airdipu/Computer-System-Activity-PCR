dataset = readtable('compactiv.dat');     % Read the .dat formate as a table
data = table2array(dataset);              % Changing data table to array

X = data(:, 1:21);                        % Computer systems activity
Y = data(:, 22);                          % Usr data
TF = ismissing(data);                     % checking the missing data
tot = sum(TF);
tot;                                      % If missing the sum >0
x = zscore(X);                            % Standarised system activity
y = zscore(Y);                            % Standarised Usr data
plot(X);                                  % Checking data center
[n, p] = size(x); 

% apply PCA
[PCALoadings, PCAScores, EigenVals, PCAVar] = pca(x, 'Economy', false);

figure;
plot(1:20, 100*cumsum(PCAVar(1:20))/sum(PCAVar(1:20)));
xlabel('Number of Principal Component')
ylabel('Explained Variance in x')

% Get regression factors for each Principal Component
betaPCR = regress(y, PCAScores(:,1:9));

% Transform B's from PCs to Beta coefficient for actual variable
betaPCR = PCALoadings(:,1:9)*betaPCR;
betaPCR = [mean(y) - mean(x)*betaPCR; betaPCR];

yfitPCR = [ones(n, 1) x]*betaPCR;

figure;
plot(y, yfitPCR, 'bo');
xlabel('Observed Response');
ylabel('Fitted Response');




% Calculation
TSS = sum(y.^2);
RSS = sum((y-yfitPCR).^2);
rsquaredPCR = 1- (RSS/TSS);


