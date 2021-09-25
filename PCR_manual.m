dataset = readtable('compactiv.dat');     % Read the .dat formate as a table
data = table2array(dataset);              % Changing data table to array
TF = ismissing(data);                     % checking the missing data
tot = sum(TF);                            % If missing the sum >0
tot; 
X = data(:, 1:21);                        % Original Computer systems activity
Y = data(:, 22);                          % Original Usr data

data = zscore(data);                      % Standardisation of data

% Outliers identify and deleting
idx = find(data(:,22)<-4.0);
data(idx,:) = [];

x = data(:, 1:21);                        % Computer systems activity
y = data(:, 22);                          % Usr data

plot(x);                                  % Checking data center
[n1, p1] = size(x); 

% apply PCA
[PCALoadings, PCAScores, EigenVals, PCAVar, Explained] = pca(xTrain, 'Economy', false);

figure;
plot(1:10, 100*cumsum(PCAVar(1:10))/sum(PCAVar(1:10)));
xlabel('Number of Principal Component')
ylabel('Explained Variance in x')

% Get regression factors for each Principal Component
betaPCR = regress(yTrain, PCAScores(:,1:9));

% Transform B's from PCs to Beta coefficient for actual variable
betaPCR = PCALoadings(:,1:9)*betaPCR;
betaPCR = [mean(yTest) - mean(xTest)*betaPCR; betaPCR];

yfitPCR = [ones(n, 1) xTest]*betaPCR;

figure;
plot(yTest, yfitPCR, 'bo');
xlabel('Observed Response');
ylabel('Fitted Response');

% Calculation
TSS = sum(yTest.^2);
RSS = sum((yTest - yfitPCR).^2);
rsquaredPCR = 1- (RSS/TSS);


