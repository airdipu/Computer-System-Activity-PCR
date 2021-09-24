dataset = readtable('compactiv.dat');     % Read the .dat formate as a table
data = zscore(table2array(dataset));      % Changing data table to array Standaraization
plot(data);                               % Checking data center
TF = ismissing(data);                     % checking the missing data
tot = sum(TF);
tot;                                      % If missing the sum >0
x = data(:, 1:21);                        % Computer systems activity
y = data(:, 22);                          % Usr data
[n, p] = size(x);
[PCALoadings, PCAScores, EigenVals, PCAVar] = pca(x, 'Economy', false);


rsquaredPCR = 0;
i = 0;

while rsquaredPCR <0.70
    i = i+1;
    betaPCR = regress(y, PCAScores(:,1:i));
    % Transform Beta PCs into Beta Variables
    betaPCR = PCALoadings(:,1:i)*betaPCR;
    betaPCR = [mean(y) - mean(x)*betaPCR; betaPCR];

    % Making predictions
    yfitPCR = [ones(n, 1) x]*betaPCR;
    
    % Calculation
    TSS = sum(y.^2);
    RSS = sum((y-yfitPCR).^2);
    rsquaredPCR(i) = 1- (RSS/TSS);
    
end

PCRmsep = sum(crossval(@pcrsse, x, y, 'KFold', 10),1)/n;

figure;
title("Mean Squared Error");
plot(0:10, PCRmsep, 'b-o');
xlabel("Number of components");
ylabel("Estimated Mean Squared Prediction Error");

