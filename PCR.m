dataset = readtable('compactiv.dat');     % Read the .dat formate as a table
data = zscore(table2array(dataset));      % Changing data table to array Standaraization
plot(data);                               % Checking data center
TF = ismissing(data);                     % checking the missing data
tot = sum(TF);
tot;                                      % If missing the sum >0
x = data(:, 1:21);                        % Computer systems activity
y = data(:, 22);                          % Usr data
[n, p] = size(x);
yc = y - mean(y);
[PCALoadings, PCAScores, EigenVals, PCAVar] = pca(x, 'Economy', false);

figure;
plot(1:10, 100*cumsum(PCAVar(1:10))/sum(PCAVar(1:10)));
xlabel('Number of Principal Component')
ylabel('Explained Variance in x')



rsquaredPCR = 0;
i = 0;

while rsquaredPCR <0.72
    i = i+1;
    betaPCR = regress(y, PCAScores(:,1:i));
    % Transform Beta PCs into Beta Variables
    betaPCR = PCALoadings(:,1:i)*betaPCR;
    betaPCR = [mean(y) - mean(x)*betaPCR; betaPCR];

    % Making predictions
    yfitPCR = [ones(n, 1) x]*betaPCR;
    
    % Calculation
    TSS = sum(yc.^2);
    RSS = sum((y-yfitPCR).^2);
    rsquaredPCR(i) = 1- (RSS/TSS);
    
end

    
