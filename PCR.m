dataset = readtable('compactiv.dat');       % Read the .dat formate as a table
data = table2array(dataset);                % Changing data table to array
TF = ismissing(data);                       % checking the missing data
tot = sum(TF);
tot;                                        % If missing the sum >0
x = data(:, 1:21);                          % Computer systems activity
y = data(:, 22);                            % Usr data
yc = y - mean(y);
ndata = zscore(x);                          % Standaraization 
plot(ndata);                                % Checking data center
[PCALoadings, PCAScores, EigenVals, PCAVar] = pca(ndata, 'Economy', false);
%idx = find(cumsum(Explained)>95);

figure;
plot(1:200, 100*cumsum(PCAVar(1:200))/sum(PCAVar(1:200)));
xlabel('Number of Principal Component')
ylabel('Explained Variance in x')

betaPCR = regress(yc, PCAScores(:,13:21));
betaPCR = PCALoadings(:,13:21)*betaPCR;
betaPCR = [mean(y) - mean(x)*betaPCR; betaPCR];

yfitPCR = [ones(n, 1) x]*betaPCR;

figure;
plot(yc, yfitPCR, 'bo');
xlabel('Observed Response');
ylabel('Fitted Response');
