result = importdata('compactiv.dat');       % Read the .dat formate as a table
A = result.data;
TF = ismissing(A);                          % checking the missing data
tot = sum(TF);
tot;                                        % If missing the sum >0

ndata = zscore(A); % Standaraization 
plot(ndata); % checking data center so the mean of the data stands in the origin
[Loadings, Scores, EigenVals, T2, Explained, mu] = pca(ndata); % Simple PCA


