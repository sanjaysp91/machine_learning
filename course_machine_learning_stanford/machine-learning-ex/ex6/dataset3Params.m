function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% For both  and  we suggest trying values in multiplicative steps 
% (e.g., 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)
C_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
Err_CV = zeros(length(C_array), length(sigma_array));     % cross-validation error matrix 
combination = 0; 
% Try all combinations of C and sigma (e.g. 8 x 8 = 64) 
for i = 1:length(C_array)   % run thru for various C
    for j = 1:length(sigma_array)   % run thru for various sigma 
        combination = combination + 1;
        % SVM Parameters
        C = C_array(i); sigma = sigma_array(j);
%         fprintf("\nCombination = %d, \t C = %f , \t sigma = %f ", combination, C, sigma);
        % Train model using training set
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        
        % Predict for validation set
        predictions = svmPredict(model, Xval);
                
        % Compute cross-validation classification error 
        Err_CV(i,j) = mean(double(predictions ~= yval));
        Err_CV(i,j) = Err_CV(i,j) / length(yval);
        fprintf("\nCombination = %d, \t C = %f, \t sigma = %f, \t Cross-validation error = %f ", ... 
                   combination, C, sigma, Err_CV(i,j));
    end
end

% Compare cross-validation error to choose best C and Sigma 
% Choose C and sigma that correspond to minimum cross-validation error 
[Err_min_row, i] = min(Err_CV,[],1);     % gives index array for C 
[Err_min, j] = min(Err_min_row,[],2);     % gives index for sigma 
C = C_array(i(j));  % note this step, find C index using j  
sigma = sigma_array(j); 
fprintf("\nOptimized or learned C_index = %d , \t sigma_index = %d ", i(j), j);
fprintf("\nOptimized or learned C = %f , \t sigma = %f \t min Err_CV = %f ", C, sigma, Err_min);

% =========================================================================

end
