function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
h = zeros(m,1); % hypothesis 
j = zeros(m,1); % cost for each sample 
n = length(theta); % number of parameters 

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% cost
h = sigmoid(X*theta); % calc hypothesis vector 
j = -y .* log(h) - (1-y) .* log(1-h); % calc cost vector with cost for each sample 
J = sum(j)/m; % total cost 

% gradient 
grad = ( (h-y)' * X )./m;
% =============================================================

end
