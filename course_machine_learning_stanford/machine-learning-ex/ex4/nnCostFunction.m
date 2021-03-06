function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
 
j_mat = zeros(m, num_labels); % cost for each sample (row), each class of sample (col)  
% layer 2: hidden 
Z2 = zeros(m, hidden_layer_size);
A2 = zeros(m, hidden_layer_size);
% layer 3: output 
Z3 = zeros(m, num_labels); 
A3 = zeros(m, num_labels); % h = hypothesis

H = zeros(m, num_labels); % hypothesis
Y_logical = zeros(m, num_labels); % row: logical vector representing class 

D3 = zeros(m, num_labels); % row: output error vector for a sample 
D2 = zeros(m, hidden_layer_size + 1); % row: hidden error vector for a sample

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% layer 1: activation value 
% A1 = X;

% layer 2: activation value 
X = [ones(m,1), X]; % add bias col of 1s'
Z2 = X*Theta1';
A2 = sigmoid(Z2); % row: samplewise activation value 
% disp('A2'), disp(A2(1:5, 1:10)); 

% layer 2: activation value 
A2 = [ones(m,1), A2]; % add bias col of 1s'
Z3 = A2*Theta2';
A3 = sigmoid(Z3); % row: samplewise hypothesis 
% disp('A3'), disp(A3(1:5, 1:10));

H = A3; % for simplicity 
% % predict (same as predict.m) 
% prob_matrix = A3;   % row: prob for all classes, col: example
% [prob_class, class] = max(prob_matrix, [], 2);  % dim=2: max for each row 
% h = class;
% disp('h'), disp(h(1:5));

% one liner 
% [~,p] = max(sigmoid([ones(size(sigmoid(X*Theta1'), 1), 1) sigmoid(X*Theta1')]*Theta2'), [], 2);

% Y_logical 
% for i=1:m
%     Y_logical(i,y(i)) = 1;
% end

% one liner 
Y_temp = Y_logical';
Y_temp(y+[0:num_labels:(m-1)*num_labels]') = 1;
Y_logical = Y_temp';

% cost (same as costFunctionReg.m / lrCostFunction.m ) but now Y and H are
% matrices 
j_mat = -Y_logical .* log(H) - (1-Y_logical) .* log(1-H); % calc cost row: cost w.r.t each class for an example  
% disp('j_mat'), disp(j_mat(1:5,:)); 

% J = sum(j)/m + (lambda/(2*m)) * sum(theta(2:end).^2); % total cost 
J = sum(j_mat, 'all')/m ; % total cost
% disp('J'), disp(J);

% add cost for regularization terms 
J = J + (lambda/2/m) * (sum(Theta1(:,2:end).^2, 'all') + sum(Theta2(:,2:end).^2, 'all'));
% =========================================================================

% gradient 

% step 1 - done above 
% find Z2, A2, Z3, A3 

% step 2 - delta for output layer 
D3 = A3 - Y_logical; % row: error term for a sample 

% step 3 - delta for hidden layer
% size(D3)
% size(Theta2)
% size(sigmoidGradient(Z2))

D2 = (D3 * Theta2);
D2 = D2(:, 2:end); % skip or remove the bias error term 
D2 = D2 .* sigmoidGradient(Z2);

% Theta2_grad
Theta2_grad = zeros(size(Theta2));
% Theta1_grad 
Theta1_grad = zeros(size(Theta1));

for i=1:m
    Theta2_grad = Theta2_grad + D3(i,:)' * A2(i,:);
    Theta1_grad = Theta1_grad + D2(i,:)' * X(i,:);
end
Theta2_grad = Theta2_grad ./ m;
Theta1_grad = Theta1_grad ./ m;

% add regularization terms 
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) .* Theta2(:,2:end);
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) .* Theta1(:,2:end);

% grad = ( (h-y)' * X )./m + [0, (lambda/m) .* theta(2:end)'];
% grad = grad(:);
% -------------------------------------------------------------

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
