function xx = degexpand(x, deg, addOnes)
% Expand input vectors to contain powers of the input features

% This file is from pmtk3.googlecode.com


[n,m] = size(x);
if nargin < 3, addOnes = 0; end

xx = repmat(x, [1 1 deg]);
degs = repmat(reshape(1:deg, [1 1 deg]), [n m]);
xx = xx .^ degs;
xx = reshape(xx, [n, m*deg]);

if addOnes
  xx = [ones(n,1) xx];
end