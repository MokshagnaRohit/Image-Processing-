function index = FindElement(array)

% Initialize the output array
index = [];

% The dimensions of the input array
[m, n] = size(array);

% Loop over the elements of the input array
for i = 1:n
    for j = 1:m
        % If the current element has a value of 1, add its index to the output array
        if array(j,i) == 1
            index = [index, (i-1)*m + j];
        end
    end
end
end
