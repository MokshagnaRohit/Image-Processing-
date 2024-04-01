function out = convolution2D(A, B)

% Getting the dimensions of the input matrices
[m, n] = size(A);
[p, q] = size(B);

% The dimensions of the output matrix
out_rows = m + p - 1;
out_cols = n + q - 1;

% Initialize the output matrix
out = zeros(out_rows, out_cols);

% Loop over the rows and columns of the output matrix
for i = 1:out_rows
    for j = 1:out_cols
        
        % Loop over the rows and columns of B (kernel)
        for k = 1:p
            for l = 1:q     
                % Check if the kernel extends beyond the bounds of the input matrix
                if (i-k+1 > 0 && i-k+1 <= m) && (j-l+1 > 0 && j-l+1 <= n)
                    out(i,j) = out(i,j) + A(i-k+1, j-l+1)*B(k,l);
                end
            end
        end
    end
end
