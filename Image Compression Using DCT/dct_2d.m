function y = dct_2d(x)
    [m,n] = size(x);
    x = double(x);
    y = zeros(m,n);
    for u = 0:m-1
        for v = 0:n-1
            if u == 0
                cu = sqrt(1/m);
            else
                cu = sqrt(2/m);
            end
            if v == 0
                cv = sqrt(1/n);
            else
                cv = sqrt(2/n);
            end
            sum = 0;
            for i = 0:m-1
                for j = 0:n-1
                    sum = sum + x(i+1,j+1)*cos(pi*(2*i+1)*u/(2*m))*cos(pi*(2*j+1)*v/(2*n));
                end
            end
            y(u+1,v+1) = cu*cv*sum;
        end
    end
end
