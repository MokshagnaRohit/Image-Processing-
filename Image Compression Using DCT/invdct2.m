
function X = invdct2(Y)
[M,N] =size(Y);
X = zeros(M,N);
 for m =0:M-1
     for n =0:N-1
         sum = 0;
            for p = 0:M-1
              for q =0:N-1
                 x= 1;
                 y =1;
                    if p == 0
                     x = sqrt(1/M);
                    else
                        x =sqrt(2/M);
                    end
                    if q == 0
                      y =sqrt(1/N);
                    else
                        y= sqrt(2/N);
                    end
                    sum = sum +x*y*Y(p+1,q+1)*cos(pi*(2*m+1)*p/(2*M))*cos(pi*(2*n+1)*q/(2*N));
                end
            end
            X(m+1,n+1) =sum;
      end
    end
end
