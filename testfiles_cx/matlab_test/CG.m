function x = CG(A, b, x)

r = b - A*x;
p = r;
rTr = r'*r;

for i = 1:1
    pAp = p'*A*p;
    A*p
    alpha = rTr / pAp;
    x = x + alpha * p;
    r = r - A * p * alpha;
    
    rTr_new = r'*r
    if rTr_new < 1e-8
        break
    end
    
    beta = rTr_new / rTr;
    rTr = rTr_new;
    
    p = r + beta * p;
end



end