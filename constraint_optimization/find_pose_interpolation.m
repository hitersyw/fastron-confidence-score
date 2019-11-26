function [x] = find_pose_interpolation(x0, lb, ub, F)
    fprintf("Initial position: [%.2f, %.2f, %.2f]\n", x0);
    fprintf("Upper bound:[%.2f, %.2f, %.2f]; Lower bound:[%.2f, %.2f, %.2f]", ub, lb);
    fun = @(x) -F(x); 
    % Add inequality constraints if needed;
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    
    options = optimoptions('fmincon',...
    'Algorithm','sqp','OptimalityTolerance',0.000001);
    x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,[],options);