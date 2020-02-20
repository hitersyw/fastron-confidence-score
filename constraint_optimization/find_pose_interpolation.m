function [x, fval, exitflag, output] = find_pose_interpolation(x0, lb, ub, F_reach, F_self_collision, F_env_collision)
    % rng(0);
    % gs = GlobalSearch;
    ms = MultiStart;
    opts = optimoptions(@fmincon,'Algorithm','sqp', 'OptimalityTolerance',1e-6);
    
    % alphas = [1, 1, 1];
    % Function for computing the optimal pose, with a start configuration;
    fprintf("Initial position: [%.2f, %.2f, %.2f]\n", x0);
    fprintf("Upper bound:[%.2f, %.2f, %.2f]; Lower bound:[%.2f, %.2f, %.2f]", ub, lb);

    fun = @(x) -(F_reach(x) + F_self_collision(x) + F_env_collision(x)); 
        
    % Add inequality constraints if needed;
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    
    problem = createOptimProblem('fmincon','x0', x0,...
        'objective',fun,'lb', lb, 'ub', ub,...
        'options',opts);
    % [x, fval, exitflag,output, ~] = run(gs,problem);
    [x, fval, exitflag,output, ~] = run(ms,problem, 50);