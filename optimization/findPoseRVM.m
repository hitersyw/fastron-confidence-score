function [x,fval,exitflag, output] = findPoseRVM(x0, X, g, lb, ub, self_collision_param, reachability_param, ...
            env_collision_param, scale_input, self_collision_scale_output, reachability_scale_output)
    
    % alphas = [1, 1, 1];
    % Function for computing the optimal pose, with a start configuration
    
    ms = MultiStart;
    opts = optimoptions(@fmincon,'Algorithm','sqp', 'OptimalityTolerance',1e-6);
    
    fprintf("Initial position: [%.2f, %.2f, %.2f]\n", x0);
    fprintf("Upper bound:[%.2f, %.2f, %.2f]; Lower bound:[%.2f, %.2f, %.2f]", ub, lb);
    
    K = exp(-g*pdist2(scale_input(x0),X).^2);
    fun = @(x) -(rvmPredict(K, self_collision_param)...
          + rvmPredict(K, reachability_param)...
          + rvmPredict(K, env_collision_param)...
          );
        
    % Add inequality constraints if needed;
    A = [];
    b = [];
    Aeq = [];
    beq = [];
  
    problem = createOptimProblem('fmincon','x0', x0,...
        'objective',fun,'lb', lb, 'ub', ub,...
        'options',opts);
    % [x, fval, exitflag,output, ~] = run(gs,problem);
    [x, fval, exitflag,output, ~] = run(ms,problem,50);
    
    