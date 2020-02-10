function [x,fval,exitflag, output] = find_optimal_pose_global_search(x0, lb, ub, self_collision_mdl, reachability_mdl, ...
            env_collision_mdl, self_collision_scale_output, reachability_scale_output)

    % rng(0);
    % gs = GlobalSearch;
    ms = MultiStart;
    opts = optimoptions(@fmincon,'Algorithm','sqp', 'OptimalityTolerance',1e-6);
    
    % alphas = [1, 1, 1];
    % Function for computing the optimal pose, with a start configuration;
    fprintf("Initial position: [%.2f, %.2f, %.2f]\n", x0);
    fprintf("Upper bound:[%.2f, %.2f, %.2f]; Lower bound:[%.2f, %.2f, %.2f]", ub, lb);

    fun = @(x) -(self_collision_scale_output(predict(self_collision_mdl, x))...
          + reachability_scale_output(predict(reachability_mdl, x))...
          + predict(env_collision_mdl, x)...
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
    [x, fval, exitflag,output, ~] = run(ms,problem, 50);