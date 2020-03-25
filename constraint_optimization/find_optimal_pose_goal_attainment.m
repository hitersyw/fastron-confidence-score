function [x, F] = find_optimal_pose_goal_attainment(x0, n_init, lb, ub, ...
            goal, self_collision_mdl, reachability_mdl, ...
            env_collision_mdl)

    % rng(0);
    % gs = GlobalSearch;
    weight = [1, 1, 1];
    opts = optimoptions(@fmincon,'Algorithm','sqp', 'OptimalityTolerance',1e-6);
    
    % alphas = [1, 1, 1];
    % Function for computing the optimal pose, with a start configuration;
    fprintf("Initial position: [%.2f, %.2f, %.2f]\n", x0);
    fprintf("Upper bound:[%.2f, %.2f, %.2f]; Lower bound:[%.2f, %.2f, %.2f]", ub, lb);

    fun = @(x) [clip(predict(self_collision_mdl, x), 0.00001), ...
                clip(predict(reachability_mdl, x), 0.00001), ...
                clip(predict(env_collision_mdl, x), 0.00001)...
          ];
        
    % Add inequality constraints if needed;
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    nonlcon = [];
    
    x = fgoalattain(fun,x0,goal,weight,A,b,Aeq,beq,lb,ub,nonlcon,opts);
    F = fun(x);