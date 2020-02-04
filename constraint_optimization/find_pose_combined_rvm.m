function [x,fval,exitflag, output] = find_pose_combined_rvm(x0, X, g, lb, ub, self_collision_param, reachability_param, ...
            env_collision_param, scale_input, self_collision_scale_output, reachability_scale_output)
    
    % alphas = [1, 1, 1];
    % Function for computing the optimal pose, with a start configuration;
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
    
    options = optimoptions('fmincon',...
    'Algorithm','sqp','OptimalityTolerance',1e-6);
    [x,fval,exitflag, output] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,[],options);