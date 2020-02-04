function [x,fval,exitflag, output] = find_pose_combined(x0, lb, ub, self_collision_mdl, reachability_mdl, ...
            env_collision_mdl, scale_input, self_collision_scale_output, reachability_scale_output)
    
    % alphas = [1, 1, 1];
    % Function for computing the optimal pose, with a start configuration;
    fprintf("Initial position: [%.2f, %.2f, %.2f]\n", x0);
    fprintf("Upper bound:[%.2f, %.2f, %.2f]; Lower bound:[%.2f, %.2f, %.2f]", ub, lb);
    
    
    fun = @(x) -(self_collision_scale_output(predict(self_collision_mdl, scale_input(x)))...
          + reachability_scale_output(predict(reachability_mdl, scale_input(x)))...
          + predict(env_collision_mdl, scale_input(x))...
          );
        
    % Add inequality constraints if needed;
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    
    options = optimoptions('fmincon',...
    'Algorithm','sqp','OptimalityTolerance',1e-6);
    [x,fval,exitflag, output] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,[],options);