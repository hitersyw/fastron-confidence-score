function [t] = loadDataGenerationTime(path, n)
    t = zeros(3, 1);
    load(sprintf(path + "reachability_time_n%d.mat", n));
    t(1) = reachbility_time; 
    load(sprintf(path + "self_collision_time_n%d.mat", n));
    t(2) = self_collision_time; 
    load(sprintf(path + "env_collision_time_n%d.mat", n));
    t(3) = env_collision_time; 
end