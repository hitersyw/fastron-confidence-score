function [t] = load_data_generation_time(path, use_fastron, n)
    if use_fastron
        fastron_suffix = "_fastron";
    else
        fastron_suffix = "";
    end
    t = zeros(3, 1);
    load(sprintf(path + "reachability_time%s_n%d.mat", fastron_suffix, n));
    t(1) = reachbility_time; 
    load(sprintf(path + "self_collision_time%s_n%d.mat", fastron_suffix, n));
    t(2) = self_collision_time; 
    load(sprintf(path + "env_collision_time%s_n%d.mat", fastron_suffix, n));
    t(3) = env_collision_time; 
end