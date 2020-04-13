function [] = saveDvrk(X, y, output_path, dataset, n, use_fastron)
    if use_fastron
        fastron_suffix = "_fastron";
    else
        fastron_suffix = "";
    end
    
    data = [X, y];
    S.(dataset) = data;  % save data in the name stored in dataset; 
    filename = sprintf(output_path, dataset + fastron_suffix, n);
    fprintf("Saving to file: %s\n", filename);
    save(filename, '-struct', 'S');
end