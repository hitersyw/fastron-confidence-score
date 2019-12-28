function h = colorScatter3(X, Y, Z, V, cm)
V = (V-min(V))/(max(V) - min(V));

cm_idx = uint32(round((size(cm,1)-1)*V)+1);
for i = 1:length(X)
    h(i) = plot3(X(i), Y(i), Z(i), '.', 'color', cm(cm_idx(i), :)); hold on;
end
end