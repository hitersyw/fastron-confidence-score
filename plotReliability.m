% Reliability plot for discriminative models;
function plotReliability(prob, y_true, num_bins, model)
  res = 1./ num_bins; 
  bins = ceil(prob/res);
  predicted = zeros(size(prob));
  fraction = zeros(size(prob));
  counts = zeros(size(prob));
  % Update the predicted_sum, fraction of positives and counts;
  for j = 1:size(model)
      for i = 1:size(prob, 1)
          predicted(bins(i,j), j) = predicted(bins(i,j), j) + prob(i,j);
          if y_true(i) == 1
              fraction(bins(i,j),j) = fraction(bins(i,j), j) + 1;
          end
          counts(bins(i,j)) = counts(bins(i,j)) + 1;
      end
  end
  
  figure('NumberTitle', 'off', 'Name', 'Reliability plots');
  
  % plot reliability curve for each model;
  subplot(2,1,1);
  a1 = [];
  l1 = [];
  for j = 1:size(model,2)
      a = plot(predicted(:,j)./counts(:,j), fraction(:,j)./counts(:,j));
      l = model(j);
      a1 = [a1; a];
      l1 = [l1; l];
      hold on;
  end
  a = plot(linspace(0, 1, num_bins), linspace(0, 1, num_bins), '--');
  l = "Perfectly Calibrated";
  a1 = [a1; a];
  l1 = [l1; l];
  legend(a1, l1);
  ylabel('fraction');
  
  % plot histogram;
  subplot(2,1,2);
  a2 = [];
  l2 = [];
  for j = 1:size(model, 2)
      b = histogram(prob(:,j), num_bins);
      l = model(j);
      a2 = [a2; b];
      l2 = [l2; l];
      hold on;
  end
  legend(a2, l2);
  xlabel('mean predicted values');
  ylabel('counts');
end
