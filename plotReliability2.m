% Reliability plot for discriminative models;
function plotReliability2(prob, y_true, num_bins, model)
  res = 1./ num_bins; 
  prob(prob == 0)=0.1^(12); % infinitestimal jitter;
  bins = ceil(prob/res);
  n = size(prob, 2);
  predicted = zeros(num_bins, n);
  fraction = zeros(num_bins, n);
  counts = zeros(num_bins, n);
  % Update the predicted_sum, fraction of positives and counts;
  for i=1:size(prob, 1)
      for j=1:size(prob, 2)
          predicted(bins(i,j),j) = predicted(bins(i,j),j) + prob(i,j);
          if y_true(i) == 1
              fraction(bins(i,j),j) = fraction(bins(i,j),j) + 1;
          end
          counts(bins(i,j), j) = counts(bins(i,j), j) + 1;
      end
  end
  
  figure('NumberTitle', 'off', 'Name', 'Reliability plots');
  
  for j = 1:(n/2)
    % reliability curve;
    subplot(n,4,1+4*(j-1));
    plot(predicted(:,2*j-1)./counts(:,2*j-1), fraction(:,2*j-1)./counts(:,2*j-1));
    hold on;
    plot(linspace(0,1, num_bins), linspace(0, 1, num_bins), '--');
    ylabel(model(j));
    
    % histogram counts;
    subplot(n,4,2+4*(j-1));
    histogram(prob(:,2*j-1), 'NumBins', num_bins, 'DisplayStyle','stairs');
    
    % reliability curve calibrated;
    subplot(n,4,3+4*(j-1));
    plot(predicted(:,2*j)./counts(:,2*j), fraction(:,2*j)./counts(:,2*j));
    hold on;
    plot(linspace(0,1,num_bins), linspace(0,1,num_bins), '--');
        
    % histogram counts calibrated;
    subplot(n,4,4+4*(j-1));
    histogram(prob(:,2*j), 'NumBins', num_bins, 'DisplayStyle','stairs');
  end
end
