function [model_rmse, max_absolute_error, x, errors_x_domain] = erf_nonlinear()
%erf_fitting using linear least squares to fit error function

% Points for evaluation
x = -10:0.1:10;

% Evaluate error function at points for fitting
y_true = erf(x);

% Evaluate error function more generally for error test points
errors_x_domain = -10:0.001:10;
true_erf_values = erf(errors_x_domain);


% Create exponential model
exponential_model = @(z) 1 + exp(-(z-1).^2) * 1  + exp(-(z-1).^2) .* z.^2 + ...
    exp(-(z-1).^2) .* z.^2 + exp(-(z-1).^2) .* z.^3;

exp_figure = figure;
exp_axes = axes;
hold(exp_axes);

exp_fit = linlsqfit(x, y_true, exponential_model);

% Evaluate model at error testing points
coefficients = splitfunction(exponential_model);
value_evaluations = 0;
for term=1:5
    value_evaluations = value_evaluations + ...
        exp_fit(term) * coefficients{term}(errors_x_domain);
end

model_evaluation = value_evaluations;

% Compute Error
exponential_model_error = model_evaluation - true_erf_values;

model_rmse = rms(exponential_model_error);
max_absolute_error = max(abs(exponential_model_error));

plot(exp_axes, errors_x_domain, log(abs(exponential_model_error)), 'LineWidth', 4);

func_title = title('Exponential Fit Error');
set(func_title, 'FontSize', 18);

x_label = xlabel('t');
y_label_ = ylabel(['log|erf(t) - c_1 + e^{-(z-1)^2} ' ...
    '(c_2 + c_3 z + c_4 z^2 + c_5 z^3)|']);
set(x_label, 'FontSize', 12);
set(x_label, 'FontSize', 12);

% Present axes for visual
hold(exp_axes);

    
end

