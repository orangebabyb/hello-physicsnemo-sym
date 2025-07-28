% --- Parameters ---
L = 1;
alpha = 0.01;
Nx = 64; Ny = 64; Nt = 480;
x = linspace(0, L, Nx);
y = linspace(0, L, Ny);
t = linspace(0, 3.0, Nt);
[X, Y, T] = meshgrid(x, y, t);

% --- Analytical ssolution ---
k = 2 * pi;
U = sin(k * X) .* sin(k * Y) .* exp(-2 * alpha * k^2 * T);

target_time = 0;
[~, t_idx] = min(abs(t - target_time));
U_slice = U(:, :, t_idx);

% --- figure ---
figure('Position', [100, 100, 1600, 600]);

% --- 3D Plot ---
subplot('Position', [0.06, 0.15, 0.35, 0.75]);  % left, bottom, width, height
surf(x, y, U_slice');
xlabel('x', 'FontSize', 12); ylabel('y', 'FontSize', 12); zlabel('u(x, y, t)', 'FontSize', 12);
title(['3D View of Analytical Solution at t = ', num2str(t(t_idx))], 'FontWeight', 'bold', 'FontSize', 8);
view(45, 30);
pbaspect([1 1 0.7]);
cb1 = colorbar;
cb1.Position = [0.42, 0.2, 0.01, 0.6];

% --- 2D Plot ---
subplot('Position', [0.55, 0.15, 0.35, 0.75]);
imagesc(x, y, U_slice');
set(gca, 'YDir', 'normal');
xlabel('x', 'FontSize', 12); ylabel('y', 'FontSize', 12);
title(['2D Heatmap of Solution at t = ', num2str(t(t_idx))], 'FontWeight', 'bold', 'FontSize', 8);
axis equal tight;
cb2 = colorbar;
cb2.Position = [0.92, 0.2, 0.01, 0.6];

% Save to .mat file
X_flat = reshape(X, [], 1);
Y_flat = reshape(Y, [], 1);
T_flat = reshape(T, [], 1);
U_flat = reshape(U, [], 1);
save('heat_ground_truth.mat', 'X_flat', 'Y_flat', 'T_flat', 'U_flat');