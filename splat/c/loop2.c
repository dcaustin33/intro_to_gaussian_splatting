torch::Tensor render_pixel_batch(
    const torch::Tensor& points_mean,
    const torch::Tensor& pixel_coords,
    const torch::Tensor& inverse_covariance,
    const torch::Tensor& pixel_colors,
    const torch::Tensor& opacities)
{
    int batch_size = pixel_coords.size(0);
    int num_points = points_mean.size(0);

    // Expand and repeat tensors for batch processing
    auto points_mean_expanded = points_mean.unsqueeze(0).expand({batch_size, num_points, 2});
    auto pixel_coords_expanded = pixel_coords.unsqueeze(1).expand({batch_size, num_points, 2});
    auto diff = points_mean_expanded - pixel_coords_expanded;

    auto weight = torch::exp(-0.5 * (diff.unsqueeze(3).matmul(inverse_covariance).matmul(diff.unsqueeze(2))).squeeze());

    auto alpha = weight * torch::sigmoid(opacities).unsqueeze(0).expand_as(weight);
    auto total_weight = torch::ones({batch_size}, torch::kFloat32).to(points_mean.device());
    auto pixel_color = torch::zeros({batch_size, 3}, torch::kFloat32).to(points_mean.device());

    for (int i = 0; i < num_points; ++i)
    {
        auto current_alpha = alpha.index({torch::indexing::Slice(), i});
        auto test_weight = total_weight * (1 - current_alpha);

        // Mask to break the loop early for some pixels
        auto mask = test_weight < 0.0001;
        if (mask.sum().item<int>() == batch_size)
        {
            break;
        }

        pixel_color += total_weight.unsqueeze(1) * current_alpha.unsqueeze(1) * pixel_colors.index({i}).unsqueeze(0);
        total_weight = test_weight.masked_fill(mask, 0);
    }

    return pixel_color;
}

torch::Tensor render_tile_batch(
    int x_min,
    int y_min,
    const torch::Tensor& points_mean,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& inverse_covariances,
    int tile_size)
{
    auto pixel_coords = torch::stack(torch::meshgrid(
        torch::arange(x_min, x_min + tile_size, torch::kFloat32),
        torch::arange(y_min, y_min + tile_size, torch::kFloat32)), -1).reshape({-1, 2}).to(points_mean.device());

    return render_pixel_batch(points_mean, pixel_coords, inverse_covariances, colors, opacities)
        .reshape({tile_size, tile_size, 3});
}

torch::Tensor render_image(
    const torch::Tensor& points_min_x,
    const torch::Tensor& points_max_x,
    const torch::Tensor& points_min_y,
    const torch::Tensor& points_max_y,
    const torch::Tensor& points_mean,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& inverse_covariances)
{
    int height = 3200;
    int width = 3200;
    int tile_size = 16;

    auto image = torch::zeros({height, width, 3}, torch::kFloat32).to(points_mean.device());

    auto render_tile_fn = [&](int x, int y)
    {
        auto x_in_tile = (points_min_x < x + tile_size) & (points_max_x > x);
        if (x_in_tile.sum().item<int>() == 0)
        {
            return;
        }

        auto y_in_tile = (points_min_y < y + tile_size) & (points_max_y > y);
        if (y_in_tile.sum().item<int>() == 0)
        {
            return;
        }

        auto in_tile = y_in_tile & x_in_tile;

        auto tile_points_mean = points_mean.index({in_tile});
        auto tile_colors = colors.index({in_tile});
        auto tile_opacities = opacities.index({in_tile});
        auto tile_inverse_covariances = inverse_covariances.index({in_tile});

        auto tile = render_tile_batch(x, y, tile_points_mean, tile_colors, tile_opacities, tile_inverse_covariances, tile_size);

        int y_end = std::min(y + tile_size, height);
        int x_end = std::min(x + tile_size, width);

        image.index_put_({torch::indexing::Slice(x, x_end), torch::indexing::Slice(y, y_end)},
                         tile.index({torch::indexing::Slice(0, x_end - x), torch::indexing::Slice(0, y_end - y)}));
    };

    // Parallel processing using multiple threads
    #pragma omp parallel for collapse(2)
    for (int x = 2000; x < width; x += tile_size)
    {
        for (int y = 0; y < height; y += tile_size)
        {
            render_tile_fn(x, y);
        }
    }

    return image;
}
