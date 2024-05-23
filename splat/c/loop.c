// torch::Tensor render_pixel(
//     torch::Tensor points_mean,
//     torch::Tensor pixel_coords,
//     torch::Tensor inverse_covariance,
//     torch::Tensor pixel_colors,
//     torch::Tensor opacities)
// {

//     torch::Tensor total_weight = torch::ones({1}, torch::kFloat32);
//     torch::Tensor pixel_color = torch::zeros({3}, torch::kFloat32);
//     for (int i = 0; i < points_mean.size(0); i++)
//     {
//         torch::Tensor point = points_mean[i];
//         torch::Tensor diff = point - pixel_coords;
//         torch::Tensor diff_t = diff.t();
//         torch::Tensor weight = torch::exp(-0.5 * diff_t.matmul(inverse_covariance[i]).matmul(diff));
//         torch::Tensor alpha = weight * torch::sigmoid(opacities[i]);
//         torch::Tensor test_weight = total_weight * (1 - alpha);

//         if (test_weight.item<float>() < 0.0001)
//         {
//             return pixel_color;
//         }
//         pixel_color += total_weight * alpha * pixel_colors[i];
//         total_weight = test_weight;
//     }
//     return pixel_color;
// }

// torch::Tensor render_tile(
//     int x_min,
//     int y_min,
//     torch::Tensor points_mean,
//     torch::Tensor colors,
//     torch::Tensor opacities,
//     torch::Tensor inverse_covariances,
//     int tile_size)
// {
//     torch::Tensor result = torch::zeros({tile_size, tile_size, 3}, torch::kFloat32);
//     for (int x = 0; x < tile_size; x++)
//     {
//         for (int y = 0; y < tile_size; y++)
//         {
//             torch::Tensor pixel_coords = torch::tensor({x_min + x, y_min + y}, torch::kFloat32);
//             result[x][y] = render_pixel(points_mean, pixel_coords, inverse_covariances, colors, opacities);
//         }
//     }
//     return result;
// }

torch::Tensor render_pixel(
    const torch::Tensor& points_mean,
    const torch::Tensor& pixel_coords,
    const torch::Tensor& inverse_covariance,
    const torch::Tensor& pixel_colors,
    const torch::Tensor& opacities)
{
    auto total_weight = torch::ones({1}, torch::kFloat32);
    auto pixel_color = torch::zeros({3}, torch::kFloat32);

    for (int i = 0; i < points_mean.size(0); ++i)
    {
        auto point = points_mean[i];
        auto diff = point - pixel_coords;
        auto weight = torch::exp(-0.5 * diff.matmul(inverse_covariance[i]).matmul(diff.t()));
        auto alpha = weight *opacities[i];
        auto test_weight = total_weight * (1 - alpha);

        if (test_weight.item<float>() < 0.0001)
        {
            break;
        }

        pixel_color += total_weight * alpha * pixel_colors[i];
        total_weight = test_weight;
    }

    return pixel_color;
}


torch::Tensor render_tile(
    int x_min,
    int y_min,
    const torch::Tensor& points_mean,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& inverse_covariances,
    int tile_size)
{
    auto result = torch::zeros({tile_size, tile_size, 3}, torch::kFloat32);
    auto pixel_coords = torch::zeros({tile_size * tile_size, 2}, torch::kFloat32);

    for (int x = 0; x < tile_size; ++x)
    {
        for (int y = 0; y < tile_size; ++y)
        {
            pixel_coords[x * tile_size + y][0] = x_min + x;
            pixel_coords[x * tile_size + y][1] = y_min + y;
        }
    }

    for (int i = 0; i < tile_size * tile_size; ++i)
    {
        result.view({-1, 3})[i] = render_pixel(points_mean, pixel_coords[i], inverse_covariances, colors, opacities);
    }

    return result;
}

torch::Tensor render_image(
    torch::Tensor points_min_x,
    torch::Tensor points_max_x,
    torch::Tensor points_min_y,
    torch::Tensor points_max_y,
    torch::Tensor points_mean,
    torch::Tensor colors,
    torch::Tensor opacities,
    torch::Tensor inverse_covariances)
{
    int height = 3200;
    int width = 3200;

    torch::Tensor image = torch::zeros({height, width, 3}, torch::kFloat32);
    int tile_size = 16;

    for (int x = 2000; x < width; x += tile_size)
    {
        torch::Tensor x_in_tile = (points_min_x < x + tile_size) & (points_max_x > x);
        if (x_in_tile.sum().item<int>() == 0)
        {
            continue;
        }
        for (int y = 0; y < height; y += tile_size)
        {
            torch::Tensor y_in_tile = (points_min_y < y + tile_size) & (points_max_y > y);
            if (y_in_tile.sum().item<int>() == 0)
            {
                continue;
            }
            torch::Tensor in_tile = y_in_tile & x_in_tile;

            // Copy relevant points and attributes into the tile
            torch::Tensor tile_points_mean = points_mean.index({in_tile});
            torch::Tensor tile_colors = colors.index({in_tile});
            torch::Tensor tile_opacities = opacities.index({in_tile});
            torch::Tensor tile_inverse_covariances = inverse_covariances.index({in_tile});

            torch::Tensor tile = render_tile(x, y, tile_points_mean, tile_colors, tile_opacities, tile_inverse_covariances, tile_size);

            // Ensure the tile fits within the image bounds
            int y_end = std::min(y + tile_size, height);
            int x_end = std::min(x + tile_size, width);

            // Use slicing with index_put_ for assignment
            image.index_put_({torch::indexing::Slice(x, x_end), torch::indexing::Slice(y, y_end)},
                             tile.index({torch::indexing::Slice(0, x_end - x), torch::indexing::Slice(0, y_end - y)}));
        }
    }

    return image;
}