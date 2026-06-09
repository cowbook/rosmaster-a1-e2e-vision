# Training data format

## CSV schema
The training CSV must include:

- `image_path`: image path relative to `--image-root` (or absolute path)
- `steering`: normalized steering in `[-1, 1]`
- `go`: `1` for drive samples, `0` for stop samples

Example: see `training/sample_dataset.csv`.

## Notes for Ackermann platforms
- Keep steering labels in normalized rack command space `[-1, 1]`.
- At runtime the normalized steering is converted to platform steering angle by `max_steering_abs` and filtered by low-pass + rate limit.
- Steering labels for `go=0` samples are ignored by masked loss.

## Dataset collection minimum
- Capture lane following and turns at fixed low speed.
- Include start/stop states and clear stop markers.
- Add low-light and bright-light passes for robustness.
