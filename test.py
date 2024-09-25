from main import get_closest_idx
import torch
import unittest


class QuantTest(unittest.TestCase):
    def test_closest_to_simple_grid(self):
        x = torch.tensor([
            0.0, 0.4, 0.6, 1.0,
        ], dtype=torch.float16)
        grid = torch.tensor([0.0, 1.0], dtype=torch.float16)

        torch.testing.assert_close(
            get_closest_idx(x, grid),
            torch.tensor([0, 0, 1, 1])
        )

    def test_closest_to_larger_grid(self):
        x = torch.tensor([
            -1.0, 0.0, 0.09, 0.11, 0.39, 0.41, 0.79, 0.81, 1.0, 2.0,
        ], dtype=torch.float16)
        grid = torch.tensor([0.0, 0.2, 0.6, 1.0], dtype=torch.float16)

        print(get_closest_idx(x, grid))

        torch.testing.assert_close(
            get_closest_idx(x, grid),
            torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 3])
        )


if __name__ == '__main__':
    unittest.main()
