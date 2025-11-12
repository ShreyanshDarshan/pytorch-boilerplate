import numpy as np
import gin

from torch.utils.data import Dataset, DataLoader

@gin.configurable
class PolynomialDataset(Dataset):
    def __init__(
        self,
        split='train',
        test_ratio=0.1,
        val_ratio=0.1,
        polynomial_coeffs = [1, 2, 3, 4, 5, 6]
        ):
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.split = split
        self.polynomial_coeffs = polynomial_coeffs
        self.data = []
        
        self.inputs_x = np.linspace(-1, 1, 100).astype(np.float32)
        self.inputs_y = np.linspace(-1, 1, 100).astype(np.float32)

        self.inputs = np.stack(np.meshgrid(self.inputs_x, self.inputs_y), -1).reshape(-1, 2)  # [N, 2]
        self.data = self.evaluate_polynomial(self.inputs)  # [N, 1]

        val_size = int(len(self.inputs) * val_ratio)
        test_size = int(len(self.inputs) * test_ratio)
        if 'split' == 'val':
            self.inputs = self.inputs[:val_size]
            self.data = self.data[:val_size]
        elif 'split' == 'test':
            self.inputs = self.inputs[-test_size:]
            self.data = self.data[-test_size:]
        else:
            self.inputs = self.inputs[val_size:-test_size]
            self.data = self.data[val_size:-test_size]


    def __len__(self):
        if 'train' in self.split:
            return len(self.inputs) * 1000
        elif 'val' in self.split:
            return len(self.inputs)
        elif 'test' in self.split:
            return len(self.inputs)
        else:
            print(f'Invalid split : {self.split}')
            exit(0)


    def evaluate_polynomial(self, inputs):
        """Evaluate polynomial at given inputs.

        Args:
            inputs (np.ndarray): Input array of shape [N, 2] where each row is (x, y).

        Returns:
            np.ndarray: Output array of shape [N, 1] with polynomial values.
        """
        x = inputs[:, 0]
        y = inputs[:, 1]
        z = np.zeros_like(x)

        # Evaluate polynomial using the coefficients
        for i, coeff in enumerate(self.polynomial_coeffs):
            z += coeff * (x ** i) * (y ** (len(self.polynomial_coeffs) - 1 - i))

        return z.reshape(-1, 1)

    def __getitem__(self, idx):
        """Get item at the given index."""
        
        # Wrap around if needed
        idx = idx % len(self.inputs)
        data = self.data[idx]

        # Prepare output dictionary
        ret = {
            "inputs": self.inputs[idx],  # (x, y)
            "output": data,              # polynomial value at (x, y)
        }

        return ret