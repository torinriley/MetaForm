import unittest
from unittest.mock import Mock
import sys
from tools.training.gradient_checkpointing import GradientCheckpointing



class TestGradientCheckpointing(unittest.TestCase):

    def setUp(self):
        self.mock_model = Mock()
        self.checkpointing = GradientCheckpointing(self.mock_model)

    def test_checkpoint_forward_pass(self):
        mock_function = Mock()
        inputs = [Mock(), Mock()]

        mock_output = Mock()
        mock_function.return_value = mock_output

        output, backward_fn = self.checkpointing.checkpoint(mock_function, *inputs)

        mock_function.assert_called_with(*inputs)
        self.assertEqual(output, mock_output)

    def test_checkpoint_backward_pass(self):
        mock_function = Mock()
        inputs = [Mock(), Mock()]

        mock_output = Mock()
        mock_function.return_value = mock_output

        _, backward_fn = self.checkpointing.checkpoint(mock_function, *inputs)

        mock_grad_output = Mock()
        mock_function.return_value.backward.return_value = [Mock() for _ in inputs]

        grads = backward_fn(mock_grad_output)

        mock_function.assert_called_with(*inputs)
        self.assertTrue(all(isinstance(grad, Mock) for grad in grads))

    def test_save_and_clear_tensors(self):
        mock_tensors = [Mock(), Mock()]

        self.checkpointing.save_for_backward(*mock_tensors)
        self.checkpointing.clear_saved_tensors()

        self.assertIsNone(self.checkpointing.saved_tensors)

if __name__ == '__main__':
    unittest.main()