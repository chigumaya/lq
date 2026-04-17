import unittest
from lq import parse_size, _mask_api_key, _process_attachment_data

class TestLQ(unittest.TestCase):
    def test_parse_size(self):
        self.assertEqual(parse_size("1024"), 1024)
        self.assertEqual(parse_size("5MB"), 5 * 1024 * 1024)
        self.assertEqual(parse_size("10KB"), 10 * 1024)
        self.assertEqual(parse_size("1GB"), 1024 * 1024 * 1024)
        with self.assertRaises(ValueError):
            parse_size("-1MB")
        with self.assertRaises(ValueError):
            parse_size("abc")

    def test_mask_api_key(self):
        self.assertEqual(_mask_api_key("sk-1234567890"), "sk-123...7890")
        self.assertEqual(_mask_api_key("short"), "****")
        self.assertEqual(_mask_api_key(""), "****")

    def test_process_attachment_data(self):
        # UTF-8 text
        text_data = b"hello"
        result = _process_attachment_data(text_data, "stdin")
        self.assertIn("encoding=\"utf-8\"", result)
        self.assertIn("hello", result)

        # Binary data
        binary_data = b"\xff\xd8\xff"
        result = _process_attachment_data(binary_data, "stdin")
        self.assertIn("encoding=\"base64\"", result)

if __name__ == "__main__":
    unittest.main()
