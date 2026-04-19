import json
import unittest
from io import BytesIO
from lq import (
    parse_size,
    _mask_api_key,
    _process_attachment_data,
    _parse_sse_line,
    _handle_streaming,
)

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


class TestSSEParsing(unittest.TestCase):
    def test_standard_format(self):
        payload = {"choices": [{"delta": {"content": "Hello"}}]}
        line = f'data: {json.dumps(payload)}'
        is_done, content = _parse_sse_line(line)
        self.assertFalse(is_done)
        self.assertEqual(content, "Hello")

    def test_no_space_format(self):
        payload = {"choices": [{"delta": {"content": "World"}}]}
        line = f'data:{json.dumps(payload)}'
        is_done, content = _parse_sse_line(line)
        self.assertFalse(is_done)
        self.assertEqual(content, "World")

    def test_done_signal(self):
        result = _parse_sse_line("data: [DONE]")
        is_done, content = result
        self.assertTrue(is_done)
        self.assertIsNone(content)

    def test_non_data_line(self):
        result = _parse_sse_line("id: 123")
        self.assertIsNone(result)

    def test_empty_content(self):
        payload = {"choices": [{"delta": {}}]}
        line = f'data: {json.dumps(payload)}'
        result = _parse_sse_line(line)
        self.assertIsNone(result)

    def test_no_choices(self):
        payload = {"choices": []}
        line = f'data: {json.dumps(payload)}'
        result = _parse_sse_line(line)
        self.assertIsNone(result)

    def test_invalid_json(self):
        result = _parse_sse_line("data: not json")
        self.assertIsNone(result)


class TestStreaming(unittest.TestCase):
    def _make_mock_resp(self, lines):
        class MockResp:
            def __init__(self, data_lines):
                self.data_lines = data_lines
            def __iter__(self):
                return iter(self.data_lines)
        return MockResp(lines)

    def test_basic_streaming(self):
        payload1 = {"choices": [{"delta": {"content": "Hello"}}]}
        payload2 = {"choices": [{"delta": {"content": " World"}}]}
        lines = [
            f'data: {json.dumps(payload1)}\n'.encode(),
            f'data:{json.dumps(payload2)}\n'.encode(),
            b'\ndata: [DONE]\n',
        ]
        result = _handle_streaming(self._make_mock_resp(lines), False, False)
        self.assertEqual(result, "Hello World")

    def test_empty_lines_skipped(self):
        payload = {"choices": [{"delta": {"content": "A"}}]}
        lines = [
            f'data: {json.dumps(payload)}\n'.encode(),
            b'\n',
            b'\ndata: [DONE]\n',
        ]
        result = _handle_streaming(self._make_mock_resp(lines), False, False)
        self.assertEqual(result, "A")

    def test_json_output_mode(self):
        lines = [b'data: {"choices": [{"delta": {"content": "X"}}]}\n', b'\ndata: [DONE]\n']
        result = _handle_streaming(self._make_mock_resp(lines), False, True)
        parsed = json.loads(result)
        self.assertIsInstance(parsed, list)

    def test_no_content_delta(self):
        payload = {"choices": [{"delta": {}}]}
        lines = [
            f'data: {json.dumps(payload)}\n'.encode(),
            b'\ndata: [DONE]\n',
        ]
        result = _handle_streaming(self._make_mock_resp(lines), False, False)
        self.assertEqual(result, "")

    def test_multiple_chunks(self):
        chunks = []
        for i in range(5):
            payload = {"choices": [{"delta": {"content": str(i)}}]}
            chunks.append(f'data: {json.dumps(payload)}\n'.encode())
        chunks.append(b'\ndata: [DONE]\n')
        result = _handle_streaming(self._make_mock_resp(chunks), False, False)
        self.assertEqual(result, "01234")


if __name__ == "__main__":
    unittest.main()
