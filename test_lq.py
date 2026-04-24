import json
import os
import sys
import tempfile
import unittest
from io import BytesIO
from unittest.mock import patch, MagicMock
from lq import (
    parse_size,
    _mask_api_key,
    _process_attachment_data,
    _parse_sse_line,
    _handle_streaming,
    resolve_template,
    load_templates,
    build_payload,
    assemble_prompt,
    call_api,
    _try_api_call,
    Config,
    ChatSession,
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


class TestResolveTemplate(unittest.TestCase):
    def test_no_placeholders(self):
        result = resolve_template("explain", "explain this code", [], [])
        self.assertEqual(result, "explain this code")

    def test_single_placeholder_with_default(self):
        result = resolve_template("sum", "summarize in %s lines", [3], [])
        self.assertEqual(result, "summarize in 3 lines")

    def test_single_placeholder_cli_override(self):
        result = resolve_template("sum", "summarize in %s lines", [3], ["5"])
        self.assertEqual(result, "summarize in 5 lines")

    def test_multiple_placeholders_all_provided(self):
        result = resolve_template("count", "count \"%s\" and \"%s\"", [], ["Python", "Java"])
        self.assertEqual(result, 'count "Python" and "Java"')

    def test_mixed_defaults_none_and_value(self):
        result = resolve_template("test", "%s is %s", [None, "great"], ["AI"])
        self.assertEqual(result, "AI is great")

    def test_all_cli_args_override_defaults(self):
        result = resolve_template("sum", "explain %s in %s words", [100, "simple"], ["Python", "50"])
        self.assertEqual(result, "explain Python in 50 words")

    def test_missing_required_arg_raises_error(self):
        with self.assertRaises(SystemExit):
            resolve_template("count", "count \"%s\" and \"%s\"", [], ["Python"])

    def test_missing_default_value_raises_error(self):
        with self.assertRaises(SystemExit):
            resolve_template("sum", "in %s lines", [None], [])

    def test_numeric_default_converted_to_string(self):
        result = resolve_template("sum", "in %s lines", [3], [])
        self.assertEqual(result, "in 3 lines")

    def test_multiple_placeholders_with_partial_defaults(self):
        # 3 placeholders, defaults [auto, None] -> arg 1 (first), arg 2 (auto), arg 3 (None -> missing)
        # Wait, the prompt has 3 placeholders. defaults has 2.
        # This test case seems fundamentally flawed if it expects "None" as a string.
        # Let's fix the test expectation to match the logic (error on missing).
        with self.assertRaises(SystemExit):
            resolve_template("test", "%s and %s and %s", ["auto", None], ["first"])

    def test_extra_cli_args_raises_error(self):
        with self.assertRaises(SystemExit):
            resolve_template("sum", "summarize in %s lines", [], ["5", "extra_arg"])


class TestLoadTemplates(unittest.TestCase):
    def test_no_templates_key(self):
        result = load_templates({})
        self.assertEqual(result, {})

    def test_empty_templates_array(self):
        result = load_templates({"templates": []})
        self.assertEqual(result, {})

    def test_valid_template_entry(self):
        config = {"templates": [{"name": "sum", "prompt": "summarize %s"}]}
        result = load_templates(config)
        self.assertIn("sum", result)
        prompt, defaults = result["sum"]
        self.assertEqual(prompt, "summarize %s")
        self.assertEqual(defaults, [])

    def test_template_with_defaults(self):
        config = {"templates": [{"name": "sum", "prompt": "%s lines", "defaults": [3]}]}
        result = load_templates(config)
        prompt, defaults = result["sum"]
        self.assertEqual(prompt, "%s lines")
        self.assertEqual(defaults, [3])

    def test_multiple_templates(self):
        config = {
            "templates": [
                {"name": "a", "prompt": "first %s"},
                {"name": "b", "prompt": "second %s"}
            ]
        }
        result = load_templates(config)
        self.assertIn("a", result)
        self.assertIn("b", result)

    def test_template_with_null_default(self):
        config = {"templates": [{"name": "t", "prompt": "%s and %s", "defaults": [None, "foo"]}]}
        result = load_templates(config)
        prompt, defaults = result["t"]
        self.assertEqual(defaults, [None, "foo"])

    def test_invalid_templates_not_array(self):
        with self.assertRaises(SystemExit):
            load_templates({"templates": "invalid"})

    def test_template_missing_name_raises_error(self):
        with self.assertRaises(SystemExit):
            load_templates({"templates": [{"prompt": "no name"}]})

    def test_template_prompt_must_be_string(self):
        with self.assertRaises(SystemExit):
            load_templates({"templates": [{"name": "t", "prompt": 123}]})

    def test_template_defaults_must_be_array(self):
        with self.assertRaises(SystemExit):
            load_templates({"templates": [{"name": "t", "prompt": "x", "defaults": "not array"}]})


class TestBuildPayloadWithChatSession(unittest.TestCase):
    def test_build_payload_with_session(self):
        cfg = Config(
            api_url="http://localhost:1234/v1",
            api_key="test-key",
            model="test-model",
            system_prompt="You are helpful.",
            files=[],
            images=[],
            prompt=["Hello"],
            max_size=10 * 1024 * 1024,
            output_json=False,
            debug=False,
            stream=True,
        )

        session = ChatSession()
        session.add_user_message([{"type": "text", "text": "Previous message"}])
        session.add_assistant_message("Previous response")

        user_content = [{"type": "text", "text": "New message"}]
        payload_bytes = build_payload(cfg, user_content, session)
        payload = json.loads(payload_bytes)

        self.assertEqual(payload["model"], "test-model")
        self.assertTrue(payload["stream"])

        messages = payload["messages"]
        self.assertEqual(len(messages), 4)  # system + previous user + previous assistant + new user
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are helpful.")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], [{"type": "text", "text": "Previous message"}])
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "Previous response")
        self.assertEqual(messages[3]["role"], "user")
        self.assertEqual(messages[3]["content"], user_content)  # New message as content array

    def test_basic_payload(self):
        cfg = Config(
            api_url="http://localhost:1234/v1",
            api_key="test-key",
            model="test-model",
            system_prompt="You are helpful.",
            files=[],
            images=[],
            prompt=["Hello"],
            max_size=10 * 1024 * 1024,
            output_json=False,
            debug=False,
            stream=True,
        )
        session = ChatSession()
        user_content = [{"type": "text", "text": "Hello"}]
        payload_bytes = build_payload(cfg, user_content, session)
        payload = json.loads(payload_bytes)

        self.assertEqual(payload["model"], "test-model")
        self.assertTrue(payload["stream"])
        messages = payload["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are helpful.")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], user_content)

    def test_payload_no_system_prompt(self):
        cfg = Config(
            api_url="http://localhost:1234/v1",
            api_key="",
            model="test-model",
            system_prompt=None,
            files=[],
            images=[],
            prompt=["Hello"],
            max_size=10 * 1024 * 1024,
            output_json=False,
            debug=False,
            stream=True,
        )
        session = ChatSession()
        user_content = [{"type": "text", "text": "Hello"}]
        payload_bytes = build_payload(cfg, user_content, session)
        payload = json.loads(payload_bytes)

        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["role"], "user")

    def test_payload_stream_false(self):
        cfg = Config(
            api_url="http://localhost:1234/v1",
            api_key="",
            model="test-model",
            system_prompt=None,
            files=[],
            images=[],
            prompt=["Hello"],
            max_size=10 * 1024 * 1024,
            output_json=False,
            debug=False,
            stream=False,
        )
        session = ChatSession()
        user_content = [{"type": "text", "text": "Hello"}]
        payload_bytes = build_payload(cfg, user_content, session)
        payload = json.loads(payload_bytes)

        self.assertFalse(payload["stream"])

    def test_payload_multiple_user_contents(self):
        cfg = Config(
            api_url="http://localhost:1234/v1",
            api_key="",
            model="test-model",
            system_prompt=None,
            files=[],
            images=[],
            prompt=["Hello"],
            max_size=10 * 1024 * 1024,
            output_json=False,
            debug=False,
            stream=True,
        )
        session = ChatSession()
        user_content = [
            {"type": "text", "text": "First block"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        payload_bytes = build_payload(cfg, user_content, session)
        payload = json.loads(payload_bytes)

        messages = payload["messages"]
        self.assertEqual(len(messages), 2)
        # Each content block gets its own message
        self.assertEqual(messages[0]["content"], [{"type": "text", "text": "First block"}])
        self.assertEqual(messages[1]["content"], [user_content[1]])


class TestAssemblePromptStdin(unittest.TestCase):
    def test_stdin_read_when_not_tty(self):
        cfg = Config(
            api_url="http://localhost:1234/v1",
            api_key="",
            model="test-model",
            system_prompt=None,
            files=[],
            images=[],
            prompt=["Hello"],
            max_size=10 * 1024 * 1024,
            output_json=False,
            debug=False,
            stream=True,
        )

        stdin_data = b"stdin content here"
        with patch("sys.stdin", MagicMock()) as mock_stdin:
            mock_stdin.isatty.return_value = False
            mock_stdin.buffer.read.return_value = stdin_data

            result = assemble_prompt(cfg)

        # Should contain stdin attachment + prompt
        self.assertEqual(len(result), 2)
        self.assertIn("stdin content here", result[0]["text"])
        self.assertIn('source="stdin"', result[0]["text"])
        self.assertEqual(result[1]["text"], "Hello")

    def test_stdin_empty_when_tty(self):
        cfg = Config(
            api_url="http://localhost:1234/v1",
            api_key="",
            model="test-model",
            system_prompt=None,
            files=[],
            images=[],
            prompt=["Hello"],
            max_size=10 * 1024 * 1024,
            output_json=False,
            debug=False,
            stream=True,
        )

        with patch("sys.stdin", MagicMock()) as mock_stdin:
            mock_stdin.isatty.return_value = True

            result = assemble_prompt(cfg)

        # Should only contain prompt (no stdin attachment)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Hello")

    def test_stdin_exceeds_size_limit(self):
        cfg = Config(
            api_url="http://localhost:1234/v1",
            api_key="",
            model="test-model",
            system_prompt=None,
            files=[],
            images=[],
            prompt=["Hello"],
            max_size=5,  # Very small limit for testing
            output_json=False,
            debug=False,
            stream=True,
        )

        stdin_data = b"this is way too long"
        with patch("sys.stdin", MagicMock()) as mock_stdin:
            mock_stdin.isatty.return_value = False
            mock_stdin.buffer.read.return_value = stdin_data

            with self.assertRaises(SystemExit):
                assemble_prompt(cfg)


class TestTryApiCallErrors(unittest.TestCase):
    def test_http_error(self):
        from urllib.error import HTTPError

        req_mock = MagicMock()
        req_mock.add_header = MagicMock()

        http_err = HTTPError(
            "http://example.com", 401, "Unauthorized", {}, BytesIO(b'{"error": "invalid key"}')
        )

        with patch("lq.urllib.request.Request", return_value=req_mock):
            with patch("lq.urllib.request.urlopen", side_effect=http_err):
                with self.assertRaises(SystemExit):
                    _try_api_call(
                        endpoint="http://example.com/chat/completions",
                        payload=b'{"model": "test"}',
                        auth_header="Bearer test-key",
                        debug=False,
                        output_json=False,
                        stream=False,
                    )

    def test_url_error(self):
        from urllib.error import URLError

        req_mock = MagicMock()
        req_mock.add_header = MagicMock()

        url_err = URLError("Connection refused")

        with patch("lq.urllib.request.Request", return_value=req_mock):
            with patch("lq.urllib.request.urlopen", side_effect=url_err):
                with self.assertRaises(SystemExit):
                    _try_api_call(
                        endpoint="http://example.com/chat/completions",
                        payload=b'{"model": "test"}',
                        auth_header=None,
                        debug=False,
                        output_json=False,
                        stream=False,
                    )

    def test_no_choices_in_response(self):
        req_mock = MagicMock()
        req_mock.add_header = MagicMock()

        class MockResp:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
            def read(self):
                return json.dumps({"choices": []}).encode()

        with patch("lq.urllib.request.Request", return_value=req_mock):
            with patch("lq.urllib.request.urlopen", return_value=MockResp()):
                with self.assertRaises(SystemExit):
                    _try_api_call(
                        endpoint="http://example.com/chat/completions",
                        payload=b'{"model": "test"}',
                        auth_header=None,
                        debug=False,
                        output_json=False,
                        stream=False,
                    )

    def test_debug_mode_no_crash(self):
        req_mock = MagicMock()
        req_mock.add_header = MagicMock()

        class MockResp:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": "test response"}}]
                }).encode()

        with patch("lq.urllib.request.Request", return_value=req_mock):
            with patch("lq.urllib.request.urlopen", return_value=MockResp()):
                _try_api_call(
                    endpoint="http://example.com/chat/completions",
                    payload=b'{"model": "test"}',
                    auth_header="Bearer test-key",
                    debug=True,
                    output_json=False,
                    stream=False,
                )


class TestChatSession(unittest.TestCase):
    def test_initialization(self):
        session = ChatSession()
        self.assertEqual(session.history, [])

    def test_add_user_message(self):
        session = ChatSession()
        session.add_user_message([{"type": "text", "text": "Hello"}])
        self.assertEqual(len(session.history), 1)
        self.assertEqual(session.history[0]["role"], "user")
        self.assertEqual(session.history[0]["content"], [{"type": "text", "text": "Hello"}])

    def test_add_assistant_message(self):
        session = ChatSession()
        session.add_assistant_message("Hi there!")
        self.assertEqual(len(session.history), 1)
        self.assertEqual(session.history[0]["role"], "assistant")
        self.assertEqual(session.history[0]["content"], "Hi there!")

    def test_get_messages_with_system_prompt(self):
        session = ChatSession()
        session.add_user_message([{"type": "text", "text": "Hello"}])
        session.add_assistant_message("Hi there!")

        messages = session.get_messages("You are helpful.")
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are helpful.")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], [{"type": "text", "text": "Hello"}])
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "Hi there!")

    def test_get_messages_without_system_prompt(self):
        session = ChatSession()
        session.add_user_message([{"type": "text", "text": "Hello"}])

        messages = session.get_messages("")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], [{"type": "text", "text": "Hello"}])

    def test_add_user_message_with_attachments(self):
        session = ChatSession()
        content = [{"type": "text", "text": "What's in this file?"}]
        content[0]["text"] += "\n\n[Attachments: file:README.md, image:screenshot.png]"
        session.add_user_message(content)
        self.assertEqual(len(session.history), 1)
        self.assertEqual(session.history[0]["role"], "user")
        self.assertIn("[Attachments:", session.history[0]["content"][0]["text"])

    def test_add_user_message_without_attachments(self):
        session = ChatSession()
        session.add_user_message([{"type": "text", "text": "Hello"}])
        self.assertEqual(session.history[0]["content"], [{"type": "text", "text": "Hello"}])


if __name__ == "__main__":
    unittest.main()
